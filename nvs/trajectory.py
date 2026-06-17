"""Camera-pose math and novel-view arc generation (COLMAP world-to-camera)."""
from __future__ import annotations

import numpy as np


def qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_center(qvec, tvec) -> np.ndarray:
    R = qvec_to_rotmat(qvec)
    return -R.T @ np.asarray(tvec, dtype=float)


def scene_stats(centers: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (centroid, mean_radius, mean_height) of camera centers."""
    centroid = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - centroid, axis=1).mean())
    height = float(centers[:, 1].mean())
    return centroid, radius, height


def _look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build world-to-camera (R, t) for a camera at `eye` looking at `target`."""
    up = np.array([0.0, 1.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    R = np.stack([right, -true_up, fwd], axis=0)
    t = -R @ eye
    return R, t


def arc_poses_around(
    centroid: np.ndarray, radius: float, base_height: float,
    start_angle: float, n: int, arc_span: float, elevation: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n` poses on an elevated horizontal arc around `centroid`, all looking at it."""
    poses = []
    angles = (np.linspace(0, arc_span, n) + start_angle) if n > 1 else [start_angle]
    for a in angles:
        eye = centroid + np.array([
            radius * np.cos(a),
            base_height + elevation,
            radius * np.sin(a),
        ])
        poses.append(_look_at(eye, centroid))
    return poses
