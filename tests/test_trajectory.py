import numpy as np
from nvs.trajectory import (
    qvec_to_rotmat, camera_center, scene_stats, arc_poses_around,
)


def test_qvec_identity():
    R = qvec_to_rotmat((1.0, 0.0, 0.0, 0.0))
    assert np.allclose(R, np.eye(3))


def test_camera_center_identity_pose():
    c = camera_center((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert np.allclose(c, [-1.0, -2.0, -3.0])


def test_scene_stats_centroid_and_radius():
    centers = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    centroid, radius, height = scene_stats(centers)
    assert np.allclose(centroid, [0, 0, 0])
    assert abs(radius - 1.0) < 1e-6


def test_arc_poses_count_and_look_at_centroid():
    poses = arc_poses_around(
        centroid=np.zeros(3), radius=2.0, base_height=0.0,
        start_angle=0.0, n=4, arc_span=np.pi / 2, elevation=0.5,
    )
    assert len(poses) == 4
    for R, t in poses:
        C = -R.T @ t
        forward = R[2]
        to_center = -C / np.linalg.norm(C)
        assert np.dot(forward / np.linalg.norm(forward), to_center) > 0.9
