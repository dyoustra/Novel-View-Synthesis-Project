"""Frame sharpness scoring (variance of Laplacian) and even-coverage selection."""
from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian. Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharp_frames(
    scores: list[float], target: int, min_score: float
) -> list[int]:
    """Pick ~`target` frame indices with even temporal coverage, dropping any
    frame whose sharpness is below `min_score`. Within each evenly-spaced bin,
    keep the sharpest surviving frame."""
    if target <= 0:
        return []
    eligible = [i for i, s in enumerate(scores) if s >= min_score]
    if not eligible:
        return []
    if len(eligible) <= target:
        return eligible
    bins = np.array_split(np.array(eligible), target)
    chosen = []
    for b in bins:
        if len(b) == 0:
            continue
        best = max(b, key=lambda i: scores[i])
        chosen.append(int(best))
    return sorted(chosen)
