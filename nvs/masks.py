"""Convert SAM2 occluder masks (True = arm) to downstream mask conventions."""
from __future__ import annotations

import numpy as np


def occluder_to_colmap_mask(occluder: np.ndarray) -> np.ndarray:
    """COLMAP mask PNG: 0 = ignore pixel, 255 = use pixel."""
    keep = ~occluder.astype(bool)
    return (keep.astype(np.uint8) * 255)


def occluder_to_keep_alpha(occluder: np.ndarray) -> np.ndarray:
    """gsplat loss mask: 1.0 = keep (supervise), 0.0 = ignore."""
    keep = ~occluder.astype(bool)
    return keep.astype(np.float32)
