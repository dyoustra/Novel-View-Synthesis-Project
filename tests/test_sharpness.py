import numpy as np
from nvs.sharpness import laplacian_variance, select_sharp_frames


def test_laplacian_variance_higher_for_sharper():
    flat = np.zeros((64, 64), dtype=np.uint8)
    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[::2, :] = 255  # high-frequency stripes
    assert laplacian_variance(sharp) > laplacian_variance(flat)


def test_select_sharp_frames_picks_target_count_evenly():
    scores = [10.0] * 100
    idx = select_sharp_frames(scores, target=10, min_score=1.0)
    assert len(idx) == 10
    assert idx[0] == 0
    assert idx == sorted(idx)
    assert len(set(idx)) == 10


def test_select_sharp_frames_drops_below_min_score():
    scores = [0.1] * 50 + [50.0] * 50
    idx = select_sharp_frames(scores, target=10, min_score=1.0)
    assert all(i >= 50 for i in idx)


def test_select_sharp_frames_target_zero_returns_empty():
    assert select_sharp_frames([10.0] * 5, target=0, min_score=1.0) == []
