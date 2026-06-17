import numpy as np
from nvs.masks import occluder_to_colmap_mask, occluder_to_keep_alpha


def test_occluder_to_colmap_mask_inverts():
    occ = np.array([[True, False], [False, True]])
    out = occluder_to_colmap_mask(occ)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0 and out[1, 1] == 0      # arm -> ignore
    assert out[0, 1] == 255 and out[1, 0] == 255  # scene -> use


def test_occluder_to_keep_alpha_is_float_complement():
    occ = np.array([[True, False]])
    alpha = occluder_to_keep_alpha(occ)
    assert alpha.dtype == np.float32
    assert alpha[0, 0] == 0.0 and alpha[0, 1] == 1.0
