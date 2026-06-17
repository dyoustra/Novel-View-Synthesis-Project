import numpy as np
from nvs.metrics import psnr, ssim


def test_psnr_identical_is_infinite():
    img = np.random.rand(32, 32, 3).astype(np.float32)
    assert psnr(img, img) == float("inf")


def test_psnr_decreases_with_noise():
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    noisy = np.clip(img + 0.1, 0, 1).astype(np.float32)
    assert 0 < psnr(img, noisy) < 100


def test_ssim_identical_is_one():
    img = np.random.rand(64, 64, 3).astype(np.float32)
    assert abs(ssim(img, img) - 1.0) < 1e-6
