"""Image-quality metrics. PSNR/SSIM are pure numpy/skimage; LPIPS is lazy (torch)."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """a, b are float images in [0, 1]."""
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity for [0,1] RGB images."""
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))


def lpips_fn(device: str = "cuda"):
    """Return a callable (a,b)->float using LPIPS(AlexNet). Lazy: imports torch."""
    import lpips as _lpips
    import torch

    net = _lpips.LPIPS(net="alex").to(device).eval()

    def _score(a: np.ndarray, b: np.ndarray) -> float:
        def to_t(x):
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
            return (t * 2 - 1).to(device)  # LPIPS expects [-1,1]
        with torch.no_grad():
            return float(net(to_t(a), to_t(b)).item())

    return _score
