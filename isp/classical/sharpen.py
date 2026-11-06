"""Noise-adaptive, edge-aware unsharp masking on luma."""
import cv2
import numpy as np

from .colorspace import rgb_to_ycrcb, ycrcb_to_rgb


def adaptive_sharpen(rgb: np.ndarray, sigma: float, base_amount: float = 0.6, radius: float = 1.2) -> tuple:
    """Returns (image, amount_used). Strength falls as noise rises; flat (noise-level) detail is not boosted."""
    amount = base_amount / (1.0 + (sigma / 0.01) ** 2)
    if amount < 0.02:
        return rgb, 0.0
    ycc = rgb_to_ycrcb(rgb)
    y = ycc[..., 0]
    detail = y - cv2.GaussianBlur(y, (0, 0), radius)
    t = 2.5 * max(sigma, 1e-3)
    mask = np.clip((np.abs(detail) - t) / t, 0.0, 1.0)  # only real edges, not noise
    ycc[..., 0] = np.clip(y + amount * mask * detail, 0, 1)
    return ycrcb_to_rgb(ycc), float(amount)
