"""Edge-preserving classical denoising, strength driven by the estimated noise level."""
import cv2
import numpy as np

from .colorspace import rgb_to_ycrcb, ycrcb_to_rgb


def denoise_srgb(rgb: np.ndarray, sigma: float, method: str = "bilateral") -> np.ndarray:
    """Denoise a display-referred image in [0,1]. sigma = noise std in the same units."""
    if sigma < 1e-4:
        return rgb
    if method == "nlm":
        u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h = float(np.clip(sigma * 255 * 0.9, 1, 30))
        out = cv2.fastNlMeansDenoisingColored(u8, None, h, h * 1.5, 7, 21)
        return out.astype(np.float32) / 255.0
    # Bilateral on luma (edge-preserving), stronger smoothing on chroma (noise is mostly chroma).
    ycc = rgb_to_ycrcb(rgb)
    d = int(np.clip(3 + sigma * 60, 5, 11)) | 1
    ycc[..., 0] = cv2.bilateralFilter(ycc[..., 0], d, sigmaColor=float(3.0 * sigma), sigmaSpace=d / 2)
    for c in (1, 2):
        ycc[..., c] = cv2.bilateralFilter(ycc[..., c], d + 4, sigmaColor=float(6.0 * sigma), sigmaSpace=d)
    return ycrcb_to_rgb(ycc)
