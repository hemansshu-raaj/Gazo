"""White balance: metadata gains, with classical illuminant estimators as fallback."""
import numpy as np

from .bayer import pack


def gray_world(bayer: np.ndarray) -> np.ndarray:
    p = pack(bayer)
    valid = p.max(axis=-1) < 0.95  # ignore clipped pixels
    m = p[valid].mean(axis=0) if valid.any() else p.reshape(-1, 4).mean(axis=0)
    rgb = np.array([m[0], 0.5 * (m[1] + m[2]), m[3]]) + 1e-8
    return (rgb[1] / rgb).astype(np.float32)


def gray_edge(bayer: np.ndarray, p_norm: float = 6.0, sigma: float = 2.0) -> np.ndarray:
    """Gray-edge (van de Weijer 2007): the average edge colour is achromatic.
    The image is Gaussian-smoothed first (as in the paper) so sensor noise is not mistaken for edges."""
    import cv2
    p = pack(bayer).astype(np.float32)
    rgb = np.stack([p[..., 0], 0.5 * (p[..., 1] + p[..., 2]), p[..., 3]], axis=-1)
    rgb = cv2.GaussianBlur(rgb, (0, 0), sigma)
    valid = (rgb.max(axis=-1) < 0.95)[:-1, :-1, None]
    gx = np.abs(np.diff(rgb, axis=1))[:-1]
    gy = np.abs(np.diff(rgb, axis=0))[:, :-1]
    g = (np.sqrt(gx ** 2 + gy ** 2) * valid).reshape(-1, 3)
    e = (g ** p_norm).mean(axis=0) ** (1 / p_norm) + 1e-8
    return (e[1] / e).astype(np.float32)


def estimate_gains(bayer: np.ndarray, method: str = "gray_world") -> np.ndarray:
    return gray_edge(bayer) if method == "gray_edge" else gray_world(bayer)


def apply_wb(bayer: np.ndarray, gains: np.ndarray, clip: bool = True) -> np.ndarray:
    out = bayer.copy()
    out[0::2, 0::2] *= gains[0]
    out[0::2, 1::2] *= gains[1]
    out[1::2, 0::2] *= gains[1]
    out[1::2, 1::2] *= gains[2]
    return np.clip(out, 0.0, 1.0) if clip else out
