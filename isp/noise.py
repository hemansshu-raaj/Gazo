"""Noise estimation: heteroscedastic Poisson-Gaussian model, variance = a * x + b."""
import numpy as np
from scipy.ndimage import uniform_filter

from .classical.bayer import pack


def estimate_nlf(bayer: np.ndarray, n_bins: int = 16) -> tuple:
    """Fit (a, b) from the image itself (used when the file has no NoiseProfile).

    Local means vs robust local variance of a high-pass residual, binned by intensity,
    then a non-negative least-squares line fit.
    """
    p = pack(bayer).astype(np.float32)
    g = 0.5 * (p[..., 1] + p[..., 2])
    # Difference of the two greens removes the signal, leaving noise with variance 2*var/2 = var
    resid = (p[..., 1] - p[..., 2]) / np.sqrt(2.0)
    mean = uniform_filter(g, 7)
    m, r = mean[::2, ::2].ravel(), resid[::2, ::2].ravel()
    keep = (m > 0.002) & (m < 0.9)
    m, r = m[keep], r[keep]
    if m.size < 1000:
        return 1e-4, 1e-6
    edges = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    xs, vs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (m >= lo) & (m < hi)
        if sel.sum() > 200:
            mad = np.median(np.abs(r[sel] - np.median(r[sel])))
            xs.append(m[sel].mean())
            vs.append((1.4826 * mad) ** 2)
    if len(xs) < 3:
        return 1e-4, 1e-6
    A = np.stack([xs, np.ones(len(xs))], axis=1)
    a, b = np.linalg.lstsq(A, np.array(vs), rcond=None)[0]
    return float(max(a, 1e-7)), float(max(b, 1e-9))


def noise_sigma_map(bayer_wb_packed: np.ndarray, nlf: tuple, wb_gains: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Per-pixel noise std at packed resolution, after white balance and digital gain."""
    a, b = nlf
    g4 = np.array([wb_gains[0], wb_gains[1], wb_gains[1], wb_gains[2]], dtype=np.float32) * gain
    # signal before WB/gain = observed / gain; std scales with the gain
    x = np.clip(bayer_wb_packed / g4, 0, 1)
    sig = np.sqrt(np.maximum(a * x + b, 1e-10)) * g4
    return sig.mean(axis=-1).astype(np.float32)


def sigma_srgb(img: np.ndarray) -> float:
    """Donoho MAD estimate of noise std on luma of a display image (Haar HH band)."""
    y = img.mean(axis=-1) if img.ndim == 3 else img
    h, w = y.shape
    y = y[: h - h % 2, : w - w % 2]
    hh = (y[0::2, 0::2] - y[0::2, 1::2] - y[1::2, 0::2] + y[1::2, 1::2]) / 2.0
    return float(np.median(np.abs(hh)) / 0.6745)
