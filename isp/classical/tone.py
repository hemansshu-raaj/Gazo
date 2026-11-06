"""sRGB transfer function, global tone curve and exposure analysis."""
import numpy as np

from .tone_curve_data import TONE_CURVE

_TC = np.asarray(TONE_CURVE, dtype=np.float32)
_TC_X = np.linspace(0.0, 1.0, len(_TC), dtype=np.float32)


def srgb_gamma(x: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB OETF (linear toe keeps gradients finite near black)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055).astype(np.float32)


def srgb_degamma(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4).astype(np.float32)


def tone_curve(x: np.ndarray) -> np.ndarray:
    return np.interp(np.clip(x, 0, 1), _TC_X, _TC).astype(np.float32)


def exposure_gain(bayer_wb: np.ndarray, target: float = 0.6, max_gain: float = 8.0) -> float:
    """Adaptive digital gain: lift dark captures so the 99th percentile lands near `target`."""
    p99 = float(np.percentile(bayer_wb[::4, ::4], 99))
    if p99 <= 1e-6:
        return 1.0
    return float(np.clip(target / p99, 1.0, max_gain))


_LUT_N = 65536
_DISPLAY_LUT = None


def display_transform(x: np.ndarray, with_tone: bool = True) -> np.ndarray:
    """Fast sRGB gamma (+ tone curve) through a 16-bit lookup table on linear input."""
    global _DISPLAY_LUT
    if _DISPLAY_LUT is None:
        lin = np.linspace(0, 1, _LUT_N, dtype=np.float32)
        g = srgb_gamma(lin)
        _DISPLAY_LUT = (g, tone_curve(g))
    idx = (np.clip(x, 0, 1) * (_LUT_N - 1) + 0.5).astype(np.uint16)
    return _DISPLAY_LUT[1 if with_tone else 0][idx]
