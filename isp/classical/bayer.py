"""Bayer helpers: CFA normalisation to RGGB, packing and unpacking."""
import numpy as np


def rggb_offset(pattern_letters) -> tuple:
    """Find (dy, dx) so that cropping the mosaic by it gives an RGGB layout.

    pattern_letters: 2x2 nested list of 'R'/'G'/'B' for the top-left 2x2 block.
    """
    p = [[c.upper() for c in row] for row in pattern_letters]
    for dy in (0, 1):
        for dx in (0, 1):
            block = [[p[(dy + i) % 2][(dx + j) % 2] for j in (0, 1)] for i in (0, 1)]
            if block == [["R", "G"], ["G", "B"]]:
                return dy, dx
    raise ValueError(f"Unsupported CFA pattern {pattern_letters}")


def crop_to_rggb(mosaic: np.ndarray, pattern_letters) -> np.ndarray:
    dy, dx = rggb_offset(pattern_letters)
    m = mosaic[dy:, dx:]
    h, w = m.shape[:2]
    return m[: h - h % 2, : w - w % 2]


def pack(bayer: np.ndarray) -> np.ndarray:
    """RGGB HxW -> (H/2, W/2, 4) with channels R, G1, G2, B."""
    return np.stack([bayer[0::2, 0::2], bayer[0::2, 1::2], bayer[1::2, 0::2], bayer[1::2, 1::2]], axis=-1)


def unpack(packed: np.ndarray) -> np.ndarray:
    h, w, _ = packed.shape
    out = np.empty((h * 2, w * 2), dtype=packed.dtype)
    out[0::2, 0::2] = packed[..., 0]
    out[0::2, 1::2] = packed[..., 1]
    out[1::2, 0::2] = packed[..., 2]
    out[1::2, 1::2] = packed[..., 3]
    return out


def mosaic_rgb(rgb: np.ndarray) -> np.ndarray:
    """Sample a 3-channel image on an RGGB grid (used for linear DNGs / sRGB inputs)."""
    h, w = rgb.shape[:2]
    rgb = rgb[: h - h % 2, : w - w % 2]
    out = np.empty(rgb.shape[:2], dtype=rgb.dtype)
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[0::2, 1::2] = rgb[0::2, 1::2, 1]
    out[1::2, 0::2] = rgb[1::2, 0::2, 1]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return out


def visualize(bayer: np.ndarray) -> np.ndarray:
    """Quick RGB preview of a Bayer image at half resolution (for the stage gallery)."""
    p = pack(bayer)
    rgb = np.stack([p[..., 0], 0.5 * (p[..., 1] + p[..., 2]), p[..., 3]], axis=-1)
    scale = np.percentile(rgb, 99) + 1e-6
    return np.clip(rgb / scale, 0, 1) ** (1 / 2.2)
