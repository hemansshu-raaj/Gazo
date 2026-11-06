"""Colour correction: white-balanced camera RGB -> linear sRGB (via CIE XYZ), as in SIDD's renderer."""
import numpy as np

# XYZ (D65) -> linear sRGB
XYZ2SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                     [-0.9692660, 1.8760108, 0.0415560],
                     [0.0556434, -0.2040259, 1.0572252]])


def cam2srgb_from_color_matrix(xyz2cam: np.ndarray) -> np.ndarray:
    """Build the 3x3 camera->sRGB matrix from a DNG ColorMatrix (XYZ->camera).

    Rows are normalised so a white-balanced neutral (1,1,1) stays neutral.
    """
    xyz2cam = np.asarray(xyz2cam, dtype=np.float64).reshape(3, 3)
    xyz2cam = xyz2cam / xyz2cam.sum(axis=1, keepdims=True)
    cam2xyz = np.linalg.inv(xyz2cam)
    xyz2srgb = XYZ2SRGB / XYZ2SRGB.sum(axis=1, keepdims=True)
    m = xyz2srgb @ cam2xyz
    return (m / m.sum(axis=1, keepdims=True)).astype(np.float32)


def apply_ccm(rgb: np.ndarray, m: np.ndarray | None) -> np.ndarray:
    if m is None:
        return np.clip(rgb, 0, 1)
    return np.clip(rgb @ np.asarray(m, dtype=np.float32).T, 0.0, 1.0)


# DNG CalibrationIlluminant code -> correlated colour temperature (K)
ILLUMINANT_CCT = {1: 6504, 2: 4230, 3: 2856, 4: 5500, 9: 5500, 10: 6500, 11: 7500, 12: 6430, 13: 5000,
                  14: 4150, 15: 3450, 17: 2856, 18: 4874, 19: 6774, 20: 5503, 21: 6504, 22: 7504,
                  23: 5003, 24: 3200}


def _cct_from_xyz(xyz: np.ndarray) -> float:
    x, y = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
    n = (x - 0.3320) / (0.1858 - y)
    return float(np.clip(449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33, 1500, 15000))


def interpolate_color_matrix(cm1, cm2, ill1, ill2, as_shot_neutral) -> tuple:
    """DNG-spec style: find the scene CCT from the as-shot neutral and blend the two
    calibration matrices linearly in inverse CCT. Returns (xyz2cam, cct)."""
    cm1, cm2 = np.asarray(cm1, float).reshape(3, 3), np.asarray(cm2, float).reshape(3, 3)
    t1, t2 = ILLUMINANT_CCT.get(int(ill1), 2856), ILLUMINANT_CCT.get(int(ill2), 6504)
    asn = np.asarray(as_shot_neutral, float)[:3]
    w, cct = 0.5, 5000.0
    for _ in range(4):
        cm = w * cm1 + (1 - w) * cm2
        cct = _cct_from_xyz(np.linalg.solve(cm, asn))
        if t1 == t2:
            break
        w = float(np.clip((1 / cct - 1 / t2) / (1 / t1 - 1 / t2), 0.0, 1.0))
    return w * cm1 + (1 - w) * cm2, cct
