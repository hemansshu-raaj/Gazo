"""Demosaicing of RGGB mosaics (float in, float out)."""
import numpy as np


def demosaic(bayer: np.ndarray, method: str = "malvar") -> np.ndarray:
    if method == "malvar":  # Malvar-He-Cutler 2004 gradient-corrected linear interpolation
        from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
        out = demosaicing_CFA_Bayer_Malvar2004(bayer.astype(np.float64), pattern="RGGB")
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    if method == "bilinear":
        from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
        return np.clip(demosaicing_CFA_Bayer_bilinear(bayer.astype(np.float64), pattern="RGGB"), 0, 1).astype(np.float32)
    if method in ("ea", "vng"):  # OpenCV edge-aware / VNG (OpenCV names RGGB as "BayerBG")
        import cv2
        u16 = (np.clip(bayer, 0, 1) * 65535).astype(np.uint16)
        if method == "ea":
            rgb = cv2.cvtColor(u16, cv2.COLOR_BayerBG2RGB_EA)
        else:
            rgb = cv2.cvtColor((u16 >> 8).astype(np.uint8), cv2.COLOR_BayerBG2RGB_VNG).astype(np.uint16) << 8
        return rgb.astype(np.float32) / 65535.0
    raise ValueError(f"unknown demosaic method {method}")
