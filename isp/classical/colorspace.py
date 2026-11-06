"""Colour-space helpers (YCrCb split for luma-only processing)."""
import cv2
import numpy as np


def rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)


def ycrcb_to_rgb(ycc: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(ycc.astype(np.float32), cv2.COLOR_YCrCb2RGB), 0.0, 1.0)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
