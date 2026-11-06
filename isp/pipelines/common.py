"""Pieces shared by all three pipelines."""
import time
from contextlib import contextmanager

import cv2
import numpy as np

from ..classical import wb as wbmod
from ..noise import estimate_nlf
from ..types import RawImage


class Timer:
    def __init__(self):
        self.ms = {}

    @contextmanager
    def stage(self, name):
        t = time.perf_counter()
        yield
        self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - t) * 1000


def resolve_wb(raw: RawImage, cfg: dict) -> tuple:
    """Metadata white balance if present (and allowed), else a classical estimator."""
    if raw.wb_gains is not None and cfg.get("wb", "metadata") == "metadata":
        return raw.wb_gains.astype(np.float32), "metadata (AsShotNeutral)"
    method = cfg.get("wb_fallback", "gray_world")
    return wbmod.estimate_gains(raw.bayer, method), method.replace("_", "-")


def resolve_nlf(raw: RawImage, cfg: dict) -> tuple:
    if raw.nlf is not None and cfg.get("nlf", "metadata") == "metadata":
        return raw.nlf, "metadata (NoiseProfile)"
    return estimate_nlf(raw.bayer), "estimated from image"


def preview(img: np.ndarray, max_side: int = 768) -> np.ndarray:
    """Downscaled uint8 copy for the stage gallery."""
    img = np.clip(img, 0, 1)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return (img * 255 + 0.5).astype(np.uint8)


def fix_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    ops = {2: lambda x: x[:, ::-1], 3: lambda x: x[::-1, ::-1], 4: lambda x: x[::-1],
           5: lambda x: np.swapaxes(x, 0, 1), 6: lambda x: np.rot90(x, -1), 7: lambda x: np.rot90(x[:, ::-1], -1),
           8: lambda x: np.rot90(x, 1)}
    return np.ascontiguousarray(ops[orientation](img)) if orientation in ops else img
