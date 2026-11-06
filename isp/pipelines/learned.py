"""Fully learned ISP baseline: packed RAW (no white balance, no metadata) -> sRGB in one network."""
import numpy as np

from ..classical.bayer import pack, visualize
from ..engine import run_tiled
from ..types import RawImage
from .common import Timer, preview


def run(raw: RawImage, engine, cfg: dict | None = None, keep_stages: bool = False):
    t = Timer()
    stages = {}
    if keep_stages:
        stages["1 RAW (visualised)"] = preview(visualize(raw.bayer))
    with t.stage("network"):
        out = np.clip(run_tiled(engine, pack(raw.bayer).astype(np.float32)), 0, 1)
    if keep_stages:
        stages["2 Network output (sRGB)"] = preview(out)
    return out.astype(np.float32), stages, {"timings_ms": t.ms, "wb_source": "learned (implicit)"}
