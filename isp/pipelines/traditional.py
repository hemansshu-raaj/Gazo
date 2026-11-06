"""Conventional ISP: every stage is a classical algorithm.

normalise (in loader) -> white balance -> [exposure gain] -> demosaic -> CCM -> sRGB gamma
-> tone curve -> noise-adaptive edge-preserving denoise -> noise-adaptive sharpening

With cfg = REFERENCE (no denoise / sharpen / gain) it is the renderer that creates our
ground-truth sRGB from the clean (150-frame average) RAW.
"""
import numpy as np

from ..classical import ccm, demosaic, denoise, sharpen, tone, wb
from ..classical.bayer import visualize
from ..noise import sigma_srgb
from ..types import RawImage
from .common import Timer, preview, resolve_nlf, resolve_wb

DEFAULT = {"wb": "metadata", "demosaic": "malvar", "denoise": "bilateral", "sharpen": True,
           "auto_exposure": False, "ccm": True, "tone": True}
REFERENCE = {"wb": "metadata", "demosaic": "malvar", "denoise": None, "sharpen": False,
             "auto_exposure": False, "ccm": True, "tone": True}


def run(raw: RawImage, cfg: dict | None = None, keep_stages: bool = False):
    cfg = {**DEFAULT, **(cfg or {})}
    t = Timer()
    stages, info = {}, {}
    with t.stage("white_balance"):
        gains, info["wb_source"] = resolve_wb(raw, cfg)
        bayer = wb.apply_wb(raw.bayer, gains, clip=False)
    info["wb_gains"] = [round(float(g), 3) for g in gains]
    gain = 1.0
    if cfg["auto_exposure"]:
        gain = tone.exposure_gain(bayer)
        bayer = bayer * gain
    info["digital_gain"] = round(gain, 2)
    bayer = np.clip(bayer, 0, 1)
    if keep_stages:
        stages["1 RAW (visualised)"] = preview(visualize(raw.bayer))
        stages["2 White balanced"] = preview(visualize(bayer))
    with t.stage("demosaic"):
        rgb = demosaic.demosaic(bayer, cfg["demosaic"])
    if keep_stages:
        stages["3 Demosaiced (linear)"] = preview(tone.srgb_gamma(rgb))
    with t.stage("ccm"):
        rgb = ccm.apply_ccm(rgb, raw.cam2srgb if cfg["ccm"] else None)
    with t.stage("gamma_tone"):
        out = tone.display_transform(rgb, with_tone=cfg["tone"])
    if keep_stages:
        stages["4 CCM + gamma + tone"] = preview(out)
    sig = 0.0
    if cfg["denoise"] or cfg["sharpen"]:
        sig = sigma_srgb(out)
    info["noise_sigma_srgb"] = round(sig, 4)
    if cfg["denoise"]:
        with t.stage("denoise"):
            out = denoise.denoise_srgb(out, sig, cfg["denoise"])
        if keep_stages:
            stages["5 Denoised"] = preview(out)
        sig = sigma_srgb(out)
    if cfg["sharpen"]:
        with t.stage("sharpen"):
            out, amt = sharpen.adaptive_sharpen(out, sig)
        info["sharpen_amount"] = round(amt, 3)
    info["nlf"] = resolve_nlf(raw, cfg)[1] if cfg["denoise"] else "n/a"
    return np.clip(out, 0, 1).astype(np.float32), stages, {"timings_ms": t.ms, **info}


def render_reference(raw: RawImage) -> np.ndarray:
    return run(raw, REFERENCE)[0]
