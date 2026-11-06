"""Hybrid ISP: classical stages around one learned stage.

classical: normalise -> white balance (metadata | gray-world) -> [adaptive exposure gain]
           -> noise model (DNG NoiseProfile | estimated) -> per-pixel sigma map
learned:   packed RGGB + sigma map -> joint denoise + demosaic -> linear camera RGB
classical: colour correction matrix -> sRGB gamma -> tone curve -> noise-adaptive sharpening
"""
import cv2
import numpy as np

from ..classical import ccm, sharpen, tone, wb
from ..classical.bayer import pack, visualize
from ..engine import run_tiled
from ..noise import noise_sigma_map, sigma_srgb
from ..types import RawImage
from .common import Timer, preview, resolve_nlf, resolve_wb

DEFAULT = {"wb": "metadata", "nlf": "metadata", "auto_exposure": False, "sharpen": False, "ccm": True,
           "tone": True, "sigma_scale": 1.0}


def sigma_heatmap(sig: np.ndarray) -> np.ndarray:
    v = np.clip(sig / (np.percentile(sig, 99) + 1e-8), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32) / 255


def run(raw: RawImage, engine, cfg: dict | None = None, keep_stages: bool = False):
    cfg = {**DEFAULT, **(cfg or {})}
    t = Timer()
    stages, info = {}, {}
    with t.stage("white_balance"):
        gains, info["wb_source"] = resolve_wb(raw, cfg)
        bayer = wb.apply_wb(raw.bayer, gains, clip=False)
    gain = tone.exposure_gain(bayer) if cfg["auto_exposure"] else 1.0
    bayer = np.clip(bayer * gain, 0, 1)
    info["wb_gains"] = [round(float(g), 3) for g in gains]
    info["digital_gain"] = round(gain, 2)
    with t.stage("noise_model"):
        nlf, info["nlf_source"] = resolve_nlf(raw, cfg)
        packed = pack(bayer)
        sig = noise_sigma_map(packed, nlf, gains, gain) * cfg["sigma_scale"]
    info["noise_sigma_mean"] = round(float(sig.mean()), 5)
    if keep_stages:
        stages["1 RAW (visualised)"] = preview(visualize(raw.bayer))
        stages["2 White balanced" + (f" + gain x{gain:.1f}" if gain > 1 else "")] = preview(visualize(bayer))
        stages["3 Noise map (sigma)"] = preview(sigma_heatmap(sig))
    with t.stage("network"):
        inp = np.concatenate([packed, sig[..., None]], -1) if engine.in_ch == 5 else packed
        lin = np.clip(run_tiled(engine, inp.astype(np.float32)), 0, 1)
    if keep_stages:
        stages["4 Network output (linear)"] = preview(tone.srgb_gamma(lin))
    with t.stage("ccm_gamma_tone"):
        rgb = ccm.apply_ccm(lin, raw.cam2srgb if cfg["ccm"] else None)
        out = tone.display_transform(rgb, with_tone=cfg["tone"])
    if keep_stages:
        stages["5 CCM + gamma + tone"] = preview(out)
    if cfg["sharpen"]:
        with t.stage("sharpen"):
            out, info["sharpen_amount"] = sharpen.adaptive_sharpen(out, sigma_srgb(out), base_amount=0.35)
        if keep_stages:
            stages["6 Adaptive sharpening"] = preview(out)
    return np.clip(out, 0, 1).astype(np.float32), stages, {"timings_ms": t.ms, **info}
