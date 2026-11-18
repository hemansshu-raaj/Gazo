"""Stage-by-stage versions of the three ISP pipelines, for the website's walk-through.

Each `trace_*` function runs the SAME operations, with the same defaults, as the pipeline it mirrors
in `isp/pipelines/` (tests/test_api.py checks the final images are identical), but it records an
image after every stage instead of only a few.

A stage is a dict:
    {"id": "white_balance",          # key into site/content/stages.json (titles + explanations)
     "image": HxWx3 float in [0,1],  # what the photo looks like after this stage
     "info": {...}}                  # small numbers worth showing (gains, sigma, ...)

To add or reorder a stage: edit the trace function here, then add the same id to stages.json.
"""
import numpy as np

from isp.classical import ccm, demosaic, denoise, sharpen, tone, wb
from isp.classical.bayer import pack, visualize
from isp.engine import run_tiled
from isp.noise import noise_sigma_map, sigma_srgb
from isp.pipelines import hybrid as hybrid_pipe
from isp.pipelines import traditional as trad_pipe
from isp.pipelines.common import resolve_nlf, resolve_wb
from isp.types import RawImage


# ---------------------------------------------------------------- how RAW data is drawn on screen
def _display_scale(bayer: np.ndarray) -> float:
    return float(np.percentile(bayer[::2, ::2], 99)) + 1e-6


def photosite_view(bayer: np.ndarray) -> np.ndarray:
    """Each photosite's light intensity as grey (the sensor only counts photons, it sees no colour)."""
    g = np.clip(bayer / _display_scale(bayer), 0, 1) ** (1 / 2.2)
    return np.repeat(g[..., None], 3, axis=-1)


def cfa_view(bayer: np.ndarray) -> np.ndarray:
    """The same values tinted by the colour filter that sits on each photosite (R, G or B)."""
    g = np.clip(bayer / _display_scale(bayer), 0, 1) ** (1 / 2.2)
    out = np.zeros(bayer.shape + (3,), np.float32)
    out[0::2, 0::2, 0] = g[0::2, 0::2]
    out[0::2, 1::2, 1] = g[0::2, 1::2]
    out[1::2, 0::2, 1] = g[1::2, 0::2]
    out[1::2, 1::2, 2] = g[1::2, 1::2]
    return out


def packed_view(bayer: np.ndarray) -> np.ndarray:
    """The 4 colour planes (R, G1, G2, B) the networks read, laid out as a 2x2 grid."""
    p = np.clip(pack(bayer) / _display_scale(bayer), 0, 1) ** (1 / 2.2)
    h, w = p.shape[:2]
    out = np.zeros((2 * h, 2 * w, 3), np.float32)
    out[:h, :w, 0] = p[..., 0]
    out[:h, w:, 1] = p[..., 1]
    out[h:, :w, 1] = p[..., 2]
    out[h:, w:, 2] = p[..., 3]
    return out


def _stage(sid, image, **info):
    return {"id": sid, "image": np.clip(image, 0, 1).astype(np.float32), "info": info}


def _raw_stages(raw: RawImage):
    h, w = raw.shape
    return [
        _stage("photosites", photosite_view(raw.bayer), width=w, height=h),
        _stage("bayer", cfa_view(raw.bayer), pattern="RGGB"),
    ]


def _gains(raw, cfg):
    gains, source = resolve_wb(raw, cfg)
    return gains, {"gains_rgb": [round(float(g), 3) for g in gains], "source": source}


# ---------------------------------------------------------------- 1. classic (traditional) ISP
def trace_classic(raw: RawImage, cfg: dict | None = None):
    """Mirrors isp/pipelines/traditional.py with its DEFAULT settings."""
    cfg = {**trad_pipe.DEFAULT, **(cfg or {})}
    stages = _raw_stages(raw)

    gains, wb_info = _gains(raw, cfg)
    bayer = wb.apply_wb(raw.bayer, gains, clip=False)
    gain = tone.exposure_gain(bayer) if cfg["auto_exposure"] else 1.0
    bayer = np.clip(bayer * gain, 0, 1)
    stages.append(_stage("white_balance", visualize(bayer), **wb_info, exposure_gain=round(gain, 2)))

    rgb = demosaic.demosaic(bayer, cfg["demosaic"])
    stages.append(_stage("demosaic", tone.srgb_gamma(rgb), method="Malvar-He-Cutler"))

    rgb = ccm.apply_ccm(rgb, raw.cam2srgb if cfg["ccm"] else None)
    stages.append(_stage("color_correction", tone.srgb_gamma(rgb)))

    out = tone.display_transform(rgb, with_tone=cfg["tone"])
    stages.append(_stage("gamma_tone", out))

    sig = sigma_srgb(out)
    out = denoise.denoise_srgb(out, sig, cfg["denoise"])
    stages.append(_stage("denoise", out, method=cfg["denoise"], sigma=round(sig, 4)))

    if cfg["sharpen"]:
        out, amount = sharpen.adaptive_sharpen(out, sigma_srgb(out))
        stages.append(_stage("sharpen", out, amount=round(float(amount), 3)))
    return np.clip(out, 0, 1).astype(np.float32), stages


# ---------------------------------------------------------------- 2. learned (end-to-end) ISP
def trace_learned(raw: RawImage, engine):
    """Mirrors isp/pipelines/learned.py: one network does everything after packing."""
    stages = _raw_stages(raw)
    packed = pack(raw.bayer).astype(np.float32)
    stages.append(_stage("pack", packed_view(raw.bayer), planes=4))
    out = np.clip(run_tiled(engine, packed), 0, 1).astype(np.float32)
    stages.append(_stage("network_e2e", out, size_mb=None))
    return out, stages


# ---------------------------------------------------------------- 3. hybrid ISP
def trace_hybrid(raw: RawImage, engine, cfg: dict | None = None):
    """Mirrors isp/pipelines/hybrid.py with its DEFAULT settings."""
    cfg = {**hybrid_pipe.DEFAULT, **(cfg or {})}
    stages = _raw_stages(raw)

    gains, wb_info = _gains(raw, cfg)
    bayer = wb.apply_wb(raw.bayer, gains, clip=False)
    gain = tone.exposure_gain(bayer) if cfg["auto_exposure"] else 1.0
    bayer = np.clip(bayer * gain, 0, 1)
    stages.append(_stage("white_balance", visualize(bayer), **wb_info, exposure_gain=round(gain, 2)))

    nlf, nlf_source = resolve_nlf(raw, cfg)
    packed = pack(bayer)
    sig = noise_sigma_map(packed, nlf, gains, gain) * cfg["sigma_scale"]
    stages.append(_stage("noise_map", hybrid_pipe.sigma_heatmap(sig), source=nlf_source,
                         mean_sigma=round(float(sig.mean()), 5)))

    inp = np.concatenate([packed, sig[..., None]], -1) if engine.in_ch == 5 else packed
    lin = np.clip(run_tiled(engine, inp.astype(np.float32)), 0, 1)
    stages.append(_stage("network_hybrid", tone.srgb_gamma(lin)))

    rgb = ccm.apply_ccm(lin, raw.cam2srgb if cfg["ccm"] else None)
    stages.append(_stage("color_correction", tone.srgb_gamma(rgb)))

    out = tone.display_transform(rgb, with_tone=cfg["tone"])
    stages.append(_stage("gamma_tone", out))

    if cfg["sharpen"]:
        out, amount = sharpen.adaptive_sharpen(out, sigma_srgb(out), base_amount=0.35)
        stages.append(_stage("sharpen", out, amount=round(float(amount), 3)))
    return np.clip(out, 0, 1).astype(np.float32), stages
