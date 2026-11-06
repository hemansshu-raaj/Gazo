"""Evaluation of all pipelines on the test split (shared by the Modal job and local runs).

Protocol: for every test scene, 32 seeded 512x512 crops (isp.data.eval_boxes) of noisy frame 0. Each crop is
processed with a 64-px margin of context (then the margin is discarded) so tiling/borders don't bias any method.
"""
import os
import time

import cv2
import numpy as np

from .data import SceneArrays, eval_boxes
from .engine import OnnxEngine, TorchEngine
from .metrics import delta_e, lpips, psnr, ssim
from .pipelines import hybrid, learned, traditional
from .types import RawImage

CROP, MARGIN = 512, 64
EXPOSURES = [1.0, 0.5, 0.25, 0.125, 0.0625]


def load_methods(ckpt_dir: str, int8_path: str | None, device: str = "cuda") -> dict:
    """name -> (kind, engine, cfg). kind in {traditional, hybrid, learned}."""
    m = {"Traditional ISP": ("traditional", None, {"denoise": "bilateral", "sharpen": True})}
    ck = lambda r: os.path.join(ckpt_dir, r, "best.pt")
    if os.path.exists(ck("e2e")):
        m["Learned E2E"] = ("learned", TorchEngine(ck("e2e"), device=device), {})
    if os.path.exists(ck("teacher")):
        m["Hybrid teacher FP32"] = ("hybrid", TorchEngine(ck("teacher"), device=device), {})
        m["Hybrid teacher FP16"] = ("hybrid", TorchEngine(ck("teacher"), device=device, half=True), {})
    if os.path.exists(ck("nosigma")):
        m["Hybrid w/o noise map"] = ("hybrid", TorchEngine(ck("nosigma"), device=device), {})
    if os.path.exists(ck("student_scratch")):
        m["Student (no distillation)"] = ("hybrid", TorchEngine(ck("student_scratch"), device=device), {})
    if os.path.exists(ck("student_kd")):
        m["Hybrid student FP32 (KD)"] = ("hybrid", TorchEngine(ck("student_kd"), device=device), {})
    if int8_path and os.path.exists(int8_path):
        m["Hybrid student INT8 (KD)"] = ("hybrid", OnnxEngine(int8_path), {})  # INT8 weights, 16-bit activations
        w8a8 = int8_path.replace("student_int8", "student_w8a8")
        if os.path.exists(w8a8):
            m["Student full 8-bit W8A8 (KD)"] = ("hybrid", OnnxEngine(w8a8), {})
    return m


def run_method(method, raw: RawImage, cfg_override=None):
    kind, eng, cfg = method
    cfg = {**cfg, **(cfg_override or {})}
    if kind == "traditional":
        return traditional.run(raw, cfg)[0]
    if kind == "learned":
        return learned.run(raw, eng)[0]
    return hybrid.run(raw, eng, cfg)[0]


def _region(sc: SceneArrays, y, x):
    y0, x0 = max(0, y - MARGIN), max(0, x - MARGIN)
    y1, x1 = min(sc.h, y + CROP + MARGIN), min(sc.w, x + CROP + MARGIN)
    return y0, x0, (y1 - y0) // 2 * 2, (x1 - x0) // 2 * 2


def _raw(sc, bayer, nlf=None):
    return RawImage(bayer=bayer, wb_gains=sc.wb, cam2srgb=sc.cam2srgb, nlf=nlf if nlf is not None else sc.nlf)


def _metrics(out, gt, with_lpips=True):
    r = {"psnr": psnr(out, gt), "ssim": ssim(out, gt), "delta_e": delta_e(out, gt)}
    if with_lpips:
        r["lpips"] = lpips(out, gt)
    return r


def _jpg(img):
    return cv2.imencode(".jpg", cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


ABLATIONS = {
    "full (metadata WB + NoiseProfile)": {},
    "WB: gray-world instead of metadata": {"wb": "estimate", "wb_fallback": "gray_world"},
    "WB: gray-edge instead of metadata": {"wb": "estimate", "wb_fallback": "gray_edge"},
    "CCM off (camera RGB shown as sRGB)": {"ccm": False},
    "noise model estimated from image": {"nlf": "estimate"},
    "noise map x0 (network told: no noise)": {"sigma_scale": 0.0},
    "noise map x2 (network told: 2x noise)": {"sigma_scale": 2.0},
    "+ adaptive sharpening": {"sharpen": True},
}


def evaluate_scene(scene_dir: str, methods: dict, n_crops=32, n_sweep=8, n_ablation=16, sweep_methods=None,
                   ablation_method="Hybrid teacher FP32", seed=0) -> dict:
    name = os.path.basename(scene_dir.rstrip("/"))
    sc = SceneArrays(scene_dir)
    boxes = eval_boxes(name, sc.h, sc.w, n=n_crops, size=CROP)
    res = {"scene": name, "camera": sc.meta.get("camera"), "iso": int(sc.meta.get("iso", 0)),
           "brightness": sc.meta.get("brightness"), "main": {k: [] for k in methods}, "time_ms": {k: 0.0 for k in methods},
           "ablation": {}, "sweep": {}, "examples": {}}
    worst = (1e9, None)
    for i, (y, x) in enumerate(boxes):
        y0, x0, h, w = _region(sc, y, x)
        raw = _raw(sc, sc.noisy_region(0, y0, x0, h, w))
        gt = sc.gt_crop(y, x, CROP)
        outs = {}
        for k, meth in methods.items():
            t = time.perf_counter()
            out = run_method(meth, raw)[y - y0:y - y0 + CROP, x - x0:x - x0 + CROP]
            res["time_ms"][k] += (time.perf_counter() - t) * 1000
            res["main"][k].append(_metrics(out, gt))
            outs[k] = out
        key = "Hybrid teacher FP32" if "Hybrid teacher FP32" in outs else list(outs)[-1]
        if res["main"][key][-1]["psnr"] < worst[0]:
            worst = (res["main"][key][-1]["psnr"], {k: _jpg(v) for k, v in outs.items()} | {"Ground truth": _jpg(gt)})
        if i == 0:
            res["examples"]["first"] = {k: _jpg(v) for k, v in outs.items()} | {"Ground truth": _jpg(gt)}
        if i < n_ablation and ablation_method in methods:
            for ab, cfg in ABLATIONS.items():
                out = run_method(methods[ablation_method], raw, cfg)[y - y0:y - y0 + CROP, x - x0:x - x0 + CROP]
                res["ablation"].setdefault(ab, []).append(_metrics(out, gt, with_lpips=False))
    res["examples"]["worst"] = worst[1]
    res["examples"]["worst_psnr"] = worst[0]
    # Illumination / noise sweep: clean RAW darkened by s, real noise model, digital gain 1/s.
    rng = np.random.default_rng(seed)
    a, b = sc.nlf or (1e-4, 1e-6)
    sweep_methods = sweep_methods or [k for k in methods if "FP16" not in k and "INT8" not in k]
    for s in EXPOSURES:
        for y, x in boxes[:n_sweep]:
            y0, x0, h, w = _region(sc, y, x)
            clean = sc.gt_raw_region(y0, x0, h, w)
            lin = clean * s
            noisy = (lin + np.sqrt(np.maximum(a * lin + b, 0)) * rng.standard_normal(lin.shape).astype(np.float32)) / s
            raw = _raw(sc, np.clip(noisy, 0, None).astype(np.float32), (a / s, b / s ** 2))
            gt = sc.gt_crop(y, x, CROP)
            for k in sweep_methods:
                out = run_method(methods[k], raw)[y - y0:y - y0 + CROP, x - x0:x - x0 + CROP]
                res["sweep"].setdefault(k, {}).setdefault(str(s), []).append(_metrics(out, gt, with_lpips=False))
    return res
