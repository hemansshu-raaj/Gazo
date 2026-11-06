"""Gradio app: Hybrid smartphone ISP (RAW -> sRGB) with traditional and fully learned baselines.

Run:  python app.py   (http://localhost:7860)
"""
import glob
import json
import os
import time

import gradio as gr
import numpy as np

from isp.engine import OnnxEngine
from isp.io import load_any
from isp.metrics import delta_e, psnr, ssim
from isp.pipelines import hybrid, learned, traditional
from isp.pipelines.common import fix_orientation
from isp.types import ISPInputError

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(ROOT, "weights")
THREADS = int(os.environ.get("ISP_THREADS", "0"))
MAX_FAST_PIXELS = 4_000_000  # keep CPU latency well under ~10 s

MODEL_FILES = {
    "Hybrid student (INT8 weights, 0.7 MB)": "student_int8.onnx",
    "Hybrid teacher (FP32, best quality)": "teacher_fp32.onnx",
}
E2E_FILE = "e2e_fp32.onnx"
_engines = {}


def engine(fname):
    if fname not in _engines:
        path = os.path.join(WEIGHTS, fname)
        if not os.path.exists(path):
            raise gr.Error(f"Model file {fname} is missing on the server.")
        _engines[fname] = OnnxEngine(path, threads=THREADS)
    return _engines[fname]


def _center_crop(raw, max_pixels):
    h, w = raw.shape
    if h * w <= max_pixels:
        return raw, False
    s = int(np.sqrt(max_pixels)) // 2 * 2
    ch, cw = min(h, s), min(w, s)
    y, x = (h - ch) // 4 * 2, (w - cw) // 4 * 2
    raw.bayer = np.ascontiguousarray(raw.bayer[y:y + ch, x:x + cw])
    raw.meta["crop"] = [y, x, ch, cw]
    return raw, True


def _load_gt(path, raw):
    if not path.endswith(".npz"):
        return None
    z = np.load(path, allow_pickle=False)
    if "gt" not in z:
        return None
    gt = z["gt"].astype(np.float32) / 255.0
    if "crop" in raw.meta:
        y, x, h, w = raw.meta["crop"]
        gt = gt[y:y + h, x:x + w]
    return gt


def _metrics(img, gt):
    if gt is None:
        return None
    return {"PSNR (dB)": round(psnr(img, gt), 2), "SSIM": round(ssim(img, gt), 4), "ΔE2000": round(delta_e(img, gt), 2)}


def _to_u8(img, orientation, max_side=None):
    img = np.clip(fix_orientation(img, orientation), 0, 1)
    if max_side and max(img.shape[:2]) > max_side:
        import cv2
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return (img * 255 + 0.5).astype(np.uint8)


def process(file, mode, model_name, auto_exposure, sharpen, full_res, progress=gr.Progress()):
    if file is None:
        raise gr.Error("Upload a RAW file (DNG) or pick an example below.")
    path = file if isinstance(file, str) else file.name
    t_start = time.perf_counter()
    progress(0.05, desc="Reading RAW")
    try:
        raw = load_any(path)
    except ISPInputError as e:
        raise gr.Error(str(e))
    except Exception as e:  # noqa: BLE001 - never show a stack trace to the user
        raise gr.Error(f"Could not read this file ({type(e).__name__}).")
    cropped = False
    if not full_res:
        raw, cropped = _center_crop(raw, MAX_FAST_PIXELS if mode != "Compare all" else MAX_FAST_PIXELS // 2)
    gt = _load_gt(path, raw)
    cfg_h = {"auto_exposure": auto_exposure, "sharpen": sharpen}
    cfg_t = {"auto_exposure": auto_exposure, "sharpen": sharpen}
    results, stages, info = {}, {}, {}
    try:
        if mode in ("Hybrid", "Compare all"):
            progress(0.3, desc="Hybrid ISP")
            out, st, inf = hybrid.run(raw, engine(MODEL_FILES[model_name]), cfg_h, keep_stages=True)
            results["Hybrid"] = out
            stages, info["Hybrid"] = st, inf
        if mode in ("Traditional", "Compare all"):
            progress(0.6, desc="Traditional ISP")
            out, st, inf = traditional.run(raw, cfg_t, keep_stages=(mode == "Traditional"))
            results["Traditional"] = out
            info["Traditional"] = inf
            if mode == "Traditional":
                stages = st
        if mode in ("Learned end-to-end", "Compare all"):
            progress(0.8, desc="Learned end-to-end ISP")
            out, st, inf = learned.run(raw, engine(E2E_FILE), keep_stages=(mode == "Learned end-to-end"))
            results["Learned E2E"] = out
            info["Learned E2E"] = inf
            if mode == "Learned end-to-end":
                stages = st
    except gr.Error:
        raise
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Processing failed ({type(e).__name__}: {e}).")
    main_name = next(iter(results))
    o = raw.orientation
    final = _to_u8(results[main_name], o)
    compare = [(_to_u8(v, o, 1600), k) for k, v in results.items()]
    if gt is not None:
        compare.append((_to_u8(gt, o, 1600), "Ground truth (150-frame average)"))
    gallery = [(fix_orientation(v, o), k) for k, v in stages.items()]
    rows = []
    for k, v in results.items():
        m = _metrics(v, gt)
        ms = sum(info[k]["timings_ms"].values())
        rows.append(f"| {k} | " + (" | ".join(str(x) for x in m.values()) if m else "– | – | –") + f" | {ms:,.0f} |")
    table = "| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | Time (ms) |\n|---|---|---|---|---|\n" + "\n".join(rows)
    h, w = raw.shape
    notes = [f"**Input:** {os.path.basename(path)} · {w}×{h} px RAW"
             + (f" (centre {raw.shape[0] * raw.shape[1] / 1e6:.0f} MP crop for speed; tick *Full resolution* for the whole image)" if cropped else ""),
             f"**Total time:** {time.perf_counter() - t_start:.1f} s"]
    if gt is None:
        notes.append("No ground truth for this input, so only timings are shown.")
    details = {k: {kk: vv for kk, vv in v.items() if kk != "timings_ms"} | {"timings_ms": {kk: round(vv) for kk, vv in
               v["timings_ms"].items()}} for k, v in info.items()}
    return final, compare, gallery, "\n\n".join(notes) + "\n\n" + table, details


EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "examples", "*.npz"))) + sorted(glob.glob(os.path.join(ROOT, "examples", "*.dng")))

with gr.Blocks(title="Hybrid Smartphone ISP") as demo:
    gr.Markdown(
        "# Hybrid Smartphone ISP: RAW → sRGB\n"
        "Classical camera stages (white balance, noise model, colour matrix, tone curve) with a small neural "
        "network doing joint **denoising + demosaicing**, compared against a **traditional ISP** and a "
        "**fully learned ISP**. Upload a phone **DNG** (Android Pro/RAW mode) or try an example.")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            f = gr.File(label="RAW file (.dng, SIDD .npz, 16-bit Bayer .png)",
                        file_types=[".dng", ".DNG", ".npz", ".png", ".tif", ".tiff", ".nef", ".cr2", ".arw"])
            mode = gr.Radio(["Hybrid", "Traditional", "Learned end-to-end", "Compare all"], value="Compare all",
                            label="Pipeline")
            model = gr.Dropdown(list(MODEL_FILES), value=list(MODEL_FILES)[0], label="Hybrid model")
            with gr.Accordion("Options", open=False):
                ae = gr.Checkbox(False, label="Adaptive exposure (brighten dark captures)")
                sh = gr.Checkbox(False, label="Noise-adaptive sharpening")
                full = gr.Checkbox(False, label="Full resolution (slower)")
            btn = gr.Button("Process", variant="primary")
            gr.Examples(EXAMPLES, inputs=f, label="Examples (SIDD test scenes have ground truth; DNGs are unseen phones)")
        with gr.Column(scale=2, min_width=320):
            out = gr.Image(label="Output", type="numpy", format="jpeg")
            report = gr.Markdown()
    comp = gr.Gallery(label="Comparison", columns=4, height="auto", object_fit="contain", format="jpeg")
    stages = gr.Gallery(label="Intermediate stages", columns=6, height="auto", object_fit="contain", format="jpeg")
    details = gr.JSON(label="Details (white balance, noise model, timings)")
    btn.click(process, [f, mode, model, ae, sh, full], [out, comp, stages, report, details])

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
