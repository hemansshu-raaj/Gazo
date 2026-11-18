"""Everything between the HTTP routes and the ISP code: models, examples, cropping, image encoding."""
import base64
import functools
import os
import time

import cv2
import numpy as np

from isp.engine import OnnxEngine
from isp.io import load_any
from isp.metrics import delta_e, psnr, ssim
from isp.pipelines.common import fix_orientation

from . import trace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.environ.get("ISP_WEIGHTS", os.path.join(ROOT, "weights"))
EXAMPLES_DIR = os.environ.get("ISP_EXAMPLES", os.path.join(ROOT, "examples"))
THREADS = int(os.environ.get("ISP_THREADS", "0"))

MAX_PIXELS = 4_000_000        # bigger uploads are centre-cropped so a CPU answers in a few seconds
COMPARE_PIXELS = 2_000_000    # the comparison runs all three pipelines, so it uses a smaller crop
COMPARE_SIDE = 720            # comparison tiles are shown at ~350 px (x2 for sharp screens)
PREVIEW_SIDE = 960            # longest side of every stage image sent to the browser
ZOOM_SIZE = 40                # zoom window, in sensor pixels (shows the photosite grid)

METHODS = {  # url name -> (trace function, model file or None)
    "classic": ("trace_classic", None),
    "learned": ("trace_learned", "e2e_fp32.onnx"),
    "hybrid": ("trace_hybrid", "student_int8.onnx"),
}

HYBRID_MODELS = {"student": "student_int8.onnx", "teacher": "teacher_fp32.onnx"}


def clean_options(opts: dict | None) -> dict:
    """The phone's Advanced screen settings, validated. sharpen: None = each ISP's own default."""
    o = opts or {}
    model = o.get("model", "student")
    return {
        "model": model if model in HYBRID_MODELS else "student",
        "auto_exposure": bool(o.get("auto_exposure", False)),
        "sharpen": None if o.get("sharpen") in (None, "auto") else bool(o.get("sharpen")),
        "full_res": bool(o.get("full_res", False)),
    }


# The comparison uses the best-quality hybrid model (the method page keeps the fast student).
COMPARE_MODELS = {"hybrid": "teacher_fp32.onnx"}

# Bundled examples. Titles are what the site shows; edit freely.
EXAMPLES = {
    "sidd_iso10000_pixel": "Google Pixel, ISO 10,000",
    "sidd_lowlight_iphone_iso100": "iPhone, low light",
    "sidd_normal_nexus6": "Nexus 6, daylight",
    "unseen_pixel7pro": "Pixel 7 Pro (never seen in training)",
    "unseen_pixel4a": "Pixel 4a (never seen in training)",
}


# ---------------------------------------------------------------- models
@functools.lru_cache(maxsize=None)
def engine(fname: str) -> OnnxEngine:
    return OnnxEngine(os.path.join(WEIGHTS, fname), threads=THREADS)


# ---------------------------------------------------------------- examples
def example_path(example_id: str) -> str | None:
    if example_id not in EXAMPLES:
        return None
    for ext in (".npz", ".dng"):
        p = os.path.join(EXAMPLES_DIR, example_id + ext)
        if os.path.exists(p):
            return p
    return None


def list_examples() -> list[dict]:
    out = []
    for eid, title in EXAMPLES.items():
        p = example_path(eid)
        if p:
            out.append({"id": eid, "title": title, "file": os.path.basename(p),
                        "has_ground_truth": p.endswith(".npz") and "gt" in np.load(p, allow_pickle=False)})
    return out


@functools.lru_cache(maxsize=None)
def example_thumbnail(example_id: str) -> bytes:
    """Small JPEG of the classic pipeline's output, for the example cards on the front page."""
    raw = load_any(example_path(example_id))
    crop(raw, 1_000_000)
    img, _ = trace.trace_classic(raw)
    return encode_jpeg(fix_orientation(img, raw.orientation), 140, quality=82)


# ---------------------------------------------------------------- image helpers
def crop(raw, max_pixels: int) -> bool:
    """Centre crop (in place, keeping the RGGB phase) when the image is bigger than max_pixels."""
    h, w = raw.shape
    if h * w <= max_pixels:
        return False
    s = int(np.sqrt(max_pixels)) // 2 * 2
    ch, cw = min(h, s), min(w, s)
    y, x = (h - ch) // 4 * 2, (w - cw) // 4 * 2
    raw.bayer = np.ascontiguousarray(raw.bayer[y:y + ch, x:x + cw])
    raw.meta["crop"] = [y, x, ch, cw]
    return True


def encode_jpeg(img: np.ndarray, max_side: int | None = PREVIEW_SIDE, quality: int = 88) -> bytes:
    img = np.clip(img, 0, 1)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    u8 = (img * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def jpeg_entry(img: np.ndarray, max_side: int = PREVIEW_SIDE, quality: int = 88) -> dict:
    """{"src": data URL, "w", "h"}: the browser gets the size up front (no layout shift)."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return {"src": data_url(encode_jpeg(img, max_side, quality)), "w": round(w * s), "h": round(h * s)}


def encode_png(img: np.ndarray) -> bytes:
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return cv2.imencode(".png", u8[..., ::-1])[1].tobytes()


def data_url(b: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(b).decode()


def detail_spot(img: np.ndarray, size: int) -> tuple[float, float]:
    """Centre (as fractions of width/height) of the most detailed area: where the zoom looks."""
    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.resize(g, (64, max(1, round(64 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA)
    energy = cv2.blur(np.abs(cv2.Laplacian(small, cv2.CV_32F)), (5, 5))
    midtone = cv2.blur(((small > 40) & (small < 215)).astype(np.float32), (7, 7))  # skip clipped / black areas
    energy = energy * midtone ** 2
    m = 4  # keep away from the borders
    energy[:m], energy[-m:], energy[:, :m], energy[:, -m:] = 0, 0, 0, 0
    y, x = np.unravel_index(int(np.argmax(energy)), energy.shape)
    return (x + 0.5) / energy.shape[1], (y + 0.5) / energy.shape[0]


def zoom(img: np.ndarray, spot: tuple[float, float], sensor_size: tuple[int, int], px: int = ZOOM_SIZE) -> np.ndarray:
    """Nearest-neighbour blow-up of a px x px sensor-pixel window, so single photosites are visible."""
    h, w = img.shape[:2]
    sh, sw = sensor_size
    # stage images may be at half (visualize) or double (packed view) resolution: scale the window
    win = max(4, round(px * w / sw))
    cx, cy = int(spot[0] * w), int(spot[1] * h)
    x0 = int(np.clip(cx - win // 2, 0, max(0, w - win))) // 2 * 2
    y0 = int(np.clip(cy - win // 2, 0, max(0, h - win))) // 2 * 2
    patch = img[y0:y0 + win, x0:x0 + win]
    return cv2.resize(patch, (256, 256), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------- the two jobs the site asks for
def run_method(path: str, method: str, options: dict | None = None) -> dict:
    """Run one pipeline with every stage recorded. Returns JSON-ready data (images as data URLs)."""
    opts = clean_options(options)
    fn_name, model = METHODS[method]
    if method == "hybrid":
        model = HYBRID_MODELS[opts["model"]]
    cfg = {"auto_exposure": opts["auto_exposure"]}
    if opts["sharpen"] is not None:
        cfg["sharpen"] = opts["sharpen"]
    t0 = time.perf_counter()
    raw = load_any(path)
    cropped = False if opts["full_res"] else crop(raw, MAX_PIXELS)
    fn = getattr(trace, fn_name)
    if method == "learned":            # one network, no settings to pass
        final, stages = fn(raw, engine(model))
    else:
        final, stages = fn(raw, engine(model), cfg) if model else fn(raw, cfg)
    ms = (time.perf_counter() - t0) * 1000
    o = raw.orientation
    spot = detail_spot(final, ZOOM_SIZE)
    sensor = raw.shape
    out_stages = []
    for st in stages:
        img = fix_orientation(st["image"], o) if st["id"] != "pack" else st["image"]
        z = zoom(st["image"], spot, sensor) if st["id"] not in ("pack", "noise_map") else None
        out_stages.append({
            "id": st["id"],
            "image": jpeg_entry(img),
            "zoom": data_url(encode_png(fix_orientation(z, o)), "image/png") if z is not None else None,
            "info": st["info"],
        })
    h, w = raw.shape
    return {
        "method": method, "options": opts,
        "width": w, "height": h, "cropped": cropped,
        "time_ms": round(ms),
        "final": jpeg_entry(fix_orientation(final, o), 1600, quality=92),
        "stages": out_stages,
    }


@functools.lru_cache(maxsize=None)
def compare_example(example_id: str) -> dict:
    """All three pipelines on one bundled example, plus metrics when it has a ground truth."""
    path = example_path(example_id)
    gt = None
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        gt = z["gt"].astype(np.float32) / 255.0 if "gt" in z else None
    results = []
    orientation = 1
    for method, (fn_name, model) in METHODS.items():
        model = COMPARE_MODELS.get(method, model)
        raw = load_any(path)
        crop(raw, COMPARE_PIXELS)
        orientation = raw.orientation
        fn = getattr(trace, fn_name)
        t0 = time.perf_counter()
        img, _ = fn(raw, engine(model)) if model else fn(raw)
        ms = (time.perf_counter() - t0) * 1000
        row = {"method": method, "model": model or "classical", "time_ms": round(ms),
               "image": jpeg_entry(fix_orientation(img, orientation), COMPARE_SIDE, quality=88)}
        if gt is not None:
            g = gt
            if "crop" in raw.meta:
                y, x, h, w = raw.meta["crop"]
                g = gt[y:y + h, x:x + w]
            row["metrics"] = {"psnr": round(float(psnr(img, g)), 2), "ssim": round(float(ssim(img, g)), 4),
                              "delta_e": round(float(delta_e(img, g)), 2)}
            gt_crop = g
        results.append(row)
    return {
        "example": example_id,
        "title": EXAMPLES[example_id],
        "ground_truth": jpeg_entry(fix_orientation(gt_crop, orientation), COMPARE_SIDE, quality=88)
        if gt is not None else None,
        "results": results,
    }
