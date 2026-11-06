"""Load RAW inputs into a RawImage.

Supported:
  * DNG and other camera RAW files (rawpy / LibRaw), metadata from DNG tags via exifread
  * preprocessed SIDD scene folders (noisy_k.npy / gt.npy + meta.json from modal_app/prepare_data.py)
  * .npz with `bayer` (+ optional `meta` json string)
  * 16-bit single-channel PNG holding an RGGB mosaic
"""
import io as _io
import json
import os
import re

import numpy as np

from .classical.bayer import crop_to_rggb, mosaic_rgb
from .classical.ccm import cam2srgb_from_color_matrix, interpolate_color_matrix
from .types import ISPInputError, RawImage

MAX_BYTES = 80 * 1024 * 1024
RAW_EXTS = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}

_TAGS = {
    "ColorMatrix1": 0xC621, "ColorMatrix2": 0xC622, "AsShotNeutral": 0xC628,
    "NoiseProfile": 0xC761, "Orientation": 0x0112, "CalibrationIlluminant1": 0xC65A,
    "CalibrationIlluminant2": 0xC65B,
}


def _floats(values) -> list:
    """Turn exifread values / stored strings like '(0.0004,)' into floats."""
    out = []
    for v in values if isinstance(values, (list, tuple)) else [values]:
        if hasattr(v, "num") and hasattr(v, "den"):
            out.append(float(v.num) / float(v.den) if v.den else 0.0)
        elif isinstance(v, (int, float, np.floating)):
            out.append(float(v))
        else:
            out += [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v))]
    return out


def _nlf_from_profile(profile) -> tuple | None:
    vals = _floats(profile) if profile is not None else []
    if len(vals) < 2:
        return None
    a = float(np.mean(vals[0::2]))
    b = float(np.mean(vals[1::2]))
    if a <= 0 and b <= 0:
        return None
    return max(a, 0.0), max(b, 0.0)


def _read_tags(data: bytes) -> dict:
    import exifread

    try:
        tags = exifread.process_file(_io.BytesIO(data), details=False)
    except Exception:
        return {}
    by_id = {getattr(v, "tag", None): v for v in tags.values()}
    return {name: _floats(by_id[t].values) for name, t in _TAGS.items() if t in by_id}


def _build(bayer_norm: np.ndarray, letters, tags: dict, cam_wb=None, rgb_xyz=None, meta=None) -> RawImage:
    bayer = crop_to_rggb(bayer_norm, letters).astype(np.float32)
    asn = tags.get("AsShotNeutral")
    if asn and len(asn) >= 3 and min(asn[:3]) > 0:
        wb = 1.0 / np.asarray(asn[:3], dtype=np.float32)
    elif cam_wb is not None and cam_wb[1] > 0:
        wb = np.asarray([cam_wb[0], cam_wb[1], cam_wb[2]], dtype=np.float32) / cam_wb[1]
    else:
        wb = None
    if wb is not None:
        wb = wb / wb[1]
    cm, cm2 = tags.get("ColorMatrix1"), tags.get("ColorMatrix2")
    meta = dict(meta or {})
    if cm and cm2 and len(cm) >= 9 and len(cm2) >= 9 and asn and len(asn) >= 3:
        xyz2cam, cct = interpolate_color_matrix(cm[:9], cm2[:9], tags.get("CalibrationIlluminant1", [17])[0],
                                                tags.get("CalibrationIlluminant2", [21])[0], asn)
        cam2srgb = cam2srgb_from_color_matrix(xyz2cam)
        meta["scene_cct"] = round(cct)
    elif cm and len(cm) >= 9:
        cam2srgb = cam2srgb_from_color_matrix(np.asarray(cm[:9]).reshape(3, 3))
    elif rgb_xyz is not None and np.abs(np.asarray(rgb_xyz)[:3]).sum() > 0:
        cam2srgb = cam2srgb_from_color_matrix(np.asarray(rgb_xyz)[:3])
    else:
        cam2srgb = None
    orient = tags.get("Orientation", [1])
    return RawImage(
        bayer=bayer, wb_gains=wb, cam2srgb=cam2srgb, nlf=_nlf_from_profile(tags.get("NoiseProfile")),
        orientation=int(orient[0]) if orient else 1, meta=meta,
    )


def _letters(raw_pattern, color_desc: str):
    return [[color_desc[i] for i in row] for row in raw_pattern]


def normalise_mosaic(mosaic: np.ndarray, raw_pattern, black_per_channel, white_level) -> np.ndarray:
    pat = np.asarray(raw_pattern)
    h, w = mosaic.shape
    colors = np.tile(pat, (h // 2 + 1, w // 2 + 1))[:h, :w]
    black = np.asarray(black_per_channel, dtype=np.float32)[colors]
    out = (mosaic.astype(np.float32) - black) / (float(white_level) - black)
    return np.clip(out, 0.0, None)


def load_dng_bytes(data: bytes, name: str = "") -> RawImage:
    import rawpy

    if len(data) > MAX_BYTES:
        raise ISPInputError("File is larger than 80 MB.")
    try:
        raw = rawpy.imread(_io.BytesIO(data))
        raw.unpack()
    except Exception as e:  # LibRaw errors are not friendly
        raise ISPInputError(f"Could not read this RAW file ({type(e).__name__}). Is it a valid DNG/RAW?") from e
    with raw:
        tags = _read_tags(data)
        cam_wb = list(map(float, raw.camera_whitebalance))
        rgb_xyz = raw.rgb_xyz_matrix.copy()
        meta = {"source": name or "raw", "camera": "", "white_level": float(raw.white_level)}
        vis = raw.raw_image_visible
        if vis.ndim == 3 or raw.raw_pattern is None:
            # Linear (already demosaiced) DNG, e.g. iPhone ProRAW: re-mosaic so the same ISP applies.
            rgb = vis[..., :3].astype(np.float32) / float(raw.white_level)
            mosaic = mosaic_rgb(rgb)
            meta["linear_dng"] = True
            return _build(mosaic, [["R", "G"], ["G", "B"]], tags, cam_wb, rgb_xyz, meta)
        desc = raw.color_desc.decode()
        if set(desc.replace("G", "").replace("R", "").replace("B", "")):
            raise ISPInputError(f"Unsupported colour filter ({desc}); only RGB Bayer sensors are supported.")
        if raw.raw_pattern.shape != (2, 2):
            raise ISPInputError("Only 2x2 Bayer sensors are supported (not X-Trans / Quad-Bayer layouts).")
        norm = normalise_mosaic(vis, raw.raw_pattern, raw.black_level_per_channel, raw.white_level)
        return _build(norm, _letters(raw.raw_pattern, desc), tags, cam_wb, rgb_xyz, meta)


def scene_tags(meta: dict) -> dict:
    t = meta.get("dng_tags", {})
    return {k: _floats(v) for k, v in t.items()}


def load_scene(scene_dir: str, which: str = "noisy_0") -> RawImage:
    """Load one frame ('noisy_k' uint16 or 'gt' float) of a preprocessed SIDD scene."""
    with open(os.path.join(scene_dir, "meta.json")) as f:
        meta = json.load(f)
    arr = np.load(os.path.join(scene_dir, f"{which}.npy"), mmap_mode="r")
    if which == "gt":
        norm = np.asarray(arr, dtype=np.float32)
    else:
        norm = normalise_mosaic(np.asarray(arr), meta["raw_pattern"], meta["black_level_per_channel"],
                                meta["white_level"])
    img = _build(norm, _letters(meta["raw_pattern"], meta["color_desc"]), scene_tags(meta),
                 meta.get("camera_whitebalance"), meta.get("rgb_xyz_matrix"),
                 {"source": "sidd", "scene": meta["scene_name"], "camera": meta.get("camera"),
                  "iso": int(meta.get("iso", 0)), "brightness": meta.get("brightness")})
    img.orientation = 1  # keep sensor orientation so GT and input stay aligned
    return img


def load_any(path: str) -> RawImage:
    """Load a file path of any supported type (raises ISPInputError with a friendly message)."""
    if os.path.isdir(path):
        return load_scene(path)
    ext = os.path.splitext(path)[1].lower()
    if os.path.getsize(path) > MAX_BYTES:
        raise ISPInputError("File is larger than 80 MB.")
    if ext in RAW_EXTS:
        with open(path, "rb") as f:
            return load_dng_bytes(f.read(), os.path.basename(path))
    if ext == ".npz":
        z = np.load(path, allow_pickle=False)
        if "bayer" not in z:
            raise ISPInputError("NPZ must contain a 'bayer' array.")
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        tags = {k: _floats(v) for k, v in meta.get("dng_tags", {}).items()}
        letters = meta.get("cfa", [["R", "G"], ["G", "B"]])
        return _build(np.asarray(z["bayer"], dtype=np.float32), letters, tags, meta={"source": "npz"})
    if ext in (".png", ".tif", ".tiff"):
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ISPInputError("Could not decode the image file.")
        if img.ndim != 2 or img.dtype != np.uint16:
            raise ISPInputError("PNG/TIFF input must be a single-channel 16-bit RGGB Bayer mosaic. "
                                "For ordinary photos use a DNG.")
        return _build(img.astype(np.float32) / 65535.0, [["R", "G"], ["G", "B"]], {}, meta={"source": "png16"})
    raise ISPInputError(f"Unsupported file type '{ext}'. Upload a DNG/RAW file, NPZ or 16-bit Bayer PNG.")
