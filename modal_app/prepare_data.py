"""Build the training dataset from SIDD DNG bursts on Modal.

Source: Hugging Face `MhdMohammadi/SIDD-CC-DNG` (original SIDD DNGs, 115 scenes x 150 frames).
For each scene we:
  1. download the burst zip,
  2. average all 150 normalised Bayer frames -> clean ground-truth RAW (same idea SIDD used),
     dropping frames whose brightness is an outlier,
  3. keep 4 single noisy frames (uint16) as network inputs,
  4. keep 1 original DNG (for testing the real DNG path / app examples),
  5. save DNG metadata (black/white level, as-shot neutral, colour matrices, noise profile, CFA).

Usage:
  modal run modal_app/prepare_data.py --limit 1          # test one scene
  modal run --detach modal_app/prepare_data.py           # all scenes
"""
import io
import json
import os

import modal

from common import APP_NAME, DATA_DIR, data_image, data_vol

app = modal.App(APP_NAME + "-data")

HF_REPO = "MhdMohammadi/SIDD-CC-DNG"
KEEP_NOISY = (0, 50, 100, 149)

# DNG tags we keep (name -> EXIF tag id)
DNG_TAGS = {
    "CFARepeatPatternDim": 0x828D, "CFAPattern": 0x828E, "BlackLevel": 0xC61A,
    "WhiteLevel": 0xC61D, "ColorMatrix1": 0xC621, "ColorMatrix2": 0xC622,
    "CameraCalibration1": 0xC623, "CameraCalibration2": 0xC624,
    "AsShotNeutral": 0xC628, "CalibrationIlluminant1": 0xC65A,
    "CalibrationIlluminant2": 0xC65B, "NoiseProfile": 0xC761,
    "ForwardMatrix1": 0xC714, "ForwardMatrix2": 0xC715, "Orientation": 0x0112,
}


def _to_floats(values):
    out = []
    for v in values if isinstance(values, (list, tuple)) else [values]:
        if hasattr(v, "num") and hasattr(v, "den"):
            out.append(float(v.num) / float(v.den) if v.den else 0.0)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(str(v))
    return out


def read_dng_tags(dng_bytes: bytes) -> dict:
    import exifread

    tags = exifread.process_file(io.BytesIO(dng_bytes), details=False)
    by_id = {}
    for key, val in tags.items():
        tag_id = getattr(val, "tag", None)
        if tag_id is not None:
            by_id[tag_id] = val
    out = {}
    for name, tid in DNG_TAGS.items():
        if tid in by_id:
            out[name] = _to_floats(by_id[tid].values)
    return out


@app.function(image=data_image, volumes={DATA_DIR: data_vol}, cpu=4, memory=8192, timeout=3600)
def process_scene(row: dict) -> dict:
    import zipfile

    import numpy as np
    import rawpy
    from huggingface_hub import hf_hub_download

    inst = row["instance"]
    out_dir = os.path.join(DATA_DIR, "scenes", row["scene_name"])
    if os.path.exists(os.path.join(out_dir, "DONE")):
        return {"scene": row["scene_name"], "status": "cached"}
    os.makedirs(out_dir, exist_ok=True)

    zpath = hf_hub_download(HF_REPO, row["zip_file"], repo_type="dataset", local_dir="/tmp/hf")
    zf = zipfile.ZipFile(zpath)
    names = sorted(n for n in zf.namelist() if n.upper().endswith(".DNG"))

    def decode(name):
        data = zf.read(name)
        with rawpy.imread(io.BytesIO(data)) as raw:
            bayer = raw.raw_image_visible.astype(np.uint16).copy()
            colors = raw.raw_colors_visible.copy()
            black = np.array(raw.black_level_per_channel, dtype=np.float32)
            white = float(raw.white_level)
            info = {
                "raw_pattern": raw.raw_pattern.tolist(),
                "color_desc": raw.color_desc.decode(),
                "black_level_per_channel": black.tolist(),
                "white_level": white,
                "camera_whitebalance": list(map(float, raw.camera_whitebalance)),
                "rgb_xyz_matrix": raw.rgb_xyz_matrix.tolist(),
            }
        return data, bayer, colors, black, white, info

    total = None
    frame_means = []
    first_info = None
    norm_cache = {}
    for i, name in enumerate(names):
        data, bayer, colors, black, white, info = decode(name)
        if first_info is None:
            first_info = info
            first_info["dng_tags"] = read_dng_tags(data)
            with open(os.path.join(out_dir, "frame.dng"), "wb") as f:
                f.write(data)
            norm_cache["black_map"] = black[colors]
            norm_cache["scale"] = 1.0 / (white - black[colors])
        norm = (bayer.astype(np.float32) - norm_cache["black_map"]) * norm_cache["scale"]
        frame_means.append(float(norm.mean()))
        total = norm.astype(np.float64) if total is None else total + norm
        if i in KEEP_NOISY:
            np.save(os.path.join(out_dir, f"noisy_{KEEP_NOISY.index(i)}.npy"), bayer)

    # Drop frames with outlier brightness (flicker / exposure glitches): re-decode and subtract.
    fm = np.array(frame_means)
    med = np.median(fm)
    mad = np.median(np.abs(fm - med)) + 1e-9
    bad = [i for i, m in enumerate(fm) if abs(m - med) > max(6 * mad, 0.02 * med)]
    for i in bad:
        _, bayer, *_ = decode(names[i])
        total -= (bayer.astype(np.float32) - norm_cache["black_map"]) * norm_cache["scale"]
    n_used = len(names) - len(bad)
    gt = (total / n_used).astype(np.float32)
    np.save(os.path.join(out_dir, "gt.npy"), gt)

    meta = dict(row)
    meta.update(first_info)
    meta.update({"n_frames": len(names), "n_used": n_used, "dropped_frames": bad,
                 "kept_noisy_frames": list(KEEP_NOISY), "shape": list(gt.shape)})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    open(os.path.join(out_dir, "DONE"), "w").close()
    os.remove(zpath)
    data_vol.commit()
    return {"scene": row["scene_name"], "status": "ok", "dropped": len(bad), "shape": gt.shape}


@app.local_entrypoint()
def main(limit: int = 0, only: str = ""):
    import csv
    import urllib.request

    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/metadata/train_metadata.csv"
    text = urllib.request.urlopen(url).read().decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    if only:
        rows = [r for r in rows if r["scene_name"] in only.split(",")]
    if limit:
        rows = rows[:limit]
    print(f"processing {len(rows)} scenes")
    for res in process_scene.map(rows, return_exceptions=True, order_outputs=False):
        print(res)
