"""Render ground-truth sRGB for every scene + first traditional-ISP baseline on the test split.

GT = reference classical renderer (metadata WB -> Malvar demosaic -> interpolated CCM -> sRGB gamma
-> SIDD tone curve) applied to the clean 150-frame-average RAW. Saved as uint16 gt_srgb.npy.

  PYTHONPATH=. modal run modal_app/render_gt.py
"""
import json
import os

import modal

from common import APP_NAME, CKPT_DIR, DATA_DIR, ckpt_vol, data_vol, train_image

app = modal.App(APP_NAME + "-gt")

TRAD_VARIANTS = {
    "no_denoise": {"denoise": None, "sharpen": False},
    "bilateral": {"denoise": "bilateral", "sharpen": True},
    "nlm": {"denoise": "nlm", "sharpen": True},
}


@app.function(image=train_image, volumes={DATA_DIR: data_vol}, cpu=4, memory=12288, timeout=1800)
def render_scene(scene: str, split: str, score: bool = False) -> dict:
    import numpy as np

    from isp.data import eval_boxes
    from isp.io import load_scene
    from isp.metrics import all_metrics
    from isp.pipelines import traditional

    d = os.path.join(DATA_DIR, "scenes", scene)
    out_path = os.path.join(d, "gt_srgb.npy")
    if not os.path.exists(out_path):
        gt = traditional.render_reference(load_scene(d, "gt"))
        np.save(out_path, (np.clip(gt, 0, 1) * 65535 + 0.5).astype(np.uint16))
        data_vol.commit()
    res = {"scene": scene, "split": split}
    if split != "test" and not score:
        return res
    gt = np.load(out_path).astype(np.float32) / 65535.0
    noisy = load_scene(d, "noisy_0")
    boxes = eval_boxes(scene, *gt.shape[:2])
    for name, cfg in TRAD_VARIANTS.items():
        out, _, info = traditional.run(noisy, cfg)
        ms = [all_metrics(out[y:y + 512, x:x + 512], gt[y:y + 512, x:x + 512], with_lpips=False) for y, x in boxes]
        res[name] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        res[name]["ms_full"] = float(sum(info["timings_ms"].values()))
    return res


@app.local_entrypoint()
def main(only_split: str = ""):
    with open("splits/splits.json") as f:
        splits = json.load(f)
    jobs = [(s, k, k == only_split) for k, v in splits.items() for s in v if not only_split or k == only_split]
    results = []
    for r in render_scene.starmap(jobs, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("ERROR", r)
            continue
        results.append(r)
        if r["split"] == (only_split or "test"):
            print(r["scene"], {k: round(v["psnr"], 2) for k, v in r.items() if isinstance(v, dict)})
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/traditional_baseline{'_' + only_split if only_split else ''}.json", "w") as f:
        json.dump(results, f, indent=1)
    test = [r for r in results if r["split"] == (only_split or "test")]
    for name in TRAD_VARIANTS:
        vals = [r[name] for r in test if name in r]
        if vals:
            print(f"{name:12s}", {k: round(sum(v[k] for v in vals) / len(vals), 3) for k in vals[0]})
