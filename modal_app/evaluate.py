"""Full test-split evaluation on Modal (one L4 container per test scene).

  PYTHONPATH=. modal run modal_app/evaluate.py [--limit 2]
Writes runs/eval/results.pkl locally; eval/report.py turns it into tables, plots and eval_P2_T77.md.
"""
import json
import os
import pickle

import modal

from common import APP_NAME, CKPT_DIR, DATA_DIR, ckpt_vol, data_vol, train_image

app = modal.App(APP_NAME + "-eval")
INT8 = os.path.join(CKPT_DIR, "onnx", "student_int8.onnx")


@app.function(image=train_image, gpu=["L4", "A10", "T4"], cpu=4, memory=16384, timeout=3600,
              volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol})
def eval_scene(scene: str) -> dict:
    import torch

    from isp.evaluation import evaluate_scene, load_methods

    torch.backends.cudnn.benchmark = True
    methods = load_methods(CKPT_DIR, INT8, "cuda")
    return evaluate_scene(os.path.join(DATA_DIR, "scenes", scene), methods)


@app.function(image=train_image, gpu=["L4", "A10", "T4"], cpu=4, memory=16384, timeout=1800, volumes={CKPT_DIR: ckpt_vol})
def gpu_latency() -> dict:
    """GPU latency of each network on a 1024x1024 output (= 512x512 packed), batch 1, after warm-up."""
    import time

    import torch

    from isp.models import build, n_params

    out = {"gpu": torch.cuda.get_device_name(0)}
    for run in ["teacher", "e2e", "student_kd", "student_scratch"]:
        p = os.path.join(CKPT_DIR, run, "best.pt")
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location="cuda", weights_only=False)
        m = build(ck["config"]["model"]).cuda().eval()
        m.load_state_dict(ck["model"])
        in_ch = 5 if getattr(m, "use_sigma", False) else 4
        for half in (False, True):
            mm = m.half() if half else m.float()
            x = torch.rand(1, in_ch, 512, 512, device="cuda", dtype=torch.float16 if half else torch.float32)
            with torch.no_grad():
                for _ in range(5):
                    mm(x[:, :4], x[:, 4:5]) if in_ch == 5 else mm(x[:, :4])
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                ts = []
                for _ in range(20):
                    t = time.perf_counter()
                    mm(x[:, :4], x[:, 4:5]) if in_ch == 5 else mm(x[:, :4])
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t) * 1000)
            out[f"{run}_{'fp16' if half else 'fp32'}"] = {
                "ms_1MP": sorted(ts)[len(ts) // 2], "peak_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                "params": n_params(m), "best_val_psnr": ck.get("best"), "step": ck.get("step")}
    return out


@app.local_entrypoint()
def main(limit: int = 0):
    with open("splits/splits.json") as f:
        scenes = json.load(f)["test"]
    if limit:
        scenes = scenes[:limit]
    os.makedirs("runs/eval", exist_ok=True)
    lat = gpu_latency.spawn()
    results = []
    for r in eval_scene.map(scenes, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("ERROR", r)
            continue
        results.append(r)
        print(r["scene"], {k: round(sum(m["psnr"] for m in v) / len(v), 2) for k, v in r["main"].items()}, flush=True)
    payload = {"scenes": results, "gpu_latency": lat.get()}
    with open("runs/eval/results.pkl", "wb") as f:
        pickle.dump(payload, f)
    print("saved runs/eval/results.pkl with", len(results), "scenes")
