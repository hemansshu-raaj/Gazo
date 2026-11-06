"""End-to-end latency of each complete pipeline on a 1-megapixel RAW (1024x1024 output), CPU, fixed threads.

Unlike optimize/benchmark.py (network only), this times everything a user waits for: white balance, noise
model, network (tiled), colour correction, gamma/tone. Median of N runs after a warm-up.
  python -m optimize.pipeline_latency --threads 2 --out results/pipeline_latency.json
"""
import argparse
import json
import os
import time

import numpy as np

from isp.engine import OnnxEngine
from isp.io import load_any
from isp.pipelines import hybrid, learned, traditional


def main(threads=2, reps=5, out=""):
    raw = load_any("examples/sidd_normal_nexus6.npz")  # 1024x1024 SIDD crop with full metadata
    W = "weights"
    runs = {
        "Traditional ISP": lambda: traditional.run(raw, {"denoise": "bilateral", "sharpen": True}),
        "Learned E2E (FP32)": (lambda e: lambda: learned.run(raw, e))(OnnxEngine(f"{W}/e2e_fp32.onnx", threads)),
        "Hybrid teacher (FP32)": (lambda e: lambda: hybrid.run(raw, e))(OnnxEngine(f"{W}/teacher_fp32.onnx", threads)),
        "Hybrid student (FP32)": (lambda e: lambda: hybrid.run(raw, e))(OnnxEngine(f"{W}/student_fp32.onnx", threads)),
        "Hybrid student (W8A16, deployed)": (lambda e: lambda: hybrid.run(raw, e))(
            OnnxEngine(f"{W}/student_int8.onnx", threads)),
    }
    import cv2
    cv2.setNumThreads(threads)
    res = {}
    for name, fn in runs.items():
        fn()
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1000)
        res[name] = round(float(np.median(ts)), 1)
        print(f"{name:34s} {res[name]:8.1f} ms / MP", flush=True)
    payload = {"input": "1024x1024 RAW (1 MP), examples/sidd_normal_nexus6.npz", "threads": threads,
               "cpu": os.popen("lscpu | grep 'Model name' | cut -d: -f2").read().strip(), "median_ms": res}
    if out:
        json.dump(payload, open(out, "w"), indent=1)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/pipeline_latency.json")
    a = ap.parse_args()
    main(a.threads, a.reps, a.out)
