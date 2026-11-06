"""Measure model size, parameters, latency and peak memory of each ONNX model (CPU).

Each model runs in a fresh subprocess so peak RSS is attributable to that model.
  python -m optimize.benchmark runs/onnx/*.onnx --out runs/benchmark.json
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np


def count_params(path: str) -> int:
    import onnx
    from onnx import numpy_helper

    m = onnx.load(path)
    quantized_weights = {n.input[0] for n in m.graph.node if n.op_type == "DequantizeLinear"}
    total = 0
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim >= 1 and (init.name in quantized_weights or arr.dtype in (np.float32, np.float16)):
            if arr.size > 1:
                total += arr.size
    return int(total)


def _child(path: str, threads: int, size: int, reps: int):
    from isp.engine import OnnxEngine, run_tiled

    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    eng = OnnxEngine(path, threads=threads)
    x = np.random.rand(size // 2, size // 2, eng.in_ch).astype(np.float32) * 0.3
    run_tiled(eng, x)  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        run_tiled(eng, x)
        ts.append((time.perf_counter() - t) * 1000)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"latency_ms": float(np.median(ts)), "peak_rss_mb": peak / 1024,
                      "model_rss_mb": (peak - base) / 1024}))


def bench(path: str, threads: int, size: int, reps: int) -> dict:
    out = subprocess.run([sys.executable, "-m", "optimize.benchmark", "--child", path, str(threads), str(size),
                          str(reps)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main(paths, out, sizes=(1024,), threads=(2, 4), reps=5):
    rows = []
    for p in paths:
        r = {"model": os.path.basename(p), "size_mb": os.path.getsize(p) / 1e6, "params": count_params(p)}
        for s in sizes:
            for t in threads:
                b = bench(p, t, s, reps)
                r[f"cpu{t}_{s}px_ms"] = round(b["latency_ms"], 1)
                r[f"peak_mem_mb_{s}px"] = round(b["peak_rss_mb"], 1)
        rows.append(r)
        print(r, flush=True)
    if out:
        json.dump(rows, open(out, "w"), indent=1)
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--out", default="")
    ap.add_argument("--sizes", default="1024"); ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    main(a.paths, a.out, tuple(int(s) for s in a.sizes.split(",")), reps=a.reps)
