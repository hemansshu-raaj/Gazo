"""Turn runs/eval/results.pkl (+ runs/benchmark.json) into eval_P2_<TeamID>.md, plots and failure images.

  python -m eval.report --team T00 --usns "USN1,USN2,USN3,USN4"
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np

ORDER = ["Traditional ISP", "Learned E2E", "Hybrid teacher FP32", "Hybrid teacher FP16", "Hybrid w/o noise map",
         "Student (no distillation)", "Hybrid student FP32 (KD)", "Hybrid student INT8 (KD)", "Student full 8-bit W8A8 (KD)"]
ONNX_FOR = {"Hybrid teacher FP32": "teacher_fp32.onnx", "Learned E2E": "e2e_fp32.onnx",
            "Hybrid student FP32 (KD)": "student_fp32.onnx", "Hybrid student INT8 (KD)": "student_int8.onnx",
            "Student full 8-bit W8A8 (KD)": "student_kd_w8a8.onnx"}
GPU_FOR = {"Hybrid teacher FP32": "teacher_fp32", "Hybrid teacher FP16": "teacher_fp16", "Learned E2E": "e2e_fp32",
           "Hybrid student FP32 (KD)": "student_kd_fp32", "Student (no distillation)": "student_scratch_fp32"}
PARAMS = {"Learned E2E": 2.653, "Hybrid teacher FP32": 2.653, "Hybrid teacher FP16": 2.653,
          "Hybrid w/o noise map": 2.652, "Student (no distillation)": 0.630, "Hybrid student FP32 (KD)": 0.630,
          "Hybrid student INT8 (KD)": 0.630, "Student full 8-bit W8A8 (KD)": 0.630}


def mean_metrics(lst):
    return {k: float(np.mean([m[k] for m in lst])) for k in lst[0]} if lst else {}


def iso_bucket(iso):
    return "ISO ≤ 400" if iso <= 400 else "ISO 800–1600" if iso <= 1600 else "ISO ≥ 3200"


def fmt(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def build(results_path="runs/eval/results.pkl", bench_path="runs/benchmark.json", team="TXX", usns="USN1,USN2,USN3,USN4",
          out_md=None, docs="docs"):
    R = pickle.load(open(results_path, "rb"))
    scenes = R["scenes"]
    gpu = R.get("gpu_latency", {})
    bench = {b["model"]: b for b in json.load(open(bench_path))} if os.path.exists(bench_path) else {}
    methods = [m for m in ORDER if any(m in s["main"] for s in scenes)]
    pooled = {m: mean_metrics([x for s in scenes for x in s["main"].get(m, [])]) for m in methods}
    n_crops = sum(len(s["main"][methods[0]]) for s in scenes)

    L = [usns, "", f"# Evaluation: P2 Intelligent and Computationally Efficient Smartphone ISP (Team {team})", "",
         f"**Test set:** {len(scenes)} SIDD captures of 2 physical scenes (003, 007) never seen in training; "
         f"5 phones, ISO 100 to 10,000, low and normal light. {n_crops} seeded 512×512 crops "
         "(32 per capture) of a single noisy frame. Ground truth = 150-frame average rendered by the reference ISP.",
         "", "## 1. Main results (metrics in the order of the problem statement)", "",
         "| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ | Params (M) | Model size (MB) | CPU latency, 1 MP, 2 threads (ms) | Peak CPU memory (MB) | GPU latency, 1 MP, " + gpu.get("gpu", "GPU").replace("NVIDIA ", "") + " (ms) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        p = pooled[m]
        b = bench.get(ONNX_FOR.get(m, ""), {})
        g = gpu.get(GPU_FOR.get(m, ""), {})
        size = b.get("size_mb")
        if m == "Hybrid teacher FP16" and "teacher_fp32.onnx" in bench:
            size = bench["teacher_fp32.onnx"]["size_mb"] / 2
        L.append(f"| {m} | {fmt(p['psnr'])} | {fmt(p['ssim'], 4)} | {fmt(p['delta_e'])} | {fmt(p.get('lpips'), 4)} | "
                 f"{'0 (classical)' if m == 'Traditional ISP' else fmt(PARAMS.get(m), 3)} | {fmt(size)} | "
                 f"{fmt(b.get('cpu2_1024px_ms'), 0)} | {fmt(b.get('peak_mem_mb_1024px'), 0)} | {fmt(g.get('ms_1MP'), 1)} |")
    trad_ms = np.mean([s["time_ms"]["Traditional ISP"] / len(s["main"]["Traditional ISP"]) for s in scenes])
    L += ["", f"Traditional ISP CPU time: {trad_ms:.0f} ms per 512×512 crop (whole pipeline, numpy/OpenCV). "
          "Network latencies are for the network stage only; the classical stages add the same cost to every hybrid variant.", ""]

    def breakdown(title, keyfn, ms):
        groups = defaultdict(lambda: defaultdict(list))
        for s in scenes:
            for m in ms:
                groups[keyfn(s)][m] += s["main"].get(m, [])
        out = ["", f"## {title}", "", "| Group | " + " | ".join(f"{m} PSNR / ΔE" for m in ms) + " |",
               "|---|" + "---|" * len(ms)]
        for g in sorted(groups):
            out.append(f"| {g} | " + " | ".join(
                f"{fmt(mean_metrics(groups[g][m]).get('psnr'))} / {fmt(mean_metrics(groups[g][m]).get('delta_e'))}"
                if groups[g][m] else "–" for m in ms) + " |")
        return out

    key = [m for m in ["Traditional ISP", "Learned E2E", "Hybrid teacher FP32", "Hybrid student INT8 (KD)"] if m in methods]
    L += breakdown("2. Robustness by noise level (ISO)", lambda s: iso_bucket(s["iso"]), key)
    L += breakdown("3. Robustness by illumination (SIDD brightness code)", lambda s: s["brightness"], key)
    L += breakdown("4. Robustness by phone (sensor)", lambda s: s["camera"], key)

    # Sweep
    sw = defaultdict(lambda: defaultdict(list))
    for s in scenes:
        for m, d in s["sweep"].items():
            for e, lst in d.items():
                sw[m][float(e)] += lst
    exps = sorted({e for m in sw for e in sw[m]}, reverse=True)
    if exps:
        L += ["", "## 5. Degradation sweep: darker captures (clean RAW × exposure, real noise model, digital gain back)", "",
              "| Pipeline | " + " | ".join(f"×{e:g} PSNR" for e in exps) + " |", "|---|" + "---|" * len(exps)]
        for m in [x for x in ORDER if x in sw]:
            L.append(f"| {m} | " + " | ".join(fmt(mean_metrics(sw[m][e])["psnr"]) for e in exps) + " |")
        L += ["", "![sweep](docs/sweep.png)"]
        _plot_sweep(sw, exps, os.path.join(docs, "sweep.png"))

    # Ablations
    ab = defaultdict(list)
    for s in scenes:
        for k, lst in s["ablation"].items():
            ab[k] += lst
    L += ["", "## 6. Ablation study", "", "**Inference-time ablations on the hybrid teacher** (16 crops per capture):", "",
          "| Variant | PSNR (dB) | SSIM | ΔE2000 | Δ PSNR vs full |", "|---|---|---|---|---|"]
    base = mean_metrics(ab["full (metadata WB + NoiseProfile)"])["psnr"] if ab else 0
    for k, lst in ab.items():
        mm = mean_metrics(lst)
        L.append(f"| {k} | {fmt(mm['psnr'])} | {fmt(mm['ssim'], 4)} | {fmt(mm['delta_e'])} | {mm['psnr'] - base:+.2f} |")
    L += ["", "**Retrained / component ablations** (full test set):", "", "| Comparison | A | B | Δ PSNR (A − B) | Δ ΔE (A − B) |",
          "|---|---|---|---|---|"]
    for name, a, b in [("Hybrid vs fully learned (same network, same size)", "Hybrid teacher FP32", "Learned E2E"),
                       ("Hybrid vs traditional ISP", "Hybrid teacher FP32", "Traditional ISP"),
                       ("Noise-map conditioning (with vs without)", "Hybrid teacher FP32", "Hybrid w/o noise map"),
                       ("Knowledge distillation (KD vs from scratch)", "Hybrid student FP32 (KD)", "Student (no distillation)"),
                       ("INT8 weights + 16-bit activations (W8A16) vs FP32", "Hybrid student INT8 (KD)", "Hybrid student FP32 (KD)"),
                       ("Full 8-bit (W8A8) vs FP32", "Student full 8-bit W8A8 (KD)", "Hybrid student FP32 (KD)"),
                       ("FP16 inference (FP16 vs FP32)", "Hybrid teacher FP16", "Hybrid teacher FP32"),
                       ("Compression total (student INT8 vs teacher)", "Hybrid student INT8 (KD)", "Hybrid teacher FP32")]:
        if a in pooled and b in pooled:
            L.append(f"| {name} | {a} | {b} | {pooled[a]['psnr'] - pooled[b]['psnr']:+.2f} | "
                     f"{pooled[a]['delta_e'] - pooled[b]['delta_e']:+.2f} |")

    # Failure cases: worst crops for the teacher
    worst = sorted([s for s in scenes if s["examples"].get("worst")], key=lambda s: s["examples"]["worst_psnr"])[:3]
    os.makedirs(os.path.join(docs, "failures"), exist_ok=True)
    L += ["", "## 7. Failure cases (lowest-PSNR crops of the hybrid teacher)", ""]
    for i, s in enumerate(worst, 1):
        path = os.path.join(docs, "failures", f"failure_{i}_{s['scene']}.jpg")
        _montage(s["examples"]["worst"], path)
        L.append(f"{i}. **{s['scene']}** ({s['camera']}, ISO {s['iso']}, {s['brightness']} light): teacher PSNR "
                 f"{s['examples']['worst_psnr']:.2f} dB. ![]({path})")
    for s in scenes[:1]:
        _montage(s["examples"]["first"], os.path.join(docs, "example_comparison.jpg"))
    summary = {"n_scenes": len(scenes), "n_crops": n_crops, "pooled": pooled, "gpu_latency": gpu,
               "per_scene": [{"scene": s["scene"], "camera": s["camera"], "iso": s["iso"], "brightness": s["brightness"],
                              **{m: mean_metrics(v) for m, v in s["main"].items()}} for s in scenes],
               "ablation": {k: mean_metrics(v) for k, v in ab.items()},
               "sweep": {m: {str(e): mean_metrics(sw[m][e]) for e in exps} for m in sw}}
    os.makedirs("results", exist_ok=True)
    json.dump(summary, open("results/eval_summary.json", "w"), indent=1)
    md = "\n".join(L) + "\n"
    out_md = out_md or f"eval_P2_{team}.md"
    open(out_md, "w").write(md)
    print(md)
    return pooled


def _plot_sweep(sw, exps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in [x for x in ORDER if x in sw]:
        ax.plot([1 / e for e in exps], [mean_metrics(sw[m][e])["psnr"] for e in exps], marker="o", label=m)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Underexposure factor (digital gain applied)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Degradation under darker captures (test scenes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _montage(imgs: dict, path: str):
    import cv2

    tiles = []
    for name, b in imgs.items():
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (320, 320), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (320, 22), (0, 0, 0), -1)
        cv2.putText(im, name[:34], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="TXX")
    ap.add_argument("--usns", default="USN1,USN2,USN3,USN4")
    ap.add_argument("--results", default="runs/eval/results.pkl")
    a = ap.parse_args()
    build(a.results, team=a.team, usns=a.usns)
