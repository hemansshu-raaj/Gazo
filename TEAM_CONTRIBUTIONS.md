**Team T77**: Problem P2, CV Hackathon 2026

# Team Contributions: P2 Hybrid Smartphone ISP

| Member | Role | Owns (code) |
|---|---|---|
| **Hemansshu Raaj RK** (1RUA24CSE0162) | Data and evaluation lead | `modal_app/prepare_data.py`, `modal_app/render_gt.py`, `isp/data.py`, `isp/train_data.py`, `isp/metrics.py`, `isp/evaluation.py`, `modal_app/evaluate.py`, `eval/`, `splits/` |
| **Gurupreeth H** (1RUA24CSE0150) | Classical ISP lead | `isp/io.py`, `isp/classical/*`, `isp/noise.py`, `isp/pipelines/traditional.py`, `tests/` |
| **Jithesh D** (1RUA24CSE0187) | Deep learning lead | `isp/models/*`, `isp/train_loop.py`, `modal_app/train.py`, `isp/pipelines/hybrid.py`, `isp/pipelines/learned.py` |
| **Hitesh S** (1RUA24CSE0167) | Optimisation, deployment and docs lead | `optimize/*`, `isp/engine.py`, `app.py`, `modal_app/serve.py`, `Dockerfile`, `Makefile`, `docs/*`, `README.md` |

Everyone reviewed the final results together and can explain the overall architecture (`docs/architecture.png`).

---

## Hemansshu Raaj RK: Data and evaluation

**What I built**
- The dataset. The official SIDD server was down, so I rebuilt SIDD from the original DNG bursts on Hugging Face:
  115 captures × 150 frames, processed in parallel on Modal. Clean RAW = mean of the 150 frames, dropping
  117 brightness-outlier frames out of 17,206.
- Ground-truth sRGB, rendered from the clean RAW with the reference classical renderer.
- The **scene-wise split**: test = scenes 003 + 007 (26 captures), val = 008, train = the rest. No physical scene
  appears in both training and test.
- The metrics (PSNR, SSIM, CIEDE2000 ΔE, LPIPS) and the evaluation protocol: 32 seeded 512² crops per capture,
  processed with a 64-px context margin so tile borders don't bias any method.
- The robustness breakdowns (ISO, light level, phone) and the darkening sweep (×1 to ×1/16 exposure with the
  sensor's real noise model).

**Numbers to quote**
- Hybrid 34.91 dB / ΔE 2.44 vs traditional 28.03 / 5.83 vs end-to-end 29.43 / 6.77.
- ISO ≥ 3200: hybrid 30.55 vs traditional 22.14. At 16× underexposure: 27.06 vs 19.45.

**Likely viva questions**
1. *Why split by scene, not randomly?* SIDD photographs the same 10 scenes many times. A random split puts the
   same objects in train and test and inflates scores.
2. *How was ground truth made?* Mean of 150 frames shot on a tripod (not aligned), which divides the noise std by ~12, then the
   same classical renderer for every method, so the comparison is fair.
3. *Why ΔE2000 and LPIPS on top of PSNR?* PSNR misses colour shifts and perceptual blur. ΔE measures colour error
   the way humans see it; LPIPS measures perceptual similarity.
4. *Why crops with a margin?* Networks run in tiles. Evaluating on a crop's interior avoids penalising edge effects.
5. *Is the test data unseen?* Yes: different physical scenes, and its noisy frames were never used in training or
   model selection (selection used the val scene).

**Failure case I present:** F2. The ground truth is blurrier than our output because of camera shake during the
burst, so PSNR penalises a sharper result. This is a dataset limitation, not a model bug.

---

## Gurupreeth H: Classical ISP

**What I built**
- RAW loading for DNG and other formats (rawpy + DNG tags), normalisation of every Bayer layout to RGGB, and clear
  errors for bad files (wrong type, corrupt, unsupported sensor).
- White balance from the DNG's AsShotNeutral, with gray-world and gray-edge fallbacks.
- **Dual-illuminant colour correction:** estimate the scene's colour temperature, then blend the DNG's two
  calibration matrices. SIDD's own renderer used only one matrix, which washed out daylight photos from newer phones.
- sRGB gamma plus the SIDD tone curve (fast lookup table), and a differentiable PyTorch copy so the networks
  train through it. The two match to within 1e-4 (unit test).
- The noise model σ² = a·x + b: from the DNG NoiseProfile, or estimated from the image (green-green difference
  with a robust median-based estimate).
- The traditional ISP baseline: Malvar demosaic, bilateral/NLM denoise sized by measured noise, noise-adaptive
  edge-masked sharpening.

**Numbers to quote**
- Metadata WB vs gray-world: +5.8 dB. Colour matrix on vs off: +2.2 dB.
- Estimated noise model is only 0.16 dB worse than the DNG profile.

**Likely viva questions**
1. *Why keep WB/CCM classical?* They're fully determined by calibration metadata every DNG carries. Learning them
   costs capacity and fails on new sensors (end-to-end ΔE 6.77 vs 2.44).
2. *What is demosaicing?* Each sensor pixel sees one colour (RGGB pattern). Demosaicing interpolates the other two.
   Malvar-He-Cutler uses gradient correction to avoid colour fringes.
3. *Why denoise after tone mapping in the traditional ISP?* Bilateral/NLM parameters are tuned for display-referred
   images. We measure σ on the output and size the filter to it.
4. *Why does sharpening use an edge mask?* Unsharp masking amplifies noise. We only boost detail above 2.5σ and
   weaken it as noise rises.
5. *How do you handle a phone with a different Bayer layout?* Find the (dy, dx) shift that makes the 2×2 pattern
   RGGB and crop by it.

**Failure case I present:** F4. Without AsShotNeutral, gray-world white balance tints scenes dominated by one
colour, costing about 6 dB in our ablation.

---

## Jithesh D: Deep learning

**What I built**
- **Hybrid teacher:** a NAFNet-style U-Net (2.65 M params). Input: packed white-balanced RGGB plus a per-pixel
  noise map. Output: full-resolution linear RGB via PixelShuffle, added to a bilinear demosaic (it learns the residual).
- **End-to-end baseline:** the same network and size, given only the raw input, learning colour too.
- **Student:** a plain conv U-Net (0.63 M), trained from scratch and with knowledge distillation.
- **Training:** loss in sRGB through the fixed classical tail; Bayer-preserving flips; extra-noise augmentation;
  AdamW with cosine schedule; bf16; resumable runs on Modal A10/L40S GPUs.
- The no-noise-map ablation model.

**Numbers to quote**
- Hybrid vs same network end-to-end: +5.48 dB.
- Noise map: wrong σ at inference costs 1.1–1.6 dB; a model retrained without it is 0.13 dB worse.
- Distillation: +0.04 dB (negligible). Student is only 0.37 dB behind the teacher.

**Likely viva questions**
1. *Why NAFNet blocks?* They're state of the art on SIDD denoising and use no expensive attention: depthwise
   convolutions, a simple gate and cheap channel attention.
2. *Why put the loss in sRGB through the classical tail?* Errors in dark linear values become large after gamma.
   Training through the tail optimises what the viewer actually sees.
3. *Why feed a noise map?* It tells the network how strongly to denoise each pixel (FFDNet idea), so one model
   covers ISO 100 to 10,000, and it can be steered at inference.
4. *Why did distillation not help?* The small student already reaches 99% of the teacher, so there's little left
   to transfer in 30k steps. We report it as a negative result.
5. *How do you avoid overfitting with 76 training captures?* Thousands of random crops per capture, 4 noisy
   frames each, flips, noise augmentation, and model selection on a held-out scene.

**Failure case I present:** F1. At ISO 10,000, deep shadows get a red/yellow cast (black-level error amplified by
gain), and the burst ground truth contains an object that moved, so part of the error is in the GT.

---

## Hitesh S: Optimisation, deployment and docs

**What I built**
- ONNX export, FP16, and INT8 post-training quantisation calibrated on real RAW tiles.
- Found that full INT8 (W8A8) loses 6.8 dB on RAW data, and switched to **W8A16** (INT8 weights, 16-bit
  activations): 0.71 MB, −0.02 dB.
- The benchmark (parameters, size, CPU latency at 2/4 threads, peak memory) and GPU latency for FP32 vs FP16.
- Tiled inference engine (overlapping 512² tiles with blending) for any image size.
- The Gradio app: upload a DNG, choose pipeline or compare all, stage-by-stage gallery, metrics when ground truth
  exists, friendly errors, 4 MP default crop to stay under ~10 s. Hosted on Modal (scales to zero); Docker image
  and Makefile for one-command runs.
- The architecture diagram (generated from `docs/architecture.dot`, labelled with real module names), README,
  design document.

**Numbers to quote**
- Teacher 10.8 MB → deployed student 0.71 MB (15× smaller) for −0.37 dB.
- FP16 on GPU: 25.7 → 14.0 ms per MP, no quality loss.
- Student network: 365 ms per MP (FP32) / 527 ms (W8A16) on 2 CPU threads, about 170 MB peak memory. Whole deployed pipeline: 0.64 s per MP.

**Likely viva questions**
1. *Why did 8-bit activations fail?* Linear RAW is dark: shadows use ~1% of the range. One 8-bit scale per
   tensor leaves them a handful of levels, causing posterisation and colour shift. 16-bit activations fix it.
2. *Why is W8A16 slower than FP32 on CPU?* ONNX Runtime on x86 has no fast 16-bit integer kernels. The win is size
   and mobile-NPU compatibility (NPUs support 16-bit activations natively).
3. *Why ONNX Runtime and not PyTorch in the app?* Smaller image (no PyTorch), faster start, and the same file runs
   on mobile runtimes.
4. *How does the app stay under 10 seconds?* Student model, 4 MP centre crop by default, tiled batching, and the
   classical stages vectorised with lookup tables.
5. *How do I run it?* `make app`, or `docker build -t hybrid-isp . && docker run -p 7860:7860 hybrid-isp`.

**Failure case I present:** F5. An unsupported RAW variant (a Galaxy S22 DNG): the app shows a clear error instead of
crashing. Also the W8A8 quantisation failure, as an optimisation lesson.

---

## 3-minute demo script

| Time | Who | What |
|---|---|---|
| 0:00–0:30 | Hitesh S | Problem + architecture diagram: classical stages where metadata decides, network where content decides. |
| 0:30–1:15 | Gurupreeth H | Live URL: upload the judge's (or an unseen phone's) DNG → walk the stage gallery: RAW → WB → noise map → network → CCM/tone. |
| 1:15–2:00 | Hemansshu Raaj RK | "Compare all" on the ISO 10,000 example: traditional vs end-to-end vs hybrid, metrics against ground truth; results table and sweep plot. |
| 2:00–2:35 | Jithesh D | Model choices and ablations: hybrid +5.5 dB over end-to-end, noise-map steering, distillation result. |
| 2:35–3:00 | Hitesh S | 0.71 MB W8A16 model, latency numbers, one failure case (F2: GT blurrier than our output). |
