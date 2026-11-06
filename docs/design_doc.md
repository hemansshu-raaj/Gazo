# Design Document: Hybrid, Computationally Efficient Smartphone ISP (P2)

## 1. Problem and goals
Turn noisy Bayer RAW from phone sensors into high-quality sRGB, handling sensor noise, white-balance errors and
varying illumination, while staying efficient enough for mobile. The PDF requires a **hybrid** design (not a purely
classical pipeline, not an off-the-shelf network), comparison with a conventional ISP and a learned ISP, at least
one model-compression technique, and measurements of size, memory and latency.

**Goals:** (1) beat the conventional ISP on PSNR / SSIM / ΔE / LPIPS on unseen scenes; (2) a deployable model under
1 MB that runs a 4 MP photo on a CPU in about 1 s; (3) an app that accepts real phone DNGs.
**Non-goals:** burst / multi-frame fusion, HDR merging, learned "aesthetic" photo-finishing styles.

## 2. Architecture
See `docs/architecture.png` (source: `docs/architecture.dot`). Three pipelines share the classical front end:

| Stage | Hybrid | Traditional | Learned E2E |
|---|---|---|---|
| Black/white level, CFA → RGGB | classical | classical | classical |
| White balance | DNG AsShotNeutral (gray-world / gray-edge fallback) | same | learned (implicit) |
| Noise model | DNG NoiseProfile or estimated → σ map | used for denoise strength | – |
| Demosaic + denoise | **network (joint)** | Malvar-He-Cutler + bilateral / NLM | network |
| Colour correction | dual-illuminant DNG matrix | same | learned |
| Gamma + tone curve | sRGB OETF + SIDD tone curve | same | learned |
| Sharpening | noise-adaptive, edge-masked (optional) | same | – |

**Why this split:** white balance and colour correction are exactly determined by camera calibration metadata that
every DNG carries. Learning them wastes capacity and fails on unseen sensors. Denoising and demosaicing depend on local
image content, where classical filters trade detail against noise and networks do far better. Telling the network
the noise level (a σ map from the sensor's noise model) lets one model adapt across ISO 50 to 10,000 instead of
memorising noise levels.

## 3. Data and split
- **Source:** SIDD original DNG bursts (`MhdMohammadi/SIDD-CC-DNG`): 115 captures, 150 frames each, 5 phones
  (iPhone 7, Pixel, Nexus 6, Galaxy S6 Edge, LG G4), ISO 50 to 10,000, 3 illuminants, low/normal/high brightness.
  The official SIDD server was unreachable, so we rebuilt the dataset from the raw bursts (`modal_app/prepare_data.py`).
- **Ground truth RAW:** mean of all 150 normalised frames. Frames whose brightness is an outlier (flicker) are
  dropped automatically: 117 of 17,206 frames.
- **Inputs:** 4 single noisy frames per capture (frames 0, 50, 100, 149).
- **Ground truth sRGB:** our reference renderer on the clean RAW (metadata WB → Malvar demosaic → interpolated
  CCM → sRGB OETF → SIDD tone curve).
- **Split by physical scene** (`isp/data.py`): test = scenes 003 + 007 (26 captures, all 5 phones, ISO 100 to 10,000),
  val = scene 008 (13), train = the other 7 scenes (76). Random splits would leak the same scene into training.
- **Unseen cameras:** CC0 DNGs from raw.pixls.us (Pixel 4a, Pixel 7 Pro, OnePlus 6, Redmi Note 8T).

## 4. Classical stages (`isp/classical/`)
- **CFA normalisation:** every Bayer layout is cropped by (dy, dx) to RGGB, so one model handles all sensors.
- **White balance:** DNG AsShotNeutral. Fallbacks are gray-world (ignoring clipped pixels) and gray-edge (p = 6).
- **Colour correction:** SIDD's renderer used only ColorMatrix1 (tungsten-calibrated), which desaturated daylight
  photos from newer phones. We follow the DNG spec instead: estimate the scene CCT from AsShotNeutral and
  interpolate ColorMatrix1/2 in inverse-CCT space. Rows are normalised so neutrals stay neutral.
- **Tone:** IEC sRGB OETF (finite slope at black, needed for stable training) and SIDD's global tone curve, via a
  16-bit LUT for speed. A differentiable torch copy (`torch_tail.py`) is verified against numpy to within 1e-4.
- **Noise:** σ²(x) = a·x + b from the DNG NoiseProfile, otherwise estimated from the green-green difference
  (signal-free) binned by intensity with MAD (robust) variance.
- **Denoise (traditional):** bilateral on luma and stronger on chroma, or NLM, with strength set by the measured σ.
- **Sharpening:** unsharp mask on luma, gated by |detail| > 2.5σ, with strength falling as noise rises.

## 5. Learned models (`isp/models/`)
- **Hybrid teacher:** NAFNet-style U-Net (Chen et al. 2022), width 32, 3 levels, 2.65 M parameters. Input: packed
  white-balanced RGGB (4 channels at half resolution) plus a σ map. Output: 12 channels → PixelShuffle → full-res
  linear RGB, added to a bilinear demosaic (residual learning).
- **Learned E2E baseline:** the same network and size, input packed RAW without WB or metadata, output sRGB directly.
- **Student:** plain Conv-ReLU U-Net, width 16, 0.63 M parameters. No LayerNorm or attention, so it exports and
  quantises cleanly.
- **Training:** Charbonnier loss in sRGB, computed through the fixed classical tail (so the network learns what
  matters after colour and tone). 512² crops, batch 16, AdamW with cosine schedule, bf16 autocast, gradient clipping.
  Augmentation: Bayer-preserving flips and extra Poisson-Gaussian noise (with a matching σ update) to simulate higher
  ISO / darker captures. Resumable checkpoints on Modal A10 GPUs.

## 6. Optimisation (`optimize/`)
1. **Knowledge distillation:** student loss = L(student, GT) + 0.5·L(student, teacher).
2. **INT8 post-training static quantisation:** ONNX Runtime, QDQ, per-channel INT8 weights, calibrated on real RAW
   tiles. Full W8A8 loses 6.8 dB (8-bit activations posterise linear-RAW shadows), so the deployed model is
   **W8A16**: INT8 weights, 16-bit activations, 0.71 MB, −0.02 dB.
3. **Reduced precision:** FP16 teacher on GPU: same quality, 1.8× faster (25.7 → 14.0 ms per MP on an A10).

Measurements: parameters, file size, CPU latency (2 and 4 threads), peak memory (`optimize/benchmark.py`).

## 7. Evaluation and results
Test split: 26 captures (scenes 003 + 007), 832 seeded 512×512 crops. Full tables (per ISO, per illumination,
per phone, degradation sweep): `eval_P2_T77.md`.

| Pipeline | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ | Params | Size | Full pipeline, CPU ms/MP (2 thr) |
|---|---|---|---|---|---|---|---|
| Traditional ISP | 28.03 | 0.674 | 5.83 | 0.459 | – | – | 305 |
| Learned E2E | 29.43 | 0.877 | 6.77 | 0.286 | 2.65 M | 10.8 MB | 2697 |
| **Hybrid teacher** | **34.91** | **0.895** | **2.44** | **0.236** | 2.65 M | 10.8 MB | 2753 |
| Hybrid student (W8A16, deployed) | 34.54 | 0.890 | 2.57 | 0.249 | 0.63 M | **0.71 MB** | 637 |

The hybrid wins in every ISO bucket, both illumination levels and all five phones. The deployed student keeps
99% of the teacher's PSNR at 1/15 of the size.

## 8. Ablation study
Summary (details in `docs/ablation_failures.md`): metadata WB is worth ~6 dB; CCM 2.2 dB; the hybrid split beats the
same network trained end-to-end by 5.5 dB; feeding a wrong σ costs 1.1–1.6 dB; a retrained no-σ model is only
0.13 dB worse; distillation gave +0.04 dB (negligible); W8A16 −0.02 dB vs W8A8 −6.8 dB; FP16 lossless.

## 9. Failure cases
Detailed with images in `docs/ablation_failures.md`: (F1) ISO 10,000 colour cast plus scene motion in the burst GT;
(F2) GT blurrier than the output due to shake across the burst; (F3) frame-vs-mean exposure mismatch;
(F4) gray-world fallback on unseen sensors without metadata; (F5) RAW variants LibRaw cannot decode.

## 10. Limitations
- The ground truth is a fixed classical rendering, so the metrics reward fidelity to it, not phone-maker aesthetics.
- Training covers 5 older phones. Newer sensors rely on metadata (WB, CCM, NoiseProfile) being present and correct.
- Some newer DNG files (e.g. our Galaxy S22 sample) are rejected by LibRaw. The app reports this clearly
  instead of failing.
- The burst-average ground truth has flaws (motion, shake blur, exposure drift; see failure cases), which caps
  achievable PSNR and occasionally penalises sharper output.
- W8A16 is the accurate quantised model but is slower than FP32 on desktop CPUs (ONNX Runtime lacks fast 16-bit
  integer kernels). Its benefit is size and NPU-friendliness.
- Single-frame only: no burst merging, so extreme low light keeps residual noise.

## 11. Ethics
- Uploaded RAW files are processed in memory and never stored or logged.
- Dataset licences: SIDD (MIT); simple-camera-pipeline tone curve (MIT, credited in `third_party/`).
- An ISP changes how people and scenes look. Colour comes from camera calibration, not learned skin-tone
  "beautification", which limits bias, but the training set covers only 5 phones and lab/indoor scenes.

## 12. Deployment
Gradio app (`app.py`) served from a CPU container on Modal (`modal_app/serve.py`), scaling to zero when idle and
kept warm during judging. The same app runs locally (`make app`) or in Docker (`make docker`). Models are
ONNX Runtime CPU sessions, with no PyTorch in the serving image. Large photos are processed as a centre 4 MP crop by
default (full resolution is optional) to stay under ~10 s.

## 13. Team contributions
See `TEAM_CONTRIBUTIONS.md`.
