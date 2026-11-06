# Ablation Study and Failure-Case Analysis

All numbers: test split (26 SIDD captures of scenes 003 and 007, never seen in training), seeded 512×512 crops of a
single noisy frame. Full tables: `eval_P2_T77.md`. Reproduce: `make eval` then `python -m eval.report`.

## 1. Does each component earn its place?

### 1.1 Hybrid split (classical colour + learned restoration)
| Comparison | Δ PSNR | Δ ΔE2000 |
|---|---|---|
| Hybrid teacher vs **fully learned** E2E (same NAF-UNet, same 2.65 M params, same data and steps) | **+5.48 dB** | **−4.33** |
| Hybrid teacher vs **traditional ISP** (same classical colour stages, bilateral denoise) | **+6.88 dB** | **−3.39** |

The end-to-end network must learn white balance and colour per sensor from 76 captures. It gets the restoration
roughly right but its colour is the worst of all methods (ΔE 6.77). Giving the network metadata-driven WB and CCM
removes that burden: the same network capacity goes into denoising and demosaicing.

### 1.2 Classical stages, removed at inference (hybrid teacher, 16 crops per capture)
| Variant | PSNR | ΔE2000 | Δ PSNR |
|---|---|---|---|
| Full hybrid (metadata WB + DNG NoiseProfile) | 34.90 | 2.42 | – |
| White balance by gray-world instead of metadata | 29.08 | 6.77 | **−5.82** |
| White balance by gray-edge instead of metadata | 28.81 | 7.61 | **−6.09** |
| Colour correction matrix off | 32.75 | 3.42 | −2.15 |
| Noise model estimated from the image (no NoiseProfile) | 34.74 | 2.43 | −0.16 |
| Network told "no noise" (σ map × 0) | 33.76 | 2.97 | −1.14 |
| Network told "2× noise" (σ map × 2) | 33.33 | 3.13 | −1.56 |
| + noise-adaptive sharpening | 34.88 | 2.42 | −0.02 |

- **Metadata white balance is the single most important classical stage.** Illuminant estimation is hard on these
  lab scenes (dominant-colour objects), and gray-world/gray-edge cost about 6 dB. They stay as fallbacks for files
  without AsShotNeutral.
- **The noise estimator works:** estimating the noise model from the image costs only 0.16 dB versus the DNG's
  NoiseProfile, so files without metadata still get a good σ map.
- **The network uses the σ map:** feeding a wrong noise level (×0 or ×2) costs 1.1–1.6 dB. It denoises more or
  less depending on what it is told, which is the adaptive behaviour we designed for.
- **Sharpening is neutral on PSNR** (−0.02 dB), because the ground truth is not sharpened. It is off by default
  and available as an option in the app for visual crispness.

### 1.3 Retrained ablations
| Comparison | Δ PSNR | Δ ΔE | Verdict |
|---|---|---|---|
| Noise-map conditioning: trained with vs without σ input | +0.13 dB | −0.01 | Small gain overall. A no-σ network learns to infer noise from the image itself. The σ map matters more at extreme gain (sweep below) and is what lets us steer the denoiser (1.2). |
| Knowledge distillation: KD student vs same student trained from scratch | +0.04 dB | −0.03 | **No meaningful gain at this size.** The 0.63 M student is close to the teacher already (−0.37 dB), leaving little for distillation to transfer in 30 k steps. Reported as a negative result. |

### 1.4 Precision and quantisation (student, 0.63 M params)
| Variant | Size | PSNR | Δ vs FP32 | CPU ms / MP (2 threads) |
|---|---|---|---|---|
| FP32 | 2.53 MB | 34.56 | – | 546 |
| **INT8 weights + 16-bit activations (W8A16)**, deployed | **0.71 MB** | **34.54** | **−0.02** | 784 |
| Full INT8 (W8A8, all ops) | 0.71 MB | 27.72 | **−6.84** | 305 |
| Teacher FP16 (GPU) | 5.38 MB | 34.91 | 0.00 vs FP32 | GPU: 14.0 ms vs 25.7 ms FP32 (A10) |

**Why full INT8 fails here:** the network works on linear RAW values, where shadows sit within the bottom ~1% of
the range. One 8-bit scale per tensor gives dark regions only a handful of levels, so they posterise and colour
shifts (ΔE 7.9). Keeping activations at 16 bits removes the loss entirely while keeping the 3.6× size reduction.
The trade-off: W8A16 is slower than FP32 on a desktop CPU in ONNX Runtime (no fast 16-bit integer kernels), while
W8A8 is the fastest but too lossy. On mobile NPUs, which support 16-bit activations natively, W8A16 is the standard
choice for exactly this reason.

### 1.5 Robustness: darker captures (synthetic, real noise model)
PSNR as the clean RAW is darkened by 2^k and amplified back (`docs/sweep.png`):

| | ×1 | ×½ | ×¼ | ×⅛ | ×1/16 |
|---|---|---|---|---|---|
| Traditional | 30.34 | 27.29 | 24.38 | 21.75 | 19.45 |
| Learned E2E | 30.34 | 29.83 | 28.94 | 27.70 | 25.08 |
| **Hybrid teacher** | **38.44** | **37.38** | **35.35** | **31.50** | **27.06** |
| Student (KD) | 37.94 | 36.62 | 34.39 | 30.18 | 24.80 |

The hybrid stays best at every level. The gap to traditional grows from 8 dB to 11 dB as light drops. The E2E
model degrades more slowly in relative terms but starts 8 dB lower. The small student falls off fastest beyond
8× gain, the clearest cost of compression.

## 2. Failure cases

### F1. Extreme ISO (10,000): colour cast and burst motion. `docs/failures/failure_1_0065_003_GP_10000_08460_4400_N.jpg`
Worst crop: 15.0 dB (Pixel, ISO 10,000). Two causes:
1. **Colour bias in deep shadows.** At this gain, small black-level errors are amplified into a red/yellow cast in
   every method's output. The traditional ISP is also buried in chroma noise. This is the known "low-light colour
   bias" problem of RAW denoising; fixing it needs per-frame black-level estimation.
2. **The ground truth is not the same scene.** The GT (a 150-frame average) contains a dark cable that is absent from
   the single noisy frame. Something moved during the burst, so part of the PSNR loss is a GT artefact.

### F2. Ground truth blurrier than the output. `docs/failures/failure_2_0054_003_N6_00100_00160_5500_N.jpg`
Nexus 6, ISO 100: 23.3 dB, yet visually every learned output is **sharper** than the GT. Small camera shake across
the 150 frames blurred the average. PSNR then penalises the correct, sharp reconstruction. Burst alignment before
averaging would fix the GT; we kept SIDD-style averaging and report this as a dataset limitation.

### F3. Frame-to-frame exposure mismatch. `docs/failures/failure_3_0055_003_N6_00800_01000_5500_N.jpg`
Nexus 6, ISO 800: 24.1 dB. The text is reconstructed cleanly, but the whole crop is slightly brighter and greener
than the GT, because the single input frame's exposure differs from the burst mean. The model is faithful to its
input; a global brightness/colour offset dominates the error.

### F4. Unseen sensors without metadata (qualitative)
For DNGs that lack AsShotNeutral, the pipeline falls back to gray-world / gray-edge white balance, which costs
about 6 dB on the test set (ablation 1.2) and visibly tints scenes dominated by one colour (e.g. a green wall).

### F5. Unsupported RAW variants
A Samsung Galaxy S22 DNG from raw.pixls.us is rejected by LibRaw ("Unsupported file format"), likely a newer DNG variant. The app reports a clear error
("Could not read this RAW file…") instead of crashing. X-Trans and Quad-Bayer layouts are also rejected with a
message.
