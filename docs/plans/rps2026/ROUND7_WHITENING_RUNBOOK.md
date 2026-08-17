# Round 7 runbook — measure the whitening dynamic range

**Date:** 2026-08-17
**Compute:** ~5 minutes, CPU only. No training, no GPU, no new experiments.
**One command.** Step 0 is `git pull`; `tidmad/whitening_range.py` is new.

---

## Why

Round 6 found **no metric reversal at rung 4**. The `Σ⁻¹`-trained arm was better on *both* held-out criteria — `−1.15×10⁻⁴` on the normalized likelihood residual and `−57.2` on raw MSE, both intervals excluding zero, the second opposite to prediction.

The paper currently explains this by saying that whitening the band-limited spectrogram is close to a constant rescaling, so the two losses are nearly collinear and there is no metric contrast to detect. **That explanation is inferred from the effect signature, not measured.** It is the paper's account of its own central negative result, and it should be a measurement.

The two objectives are `MSE = mean(r²)` and `chi2 = mean(r²/J)`. They are the same objective up to a scale — identical minimisers — exactly when `J` is constant over the bands carrying the residual. So the question is entirely about the *spread* of the weights `1/J`.

---

## The command

```bash
cd ~/noise-weighted-subspace-reconstruction && git pull && export PYTHONPATH=src
PYTHON=/home/dwong/anaconda3/bin/python

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  timeout 15m nice -n 10 $PYTHON -m tidmad.whitening_range \
  --data-dir <resolved_tidmad_raw_dir> \
  --out results/rps2026/E6_whitening_range.json
```

`<resolved_tidmad_raw_dir>` is the directory holding `abra_science_*.h5` — the same noise-only files training fitted `J` on. Round 5 and 6 recorded its actual location; reuse that, and record it again here.

The module uses the **frozen configuration and the same estimator training used** (`estimate_band_psd`, boxcar window, median average, 512 calibration windows, `_noise_only_windows`), so the `J` it measures is the `J` the loss divided by.

---

## What it prints, and what decides the question

Statistics over the 328 kept bands, reported both with and without band 0 (DC is degenerate for zero-mean calibration noise):

| Statistic | Meaning |
|---|---|
| `ratio_max_min`, `ratio_p99_p01` | crude and robust dynamic range of `J` |
| **`cv_inverse_J`** | **decisive.** `std(1/J)/mean(1/J)`. Zero means the objectives coincide. |
| `effective_n_bands` | participation ratio of `1/J`; near 328 means near-uniform weighting |

**Gate: `cv_inverse_J`, excluding DC.**

- **Below ~0.1** — the collinearity account is supported. The paper's explanation becomes a measurement, and §4's sentence can be stated as fact with the number in it.
- **Substantially above ~0.1** — the account is **wrong**. `χ²` and MSE are genuinely different objectives over this band, and the absence of reversal at rung 4 needs a different explanation.

**This is a demanding threshold and it may well fail.** On synthetic checks, a flat `J` gives `cv_inverse_J = 0.000`, but even a mild 2× dynamic range gives **0.20** — already twice the threshold. If the SQUID noise has any real structure across 0–800 kHz, the test will refute the collinearity story.

**Report the number either way.** Do not adjust the threshold after seeing it. A refutation is more valuable now than a plausible-sounding sentence a referee dismantles later.

---

## If the gate fails, what it means

Two explanations remain, and the numbers already in hand distinguish them:

1. **Both arms are undertrained.** 25,000 steps, 569k parameters, ~14 epochs. If neither arm is near its optimum, which one wins is dominated by optimisation noise rather than by the objective. Evidence: the differences are small (0.85% and 0.50% relative) and *both* favour the same arm, which is what you would expect if the two runs simply differ in optimisation luck.
2. **The architecture absorbs the metric.** A sufficiently expressive model may reach a similar reconstruction under either weighting — this is exactly the possibility §5 already anticipates ("capacity can absorb a misspecified metric").

Distinguishing those needs a second seed per arm, which is ~9 hours of GPU and probably beyond the deadline. **In that case the honest report is that rung 4 does not reproduce the reversal and the paper cannot say why** — with the undertraining and capacity accounts offered as candidates. That is a weaker but defensible position, and §5's single-seed caveat already covers it.

---

## Sync back

```
results/rps2026/E6_whitening_range.json
```

Append a short **Round 7** section to `EXPERIMENT_RUN_LOG.md`: the resolved science-file directory, `cv_inverse_J` with and without DC, the dynamic-range ratios, the verdict the module printed, and one sentence on whether the paper's rung-4 explanation survives.

---

## Stop conditions

| Stop if | Because |
|---|---|
| no `abra_science_*.h5` under the given directory | `J` cannot be reproduced; the measured weighting would not be the one training used |
| the estimator errors or returns non-positive `J` | a positive-definite covariance is a precondition of the whole framework |

Do not substitute a different PSD estimator, window, or averaging rule to make the number come out. The point is to measure what the loss actually divided by.
