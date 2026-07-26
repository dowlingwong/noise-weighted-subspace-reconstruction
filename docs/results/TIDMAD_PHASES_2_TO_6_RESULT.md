# TIDMAD Phases 2–6 — Implementation and Synthetic Results

Date: 2026-07-27. Stage: TIDMAD (Paper 1's second real-data anchor).
Roadmap: [`../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`](../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md).
Phase 1: [`TIDMAD_PHASE1_RESULT.md`](TIDMAD_PHASE1_RESULT.md).
Code: `src/noise_geometry/tidmad/{analysis,calibration,denoise,score}.py`,
`scripts/tidmad_phases.py`. Configs: `configs/tidmad/phase{2,3,4,5}_*.yaml`.

## Scope, stated first

Phases 2–4 are **implemented and executed in synthetic mode**, where the clean
signal is an injected sinusoid of known amplitude and random phase. That
establishes the analysis is correct and the gates have power. It establishes
nothing about TIDMAD data, which is still not downloaded.

Phase 5 is **implemented but not executed**: it has no synthetic form. The
denoising score is defined by the release's own `benchmark.py`, so there is
nothing to score without the real validation files and a checkout of the
upstream repository. Its components — the Wiener filter and the overlap-add
denoiser — are verified numerically (below).

Everything is blocked on Phase 0 for the real-data run: a verified data
contract and reproduction of the published baseline scores. The loader enforces
that rather than trusting the caller.

## Phase 2 — metric reversal against truth (gate G2)

The stage's headline. Two matched-rank subspaces are fitted on the **same**
representation — the faithful real form of the rFFT — and differ only in the
metric used to fit and project. Both are scored on held-out windows.

Operating point: 4096-sample windows at 10 MS/s, 512 windows, pink noise,
per-window SNR 8, rank 2, chronological 70/30 split with a 4-window guard gap.

| Metric | PCA (Euclidean) | EMPCA (Σ⁻¹) | paired difference, 95% CI |
| --- | --- | --- | --- |
| raw MSE | **19180.1** | 22396.9 | −3216.8 [−3750.6, −2740.1] |
| weighted residual | 8.631×10¹³ | **8.488×10¹³** | +1.421×10¹² [1.344, 1.505]×10¹² |
| **amplitude \|error\| vs truth** | 20.487 | **5.792** | **+15.043 [14.698, 15.381]** |
| clean-signal MSE | 3659.6 | **145.6** | +3514.0 [3068.8, 4029.2] |

All three gates pass. The pattern is the predicted one: **PCA wins the metric
it optimizes, EMPCA wins the likelihood, and the truth sides with EMPCA.** The
first two rows on their own would be a tautology — each method wins its own
objective. The third row is the claim, and it is the row CRESST structurally
cannot produce.

The mechanism is visible in the numbers: PCA's amplitude bias is −20.47 against
an injected amplitude of ~20.5, i.e. it recovers essentially none of the signal.
With a steep 1/f floor, the leading Euclidean variance directions are
low-frequency noise, not the 1 MHz signal. EMPCA finds the signal plane but
shrinks it (bias −5.43), which is finite-sample subspace error at this SNR, not
bias in the estimator — Phase 1's scaling law predicts the magnitude.

A moving-block bootstrap (block 16) is reported alongside the window-level one
because consecutive windows of a continuous stream are dependent. Here they
agree closely ([14.77, 15.31] vs [14.70, 15.38]); on real data they may not, and
the discrepancy would itself be informative. Leakage audit: zero index overlap,
zero window-hash overlap, 4-window realized guard gap.

## Phase 3 — the rank axis against a closed-form prediction (gate G3)

Amplitude RMSE against truth, held-out:

| rank | 1 | 2 | 3 | 4 | fixed-phase rank-1 OF |
| --- | --- | --- | --- | --- | --- |
| RMSE | 13.456 | **5.792** | 5.795 | 5.794 | 24.890 |

| Gate | Value | |
| --- | --- | --- |
| G3 rank 2 beats rank 1 | +6.847 [5.996, 7.570] | PASS |
| G3 rank 3 does not improve on rank 2 | −0.0002 (tolerance 0.05) | PASS |
| G3 knee is at rank 2 | 2 | PASS |
| G3 rank 2 beats fixed-phase OF | +14.517 [12.315, 16.681] | PASS |

This is the measurement TIDMAD exists for. The unknown phase of a sinusoid
spans *exactly* span{cos, sin}, so the correct rank is known in closed form
rather than chosen empirically as CRESST's k = 6 was. The curve behaves as
predicted: a large gain from 1 → 2, and nothing from 2 → 3 (0.03%, CI spanning
zero at [−0.001, 0.005]).

Note on reporting: the raw argmin of the RMSE curve is rank 4, by 0.0004. That
is noise in the fourth decimal, and quoting it would be misleading. The reported
quantity is the **smallest rank within a predeclared tolerance of the minimum**
— the knee — which is what the rank-2 prediction is about. The maximum principal
angle between the learned rank-2 subspace and the analytic quadrature subspace
is 32.15° at this SNR, consistent with the Phase 1 scaling law; Phase 1 verifies
that this angle shrinks as 1/SNR rather than converging to a wrong plane.

## Phase 4 — the calibration boundary (control is a halt gate)

Structure deliberately mirrors the GWOSC stage, because that design was sound
even though its result was negative.

| model | χ²/dof | score dispersion (SE 0.022) |
| --- | --- | --- |
| stationary synthetic control | 1.0002 | 1.0758 (z = 3.4) |
| global PSD | 1.0023 | 1.0112 (z = 0.5) |
| local, radius 8 windows | 1.0813 | 1.0207 |
| local, radius 16 windows | 1.0432 | 0.9915 |
| local, radius 32 windows | 1.0228 | 0.9803 |

The control passes its predeclared band [0.9, 1.1] on both statistics, so the
machinery is sound and a real-data failure would be a finding rather than a bug.

Two things worth recording. First, the χ² normalization is derived and
documented in the code rather than fitted: with the repository's one-sided OF
weights every contributing bin supplies `f_s N / 2` per degree of freedom, and a
unit-tested `_effective_dof` counts two real degrees of freedom per interior bin
and one at Nyquist. It lands at 1.000 on correct-model noise, which is the
evidence that the convention is right.

Second, the local-PSD estimates sit systematically *above* one (1.081 at radius
8, falling to 1.023 at radius 32). That is the expected cost of estimating the
PSD from few neighbours, and it is the reason the neighbourhood **excludes the
window being scored** — including it would let the model absorb the fluctuation
it is meant to predict and would produce an apparently excellent calibration
that is pure leakage. A test pins that exclusion.

A caveat that the code now carries: the reported dispersion standard error
assumes the predicted σ is exact. It is not, because Σ is estimated. A five-seed
check gave a realized spread of about 0.03 against a formula value of 0.022, so
the z-score is a lower bound on the fluctuation. Treat a dispersion near the
acceptance boundary as requiring a seed ensemble, not a single run — the
seed-20260727 control value of 1.076 is a 3.4σ excursion by the formula and an
ordinary one by the empirical spread.

## Phase 5 — the detector-facing metric (implemented, not run)

Design, with its claim boundary fixed in the code before any run:

> This measures how much of the gap between doing nothing and a trained deep
> denoiser is explained by using the correct noise geometry with a linear model.
> It is not a claim to win the benchmark.

The estimator is the Wiener filter `H = S/(S+N)` — the linear MMSE estimator
under the same Gaussian model Paper 1 uses, applied to a denoising task instead
of an amplitude task. Both spectra come from **training** files and are applied
blind to validation, which is the same information the deep baselines receive.

Verified numerically:

- overlap-add with a square-root Hann window and 50% overlap reconstructs a
  stream to 7×10⁻¹⁴ when the transfer function is the identity, so no artificial
  block-edge structure enters the score;
- on synthetic pink noise the fitted filter reduces MSE to the clean signal by a
  factor of 20 (ratio 0.051);
- spectral subtraction is clipped at zero, so the transfer function cannot
  exceed one — a negative "signal" PSD would turn the filter into an amplifier.

The score itself is **never reimplemented**. `score.py` writes the release's
file layout (preserving dataset name and dtype) and shells out to the upstream
`benchmark.py`, capturing its output verbatim. If the upstream script is
missing it raises rather than falling back to an internal approximation — a
reimplementation that disagreed by a normalization factor would produce a number
that looks comparable to the published scoreboard and is not.

## Phase 6 — negative controls (all five degrade)

Phase 2's positive result is compatible with a pipeline that would report a win
no matter what it was fed. Each control breaks one ingredient and must degrade
the primary metric with a bootstrap interval excluding zero. The set is matched
one-for-one with CRESST's, so both real-data stages are falsifiable the same way.

Nominal (EMPCA) amplitude RMSE: **5.792**. Zero-recovery level (recovering
nothing): **20.515**.

| control | RMSE @ SNR 8 | paired difference vs nominal, 95% CI | RMSE @ SNR 40 |
| --- | --- | --- | --- |
| `noise_only_train` | 20.507 | +15.076 [14.766, 15.381] | 102.522 |
| `permuted_bins` | 21.043 | +15.394 [14.757, 15.923] | 102.924 |
| `random_weighted_basis` | 20.472 | +15.030 [14.685, 15.365] | 102.354 |
| `wrong_frequency` | 20.547 | +14.982 [14.445, 15.465] | 102.498 |
| `wrong_psd` | 20.496 | +15.054 [14.766, 15.379] | 59.219 |

All five gates pass. Two things the table shows that a bare pass/fail would hide:

**At SNR 8 every control saturates.** All five land at the zero-recovery level
(20.515), so they establish that the pipeline is falsifiable but cannot rank
severity — the differences between them are noise. The run record flags this
automatically (`all_controls_saturated: true`) rather than leaving a reader to
infer an ordering from it.

**At SNR 40 the controls separate, and `wrong_psd` is the weakest** — 59.2
against a zero-recovery level of 102.6, while the other four stay at ~102.5.
A wrong-but-still-coloured noise model retains partial recovery, which is
exactly what CRESST found: wrong-PSD weighting was its weakest control too.
That agreement across two unrelated detectors is a small but real check on the
control design.

Design note: `wrong_psd` uses a **spectrally flipped** PSD, not the identity
metric. The isotropic case is already Phase 2's Euclidean arm, so using it here
would duplicate that contrast instead of testing the noise model separately —
the first draft did exactly that and produced a control numerically identical
to the Phase 2 PCA arm.

## Verification

22 tests pass (`tests/test_tidmad_estimators.py`, `tests/test_tidmad_phases.py`),
covering: the predicted direction of all three metric families; the rank knee;
χ²/dof unit expectation and the degrees-of-freedom count; leave-one-out
enforcement in the local model; exact overlap-add reconstruction; the bounded
transfer function; and the release's denoised-filename pattern.

All four runnable phases also execute through the config-driven entry point
(`scripts/run_experiment.py --config configs/tidmad/phase{1,2,3,4}_*.yaml`) and
report `status: complete`.

## Next

1. **Phase 0** (blocking everything real): download two validation and two
   science files, run `scripts/tidmad_phase0_contract.py`, confirm channel
   semantics, reproduce the published `none` and `savgol` scores (gate G0), and
   set `verified: true` in the contract.
2. Re-run Phases 1–4 with `--mode file`. Expect the Phase 4 global model to
   fail on real SQUID noise — 8-bit quantization, drift, and glitches are not in
   the Gaussian model — and record what it costs via `cost_of_mismatch`, which
   is the number GWOSC could never produce.
3. Phase 5 with `--upstream <TIDMAD checkout>`.
4. Re-run Phase 6 on real data. A control that fails to degrade there is a
   result to publish, not a parameter to adjust — CRESST had exactly one such
   case and reported it.
