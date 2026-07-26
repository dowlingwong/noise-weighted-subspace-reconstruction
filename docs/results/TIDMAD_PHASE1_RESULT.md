# TIDMAD Phase 1 — Estimator Identity: Result

Date: 2026-07-27. Stage: TIDMAD (Paper 1's second real-data anchor).
Roadmap: [`../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`](../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md).
Run record: `results/metrics/TIDMAD_P1_synthetic_seed20260727.json`.
Code: `scripts/tidmad_phase1_estimator_identity.py`, `src/noise_geometry/tidmad/`.

## What this is, and what it is not

This is the **synthetic** half of Phase 1: the estimator, the whitening
convention, and the predicted-variance formula are verified on noise generated
from a declared PSD, where the right answer is known by construction. It
establishes that the machinery is correct. It establishes nothing about TIDMAD
data, which has not been downloaded yet.

The real-noise half (`--mode file`) runs the identical checks with injections
into signal-free TIDMAD science stretches. It is blocked on Phase 0: a verified
data contract and reproduction of the published baseline scores (gate G0). The
code path is written and the config is committed
(`configs/tidmad/phase1_estimator_identity_real_noise.yaml`); it needs the data.

The reason for doing it in this order is the same reason Paper 1's GWOSC
implementation checks mattered even though the calibration failed: without them,
a null result on real data is indistinguishable from a convention bug.

## Result: all six gates pass

Operating point: 4096-sample windows at 10 MS/s, 512 windows, pink (1/f) noise,
injected sinusoid at the on-grid frequency 1 000 976.5625 Hz, amplitude at
SNR 100 (predicted σ = 2.437, injected A = 243.67).

| Gate | Statement | Value | Threshold | |
| --- | --- | --- | --- | --- |
| G1a | amplitude estimate is unbiased | z = −0.55 | \|z\| < 3 | PASS |
| G1b | empirical spread matches the PSD-predicted bound | 0.9913 | within 10% of 1 | PASS |
| G1c | no-signal null is consistent with zero | z = 0.028 | \|z\| < 3 | PASS |
| G1d | learned rank-2 subspace recovers the analytic quadrature | 2.32° | < 5° | PASS |
| G1e | fixed-phase rank-1 filter loses SNR as cos² | 0.022 max abs error | < 0.05 | PASS |
| G1f | subspace-recovery error follows the 1/SNR law | slope −1.0016 | −1 ± 0.15 | PASS |

Fractional bias was −0.024% against a standard error of 0.044%, so G1a is a
genuine constraint at this operating point rather than a vacuous one.

## The two results worth reading twice

**G1f — the subspace error follows its theoretical rate.** A single principal
angle at one operating point says little; any value could be finite-sample
error or could be bias. The falsifiable statement is the *rate*. Measured over
SNR 30 → 1000:

| SNR | 30 | 100 | 300 | 1000 |
| --- | --- | --- | --- | --- |
| max principal angle | 7.810° | 2.341° | 0.770° | 0.234° |

log-log slope **−1.0016** against the theoretical −1. The learned subspace is
converging to the analytic quadrature plane at exactly the classical
perturbation rate. A slope near zero would have meant convergence to the wrong
place — a bug or a broken convention. This is a stronger statement than any
fixed-angle threshold, and it is why G1d's absolute threshold is documented as
a smoke check at a declared operating point rather than as the scientific claim.

**G1e — the cos² prediction holds, and the cos/cos² distinction is real.**
Paper 1 Eq. (16) predicts that a misaligned template loses effective SNR by the
squared cosine of the whitened angle. For the quadrature manifold that angle
*is* the phase error, which makes this an exact closed-form prediction rather
than a heuristic:

| phase error (rad) | 0.0 | 0.3 | 0.6 | 1.0 |
| --- | --- | --- | --- | --- |
| observed SNR² ratio | 1.000 | 0.926 | 0.703 | 0.307 |
| predicted cos² | 1.000 | 0.913 | 0.681 | 0.292 |
| observed amplitude ratio | 1.000 | 0.955 | 0.825 | 0.540 |
| predicted cos | 1.000 | 0.955 | 0.825 | 0.540 |

Both rows match their own prediction. The amplitude falls as cos and the SNR
*power* falls as cos²; an earlier draft of this check compared the amplitude
ratio against cos² and failed by 0.40. The two quantities are recorded side by
side in the run JSON so that the confusion cannot recur silently.

## Why this stage is worth doing at all

The unknown phase of a sinusoid spans **exactly** span{cos, sin}. That gives
TIDMAD something neither CRESST nor GWOSC can: an analytically known correct
subspace on real detector data. CRESST's rank-6 subspace was an empirical
choice, so its rank result is descriptive; here rank-1 is provably suboptimal by
a known factor, rank-2 is provably sufficient, and rank ≥ 3 can only add noise.
G1d and G1f are the synthetic rehearsal of that measurement, and Phase 3 is the
real-data version.

## Implementation notes worth carrying forward

- **Amplitude is specified in units of the estimator's own predicted σ.** A
  bias gate stated as a fixed percentage is empty when the statistical
  uncertainty exceeds that percentage. The first draft of this check used a
  raw amplitude of 1.0, which put the injection at SNR ≈ 0.4 and produced an
  apparent 14.5% "bias" that was 1.3 standard errors from zero. Fixing SNR
  fixes the power of the test; stating G1a in standard errors keeps it honest
  at any SNR.
- **Σ comes from calibration windows only.** Using the evaluation windows to
  estimate their own PSD would make the check pass for the wrong reason.
- **The estimator uses the repository's single weight convention**
  (`inverse_psd_weights` → `build_of_one_sided_weights`), and noise generation
  uses the repository's own `generate_colored_noise`, so the discrete
  DC/Nyquist convention has one definition shared with S1/S2/S5 and CRESST.
- The synthetic-PSD path also reports σ ratio computed with the *true* PSD
  (1.0152 here versus 0.9913 with the estimated PSD), which separates estimator
  error from covariance-estimation error — the effect Paper 1's S7 measures.

## Next

1. **Phase 0** (blocking): download two validation and two science files, run
   `scripts/tidmad_phase0_contract.py`, verify channel semantics, reproduce the
   published `none` and `savgol` denoising scores (gate G0), and set
   `verified: true` in `docs/tidmad_data_contract.json`.
2. **Phase 1 on real noise**: rerun this gate with
   `configs/tidmad/phase1_estimator_identity_real_noise.yaml`. Expect G1b to
   degrade somewhat — 8-bit quantization and glitches are not in the Gaussian
   model — and record by how much. That degradation is the first quantitative
   preview of the Phase 4 boundary result.
3. **Phase 2**: metric reversal against injected truth (gate G2), the headline
   experiment and the one that decides whether TIDMAD becomes a paper section.
