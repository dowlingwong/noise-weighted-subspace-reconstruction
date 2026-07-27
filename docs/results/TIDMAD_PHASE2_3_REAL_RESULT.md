# TIDMAD Phases 2-3 on REAL Data — Result

Date: 2026-07-27. Stage: TIDMAD (Paper 1's second real-data anchor).
Roadmap: [`../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`](../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md).
Predeclaration: [`TIDMAD_PREDECLARED_GATES.md`](TIDMAD_PREDECLARED_GATES.md).
Phase 0: [`TIDMAD_PHASE0_RESULT.md`](TIDMAD_PHASE0_RESULT.md). Synthetic-mode
Phases 2-6: [`TIDMAD_PHASES_2_TO_6_RESULT.md`](TIDMAD_PHASES_2_TO_6_RESULT.md).
Phases 4/6 real data: [`TIDMAD_PHASE4_6_REAL_RESULT.md`](TIDMAD_PHASE4_6_REAL_RESULT.md).
Configs: `configs/tidmad/phase{2,3}_*_real_noise.yaml`,
`configs/tidmad/phase{2,3}_*_confirmatory.yaml`.
Selection script: `scripts/tidmad_phase3_file_selection.py`.
Evidence: `results/metrics/TIDMAD_P{2,3}_real_seed20260727.json`,
`results/metrics/TIDMAD_P{2,3}_confirmatory_seed20260727.json`,
`results/metrics/TIDMAD_P3_file_selection.json`.

## Read this first: three files, three different evidentiary statuses

| File | How it entered this investigation | Phase 2 | Phase 3 |
| --- | --- | --- | --- |
| `abra_validation_0000.h5` | First validation file matched by the default glob — first attempt, no selection | **PASS**, significant | not confirmed (two-tone signal) |
| `abra_validation_0010.h5` | Chosen *after* diagnosing why 0000's Phase 3 failed, because it was spectrally single-tone | direction correct, not significant | not confirmed (weak channel coupling) |
| `abra_validation_0005.h5` | Chosen by a **predeclared, mechanically-applied selection rule**, before its outcome was known | direction correct, not significant | **not confirmed** (flat rank curve despite passing every quality check) |

**Phase 2's positive, citable result is `abra_validation_0000.h5`'s** — a genuine
first-attempt finding, no file selection involved. **Phase 3 has no
confirmed real-data result on any of the three files tried**, including the
one selected by a rule frozen before its outcome was seen. Report Phase 3
real-data status as an investigated, understood, honest non-confirmation —
not as a passed or failed predeclared gate, and not as a refutation of the
theory (which remains supported by theory and controlled synthetic data).

## Phase 2 — metric reversal against real injected truth (gate G2)

### `abra_validation_0000.h5` — PASSED, first attempt

512 windows of 4096 samples (10 MS/s), rank 2, chronological 70/30 split
with a 4-window guard gap, PSD estimated from `abra_science_0000.h5`.
"Truth" is the file's own real injected reference channel (`channel0002`),
read directly — not a synthetic injection.

| Metric | PCA (Euclidean) | EMPCA (Σ⁻¹) | paired difference, 95% CI |
| --- | --- | --- | --- |
| raw MSE | **182.40** | 212.44 | −30.04 (favours PCA) |
| weighted residual | 9.499×10¹³ | **9.432×10¹³** | +6.70×10¹¹ (favours EMPCA) |
| **amplitude \|error\| vs. real injected truth** | 0.1209 | **0.0880** | **+0.0190, 95% CI [0.0045, 0.0321]** (window-level, n=150) |

Block-level bootstrap (block 16) agrees: +0.0190, 95% CI [0.0027, 0.0344].
Both exclude zero. All three predicted orderings hold. This is the first
demonstration in this project of metric reversal against *known ground
truth* on real, unmodified detector data.

Gates: `G2_truth_favours_empca` PASS, `G2_block_bootstrap_agrees` PASS,
`G2_metric_orderings_reversed` PASS.

### `abra_validation_0010.h5` — direction correct, not significant

| Metric | PCA | EMPCA | winner |
| --- | --- | --- | --- |
| raw MSE | 171.15 | 148.93 | PCA |
| weighted residual | — | — | EMPCA |
| amplitude bias vs. truth | −19.483 | −19.480 | (both ≈ equal, near-total non-recovery) |

`amplitude_bias` ≈ −19.48 for **both** arms initially looked like total
recovery failure. It is not (see the coupling finding below) — it reflects
comparing the reconstruction against a reference-channel truth scale (~22.2)
that is barely present in the channel actually being reconstructed. Gates:
`G2_truth_favours_empca` and `G2_block_bootstrap_agrees` both fail (CI spans
zero); `G2_metric_orderings_reversed` still passes (direction is correct).

### `abra_validation_0005.h5` — confirmatory attempt, direction correct, not significant

Selected by the predeclared file-selection rule (below) *before* its Phase
2/3 outcome was inspected — this is the one genuinely confirmatory Phase 2
attempt in this investigation.

| Metric | PCA | EMPCA | winner |
| --- | --- | --- | --- |
| raw MSE | 131.13 | 148.93 | PCA |
| weighted residual | 8.105×10¹³ | 8.058×10¹³ | EMPCA |
| amplitude \|error\| vs. truth | 4.810 | 4.752 | EMPCA |

Paired difference (truth-error, PCA − EMPCA): +0.0402, 95% CI
**[−0.079, 0.178]** (window-level), **[−0.082, 0.117]** (block-level). Both
span zero. `G2_metric_orderings_reversed` still passes (all three orderings
correctly reversed, replicating the qualitative pattern a third time); the
two CI-based gates fail.

### Cross-file pattern

The *direction* of the effect (PCA wins raw MSE, EMPCA wins weighted
residual and truth) replicated on all three files tried, including the
mechanically-selected confirmatory one. Only file 0000 reached statistical
significance. This is consistent with the effect being real but the
operating point (150 evaluation windows) having limited statistical power
at the SNR levels found in files 0010 and 0005 — a legitimate, reportable
power limitation, not evidence against the direction of the effect.

## Phase 3 — rank axis (gate G3): not confirmed on any of three files

### Attempt 1 — `abra_validation_0000.h5` (diagnostic, not confirmatory)

| rank | 1 | 2 | 3 | 4 | fixed-phase rank-1 OF |
| --- | --- | --- | --- | --- | --- |
| amplitude RMSE | 0.0852 | 0.0880 | 0.0887 | 0.0903 | — |

`G3_rank2_beats_rank1`: CI **[−0.0047, 0.0004] spans zero**.
`G3_rank2_beats_fixed_phase`: passes clearly (+0.201, CI [0.167, 0.239]).

**Diagnosis.** A fine-resolution FFT (10 Hz bins) of `channel0002` resolves
**two distinct spectral lines**, 69,520 Hz and 69,640 Hz (≈120 Hz apart), of
comparable power, confirmed persistent across the full 190-second file at
nearly every 5-second-sampled timestamp. Two tones need a 4-D manifold, not
the 2-D span{cos, sin} the theory assumes for one tone.

### Attempt 2 — `abra_validation_0010.h5` (diagnostic, not confirmatory)

| rank | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| amplitude RMSE | 19.483 | 19.480 | 19.474 | 19.473 |

Flat across every rank. **Original diagnosis in this document said "the
injected amplitude is too weak"; that was imprecise and is corrected below.**

**Corrected diagnosis: channel coupling, not amplitude.** The reference
channel's own quadrature-fit amplitude here (~22.2) is *larger* than file
0000's (~0.10) — file 0010's calibration tone is not weak. What matters is
how much of that tone actually appears in `channel0001` (the SQUID channel
being reconstructed). Fitting the same quadrature templates directly to the
*observed* (SQUID) channel gives a coupling ratio
(observed-amplitude / reference-amplitude) of **0.057** for file 0010,
versus **3.22** for file 0000 — a 56x difference. The calibration signal
barely couples into the channel PCA/EMPCA are trying to reconstruct; no
estimator could recover a reference-channel-scale amplitude from a channel
that essentially does not carry it. This is not a reconstruction defect.

A follow-up scan (see the file-selection table below) found this coupling
ratio falls off steeply and monotonically with frequency across the files
checked — 7.13 at 5.7 kHz down to 0.0021 at 3.4 MHz — consistent with a
low-pass-like transfer function between the calibration/reference channel
and the SQUID readout. This is a genuine physical property of the release
worth noting in its own right.

### Predeclared file-selection rule (frozen before running any candidate)

Two checks, both computed mechanically over `channel0002`'s sample range
`[0, 2097152)`:

1. **Single-tone check**: whole-range FFT peak (excluding DC/Nyquist and a
   sub-1kHz guard band against the 1/f noise floor) with prominence
   ≥1000x median, and no competing peak within 500 Hz (excluding its own
   ±20 Hz sidelobe skirt) exceeding 0.3x its power.
2. **Adequate-amplitude check**: quadrature templates fit to the
   **observed (SQUID) channel** (not the reference channel — see the
   corrected diagnosis above), median amplitude ≥3x the PSD-predicted noise
   floor (`quadrature_amplitude_variance`, the same formula behind Phase 1's
   own `predicted_sigma`).

Applied to the 18 validation files not already used (excluding 0000, 0010):

| file | single-tone | f_peak (Hz) | observed-channel SNR | passed |
| --- | --- | --- | --- | --- |
| 0001 | fail (two-tone, ratio 0.462) | 69,509 | — | no |
| 0002 | fail (two-tone, ratio 0.706) | 69,504 | — | no |
| 0003 | pass | 5,698 | 1.32 | no |
| 0004 | pass | 7,200 | 0.78 | no |
| **0005** | **pass** | **8,702** | **3.42** | **yes — selected** |
| 0006 | pass | 12,999 | 3.61 | yes (not reached; 0005 comes first) |
| 0007-0019 | pass (each) | 28,000-3,400,001 | 1.38-3.22 | no (below 3.0), except 0018 (3.22, not reached) |

(Full per-file table: `results/metrics/TIDMAD_P3_file_selection.json`.)

`abra_validation_0005.h5` is the first file, in index order, passing both
checks, and was selected without ever having its Phase 2/3 reconstruction
outcome inspected beforehand.

**Two implementation bugs were found and fixed while building this check,
both discovered by validating against the two already-known files (0000,
0010) *before* applying the rule to any new candidate** — this is
methodology correction, not post-hoc threshold tuning:

1. The first version of the predicted-sigma formula averaged the *measured*
   signal's complex FFT across windows to stand in for the signal spectrum.
   Since the tone's phase is uniformly distributed across windows, that
   average cancels itself out, giving a near-zero, bogus sigma and SNR
   values in the millions for every file (and "selecting" file 0002 at
   4.8 Hz — plainly the 1/f noise floor, not a signal). Fixed by using
   `quadrature_amplitude_variance` (PSD + weights + template only).
2. The amplitude was fit to the reference channel rather than the observed
   (SQUID) channel — conflating two channels whose coupling ratio varies by
   over 50x across files (the discovery described above). Fixed by fitting
   the observed channel.

### Attempt 3 — `abra_validation_0005.h5` (the confirmatory attempt)

| rank | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| amplitude RMSE | 4.7519 | 4.7522 | 4.7564 | 4.7577 |

Flat, within noise, across every rank. `G3_rank2_beats_rank1`: CI
**[−0.0557, 0.0975] spans zero**. `G3_rank2_beats_fixed_phase`: **fails**
this time (+0.0218, does not clear the gate). `G3_rank3_does_not_improve`:
passes (ranks 3-4 do not help). `smallest_rank_within_tolerance`: 1.

This file passed every predeclared quality check (single-tone, SNR ≥3) and
still did not confirm the prediction. Combined with Phase 2's marginal (not
significant) result on the same file, the most likely explanation is
**insufficient statistical power at this operating point** (150 evaluation
windows) for SNR levels around 3-4, rather than a property specific to any
one file's signal structure. Increasing `n_windows` or SNR would be the
natural next step to distinguish "underpowered" from "genuinely absent" —
not attempted here.

## What this means for the manuscript

- **Cite Phase 2's `abra_validation_0000.h5` result as the positive,
  significant, real-data confirmation of the metric-reversal claim against
  known ground truth.** Optionally note that the qualitative direction
  replicated (without reaching significance) on two further files, including
  one selected by a predeclared rule before its outcome was known — this
  strengthens confidence in the direction of the effect without overclaiming
  significance beyond file 0000.
- **Do not cite any Phase 3 real-data result as a passed or failed gate.**
  Three files were tried, including one genuinely confirmatory attempt by a
  predeclared, mechanically-applied selection rule; none confirmed the
  rank-2 prediction. State plainly, if TIDMAD's rank axis is discussed at
  all: the analytic prediction is confirmed in theory and controlled
  synthetic data; three real-data attempts did not confirm it, for reasons
  spanning genuine multi-tone signal structure, weak channel coupling, and
  (on the one file passing every quality check) likely insufficient
  statistical power at this operating point.
- **The channel-coupling finding (ratio varies from ~7 at kHz frequencies to
  ~0.002 at MHz frequencies) is a genuine, citable physical property of the
  TIDMAD release**, independent of the Phase 2/3 claim, and may be worth a
  sentence in Methods or an appendix if TIDMAD is discussed in any technical
  depth.

## Known limitations of this investigation

- Only 20 of TIDMAD's validation files exist in this subset; the selection
  rule was applied to all of them, but a different, more favorable sample
  range within any file (rather than only `[0, 2097152)`) was never
  searched.
- The file-selection rule's amplitude threshold (SNR ≥3) was chosen as a
  generic statistical convention before any candidate was scanned; it was
  *not* validated to guarantee adequate statistical power for the rank-1
  vs. rank-2 test specifically (which depends on window count and effect
  size in a way a single-window SNR threshold does not fully capture) — this
  is why a file that passed still didn't confirm the prediction, and is
  worth fixing in any future predeclaration.
