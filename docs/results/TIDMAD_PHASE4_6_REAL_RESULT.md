# TIDMAD Phases 4 and 6 on REAL Data — Result

Date: 2026-07-27. Stage: TIDMAD (Paper 1's second real-data anchor).
Roadmap: [`../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`](../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md).
Predeclaration: [`TIDMAD_PREDECLARED_GATES.md`](TIDMAD_PREDECLARED_GATES.md).
Synthetic-mode Phases 2-6: [`TIDMAD_PHASES_2_TO_6_RESULT.md`](TIDMAD_PHASES_2_TO_6_RESULT.md).
Phases 2/3 real data: [`TIDMAD_PHASE2_3_REAL_RESULT.md`](TIDMAD_PHASE2_3_REAL_RESULT.md).
Configs: `configs/tidmad/phase4_calibration_boundary_real_noise.yaml`,
`configs/tidmad/phase6_negative_controls_real_noise.yaml`.
Evidence: `results/metrics/TIDMAD_P4_real_seed20260727.json`,
`results/metrics/TIDMAD_P6_real_seed20260727.json`.

Both phases use `abra_validation_0000.h5` for the parts of their computation
that need real signal-containing data (Phase 6 needs it directly; Phase 4's
own computation only consumes science-file noise, but the loader's
structural check still requires a validation file to be present) — the same
file as Phase 2's significant positive result, so there is a real, working
nominal baseline for Phase 6 to degrade from, and consistency with Phase 2's
frequency (69513.32 Hz, `docs/tidmad_data_contract.json`).

## Phase 4 — calibration boundary: real data calibrates well, unlike GWOSC

The roadmap's prior expectation, carried over from GWOSC's experience, was
that a simple Gaussian global-PSD model would show real degradation on real
SQUID noise (8-bit quantization, drift, glitches). **That is not what
happened at this file and sample range.**

| Model | χ²/dof | score dispersion |
| --- | --- | --- |
| stationary synthetic control (halt gate) | 1.00015 | 1.0140 |
| **global model, real noise** | **1.00312** | **1.0142** |
| local, radius 8 | 1.0809 | 1.0500 |
| local, radius 16 | 1.0426 | 1.0535 |
| local, radius 32 | 1.0223 | 1.0276 |

The control passes (as it must — a halt gate). The **global model on real
SQUID noise is comfortably within the predeclared [0.9, 1.1] acceptance
band on both statistics**, essentially matching the synthetic control.
`cost_of_mismatch`: sigma inflation 1.0002, threshold inflation 1.0002 —
negligible. The local models (radii 8/16/32) show the same qualitative
pattern documented in the synthetic write-up (smaller radius → more
positive bias from limited averaging) but all three remain within the
acceptance band too.

**This is a genuinely different, and genuinely interesting, result from
GWOSC's real-noise calibration failure.** At this file and sample range, a
simple global Gaussian PSD model calibrates TIDMAD's real SQUID noise
adequately — there is no boundary lesson to report here of the kind GWOSC
supplied. Report this honestly as a real, positive finding, not force it
into the "GWOSC-style boundary" narrative the roadmap anticipated. Whether
this holds at other files, sample ranges, or the full 200-second file
duration was not checked.

Gate: `P4_control_passes` PASS.

## Phase 6 — negative controls: three of five degrade, two do not

| control | paired difference vs. nominal | 95% CI | degrades? |
| --- | --- | --- | --- |
| `wrong_frequency` | +0.3567 | [0.3065, 0.4122] | yes, strongly |
| `permuted_bins` | +0.0161 | [0.0075, 0.0272] | yes |
| `wrong_psd` | +0.0172 | [0.0058, 0.0301] | yes |
| `noise_only_train` | −0.0025 | [−0.0092, 0.0047] | **no** (CI spans zero) |
| `random_weighted_basis` | −0.0062 | [−0.0116, −0.0005] | **no — improves slightly** (CI entirely negative) |

Three of five controls degrade clearly (`permuted_bins`, `wrong_frequency`,
`wrong_psd`); two do not (`noise_only_train` is statistically flat;
`random_weighted_basis` *improves* slightly, CI excluding zero on the wrong
side). This is a real, mixed finding, not massaged into a clean 5-for-5.

**Report both exceptions plainly, as CRESST's precedent requires** (CRESST
had exactly one control that did not degrade and reported it). Plausible,
not yet investigated, explanations: `random_weighted_basis` replaces the
learned rank-2 subspace with a random one in the *same, correctly-weighted*
geometry — if most of the benefit at this operating point comes from
correct whitening rather than the specific learned directions, a random
same-rank basis in the correct metric could perform comparably or even
marginally better by chance. `noise_only_train` fits the subspace on
signal-free windows instead of signal-containing ones — if whitening alone
concentrates recoverable information near the injection frequency regardless
of which windows were used to fit the subspace, training-set composition
may matter less here than in the synthetic design. Neither explanation was
tested further.

Gates: `P6_wrong_psd_degrades` PASS, `P6_wrong_frequency_degrades` PASS,
`P6_permuted_bins_degrades` PASS, `P6_noise_only_train_degrades` FAIL,
`P6_random_weighted_basis_degrades` FAIL.

## What this means for the manuscript

- Phase 4 real result is a genuine positive finding worth reporting on its
  own terms: TIDMAD's real SQUID noise, unlike GWOSC's real strain, is
  adequately calibrated by a simple global Gaussian PSD model at this
  operating point. Do not claim this generalizes to all files or the full
  file duration — it was checked at one file, one ~0.2-second sample range.
- Phase 6 real result is mixed: report 3 of 5 controls degrading, and name
  the 2 that do not, exactly as CRESST's one exception was named. Do not
  claim all five controls degrade on real data.
