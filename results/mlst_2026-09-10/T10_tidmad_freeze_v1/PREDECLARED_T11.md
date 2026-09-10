# PREDECLARED_T11 — TIDMAD frozen operating point (MLST revision)

**Frozen 2026-09-10, before any validation or final-test method outcome was
computed.**  Protocol: MS `plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md` T10–T12
and `plan/mlst_protocol_2026-09-10.json`.

Everything below was decided on **development material only**
(`abra_validation_0000/0005/0010`, stretch start 7 097 152) plus the
characterisation of every file, which uses transfer-calibration windows alone
and precedes any method fit.

## Frozen choices

| Item | Value |
|---|---|
| Window length `w` | **65536** samples (0.0065536 s) |
| Rank | 2 (fixed, never selected) |
| Endpoint | mean paired squared quadrature error difference, physical amplitude `a = 2q/√w`, readout ADC units |
| Truth label | `t_i = ĝ ρ_i`, `ĝ` complex LS on the 64 transfer-calibration windows |
| Evaluation block length `b_eval` | **16** |
| Calibration block length `b_cal` | **2** |
| Dependence verdict | **inconclusive** — see below |
| Bootstrap | 2000 replicates, seed 20260910, percentile |
| Methods | zero, calibration-mean, raw-bin, quadrature-OF, PCA, EMPCA, Wiener, shrinkage-cal |
| Allocation | 64 cal / guard 16384 / 350 fit / guard 16384 / 150 eval; stretch 36 995 072 samples |
| Pool start sample | 0 |

## Dependence assessment (declared rule, item 4)

Aggregated (maximum normalised) evaluation variance curve over the 10 contrasts
of the eligible development stretch:

| b | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| normalised V(b) | 1.000 | 1.773 | 2.431 | 3.290 | 1.895 |

No plateau candidate (1, 2, 4) qualified; the top of the grid is not locally
flat in both ratios; the curve is **non-monotone**, so the large-block variance
estimates are themselves unstable at 9 blocks.  Verdict **inconclusive**:
`b_eval = 16` is used, every interval is flagged, and **no file may receive a
"useful" verdict** — only "ordering", "not-useful" or "inconclusive".  This is
not an aggregation artefact: 7 of the 10 individual contrast curves are
independently inconclusive and 7 of 10 select b = 16.

The calibration curve stabilised at `b_cal = 2` (candidates stop at 8 because
64 calibration windows give ≥ 5 blocks only up to that length).

## Selection

Declared criterion: maximise the precision-weighted mean over
**eligible-coupled** development stretches of `Δ(zero, EMPCA)`; ties → smaller
window.

| w | eligible dev stretches | θ_IV | 95 % CI |
|---|---|---|---|
| 4096 | 1 | −0.00952 | [−0.0328, +0.0137] |
| 16384 | 1 | +0.00431 | [−0.0090, +0.0177] |
| **65536** | **1** | **+0.09727** | **[+0.0712, +0.1234]** |

**Limitation recorded in advance of evaluation:** only `abra_validation_0010.h5`
was eligible-coupled among the development files at any window length
(`0000` uncoupled at all three w; `0005` uncoupled at 4096/16384 and chirped at
65536).  The selection therefore rests on a single development stretch per
window length, and the "aggregation over development files" of the dependence
rule reduces to aggregation over contrasts of one file.

## File roles (fixed here, from the characterisation only)

* **Validation** (first three eligible-coupled pool files in index order):
  `abra_validation_0008.h5`, `abra_validation_0009.h5`, `abra_validation_0011.h5`.
* **Final test** (the remaining 14 pool files, evaluated once each):
  `0001, 0002, 0003, 0004, 0006, 0007, 0012, 0013, 0014, 0015, 0016, 0017, 0018, 0019`.

11 of 17 pool files are eligible-coupled at w = 65536, so the validation set is
complete and the fewer-than-three fallback is not exercised.

## Declared in advance

1. Only eligible-coupled stretches enter the combined estimate; every other
   outcome is evaluated and reported as the failure regime.
2. A file whose reference line SNR is below 100 supports only the historical
   reference-unit diagnostic and is labelled `diagnostic-only`; it does not
   enter the readout-unit combined estimate.  The reference-SNR rule is a
   declared necessary condition, not a demonstrated bias bound — its three
   errors-in-variables assumptions are recorded with each result.
2. The usefulness criterion requires beating **both** zero and calibration-mean
   with 95 % intervals excluding zero.  With the dependence verdict
   inconclusive, this verdict is unavailable for every file in this run.
4. Held conditional (not resampled): the 350 fit windows and the PCA/EMPCA
   bases and Wiener signal PSD estimated from them, the science-file noise bin
   powers, `k_inj`, and the block lengths.
5. The final test is executed once.  No retuning against final-test outcomes.

## Expectation recorded before evaluation

The programme anticipated that high-frequency files would come out *uncoupled*
and form the failure regime.  The characterisation **refutes** this: at
w = 65536, `0018` (1.80 MHz) and `0019` (3.40 MHz) are eligible-coupled with
readout line SNR 30.2 and 32.8.  The uncoupled and chirped stretches are instead
the *low*-frequency files (`0003`–`0006`, 5.6–13 kHz) and the ~69.6 kHz
two-tone group (`0000`, `0001`, `0002`).  This is recorded here, before the
method outcomes on those files were computed.
