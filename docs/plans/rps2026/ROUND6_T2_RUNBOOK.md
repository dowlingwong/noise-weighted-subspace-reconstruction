# Round 6 runbook — T2 and the missing interval

**Date:** 2026-08-17
**Compute:** ~35 minutes. **No training, no new experiments.** Everything reads files already on disk.
**Standing rules:** `AGENT_PROMPT_REMOTE_EXPERIMENTS.md` applies in full.
**Step 0:** `git pull` first. `score_unrounded.py` was changed after round 5 and now emits per-window products; without the pull, T2b cannot run.

---

## 0. Where round 5 got to

**T1 succeeded and its methodology audits clean.** Both arms reproduce round 3's `log_score_upstream_equivalent` exactly, so the only change is the removal of `np.round(..., 2)`. Because `score_unrounded` calls the release's own `process_iteration` with an identically ordered file list, the SNR arrays are the upstream's by construction.

Result, 16 in-band files:

| arm | raw score (unrounded) |
|---|---|
| `t_mse` | 0.024149 |
| `t_chi2` | 0.023131 |
| **difference** | **−0.001018, i.e. `t_chi2` is 4.2 % worse** |

**Two problems remain, and both are about evidence quality, not about the number.**

**Problem 1 — the contrast has no uncertainty.** The score is a mean over 320 one-second windows, and only a point difference was reported. The round-5 gate pins the raw score to ±0.005 (the rounding bin), which is five times *larger* than the 0.001 contrast being claimed. Nothing currently establishes that −0.001018 is distinguishable from zero.

**Problem 2 — T2 never ran.** The held-out χ² / raw-MSE series was not extracted, so there is no measurement of the paper's actual claim at rung 4. The benchmark score is a third metric that neither arm optimised: `getSNR` locates the peak frequency in the *reference* channel and evaluates the *denoised readout* there. It is a narrowband SNR-recovery measure, not the whitened residual the ladder is about.

Useful property established during the audit: the reference channel is copied through unchanged by `infer.py`, so `snr_sg` and its normalisation are **identical across arms**. The entire contrast is carried by `snr_squid`. The comparison is well posed.

---

## 1. Environment

```bash
export CODE_ROOT=~/noise-weighted-subspace-reconstruction
export RPS_RESULTS=$CODE_ROOT/results/rps2026
export TIDMAD_UPSTREAM=~/src/TIDMAD
export PYTHON=/home/dwong/anaconda3/bin/python
cd $CODE_ROOT && git pull && export PYTHONPATH=src
grep -c per_window_products src/tidmad/score_unrounded.py    # must be >= 1
```

Resolve and record the denoised directory as in round 5; it is not under `/ceph/dwong/datasets`.

---

## 2. T2a — held-out χ² and raw MSE series **(the paper's actual claim)**

```bash
for ARM in t_mse t_chi2; do
  LOG=$(ls -t artifacts/tidmad/$ARM/training.log artifacts/tidmad/$ARM/*/training.log 2>/dev/null | head -1)
  echo "$ARM -> $LOG"
  grep -E "^step +[0-9]+" "$LOG" > $RPS_RESULTS/E5_${ARM}_eval_series.txt
  wc -l $RPS_RESULTS/E5_${ARM}_eval_series.txt
done
```

Each line carries `step`, `train_mse`, `train_chi2`, `val_mse`, `val_chi2`, `steps_per_s`, `elapsed_s`, `eta_s`. Convert to `E5_${ARM}_eval_series.csv` with those columns.

**Gate:** both arms must have the **same number of evaluations at the same step values** — matched budget and seed handling. If the step grids differ, the arms are not comparable and everything downstream is void: stop and report.

**Report:**

- `val_chi2` and `val_mse` per arm at the final evaluation, and averaged over the last ten;
- the **per-evaluation paired difference series** `t_chi2 − t_mse` for both metrics, with a percentile bootstrap 95 % interval over evaluations.

**Predicted signature:** `t_chi2` lower on `val_chi2`; `t_mse` lower on `val_mse`. That is metric reversal at rung 4.

**If it does not reverse, report it exactly as measured.** A null here is a real finding about whether the metric question survives into a nonlinear architecture at this scale, and §5 of the paper already anticipates it. Do not tune, do not re-run with different settings, do not select a subset of evaluations.

---

## 3. T2b — an interval for the benchmark contrast

Re-run the in-band rescore so the per-window products are captured, then bootstrap over windows.

```bash
for ARM in t_mse t_chi2; do
  timeout 30m nice -n 10 $PYTHON -m tidmad.score_unrounded \
    --upstream $TIDMAD_UPSTREAM --data-dir <resolved_denoised_dir> \
    --model-tag $ARM --in-band-only $RPS_RESULTS/P1_injection_survey.json \
    --out $RPS_RESULTS/E5_${ARM}_inband_unrounded.json
done
```

Then, from the two records' `per_window_products` (320 values each, **paired by window index**):

- confirm `raw_score_unrounded` reproduces round 5 (0.024149 / 0.023131) — a determinism check;
- confirm `snr_sg_normalised` is identical between arms, as the audit predicts;
- compute the paired difference per window and a percentile bootstrap 95 % interval over the 320 windows, 10 000 resamples;
- write `$RPS_RESULTS/E5_benchmark_contrast_interval.json` with the mean difference, the interval, and whether it excludes zero.

**This decides how the benchmark result may be described.** If the interval excludes zero, the paper can say the MSE-trained arm scores better on this benchmark. If it includes zero, the honest statement is that the benchmark does not distinguish the arms — which is different from, and weaker than, what round 5's point estimate suggests.

Windows within a file are adjacent one-second segments and may be correlated. If a block bootstrap over files (16 blocks) is straightforward, report it alongside; otherwise note the caveat.

---

## 4. Sync back

```
E5_t_mse_eval_series.csv            E5_t_chi2_eval_series.csv
E5_t_mse_inband_unrounded.json      E5_t_chi2_inband_unrounded.json   (refreshed)
E5_benchmark_contrast_interval.json
```

Append a **Round 6** section to `EXPERIMENT_RUN_LOG.md`: the resolved log paths; the step-grid match; `val_chi2` / `val_mse` per arm with paired differences and intervals; the benchmark contrast with its interval; the determinism and `snr_sg` identity checks; and a plain statement of what the round does and does not support.

---

## 5. Order and stop conditions

| # | Task | Stop if |
|---|---|---|
| 1 | `git pull`; confirm `per_window_products` present | the pull fails or the symbol is absent |
| 2 | T2a extraction | step grids differ between arms |
| 3 | T2a paired statistics | — |
| 4 | T2b rescore | `raw_score_unrounded` differs from round 5 |
| 5 | T2b interval | — |

**Report every result as measured.** Two outcomes are findings rather than failures: a non-reversing held-out contrast, and a benchmark interval that includes zero. Both are reportable and both are already anticipated in the paper's limitations. The only outcome meaning "bug" is a step-grid mismatch or a determinism failure at step 4.
