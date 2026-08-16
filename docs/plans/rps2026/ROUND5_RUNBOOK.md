# Round 5 runbook — remote execution

**Date:** 2026-08-16
**Purpose:** resolve the three rung-4 placeholders and return the data the paper's figure needs.
**Compute:** ~40 minutes. **No training. No new experiments.** Both arms' checkpoints and denoised outputs already exist.
**Standing rules:** `AGENT_PROMPT_REMOTE_EXPERIMENTS.md` applies in full — one GPU pinned if any is used, `timeout` on long commands, `nice -n 10`, machine left clean.

---

## 0. Why this round exists

Round 3 reported the rung-4 contrast as exactly `0.0` and called it a publishable null. **It is not a null.** `benchmark.py:139` aggregates as

```python
score = np.round(sum(snr_sg * snr_squid)/n, decimals=2) + 1e-10
return math.log(score, 5.27)
```

The raw score is **rounded to two decimals** before the logarithm, so any two methods differing by less than 0.005 return bit-identical values. Inverting the reported logs: both arms sit at a rounded raw score of **0.02** in band and **0.01** over all twenty files. Savitzky–Golay rounds to the same 0.01, which is why round 1's reproduction matched to all 17 significant figures despite the two arms' denoised outputs differing by 60 MB.

Two tasks follow: recompute the score without the rounding (T1), and extract the held-out contrast that should carry rung 4 regardless of the benchmark (T2).

---

## 1. Environment

```bash
export CODE_ROOT=~/noise-weighted-subspace-reconstruction
export TIDMAD_UPSTREAM=~/src/TIDMAD
export TIDMAD_DENOISED_DIR=/ceph/dwong/datasets/tidmad/denoised
export RPS_RESULTS=$CODE_ROOT/results/rps2026
export PYTHON=/home/dwong/anaconda3/bin/python     # cu121 interpreter
cd $CODE_ROOT && export PYTHONPATH=src
```

`/ceph/dwong/datasets` did not exist in earlier rounds; the denoised outputs were written to whatever path round 3 used. **Confirm the location first** and record it:

```bash
ls -la $TIDMAD_DENOISED_DIR/abra_validation_denoised_t_mse_0000.h5 2>/dev/null \
  || find /ceph/dwong -name 'abra_validation_denoised_t_mse_*.h5' 2>/dev/null | head -3
```

**Gate:** 20 files for each of `t_mse` and `t_chi2`. If fewer, stop and report — inference was incomplete.

---

## 2. T1 — unrounded rescore (resolves all three placeholders)

```bash
for ARM in t_mse t_chi2; do
  timeout 30m nice -n 10 $PYTHON -m tidmad.score_unrounded \
    --upstream $TIDMAD_UPSTREAM \
    --data-dir $TIDMAD_DENOISED_DIR \
    --model-tag $ARM \
    --in-band-only $RPS_RESULTS/P1_injection_survey.json \
    --out $RPS_RESULTS/E5_${ARM}_inband_unrounded.json
done
```

Then the same without `--in-band-only`, writing `E5_${ARM}_all20_unrounded.json`.

Each record carries `raw_score_unrounded`, `log_score_unrounded`, and `log_score_upstream_equivalent` — the last should reproduce round 3's reported figure exactly, confirming the only change is the rounding.

**Gate:** `log_score_upstream_equivalent` matches round 3 (`-2.353761452599672` in band). If it does not, the scoring path has changed and everything downstream is suspect — stop and report.

**Report the paired difference** `raw_score_unrounded(t_chi2) − raw_score_unrounded(t_mse)`, with sign and magnitude.

**If the unrounded contrast is still ~0**, that is a genuine null and reportable as one — but say *"the benchmark cannot resolve the contrast at this operating point"*, not *"the metric made no difference"*. Both arms sit ~6 orders of magnitude below FCNet (0.01–0.02 against 53,439), so the score is near its floor.

`score_unrounded` loads the release's own `GetOneSecPSD`, `getSNR` and `process_iteration` verbatim by exec-ing only the source above its argparse block. **Nothing upstream is modified.** If it raises about a missing marker, the release layout changed — stop and report rather than editing it.

---

## 3. T2 — held-out contrast (this is what rung 4 should rest on)

The benchmark score is a downstream proxy. The paper's claim is the **paired contrast under the two metrics**, which the training loop already evaluates on the held-out chronological split. Round 3's final values were 11519.3 / 0.0135078 (T-mse) and 11484.4 / 0.0134244 (T-chi2), but **two scalars cannot carry an interval.**

Extract the per-evaluation series from both training logs:

```bash
for ARM in t_mse t_chi2; do
  grep -E "^step +[0-9]+" artifacts/tidmad/$ARM/training.log \
    > $RPS_RESULTS/E5_${ARM}_eval_series.txt
  wc -l $RPS_RESULTS/E5_${ARM}_eval_series.txt
done
```

Each line has `step`, `train_mse`, `train_chi2`, `val_mse`, `val_chi2`, `steps_per_s`, `elapsed_s`, `eta_s`. Convert to one tidy CSV per arm with those columns, written to `$RPS_RESULTS/E5_${ARM}_eval_series.csv`.

**This is the figure's third panel.** Without it the figure shows rung 3 only.

**Report:** for the final evaluation and for the mean over the last ten evaluations, `val_chi2` and `val_mse` per arm, and the paired differences. If the two arms' evaluation steps align (they should — matched budget and seed), also report the per-evaluation paired difference series so an interval can be computed.

**Expected signature, from the ladder:** T-chi2 lower on `val_chi2`, T-mse lower on `val_mse`. **If it does not reverse, report it plainly** — a null here is a real finding about whether the metric survives into a nonlinear architecture, and it must not be tuned.

---

## 4. T3 — figure data for Block A and B

Already on disk; just confirm both are present and readable, since the figure is built from them:

```bash
wc -l $RPS_RESULTS/E1_nfpa_blockA/block_A_colored_raw.csv \
      $RPS_RESULTS/E2_nfpa_blockB/block_B_white_raw.csv
```

**Gate:** Block A has 50 rows (10 seeds × 5 methods), Block B has 100 (20 × 5).

---

## 5. What to sync back

Small files only, all under `$RPS_RESULTS`:

```
E5_t_mse_inband_unrounded.json      E5_t_chi2_inband_unrounded.json
E5_t_mse_all20_unrounded.json       E5_t_chi2_all20_unrounded.json
E5_t_mse_eval_series.csv            E5_t_chi2_eval_series.csv
```

Checkpoints and denoised `.h5` stay on the server and are referenced by path.

Append a **Round 5** section to `EXPERIMENT_RUN_LOG.md` recording: the resolved denoised-file location; the `log_score_upstream_equivalent` match against round 3; both unrounded raw scores and their paired difference; the held-out `val_chi2` / `val_mse` per arm with paired differences; and a plain statement of what the round does and does not support.

---

## 6. Order and stop conditions

| # | Task | Stop if |
|---|---|---|
| 1 | Locate denoised outputs | fewer than 20 files per arm |
| 2 | T1 unrounded rescore | `log_score_upstream_equivalent` ≠ round 3's value |
| 3 | T2 held-out extraction | training logs missing or arms' step grids differ |
| 4 | T3 confirm Block A/B CSVs | row counts wrong |

Nothing here trains, tunes, or re-runs an experiment. **If a gate fails, report it — do not work around it.** Every number the paper will cite must trace to an archived file.
