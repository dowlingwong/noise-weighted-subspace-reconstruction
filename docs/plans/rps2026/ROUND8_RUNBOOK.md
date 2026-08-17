# Round 8 runbook — whitening measurement, then the rung-4 replication seed

**Date:** 2026-08-18
**Submission deadline:** 29 Aug 2026 AoE. There is room for one GPU round, not two.
**Standing rules:** `AGENT_PROMPT_REMOTE_EXPERIMENTS.md` applies in full.
**Supersedes:** `ROUND7_WHITENING_RUNBOOK.md` (Task A below is that runbook, with the
path resolution now done in code).

Two tasks. **Task A is 5 minutes on CPU and decides whether a sentence in the paper is
true.** **Task B is ~5 hours on two GPUs and decides how strongly the paper's main
negative result can be stated.** Run A first and report it before starting B — if A
fails, the paper's account of B's result changes, and I would rather know that before
nine GPU-hours are spent.

---

## 0. Where things stand

| Rung | Result | Status |
|---|---|---|
| 3 (NFPA, colored noise) | metric reversal: −1.86 % χ̄², +49.6 % raw MSE | 10 seeds, intervals exclude zero |
| 3 (white-noise control) | both contrasts exactly 0 | 20 seeds |
| 4 (factored attention) | **no reversal** — χ²-trained arm better on *both* metrics | **single seed each arm** |
| 4 (upstream benchmark) | interval spans zero; does not distinguish the arms | 320 windows, block bootstrap |

The paper reports the rung-4 non-reversal honestly, and explains it by saying that
whitening the band-limited spectrogram is close to a constant rescaling, so the two
objectives are nearly collinear and there is no metric contrast to detect.

**Two things are wrong with that position, and this round fixes both.**

1. The collinearity explanation is *inferred from the effect signature, not measured*.
   It is the paper's account of its own central negative result, and a referee will ask
   for the number. **Task A measures it.**
2. Each rung-4 arm is a single seed. A paired difference between two single runs cannot
   be separated from optimisation luck, and the obvious referee objection is `n=1`.
   **Task B adds a second seed.**

---

## 1. Step 0 — pull, environment, and paths

```bash
export CODE_ROOT=~/noise-weighted-subspace-reconstruction
cd $CODE_ROOT && git pull && export PYTHONPATH=src
export PYTHON=/home/dwong/anaconda3/bin/python

# New this round; without the pull neither task can run.
ls src/tidmad/paths.py && grep -c '"--seed"' src/tidmad/train.py    # must print the file and >= 1
```

### Finding the TIDMAD data — do this in code, not by hand

Every previous runbook told a human to "resolve and record the directory". That is how
rounds drift apart. `tidmad.paths` now resolves it by the only test that matters —
whether the directory actually contains the files — and prints what it found:

```bash
$PYTHON -m tidmad.paths
```

Expected on `deepthought` (the checklist's planned `/ceph/dwong/datasets/...` root does
not exist here; the data live in the archived Paper-1 tree):

```
TIDMAD_DATA_ROOT=/ceph/dwong/paper1_dataset/tidmad/raw
    2 x abra_science_*.h5, 20 x abra_validation_00??.h5
TIDMAD_DENOISED_DIR=/ceph/dwong/paper1_dataset/tidmad/denoised
    40 x abra_validation_denoised_*.h5
TIDMAD_UPSTREAM=/home/dwong/src/TIDMAD
    1 x benchmark.py
```

To load them into the shell:

```bash
eval "$($PYTHON -m tidmad.paths --shell)"
```

**If any path reports NOT FOUND**, the command prints the candidates it tried and a
`find` to run. Search, then export the correct value explicitly — an env override is
still content-checked, so a wrong path fails immediately rather than silently reading
the wrong tree:

```bash
find /ceph -maxdepth 6 -name 'abra_science_*.h5' -printf '%h\n' 2>/dev/null | sort -u
```

**Record the resolved paths in the run log.** They are written into
`E6_whitening_range.json` automatically; record them for Task B by hand.

**Stop if** the raw root resolves anywhere that is not the tree rounds 3–6 used.
Comparing a new seed against the archived one requires identical data.

---

## 2. Task A — measure the whitening dynamic range (~5 min, CPU only)

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  timeout 15m nice -n 10 $PYTHON -m tidmad.whitening_range \
  --out results/rps2026/E6_whitening_range.json
```

`--data-dir` is now optional and auto-resolves; pass it only to override.

The module uses the **frozen configuration and the same estimator training used**
(`estimate_band_psd`, boxcar window, median average, 512 calibration windows,
`_noise_only_windows`), so the `J` it measures is the `J` the loss divided by.

### The argument, in one line

`MSE = mean(r²)` and `chi2 = mean(r²/J)` are the same objective up to a scale — identical
minimisers — exactly when `J` is constant over the bands carrying the residual. So the
whole question is the *spread* of the weights `1/J`.

### The gate

**`cv_inverse_J` = `std(1/J)/mean(1/J)`, excluding DC** (band 0 is degenerate for
zero-mean calibration noise; the module reports every statistic with and without it).

| Outcome | Meaning |
|---|---|
| **below ~0.1** | collinearity account supported; §4's sentence becomes a measurement with a number in it |
| **substantially above ~0.1** | account is **wrong**; the absence of reversal at rung 4 needs a different explanation |

**This threshold is demanding and it may well fail.** On synthetic vectors a flat `J`
gives 0.000, but a mere 2× dynamic range already gives **0.20** — twice the threshold. If
the SQUID noise has any real structure across 0–800 kHz, this test refutes the
collinearity story.

**Report the number either way. Do not adjust the threshold after seeing it.** A
refutation now is worth more than a plausible sentence a referee dismantles later. Do not
substitute a different PSD estimator, window, or averaging rule to make the number come
out — the point is to measure what the loss actually divided by.

**Do not edit the paper.** If the gate fails, report it; §4 is revised at the desk.

---

## 3. Task B — a second seed for both rung-4 arms (~5 h wall clock, 2 GPUs)

**Why this and not something else.** The rung-4 non-reversal is the paper's most
interesting claim and its weakest evidence. Round 6 measured a paired difference between
one run and one run; the honest reading is that we cannot yet separate the objective from
optimisation noise. A second seed does not make the result strong, but it distinguishes
*"the effect reverses sign between seeds"* (optimisation noise, and the paper must say so)
from *"both seeds agree"* (a real, if small, effect at this scale). Either answer is
publishable; not knowing is not.

### Launch

Both arms, **the same new seed**, on separate GPUs:

```bash
cd $CODE_ROOT
export SEED=20260818

GPU_INDEX=0 TIDMAD_SEED=$SEED TIDMAD_TIMEOUT=8h PYTHON=$PYTHON \
  screen -dmS rps_mse_s$SEED -L -Logfile screenlog_t_mse_s$SEED.txt \
  ./scripts/train_tidmad_local.sh mse

GPU_INDEX=1 TIDMAD_SEED=$SEED TIDMAD_TIMEOUT=8h PYTHON=$PYTHON \
  screen -dmS rps_chi2_s$SEED -L -Logfile screenlog_t_chi2_s$SEED.txt \
  ./scripts/train_tidmad_local.sh chi2
```

The launcher resolves the data root itself, tags the run directory `t_mse_s20260818` /
`t_chi2_s20260818` so nothing overwrites the archived first seed, prints the seed in its
banner, and writes the seed into the archived run config.

### Gates, in order

| # | Gate | Stop if |
|---|---|---|
| 1 | both banners print `Seed: 20260818` and the **same resolved data root** | they differ — the arms would confound seed or data with objective |
| 2 | both print the same `model_params` (569,352) and differ only in `loss=` | any other config field differs |
| 3 | both reach step 25,000 within the 8 h cap | a run is killed at 124 — resume with `TIDMAD_RESUME=auto`, do not restart from zero |
| 4 | both produce **50 evaluations at identical step values** | grids differ; the arms are not comparable and the analysis is void |

**Do not run the `chi2_of` secondary arm (E7).** It answers a different question and there
is not time for both.

### Analysis — same procedure as round 6, applied to the new logs

```bash
export RPS_RESULTS=$CODE_ROOT/results/rps2026
for ARM in t_mse t_chi2; do
  LOG=$(ls -t artifacts/tidmad/${ARM}_s$SEED/*/training.log | head -1)
  echo "$ARM -> $LOG"
  grep -E "^step +[0-9]+" "$LOG" > $RPS_RESULTS/E7_${ARM}_s${SEED}_eval_series.txt
done
```

Convert each to `E7_${ARM}_s${SEED}_eval_series.csv` with the round-6 columns
(`step,train_mse,train_chi2,val_mse,val_chi2,steps_per_s,eta_s`), then report:

- `val_chi2` and `val_mse` per arm at step 25,000, and averaged over the last ten
  evaluations;
- the per-evaluation paired difference `t_chi2 − t_mse` for both metrics, with a
  percentile bootstrap 95 % interval (10,000 resamples, seed 20260818);
- **the seed-to-seed comparison**, which is the point of the round: put the new paired
  differences beside round 6's (`val_chi2` −1.148e-04, `val_mse` −57.18, all 50
  evaluations) and state plainly whether the **sign agrees**.

Write `results/rps2026/E7_rung4_seed_replication.json` with both seeds' paired
differences, both intervals, and a boolean for sign agreement on each metric.

### What each outcome means — decided in advance

| Outcome | The paper says |
|---|---|
| both seeds agree in sign on both metrics | rung 4 does not reproduce the reversal, and this replicates across two seeds |
| signs disagree on either metric | the rung-4 contrast is **within seed variance**; the honest report is that the experiment does not resolve the question at this scale |
| one arm fails to complete | report the partial run; do not compare unequal budgets |

Neither of the first two is a failure. The second is arguably the more useful finding,
because §5 already anticipates it and the paper would then be stating a limit it has
actually measured.

---

## 4. Sync back

```
results/rps2026/E6_whitening_range.json
results/rps2026/E7_t_mse_s20260818_eval_series.{txt,csv}
results/rps2026/E7_t_chi2_s20260818_eval_series.{txt,csv}
results/rps2026/E7_rung4_seed_replication.json
```

Checkpoints and denoised outputs stay outside Git, as before.

Append a **Round 8** section to `EXPERIMENT_RUN_LOG.md` covering: the resolved paths as
printed by `tidmad.paths`; `cv_inverse_J` with and without DC, the dynamic-range ratios,
and the verdict the module printed; the four Task B gates with their observed values; both
arms' held-out numbers with intervals; the sign-agreement comparison against round 6; and
final machine state (GPUs idle, no `run.pid`, no screen sockets).

---

## 5. Standing rules for this round

- **Report every result as measured.** A failed whitening gate and a sign disagreement
  between seeds are both findings, not failures. The only outcomes meaning "bug" are a
  step-grid mismatch, a config-equality failure, or a data root that differs between arms.
- **No tuning, no re-running with different settings, no subset selection.** If a number
  is disappointing, it is still the number.
- **Do not edit the manuscript.** Report; the desk does the writing.
- **Shared machine:** `nice -n 10`, an explicit `timeout` on everything, and four threads
  for any CPU/BLAS work.
