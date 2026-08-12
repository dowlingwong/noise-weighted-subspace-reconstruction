# Round 2 checklist — R/PS 2026

**Date:** 2026-08-12
**Supersedes:** nothing. This is the **delta** on `EXPERIMENT_CHECKLIST_2026-08-08.md`, which remains authoritative for protocol definitions, safety rules and archival contract.
**Round 1 outcome:** `EXPERIMENT_RUN_LOG.md`. Prerequisites passed; E1 stopped at a pre-fit gate; E3/E4 never launched.

---

## 0. What changed since round 1

Three blockers were fixed in commit `49096ac`. Read this before re-running anything.

| Round-1 blocker | Fix | What you must now do |
|---|---|---|
| GPU preflight rejected capability 5.2 | Gate lowered to `(5, 2)`; the loop is **pure FP32**, so 7.0 was never required. Preflight matmul now FP32 too | **Measure throughput on a TITAN X** (§2) |
| E1 stopped at `MC covariance error 8.20e-02` | Runner now gates on the **whitened** statistic at 8192 draws, matching the audited test. Tolerance unchanged at 5e-2. Verified: 1.10e-2 passes | Re-run E1 (§4) |
| P1 out-of-band → baseline comparison lost | `tidmad.score --in-band-only` restricts to the 16 in-band files | Probe Route A (§3), then score in band |

**The PyTorch/driver mismatch is *not* fixed and cannot be fixed by the gate.** The repo-locked `torch 2.12.1+cu130` cannot initialise CUDA against driver 525.147.05. Use a **cu121** interpreter — the host's `torch 2.5.1+cu121` did initialise on `deepthought`. Pass it via `PYTHON=`.

---

## 1. Preconditions

```bash
export CODE_ROOT=~/noise-weighted-subspace-reconstruction
export TIDMAD_DATA_ROOT=/ceph/dwong/paper1_dataset/tidmad/raw   # round-1 resolved path
export TIDMAD_UPSTREAM=~/src/TIDMAD
export RPS_RESULTS=$CODE_ROOT/results/rps2026
export PYTHON=/home/dwong/anaconda3/bin/python                  # cu121 torch
```

`/ceph/dwong/datasets` does not exist on `deepthought`. Round 1 recorded the `path_migration` note; keep using it and keep recording it.

**Run the full test suites first — round 1's fixes are unverified on the server.** The NFPA suite took 11 min 51 s; budget for that and cap it.

```bash
cd $CODE_ROOT
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  timeout 30m nice -n 10 PYTHONPATH=.:src $PYTHON -m pytest \
  experiments/synthetic/nfpa_confirmatory/tests src/tidmad/tests -q
```

**Gate:** all pass, including `test_runner_gate_matches_the_audited_monte_carlo_statistic`. If that specific test fails, stop — the E1 fix did not take.

---

## 2. R1 — GPU throughput (this decides everything downstream)

```bash
cd $CODE_ROOT
GPU_INDEX=0 PYTHON=$PYTHON ./scripts/train_tidmad_local.sh check
GPU_INDEX=0 PYTHON=$PYTHON TIDMAD_TIMEOUT=30m ./scripts/train_tidmad_local.sh bench
```

`bench` runs 200 steps and prints a budget table via `scripts/budget_from_rate.py`.

**Gate:** `check` prints `TIDMAD_GPU_OK`, and `bench` reports a steps/s rate.

**Then decide the budget from the measured rate, not from hope:**

- frozen 150,000 steps fits in ≲ 5 h per arm → keep `num_steps` unchanged, set `TIDMAD_TIMEOUT` above it, **no re-freeze needed**;
- it does not fit → either run resumable segments with `TIDMAD_RESUME=auto`, or reduce `num_steps` and **re-freeze**, because `num_steps` is part of the predeclared configuration.

Record the rate and the arithmetic. Both arms must use the **same** `num_steps`.

**If `check` still fails:** report the exact error. Do not lower `MIN_CAPABILITY` further and do not disable the preflight. If CUDA cannot initialise with any available interpreter, that is a hardware/driver finding, not a config problem.

---

## 3. R2 — Route A probe: is the baseline comparison recoverable?

```bash
cd $CODE_ROOT
$PYTHON scripts/probe_upstream_baselines.py \
  --upstream $TIDMAD_UPSTREAM \
  --data-dir $TIDMAD_DATA_ROOT \
  --out $RPS_RESULTS/RouteA_baseline_probe.json
```

Exit 0 = recoverable, exit 2 = not. **Both outcomes are results; archive the JSON either way.**

- **Exit 0** — regenerate the recoverable baselines' denoised outputs with the release's **own** `inference.py`, then score them with `--in-band-only` (§5). This restores a like-for-like comparison on the 16 in-band files.
- **Exit 2** — the comparison is not recoverable from this checkout. Take Route C: the published numbers become regime context, not a comparison. **Do not train the baselines yourself** — that is a separate project.

**Do not modify anything under `$TIDMAD_UPSTREAM`.** The probe only reads.

---

## 4. R3 — E1 and E2 (NFPA), now unblocked

```bash
cd $CODE_ROOT
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout 1h nice -n 10 taskset -c 0-3 $PYTHON -u \
  -m experiments.synthetic.nfpa_confirmatory.run_blocks --smoke
```

**Gate:** the covariance report now passes and the smoke run reaches method fits and writes rows. If it still stops at a covariance gate, stop and report the exact statistic and value.

Then, one at a time:

```bash
timeout 4h nice -n 10 taskset -c 0-3 $PYTHON -u -m experiments.synthetic.nfpa_confirmatory.run_blocks --block A
timeout 4h nice -n 10 taskset -c 0-3 $PYTHON -u -m experiments.synthetic.nfpa_confirmatory.run_blocks --block B
```

CPU-bound; no GPU, no `screen` needed. Copy tidy outputs to `$RPS_RESULTS/E1_nfpa_blockA/` and `$RPS_RESULTS/E2_nfpa_blockB/`.

**Acceptance gates** are unchanged (`EXPERIMENT_PLAN` §4): all seeds complete with no post-result config change; `C > k_c` with both planted projectors non-trivial; fixed-architecture reversal intervals exclude zero in the predicted directions; Block B agrees to tolerance.

Each archived row now carries **both** `covariance_mc_error_whitened` (gated) and `covariance_mc_error_raw` (diagnostic).

---

## 5. R4 — E3/E4 rung-4 arms

**Only after §2 produced a rate and a justified `TIDMAD_TIMEOUT`.**

```bash
cd $CODE_ROOT
GPU_INDEX=0 PYTHON=$PYTHON TIDMAD_TIMEOUT=<from §2> \
  screen -S rps_t_mse -L -Logfile screenlog_t_mse.txt \
  ./scripts/train_tidmad_local.sh mse
```

Then `chi2`. **Use a different `GPU_INDEX` and you may run both concurrently** — four GPUs are available and the arms are independent runs. Do not attempt data parallelism across GPUs: the model is 569k parameters and it would complicate the batching the contrast depends on.

Interrupted by the cap? `TIDMAD_RESUME=auto` continues from `latest.pt`.

**Load-bearing gate:** the two `run_config.json` files must differ **only** in `train.loss`. Confirm the assertion's output appears in both logs.

Then inference and in-band scoring:

```bash
export PYTHONPATH=src
for ARM in t_mse t_chi2; do
  $PYTHON -m tidmad.infer --config src/tidmad/configs/${ARM#t_}.yaml \
    --checkpoint artifacts/tidmad/$ARM/latest.pt \
    --data-dir $TIDMAD_DATA_ROOT --out $TIDMAD_DENOISED_DIR --model-tag $ARM
  $PYTHON -m tidmad.score --upstream $TIDMAD_UPSTREAM --data-dir $TIDMAD_DENOISED_DIR \
    --model-tag $ARM --in-band-only $RPS_RESULTS/P1_injection_survey.json \
    --out $RPS_RESULTS/E5_${ARM}_inband.json | tee $RPS_RESULTS/E5_${ARM}_stdout.txt
done
```

Score **both** in band and, for completeness, over all 20 files (omit `--in-band-only`). The in-band score is the one to report; the 20-file score documents what the four out-of-band files cost.

---

## 6. Order and stop conditions

| # | Item | Blocks | Stop if |
|---|---|---|---|
| 1 | Test suites (§1) | everything | the E1 regression test fails |
| 2 | GPU throughput (§2) | R4 | CUDA cannot initialise with any interpreter |
| 3 | Route A probe (§3) | reporting | never — archive either verdict |
| 4 | E1/E2 (§4) | — | a covariance gate still fires |
| 5 | E3/E4 (§5) | — | configs differ beyond `train.loss` |

§4 and §5 are independent and may run concurrently (CPU vs GPU).

**A null rung-4 contrast is a publishable finding.** Report it. Do not tune to produce an effect. The only outcome that means "bug" is metric reversal failing at the *linear* rungs.

---

## 7. Deliverable

Append a **Round 2** section to `EXPERIMENT_RUN_LOG.md`:

1. Test-suite results, including the E1 regression test by name.
2. Measured steps/s, the budget table, and the `TIDMAD_TIMEOUT` / `num_steps` you chose — plus whether a re-freeze was needed and its commit.
3. Route A verdict and what it implies for the paper's Table 1.
4. E1/E2: gate results, seed count, paired intervals.
5. E3/E4: GPU index, screen session, wall time, resumes, and the config-equality assertion output for both arms.
6. In-band and 20-file scores for both arms, with the file set recorded.
7. Final machine state: GPUs clear, no screen sessions, no stale PID files.
8. A plain statement of what the round does and does not support.
