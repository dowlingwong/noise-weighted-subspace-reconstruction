# Round 3 checklist — R/PS 2026

**Date:** 2026-08-12
**Delta on:** `ROUND2_CHECKLIST.md` and `EXPERIMENT_CHECKLIST_2026-08-08.md`, both still authoritative for safety rules, protocol definitions and the archival contract.
**Round 2 outcome:** `EXPERIMENT_RUN_LOG.md`. Prerequisites and GPU preflight passed; E1 smoke completed but its numbers were invalid.

---

## 0. Why there is a round 3

The round-2 E1 smoke ran to completion and wrote ten rows. **Those rows are wrong.**

`unwhiten_separable` was not the inverse of `whiten_separable`: whitening used symmetric inverse square roots, unwhitening used Cholesky factors. Both are square roots of `Sigma`, but they differ by an orthogonal factor, so the round trip returned a rotated copy of its input — relative error **5.04**, not ~0.

Only methods with `coord="whitened"` reconstruct through it: **CW-PCA, NFPA, OF**. The isotropic methods do not. So the covariance-aware arm of every comparison was scored on rotated reconstructions.

The smoke made this visible: CW-PCA's **training** normalized chi-square measured 1.0411 against a rank-6 whitened SVD optimum of 0.9959, and above IsoPCA's 1.0202. That is impossible — CW-PCA minimises exactly that objective over a feasible set that contains IsoPCA's subspace. The apparent result, that covariance-aware methods were worse, was the bug.

Fixed in `442ce55`. After the fix, seed 0 colored:

| | train χ² | test χ² | test raw MSE |
|---|---|---|---|
| CW-PCA | **0.995933** (= the optimum, exactly) | **1.017219** | 1.05e-02 |
| IsoPCA | 1.020204 | 1.040418 | **6.80e-03** |

Opposite orderings — metric reversal, as predicted.

**What must be discarded:** the round-2 E1 smoke rows only. No Block A or Block B result exists. Nothing else is affected — `tidmad/` has its own loss path and never used this function.

**What is unaffected:** the archived `C = k_c = 2` reproduction runs `src/NFPA/nfpa_demo.py`, a separate implementation, and still reproduces 0.9916 / 0.9929 and the six principal angles. **The manuscript's appendix values stand; nothing needs re-exporting.**

Two new gates were added and would each have caught this before a seed ran:

- `test_whiten_unwhiten_round_trip_is_identity`
- `test_cw_pca_attains_the_rank_k_whitened_optimum`

---

## 1. Carried forward from round 2 — do not redo

| Item | Status | Note |
|---|---|---|
| GPU preflight | **PASS** | `TIDMAD_GPU_OK`, TITAN X capability 5.2, FP32 |
| Throughput | **1.58 steps/s** | measured on GPU 0 |
| Budget re-freeze | **25,000 steps** | commit `001f87c`; 4.4 h/arm, 6 h cap |
| Route A probe | **exit 2** | no recoverable baselines → Route C |
| Block-H alignment | **fixed** | commit `dc7e1b4` |
| Covariance MC gate | **fixed** | whitened statistic, commit `49096ac` |

Do not re-measure throughput and do not re-freeze the step budget. The frozen configuration is unchanged by this round: `442ce55` touches only the whitening inverse and tests, and the NFPA config hash must remain `98001a62855cb9a5`.

---

## 2. Preconditions

```bash
export CODE_ROOT=~/noise-weighted-subspace-reconstruction
export TIDMAD_DATA_ROOT=/ceph/dwong/paper1_dataset/tidmad/raw
export TIDMAD_UPSTREAM=~/src/TIDMAD
export RPS_RESULTS=$CODE_ROOT/results/rps2026
export PYTHON=/home/dwong/anaconda3/bin/python      # cu121 torch
```

The cu121 interpreter still needs the `jaxtyping` overlay round 2 built. Reuse the same approach; do not add the locked `.venv` site-packages to the path (its torch is cu130 and must not reach GPU work).

### G1 — Full suites

```bash
cd $CODE_ROOT
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  timeout 40m nice -n 10 taskset -c 0-3 PYTHONPATH=.:src $PYTHON -m pytest \
  experiments/synthetic/nfpa_confirmatory/tests src/tidmad/tests -q
```

**Gate:** all pass, and these three by name:

- `test_whiten_unwhiten_round_trip_is_identity`
- `test_cw_pca_attains_the_rank_k_whitened_optimum`
- `test_runner_gate_matches_the_audited_monte_carlo_statistic`

If either of the first two fails, **stop** — the fix did not take and no NFPA result will be valid.

The NFPA suite took 8m47s in round 2 and has grown by two tests; 40 minutes is the cap.

---

## 3. R1 — E1 smoke, with an expected answer

```bash
cd $CODE_ROOT
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout 1h nice -n 10 taskset -c 0-3 $PYTHON -u \
  -m experiments.synthetic.nfpa_confirmatory.run_blocks --smoke
```

Round 2's smoke took 14m42s for 2 seeds × 5 methods.

**This time the smoke has a predicted outcome. Check it before launching Block A:**

| Check | Expected |
|---|---|
| CW-PCA train χ² | ≈ **0.9959**, and **below** IsoPCA's ≈ 1.0202 |
| CW-PCA test χ² | **below** IsoPCA's |
| CW-PCA test raw MSE | **above** IsoPCA's |
| NFPA vs Iso-MPCA (χ²) | NFPA **lower** — the fixed-architecture contrast, H1 |
| NFPA vs Iso-MPCA (MSE) | NFPA **higher** |
| Angle to CW-PCA | NFPA ≈ 63° ≪ Iso-MPCA ≈ 87° (unchanged by the fix) |
| Config hash | `98001a62855cb9a5` |

**Gate:** CW-PCA's train χ² must be ≤ IsoPCA's. If it is not, stop — the invariant test passed but the harness disagrees, and that is a new bug.

If the NFPA-vs-Iso-MPCA direction is *not* as predicted while CW-PCA vs IsoPCA *is*, that is a genuine scientific result about the Kronecker restriction, not a bug. **Report it; do not tune.**

Archive as `$RPS_RESULTS/E1_nfpa_blockA_smoke/` with a round-3 suffix, preserving round 2's files for traceability.

---

## 4. R2 — Block A, then Block B

Only after §3's gate passes.

```bash
timeout 4h nice -n 10 taskset -c 0-3 $PYTHON -u \
  -m experiments.synthetic.nfpa_confirmatory.run_blocks --block A
timeout 4h nice -n 10 taskset -c 0-3 $PYTHON -u \
  -m experiments.synthetic.nfpa_confirmatory.run_blocks --block B
```

Round 2 projected ~74 min for 10 seeds from the smoke rate. CPU-bound; no GPU, no `screen`.

Outputs to `$RPS_RESULTS/E1_nfpa_blockA/` and `$RPS_RESULTS/E2_nfpa_blockB/`.

**Acceptance gates** (`EXPERIMENT_PLAN` §4, unchanged): all seeds complete with no post-result configuration change; `C > k_c` with both planted projectors non-trivial; the fixed-architecture reversal intervals exclude zero in the predicted directions; Block B's projectors agree to tolerance.

Every row now carries `covariance_mc_error_whitened` (gated) and `covariance_mc_error_raw` (diagnostic).

---

## 5. R3 — Rung-4 arms

**Independent of everything above** — `tidmad/` never used the broken function — so these may run concurrently with §3–§4 (GPU vs CPU).

```bash
cd $CODE_ROOT
GPU_INDEX=0 PYTHON=$PYTHON TIDMAD_TIMEOUT=6h \
  screen -S rps_t_mse -L -Logfile screenlog_t_mse.txt ./scripts/train_tidmad_local.sh mse
GPU_INDEX=1 PYTHON=$PYTHON TIDMAD_TIMEOUT=6h \
  screen -S rps_t_chi2 -L -Logfile screenlog_t_chi2.txt ./scripts/train_tidmad_local.sh chi2
```

Two arms on two GPUs concurrently is fine — they are independent runs at 25,000 steps, ~4.4 h each. Do not attempt data parallelism.

Resume after a cap with `TIDMAD_RESUME=auto`.

**Load-bearing gate:** the two `run_config.json` files must differ **only** in `train.loss`. Confirm the assertion fired in both logs.

Then inference and scoring, both in band and over all 20:

```bash
export PYTHONPATH=src
for ARM in t_mse t_chi2; do
  $PYTHON -m tidmad.infer --config src/tidmad/configs/${ARM#t_}.yaml \
    --checkpoint artifacts/tidmad/$ARM/latest.pt \
    --data-dir $TIDMAD_DATA_ROOT --out $TIDMAD_DENOISED_DIR --model-tag $ARM
  $PYTHON -m tidmad.score --upstream $TIDMAD_UPSTREAM --data-dir $TIDMAD_DENOISED_DIR \
    --model-tag $ARM --in-band-only $RPS_RESULTS/P1_injection_survey.json \
    --out $RPS_RESULTS/E5_${ARM}_inband.json | tee $RPS_RESULTS/E5_${ARM}_inband_stdout.txt
  $PYTHON -m tidmad.score --upstream $TIDMAD_UPSTREAM --data-dir $TIDMAD_DENOISED_DIR \
    --model-tag $ARM --out $RPS_RESULTS/E5_${ARM}_all20.json \
    | tee $RPS_RESULTS/E5_${ARM}_all20_stdout.txt
done
```

The **in-band** score is the one to report; the 20-file score documents what the four out-of-band files cost.

---

## 6. Order and stop conditions

| # | Item | Blocks | Stop if |
|---|---|---|---|
| 1 | Full suites (§2) | everything | either new invariant test fails |
| 2 | E1 smoke (§3) | Block A/B | CW-PCA train χ² > IsoPCA's |
| 3 | Block A, then B (§4) | — | a seed fails or a gate fires |
| 4 | Rung-4 arms (§5) | scoring | configs differ beyond `train.loss` |
| 5 | Inference + scoring (§5) | — | benchmark returncode ≠ 0 |

§4 (CPU) and §5 (GPU) may overlap. **A null rung-4 contrast remains a publishable finding** — report it rather than tuning. The only outcome meaning "bug" is metric reversal failing at the linear rungs *after* §2 and §3 have both passed.

---

## 7. Deliverable

Append a **Round 3** section to `EXPERIMENT_RUN_LOG.md`:

1. Suite results, naming the two new invariant tests.
2. E1 smoke: the predicted-outcome table from §3 filled in with measured values, and whether each matched.
3. Block A/B: seed counts, paired intervals, acceptance-gate results.
4. Rung-4: GPU indices, screen sessions, wall time, resumes, config-equality assertion output for both arms.
5. In-band **and** 20-file scores for both arms, with file sets recorded.
6. Final machine state: GPUs clear, no screen sessions, no stale PID files.
7. A plain statement of what round 3 does and does not support — and explicitly, that the round-2 E1 smoke numbers are void.
