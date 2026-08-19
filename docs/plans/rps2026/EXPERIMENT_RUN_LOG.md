# R/PS 2026 experiment run log

## Phase 0 — reconnaissance

Snapshot taken 2026-08-11 16:43–16:46 CEST before environment setup or any
experiment launch.

- Host: `deepthought`; Red Hat Enterprise Linux 8.10; kernel
  `4.18.0-553.120.1.el8_10.x86_64`.
- NVIDIA driver 525.147.05; reported CUDA 12.0. Four NVIDIA GeForce GTX TITAN X
  GPUs (12,288 MiB each). `nvidia-smi` reported no compute processes and 0 MiB
  in use on all four GPUs. GPU UUIDs were recorded by the reconnaissance
  command; GPU 0 is `GPU-c89fb4b0-1365-2f30-ba0b-4008046662ca`.
- Other users: `tmohr` was logged in on `pts/9` from `193.190.84.5` since 11:34.
  No other user had a GPU process.
- Uptime 103 days 4:22; load averages 0.63, 0.31, 0.25.
- RAM: 503 GiB total, 34 GiB used, 432 GiB free, 462 GiB available. Swap was
  fully used (3/3 GiB), but RAM headroom was ample.
- Disk: `/ceph` and the repository filesystem had 326 TiB available of 1.3 PiB
  (75% used); `/home` had 100 GiB available of 100 GiB. `quota -s` returned no
  quota record.
- Python 3.12.2 at `/home/dwong/anaconda3/bin/python`; `uv` 0.11.29 at
  `/home/dwong/.local/bin/uv`; `.venv` resolves to the repository-local
  `/ceph/dwong/noise-weighted-subspace-reconstruction/.venv`.
- No pre-existing screen sessions. Existing processes owned by `dwong` were the
  VS Code remote/Codex session and an interactive-shell `sleep 180`; none was an
  experiment or GPU process, so no pre-existing process was stopped.
- Starting git state: branch `main`, clean worktree, commit
  `3461a4300ed1b54253d2fba781a675937b9fdc26`.

Gate: **PASS**. The untouched starting state is recorded above.

## Phase 1 — environment, data, and provenance

The checklist's prescribed root `/ceph/dwong/datasets/tidmad/raw` does not
exist on this host. The release data remain at the archived Paper-1 path
`/ceph/dwong/paper1_dataset/tidmad/raw` (125 GiB TIDMAD tree). No path under
`/ceph/dwong/datasets` was created or modified. Runs on this host therefore use
the resolved legacy root and every new run record must carry:

```text
path_migration: planned root /ceph/dwong/datasets/tidmad/raw is absent on
deepthought as of 2026-08-11; actual read-only source remains
/ceph/dwong/paper1_dataset/tidmad/raw
```

The legacy root contains one training file, two science files, 20 raw
validation files, and 20 pre-existing `abra_validation_denoised_savgol_*`
files. Consequently, P1 uses the exact raw-file glob
`abra_validation_00??.h5`, not the checklist's broad
`abra_validation_*.h5`, to prevent derived files from entering the survey.

Integrity probe for `abra_validation_0000.h5`:

- size 1,960,812,142 bytes; modification time 2026-07-27 03:35:20 CEST;
- MD5 `eb56fd0014c4088159d6024516a05924` (no release manifest was found in the
  upstream checkout, so this is recorded for traceability rather than claimed
  as manifest-verified);
- groups are exactly `channel0001` and `channel0002`; both datasets have shape
  `(2010000000,)` and dtype `int8`.

Environment setup used the locked optional extras required by the actual entry
points:

```text
uv sync --frozen --extra dev --extra paper2 --extra tidmad
```

A plain `uv sync` omits optional PyTorch/W&B/test dependencies. The `tidmad`
extra also omitted the backbone's imported `jaxtyping` package; that dependency
was added explicitly and the lock updated. The resulting locked stack contains
PyTorch 2.12.1+cu130 and W&B 0.27.2.

## P1 — injection-frequency survey

Command (return code 2, the documented out-of-band scientific outcome):

```text
PYTHONPATH=src timeout 30m nice -n 10 .venv/bin/python -u \
  -m tidmad.characterise_injections \
  --data-dir /ceph/dwong/paper1_dataset/tidmad/raw \
  --glob 'abra_validation_00??.h5' \
  --out results/rps2026/P1_injection_survey.json
```

Gate: **OUT OF BAND**. The deterministic survey record is
`results/rps2026/P1_injection_survey.json`. Files 0016, 0017, 0018, and 0019
cross the 800.8 kHz modeled-band edge; the last reaches approximately 4.8 MHz.
Decision: retain the identical 0–800.8 kHz representation for the paired
T-mse/T-chi2 contrast and drop the published full-band baseline comparison.
Widening after this gate would materially redesign the representation and its
frequency resolution. The paired contrast survives because both arms retain
the same band limit.

## P2 — `none` anchor

Command:

```text
cd /home/dwong/src/TIDMAD && PYTHONHASHSEED=0 timeout 30m nice -n 10 \
  /ceph/dwong/noise-weighted-subspace-reconstruction/.venv/bin/python \
  benchmark.py -d /ceph/dwong/paper1_dataset/tidmad/raw -m none -c
```

Return code 0. The unmodified upstream script reported
`Coarse Denoising Score for Model none: 1.1026161164402752`, matching the
1.1026 gate. Raw output is archived at
`results/rps2026/P2_none_anchor/benchmark_stdout.txt`.

Gate: **PASS**. Because P1 forced removal of the external full-band comparison,
this anchor validates the channel semantics and unmodified future scoring path;
it is not used to claim leaderboard comparability for the band-limited arms.

## P3 — architecture decisions

- D1 adopted: the external measurement/loss representation remains 656 real
  rows, but the model pairs real and imaginary components as two features of
  each of 328 physical-frequency tokens. Cross-band attention therefore sees
  328 bands, not 656 unrelated component tokens.
- D2 adopted: `patch_len` is reduced from 8 to 4. With 96 frames, temporal
  attention now has 24 positions instead of 12.
- The config-equality assertion passes with the only arm difference at
  `train.loss`. The frozen candidate has 569,352 trainable parameters.

CPU validation after the changes: 24/24 TIDMAD tests pass, including the
non-proportional-loss gate, complex-feature shape, 24-position temporal axis,
config equality, two-step optimiser smoke, checkpoint round-trip, and split
guard. Syntax compilation and `git diff --check` also pass.

## NFPA prerequisite audit

All 14 correctness/leakage tests pass under a 30-minute cap with BLAS/OMP
limited to four threads (11 min 51 s). This includes all seven mandatory
correctness checks, archived degenerate-case reproduction, train-only centering,
oracle covariance isolation, and the post-fit test-access guard.

An initial combined diagnostic test invocation allowed BLAS to use all 40 CPU
cores and briefly drove load average to 41.5. It was immediately constrained to
four cores and nice level 10, then stopped because it lacked a hard timeout.
All subsequent NFPA commands set `OPENBLAS_NUM_THREADS=4`, `OMP_NUM_THREADS=4`,
and `MKL_NUM_THREADS=4` from process start and use `nice -n 10` plus `timeout`.

## P4 — resumability and throughput gate

**GPU portion blocked before training launch by the GPU preflight.** The repository-locked
PyTorch 2.12.1+cu130 cannot initialize CUDA with driver 525.147.05 (driver
reports CUDA 12.0). The host's preinstalled PyTorch 2.5.1+cu121 can initialize
CUDA, but independently reports compute capability 5.2 for every GTX TITAN X;
the prescribed launcher rejects capability below 7.0. Exact second preflight:

```text
TIDMAD_DATA_ROOT=/ceph/dwong/paper1_dataset/tidmad/raw GPU_INDEX=0 \
PYTHON=/home/dwong/anaconda3/bin/python TIDMAD_TIMEOUT=5m timeout 6m \
./scripts/train_tidmad_local.sh check
```

Result: `RuntimeError: Compute capability (5, 2) < 7.0; BF16/FP16 path
unsupported.` No GPU training process, screen session, GPU throughput
measurement, or W&B run was started. Consequently no full-arm timeout can yet
be justified and E3/E4 must not be launched on this host.

The mechanical real-data path was additionally exercised on CPU under a
10-minute cap, four-thread limit, and nice level 10. Three failures were found
and fixed before the successful attempt: the migrated contract path pointed to
the repository root rather than `docs/`; the HDF5 reader prepended a group to
already-full contract paths; and the NumPy `float64` STFT was not cast to the
model's `float32` dtype. A fourth host-toolchain issue was resolved by selecting
Muon's documented eager Newton--Schulz path; the RHEL 8 compiler does not accept
the generated kernel's `-std=c++20` flag. The launcher now sets that identical
path for both arms.

Successful CPU smoke output:

- initial run: one step, 0.45 steps/s, wrote
  `artifacts/tidmad/smoke_cpu_p4/checkpoint_00000001.pt` and `latest.pt`;
- resume: restored `latest.pt` at step 1, advanced to step 2 at 0.44 steps/s,
  and wrote `checkpoint_00000002.pt` plus a refreshed `latest.pt`;
- logs: `artifacts/tidmad/smoke_cpu_p4/smoke_initial.log` and
  `artifacts/tidmad/smoke_cpu_p4/smoke_resume.log` (large artifacts remain
  outside the tracked results tree).

Mechanical checkpoint/resume gate: **PASS on CPU**. GPU throughput and the
full-run wall-clock arithmetic remain **BLOCKED**, so the CPU rate is not used
to choose an E3/E4 timeout.

## Freeze record

The first freeze candidate was committed as
`023e18031e0aca32e80c3050c3315a60a0b1e58c` at 2026-08-11 17:53:03 CEST.
During the mandatory pre-smoke command review, before any NFPA aggregate was
inspected, two stale pre-amendment settings were found: Block A still encoded
30 seeds rather than 10, and the harness defaulted to the gitignored
`results/synthetic/nfpa_confirmatory` tree. Both are corrected in the follow-up
freeze commit: Block A reads `PRIMARY.n_seeds = 10`, and smoke/E1/E2 write only
to their experiment IDs below `results/rps2026/`. No confirmatory or smoke
result existed when this correction was made.

Corrected frozen commit:
`60f110ea928c42f5a4e69c286efba083b6bb11b4`, timestamp
2026-08-11 17:54:56 CEST. This is the frozen basis for subsequent runs.

## E1 — NFPA Block A

Two-seed smoke command:

```text
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 1h \
  nice -n 10 taskset -c 0-3 /usr/bin/time -v \
  .venv/bin/python -u \
  -m experiments.synthetic.nfpa_confirmatory.run_blocks --smoke
```

GPU: none. Screen: none (CPU-bound harness, per checklist). Wall time: 17.89 s.
Maximum RSS: 492,240 KiB. Config hash: `98001a62855cb9a5`. Return code 1.

Gate: **FAIL before any method fit or result row**. The runner's deterministic
covariance report returned raw-Frobenius Monte Carlo relative error 0.0820,
above the frozen 0.05 tolerance. No raw CSV was written. The exact command and
stderr are archived in `results/rps2026/E1_nfpa_blockA_smoke/run.log` and the
metadata in the adjacent `run_config.json`.

The failure also exposes an implementation mismatch: the audited correctness
test deliberately checks Monte Carlo covariance agreement in whitened space,
with a source comment that raw Frobenius error is dominated by the Brownian
covariance's large low-frequency eigenvalues. The runner instead applies the
0.05 threshold to raw covariance. This was not changed after observing the
failure. The 10-seed Block A run was **not launched**.

## E2 — NFPA Block B

**Not launched.** It uses the same runner-level raw Monte Carlo covariance
gate that stopped E1 before fitting. Changing the statistic or tolerance after
the failed smoke would be a post-gate configuration change, which the checklist
forbids. E2 remains pending a new author-approved predeclaration/freeze.

## E3/E4 — rung-4 GPU arms

**Not launched.** P4's GPU gate is blocked by the CUDA 13/driver mismatch and,
independently, by the GTX TITAN X compute capability 5.2 being below the
launcher's required 7.0. With no measured GPU steps/s, no defensible full-arm
timeout exists. The supplied W&B credential was therefore not used and no W&B
run was created.

## Final machine and archival state

Checked 2026-08-11 17:57 CEST:

- `nvidia-smi`: all four GPUs at 0 MiB with no running compute processes;
- `screen -ls`: no sessions;
- no `run.pid` below `artifacts/` and no experiment/test/training process owned
  by `dwong`; only the pre-existing VS Code/Codex session and interactive-shell
  sleep remained;
- load averages 0.27, 0.95, 5.22; 461 GiB RAM available;
- `results/rps2026` is 49 KiB, containing only small tracked JSON, raw stdout,
  and logs;
- CPU smoke checkpoints/logs remain under
  `artifacts/tidmad/smoke_cpu_p4/` and are intentionally outside the tracked
  results tree;
- nothing under `/ceph/dwong/paper1_dataset/tidmad/raw` or the absent
  `/ceph/dwong/datasets` path was modified.

## What these runs do and do not support

The prerequisites support the TIDMAD channel contract on this server, reproduce
the unmodified `none` anchor, show that four validation files lie above the
frozen 800.8 kHz band, and mechanically validate the revised D1/D2 architecture
and exact checkpoint resume on real data using CPU. They therefore support only
the decision to retain a band-limited paired metric contrast and drop the
published full-band baseline comparison. They do **not** provide an NFPA Block
A/B result or a rung-4 T-mse/T-chi2 result: NFPA stopped at a pre-fit covariance
gate, and the GPU arms stopped at incompatible hardware/runtime preflight. No
claim about metric reversal, nonlinear performance, leaderboard position, or
training-seed variability can be made from this execution.

# Round 3 remote execution (2026-08-14 to 2026-08-15)

## Scope and revisions

Execution started from `38b64749dacf69988dd83c84461e220da81b5c1b`, a
descendant of the prescribed `89a4a06` and whitening fix `442ce55`. The accepted
rung-4 training runs themselves record Git commit `89a4a06`. Inference exposed
four scaffold defects: the loader did not unwrap resumable checkpoints,
normalization statistics stayed on CPU, inference transformed an unused target
channel one window at a time, and non-divisible file tails were left
uninitialized. These were fixed with regression coverage in
`c8617997ee72b86a1fe192fe7db42eac25c64012`; the final inference outputs were
generated only after all four fixes. The NFPA frozen config was not edited and
retained hash `98001a62855cb9a5`.

**The round-2 E1 smoke numbers are void.** Only those smoke rows are discarded.
The archived `C = k_c = 2` appendix reproduction uses the separate
`src/NFPA/nfpa_demo.py` implementation and remains unaffected.

## Test gates

The prescribed initial combined suites passed 42/42 under the cu121 interpreter.
The three explicitly named gates then passed 3/3:

- `test_whiten_unwhiten_round_trip_is_identity`;
- `test_cw_pca_attains_the_rank_k_whitened_optimum`;
- `test_runner_gate_matches_the_audited_monte_carlo_statistic`.

After the inference fixes, the complete combined suites passed 51/51 (18 NFPA,
33 TIDMAD) in 583.19 s with two known warnings. Logs are in
`results/rps2026/Round3_tests/`, including
`full_suites_remote_20260814.log`, `named_gates_remote_20260814.log`, and
`full_suites_post_inference_20260815.log`.

## E1 smoke and Blocks A/B

The corrected two-seed smoke completed 10 rows. Values below are method means
generated from `results/rps2026/E1_nfpa_blockA_smoke/block_A_colored_raw.csv`:

| Predicted comparison | Measured values | Result |
|---|---:|---|
| CW-PCA train chi2 <= IsoPCA | 0.998069 <= 1.022363 | matched, gate pass |
| CW-PCA test chi2 < IsoPCA | 1.016815 < 1.040109 | matched |
| CW-PCA test MSE > IsoPCA | 0.0104335 > 0.00678418 | matched |
| NFPA test chi2 < Iso-MPCA | 1.014236 < 1.033337 | matched |
| NFPA test MSE > Iso-MPCA | 0.0103761 > 0.00695559 | matched |
| NFPA angle < Iso-MPCA angle | 63.5741 deg < 87.5661 deg | matched |

Block A completed seeds 0-9, 50/50 rows, with all fits converged, `C = 8`,
`k_c = 2`, `C > k_c`, `k_t = 3`, and both planted projectors non-trivial. Seed-paired
percentile 95% intervals used 10,000 resamples and seed 20260808:

| Block A contrast | Mean difference | Median difference | 95% CI | Gate |
|---|---:|---:|---:|---|
| NFPA - Iso-MPCA, test chi2 | -0.0192011 | -0.0192433 | [-0.0194141, -0.0190001] | pass, below zero |
| NFPA - Iso-MPCA, test raw MSE | +0.00341426 | +0.00340028 | [0.00339459, 0.00343451] | pass, above zero |
| NFPA - CW-PCA, test chi2 | -0.00255759 | -0.00254873 | [-0.00259164, -0.00251996] | held-out finite-sample result |

Block B completed seeds 0-19, 100/100 rows, with all fits converged. NFPA minus
Iso-MPCA was exactly 0.0 for both test chi2 and raw MSE, with 95% intervals
`[0.0, 0.0]`, inside the predeclared chi2 margin 0.005. The maximum white-noise
CW-PCA/IsoPCA projector angle was `2.41484e-6` degrees, inside the predeclared
0.01-degree tolerance. Block B gate: pass.

Raw evidence is in `results/rps2026/E1_nfpa_blockA/` and
`results/rps2026/E2_nfpa_blockB/`. The deterministic aggregate is
`results/rps2026/Round3_summary.json`.

## Rung-4 training

Both accepted runs used 25,000 steps, seed 20260808, 569,352 parameters, and no
resume. Earlier unbounded-thread diagnostic attempts were stopped and excluded.

| Arm | GPU | Screen | Wall time | Throughput | Final held-out MSE | Final held-out chi2 |
|---|---:|---|---:|---:|---:|---:|
| T-mse | 0 | `rps_t_mse` | 4.23 h | 1.64 steps/s | 11519.3 | 0.0135078 |
| T-chi2 | 1 | `rps_t_chi2` | 4.25 h | 1.63 steps/s | 11484.4 | 0.0134244 |

`CONFIG_EQUALITY_ASSERTION=PASS` for both arms. Structural comparison of
`artifacts/tidmad/t_mse/run_config.json` and
`artifacts/tidmad/t_chi2/run_config.json` found exactly one differing path:
`/run_config/train/loss`, with values `mse` and `chi2`; both record 25,000
steps. Training logs and checkpoints remain under `artifacts/tidmad/t_mse/`
and `artifacts/tidmad/t_chi2/`; screen logs are `screenlog_t_mse.txt` and
`screenlog_t_chi2.txt`.

## Inference and upstream scores

Each final inference produced and metadata-validated 20/20 release-format HDF5
files under `/ceph/dwong/paper1_dataset/tidmad/denoised/`. All four scoring passes
used the unmodified `/home/dwong/src/TIDMAD/benchmark.py -c` and returned 0.

| Arm | In-band score, files 0000-0015 | All-20 score, files 0000-0019 |
|---|---:|---:|
| T-mse | -2.353761452599672 | -2.7708098959837675 |
| T-chi2 | -2.353761452599672 | -2.7708098959837675 |
| T-chi2 minus T-mse | 0.0 | 0.0 |

The in-band score is the reported result. Files 0016-0019 are the registered
out-of-band set excluded from that comparison; the all-20 score documents their
cost. Benchmark JSON and raw stdout are `results/rps2026/E5_t_mse_{inband,all20}.*`
and `results/rps2026/E5_t_chi2_{inband,all20}.*`.

> **WITHDRAWN 2026-08-16.** The paragraph below reported the rung-4 contrast as
> a publishable null. **It is not a null result; it is a display-resolution
> artifact, and no claim may be based on it.**
>
> `benchmark.py:139` aggregates as
> `np.round(sum(snr_sg * snr_squid)/n, decimals=2) + 1e-10` before taking
> `log(score, 5.27)`. The raw score is **rounded to two decimal places**, so any
> two methods differing by less than 0.005 return bit-identical values.
> Inverting the logarithms: both arms sit at a rounded raw score of **0.02**
> in band and **0.01** over all twenty files. Savitzky-Golay rounds to the same
> 0.01, which is why round 1's reproduction matched to all 17 figures.
>
> Two consequences. First, the benchmark cannot resolve this contrast at this
> operating point, so the correct statement is *"the upstream coarse score
> cannot resolve the comparison"*, not *"changing the metric made no
> difference"*. Second, both band-limited arms sit at raw 0.01-0.02 against
> FCNet's 53,439 -- roughly six orders of magnitude below the published
> baselines, which is expected for a band-limited 569k-parameter model at
> 25,000 steps but should be stated plainly.
>
> Resolution: `tidmad.score_unrounded` recomputes the same quantity from the
> release's own `GetOneSecPSD`/`getSNR`/`process_iteration`, without the
> rounding. Rung-4 evidence should rest on the held-out chi-square and raw-MSE
> contrast from training, with the unrounded benchmark score as support.

~~The rung-4 contrast is exactly null at reported precision. This is the
predeclared publishable finding and was not tuned. It suggests that, in this
single-seed attention experiment, changing only the training metric did not
change the upstream denoising score.~~ Both negative scores also show that neither
band-limited arm is competitive on this benchmark; the 16-file scores are not
comparable to published 20-file baselines. Training-seed variance remains
uncharacterized.

## Final machine state

Checked 2026-08-15 19:42:49 UTC on `deepthought`: all four GPUs reported 0 MiB
used and no compute processes; `screen -ls` reported no sockets; no `run.pid`
remained below `artifacts/tidmad`; and no experiment, benchmark, or pytest
process remained. The host had 465 GiB available RAM, load averages
2.96/3.00/2.06, and `results/rps2026` occupied 439 KiB. Large denoised outputs
remain outside Git as required (20 T-mse files, 24,625,288,802 bytes; 20 T-chi2
files, 24,564,667,015 bytes).

# Round 6 (2026-08-17) — T2a held-out series and T2b benchmark interval

No training, no new experiments; everything read files already on disk.

## T2a — held-out χ² and raw MSE series

Resolved logs (latest `training.log`, top-level):
`artifacts/tidmad/t_mse/training.log` and
`artifacts/tidmad/t_chi2/training.log`. Extracted 50 evaluation lines per arm
(steps 500–25000, period 500) to
`results/rps2026/E5_{t_mse,t_chi2}_eval_series.txt`, converted to
`E5_{t_mse,t_chi2}_eval_series.csv` with columns
`step,train_mse,train_chi2,val_mse,val_chi2,steps_per_s,eta_s`
(the console line does not carry `elapsed_s`, which is wandb-only, so that
column was dropped and `eta_s` converted from the logged hours).

**Step-grid gate passed:** both arms have exactly 50 evaluations at the same
step values (500, 1000, …, 25000); matched budget and seed handling.

Final evaluation (step 25000), lower is better:

| metric | `t_mse` | `t_chi2` | paired `t_chi2 − t_mse` |
|---|---|---|---|
| `val_chi2` | 0.0135078 | 0.0134244 | −8.34e-05 |
| `val_mse` | 11519.3 | 11484.4 | −34.9 |

Mean over the last ten evaluations (steps 20500–25000): `val_chi2` 0.0135121
vs 0.0134277 (paired −8.44e-05); `val_mse` 11525.8 vs 11487.8 (paired −37.99).

Per-evaluation paired difference series `t_chi2 − t_mse` (50 evaluations),
percentile bootstrap 95 %, 100 000 resamples, seed 20260817:

| metric | mean paired diff | 95 % interval | excludes zero |
|---|---|---|---|
| `val_chi2`, all 50 | −1.148e-04 | [−1.269e-04, −1.034e-04] | yes |
| `val_chi2`, last 10 | −8.444e-05 | [−8.603e-05, −8.297e-05] | yes |
| `val_mse`, all 50 | −57.18 | [−71.52, −42.52] | yes |
| `val_mse`, last 10 | −37.99 | [−42.26, −34.93] | yes |

**The predicted signature does not occur.** `t_chi2` is better on *both*
metrics; there is no metric reversal at rung 4. The Σ⁻¹-weighted objective
improves the whitened residual and does not sacrifice raw MSE. This is a real,
pre-anticipated finding about whether the metric question survives into a
nonlinear architecture at this scale, reported exactly as measured — no
re-running, no subset selection.

## T2b — interval for the benchmark contrast

Re-ran the in-band rescore per `ROUND6_T2_RUNBOOK.md` §3 with the same
`~`-resolved denoised directory as round 5
(`/ceph/dwong/paper1_dataset/tidmad/denoised`; the `/ceph/dwong/datasets`
path from round 5's plan does not exist). 16 in-band files, 320 one-second
windows scored, `per_window_products` captured in the refreshed
`results/rps2026/E5_{t_mse,t_chi2}_inband_unrounded.json`.

- **Determinism:** `raw_score_unrounded` reproduces round 5 exactly —
  t_mse 0.024149077851187167, t_chi2 0.023130609616883405 (bit-identical).
- **Reference-channel identity:** `snr_sg_normalised` is identical between
  arms (320/320 entries), confirming the audit prediction that the entire
  contrast is carried by `snr_squid`.
- **Paired difference per window** (mean, `t_chi2 − t_mse`): −0.0010184682.
- **Percentile bootstrap 95 %, 10 000 resamples, seed 20260817:**
  window-level [−0.0033273, +0.0013807]; file-block bootstrap (16 blocks × 20
  windows) [−0.0046086, +0.0023744]. **Both include zero.**
- Result written to `results/rps2026/E5_benchmark_contrast_interval.json`.

**Statement the round supports:** the benchmark does not distinguish the arms
at this operating point; round 5's point estimate (4.4 %) is not resolvable
once the per-window structure is taken into account. This is weaker than the
point estimate suggested, but is not a null about the training metric — the
benchmark is a narrowband SNR-recovery measure neither arm optimised, both
arms sit ~6 orders of magnitude below the published baselines, and the
held-out contrast in T2a does resolve.

## What Round 6 does and does not support

Supports: a resolvable held-out paired contrast (χ²-trained arm better on
both metrics, intervals excluding zero, matched step grids); the benchmark
score's point estimate and the fact that its uncertainty spans zero; the
determinism of the unrounded scorer; the reference-channel identity across
arms. Does not support: metric reversal at rung 4 (predicted, not observed);
any claim that the benchmark distinguishes the arms (interval includes zero);
any claim stronger than a paired difference for the held-out contrast, since
each arm is a single seed.

# Round 8 (2026-08-18) — Task A: whitening dynamic range

Data root auto-resolved by `tidmad.paths` to `/ceph/dwong/paper1_dataset/tidmad/raw`,
the same tree rounds 3–6 used. Record: `results/rps2026/E6_whitening_range.json`.

| statistic | value (including DC) |
|---|---|
| `J` range | 1.27e+05 .. 5.602e+07 (median 7.455e+05) |
| `ratio_max_min` | 441 |
| `ratio_p99_p01` | 124 |
| **`cv_inverse_J`** | **0.6914** |
| `effective_n_bands` | 221.9 of 328 |

**Gate: FAILED, and decisively.** The threshold was `cv_inverse_J` ≲ 0.1. The measured
0.69 is nearly seven times that, and about 3.5× what a mere 2× dynamic range produces
(0.20 on synthetic vectors). The SQUID noise has real structure across 0–800 kHz.

**Consequence: the paper's near-collinearity explanation for the rung-4 non-reversal is
refuted.** `chi2 = mean(r²/J)` and `mean(r²)` are genuinely different objectives over
this band, so the absence of reversal at rung 4 is not metric degeneracy. This was
pre-declared as a possible outcome in `ROUND7_WHITENING_RUNBOOK.md` §"If the gate fails"
and the threshold was not adjusted after seeing the number.

Environment note from the operator: the runbook's
`PYTHON=/home/dwong/anaconda3/bin/python` lacks `jaxtyping`, which the module needs via
the `train.py` import chain. Task A was run with the repository `.venv` interpreter —
the launcher's default and the one rounds 3–6 used. The runbook has been corrected.

## Desk follow-up (same day)

Manuscript revised to match the measurement rather than the prior inference:

- §4 now states the test and its outcome — `J` spans a factor 441 over the modelled
  band, whitening weights have coefficient of variation 0.69, far from proportional —
  and attributes the absence to this instantiation (capacity absorbing the weighting)
  with one seed per arm, rather than to a degenerate metric.
- The abstract's causal clause ("because the two objectives stop separating once
  whitening approaches a constant rescaling") was **false as written** and is replaced
  by a statement that the obvious explanation is ruled out by measurement.
- §5's caveat "the near-collinearity account is inferred from the effect signature, not
  measured" is replaced: the objectives are measurably non-degenerate, and what does
  account for the absence is unknown.

Rebuilt: main content 4 pages, references from p5, 0 overfull boxes, 0 undefined
references or citations.

An additional check on data already on disk, from the round-6 series: over 50
evaluations the χ²-trained arm's advantage on *training* MSE is **not** resolved
(paired −14.3, 95% [−30.3, +2.1], favouring it in 60% of evaluations), while its
advantage on training χ² is (paired −9.94e-05, [−1.14, −0.86]e-04, 98% of
evaluations). So the raw-MSE advantage seen on held-out data is not explained by the
χ² arm simply optimising better; note these intervals are over evaluation points on a
single trajectory, not over seeds, which is exactly what Task B addresses.

---

# Round 9 — overnight remote run (2026-08-18/19, `deepthought`)

Executed from `ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md` on branch
`round9/remote-evidence`, created from the pinned handoff content commit
`ff50d7184a2ab66275f01d076e9750ae8c367107`. No manuscript file was touched, no
scientific threshold was changed, and the archived Block A CSV was neither
overwritten nor appended to.

## Machine at preflight

`deepthought`, Linux 4.18.0, 40 cores, 503 GB RAM (460 GB available), `/ceph`
318 TB free. No interactive users, no screen sessions, load average 4.18. All
four GPUs were occupied by this account's own `llama-server` processes
(GPUs 1–3 at 98% utilisation, ~7.7 GB resident each). No E7 process, PID file
or screen session existed. Interpreter: `.venv/bin/python` → Python 3.12.2,
NumPy 2.4.6, SciPy 1.17.1.

## Gate deviation: no network credentials

`git fetch origin codex/rps-round9-overnight-handoff` failed with
`could not read Username for 'https://github.com'` (exit 128). There is no
credential helper in this non-interactive session, so the runbook's Phase 0
`ROUND9_HANDOFF_FETCH_FAILED` condition is met and is recorded as an
infrastructure gate failure. Identity and ancestry were verified against the
local object store instead: `ff50d71` is present by exact SHA and is an
ancestor of the fetched remote-tracking ref. The same missing credentials mean
`git push -u origin round9/remote-evidence` cannot be run from this machine.

## E6 — whitening range

`results/rps2026/E6_whitening_range.json` was absent from this branch's
ancestry. It was produced by the Round 8 remote run and committed on `8ee9eb1`,
which sits on an unmerged side branch that is *not* an ancestor of the Round 9
handoff commit (they diverge at `27fcfc5`). `scripts/round9_verify_e6.py`
recovered it from the Git object store and verified, in code:

| Check | Value |
|---|---|
| `excluding_dc.cv_inverse_J` | 0.6883439500348468 → rounds to 0.69 |
| `all_bands.cv_inverse_J` (not the gated one) | 0.6914499950682564 |
| `collinearity_supported` | `false` |
| bands before DC exclusion | 328 |
| estimator | boxcar window, median average, 512 calibration windows |
| source = destination SHA-256 | `7f85a6b5b7ebd14c…` |

Not recomputed: the Round 7 data root and estimator were not re-derived.

## E7 — second rung-4 seed (20260818)

**Already complete on arrival; nothing was relaunched.** Both arms held
`checkpoint_00025000.pt` and a 50-evaluation log from `20260817_195350`.

| Gate | Result |
|---|---|
| 1. same seed and data root | PASS — both banners `seed=20260818`; identical window partition 10067/7046/3017 |
| 2. configs differ only in loss | PASS — both `model_params=569352`; the only differing field is `train.loss` |
| 3. reached 25000 steps in cap | PASS — 4.98 h and 4.93 h against an 8 h cap |
| 4. identical evaluation grids | PASS — 50 evaluations each, identical step values |

Gate 1 note: the seed-specific screen logs were not retained, so data-root
identity rests on the window-partition content signature rather than a logged
path string.

Paired differences, `t_chi2 − t_mse`, all four metrics errors:

| Metric | Round 9 (20260818) | 95% CI | Round 6 (20260808) | Sign agrees |
|---|---|---|---|---|
| `val_chi2` | −1.169e-04 | [−1.330e-04, −1.017e-04] | −1.148e-04 | yes |
| `val_mse` | −57.41 | [−76.15, −39.37] | −57.18 | yes |
| `train_chi2` | −1.036e-04 | [−1.183e-04, −8.879e-05] | −9.943e-05 | yes |
| `train_mse` | −19.86 | [−37.54, −1.71] | −14.26 | yes |

Every round-9 interval excludes zero. Round 6's `train_mse` interval included
zero; round 9's does not. This is the Round 8 table's "both seeds agree" branch:
rung 4 does not reproduce the metric reversal, and the non-reversal replicates
across two seeds. Bootstrap unit is the evaluation, not an independent run, so
seed-level uncertainty is still unquantified at n=2.

## Implementation freeze

- `51711aa` — output-safe E8/E9/E10 runners, method-aware semantics, tests.
- `391cd44` — resumability, evaluate-once-per-fit, caps raised from measured
  timings. No scientific choice changed; every output value is unchanged.

Caps were raised, never lowered, and justified from two recorded measurements:
the archived Block A run took 1:12:05 for 10 seeds × 5 methods (432 s/seed end
to end), and this run's E8 one-seed gate took 488 s. E8 2h→3h per condition,
E9 4h→6h, E10 unchanged at 3h.

## Phase 3 execution

Screen session `rps_round9_cpu`, `nice -n 10`, four BLAS threads, every command
under `timeout`. All steps returned 0.

| Step | rc | Wall (s) | Cap |
|---|---:|---:|---:|
| E8 one-seed reproduction gate | 0 | 488 | 30 m |
| E8 primary (α=0.1, seeds 0–9) | 0 | 5084 | 3 h |
| E8 separable (α=0, seeds 0–9) | 0 | 5100 | 3 h |
| E9 replication (seeds 10–29) | 0 | 10551 | 6 h |
| E10 estimated covariance (seeds 0–9) | 0 | 3885 | 3 h |

### E8 — archived-fit reproduction gate

All 50 rows (10 seeds × 5 methods) reproduce the archived
`test_normalized_chi2`. Maximum relative error **6.356e-15** against a 1e-9
tolerance. The refits are the archived fits.

### E8 — `primary_alpha_0p1`, `analysis_status: post_hoc_exploratory`

Post-hoc completion of the superseded Block H, not a confirmatory result.
Paired NFPA − Iso-MPCA (negative favours NFPA; every quantity is an error):

| Metric | Mean | 95% CI | Excludes 0 | Seeds negative |
|---|---:|---|:--:|:--:|
| `amplitude_rmse` | −0.046481 | [−0.04857, −0.04424] | yes | 10/10 |
| `rise_time_rmse` | −0.090324 | [−0.09180, −0.08877] | yes | 10/10 |
| `decay_time_rmse` | −0.015516 | [−0.01685, −0.01422] | yes | 10/10 |
| `channel_projector_error` | −0.97811 | [−0.9857, −0.9708] | yes | 10/10 |
| `temporal_projector_error` | −0.46270 | [−0.4754, −0.4507] | yes | 10/10 |
| `joint_projector_error` | −0.53286 | [−0.5457, −0.5212] | yes | 10/10 |
| `joint_projector_error_whitened` | −0.58591 | [−0.5987, −0.5742] | yes | 10/10 |

Per-method means: NFPA channel 0.0432 / temporal 0.8186 / joint 0.8194;
Iso-MPCA 1.021 / 1.281 / 1.352; CW-PCA and IsoPCA report joint only (1.189 and
1.354) because an unrestricted subspace identifies no axis factors; OF reports
amplitude only. Those fields are null, not zero.

### E8 — `separable_alpha_0`, `analysis_status: delayed_predeclared_no_numeric_threshold`

The delayed execution of H3. Config hash `09f7dbd811814c55`, distinct from the
archived `98001a62855cb9a5`. No archived-fit gate applies.

| Metric | Mean | 95% CI | Excludes 0 | Seeds negative |
|---|---:|---|:--:|:--:|
| `amplitude_rmse` | −0.055033 | [−0.05724, −0.05278] | yes | 10/10 |
| `rise_time_rmse` | −0.096727 | [−0.09828, −0.09512] | yes | 10/10 |
| `decay_time_rmse` | −0.021259 | [−0.02387, −0.01861] | yes | 10/10 |
| `channel_projector_error` | −0.98580 | [−0.9935, −0.9784] | yes | 10/10 |
| `temporal_projector_error` | −0.51190 | [−0.5342, −0.4890] | yes | 10/10 |
| `joint_projector_error` | −0.58309 | [−0.6055, −0.5604] | yes | 10/10 |
| `joint_projector_error_whitened` | −0.63517 | [−0.6578, −0.6123] | yes | 10/10 |

Per-method planted-factor means: NFPA channel 0.0349 / temporal 0.7677 / joint
0.7683; Iso-MPCA 1.021 / 1.280 / 1.351. NFPA recovers the planted channel
subspace almost exactly while temporal recovery is much weaker — an asymmetry
the desk should look at rather than average away.

The original protocol supplied no quantitative recovery threshold, so these are
reported as continuous errors and paired intervals. **No binary H3 verdict is
derived here**, and the two conditions are never pooled.

### E9 — 20-seed post-hoc replication

Seeds 10–29, frozen `PRIMARY` unchanged, written only to
`E9_blockA_replication/`. The replication summary was generated before any
pooling.

| Contrast | Seeds 0–9 (`frozen_confirmatory`) | Seeds 10–29 (`post_hoc_replication`) | Sign agrees |
|---|---:|---:|:--:|
| `test_normalized_chi2` | −0.0192011 | −0.0190460 | yes |
| `test_raw_mse` | +0.00341426 | +0.00341312 | yes |
| `angle_to_cw_pca_mean_deg` (secondary) | −24.6215 | −24.4378 | yes |

All intervals exclude zero. The opposite signs of the two primary contrasts
*are* the metric reversal: NFPA is better on normalized chi-square and worse on
raw MSE. Both directions agree between subsets, so `pooling_permitted: true`
and `pooled_summary.json` was written, displaying all three sets side by side:
pooled `test_normalized_chi2` −0.0190977 [−0.019210, −0.018988], `test_raw_mse`
+0.0034135 [+0.0033978, +0.0034281]. The 30-seed aggregate is labelled
`post_hoc_pooled` and is **not** a 30-seed confirmatory experiment.

Secondary-contrast caveat: `angle_to_cw_pca_*` is computed by
`methods.basis_whitened`, whose raw-plus-factored branch is defective (see
below). It affects Iso-MPCA only. The two primary contrasts do not use that
function.

### E10 — estimated covariance

Seeds 0–9, `N_cal = 2048` noise-only calibration events per seed, calibration
RNG `SeedSequence([20260819, seed, 10])`, `regularize_covariance(shrinkage=0.0,
floor=None)`. NFPA fits and reconstructs through the estimated axis
covariances; every reported metric is computed with the **oracle** covariance
on the unchanged held-out test set. The Iso-MPCA control reproduced its
archived primary metrics to a maximum relative error of **6.022e-15**.

Degradation, estimated minus archived oracle NFPA (positive = estimation made
it worse):

| Metric | Mean | 95% CI | Excludes 0 |
|---|---:|---|:--:|
| `normalized_chi2` | +7.905e-05 | [+6.301e-05, +9.635e-05] | yes |
| `raw_mse` | +3.548e-06 | [+1.154e-06, +6.692e-06] | yes |

Against an archived NFPA held-out chi-square near 1.015 that is a relative
degradation of order 0.008%: resolved, but tiny.

NFPA − Iso-MPCA under estimated whitening: `test_normalized_chi2` −0.0191221
[−0.019340, −0.018914], `test_raw_mse` +0.00341781 [+0.0033963, +0.0034393].
Compared with the oracle values (−0.0192011 and +0.00341426), **the reversal
survives estimated whitening essentially intact.**

Covariance diagnostics (means over 10 seeds): channel shape error 0.0282,
temporal shape error 0.0197, channel condition number 4.79, temporal condition
number 34654, channel minimum eigenvalue 4.52e-03, temporal minimum eigenvalue
2.88e-05, Kronecker scale ratio 0.0103, Kronecker relative Frobenius error
0.9897, oracle-whitening residual 2.197.

The last three are dominated by the *non-identified* reciprocal Kronecker
scale, not by estimation error: the per-axis estimators satisfy
`E[Sigma_c_hat] = Sigma_c · tr(Sigma_t)/T` and
`E[Sigma_t_hat] = Sigma_t · tr(Sigma_c)/C`, so the scale ratio's expected value
is `tr(Sigma_t)·tr(Sigma_c)/(C·T)`, not 1. This was verified numerically on a
small fixture (predicted 93.89 vs observed 93.75). The scale-free accuracy
statements are the two shape errors, and NFPA's fitted subspace and
reconstruction are invariant to a global rescaling of the whitening transform,
so the factor does not propagate into the reported metrics. This is one fixed
condition — a measured sensitivity check, not a general robustness boundary.

## Defect found and deliberately not fixed

`methods.basis_whitened` unstacks a `(C·T, rank)` basis with
`B.reshape(rank, C, T, order="F")` where the vec convention requires
`B.T.reshape(rank, C, T, order="F")`. The two have the same shape but the
former interleaves the basis index with the channel index. Verified against a
column-by-column ground truth.

Blast radius: the defective branch is reached **only by Iso-MPCA** (CW-PCA and
NFPA are whitened-coordinate and take a different branch; IsoPCA's `P`-branch
is correct), and `basis_whitened` feeds **only** `angle_to_cw_pca_*`.
Reconstruction metrics, the E8 reproduction gate and both E9 primary contrasts
are unaffected.

It was left unchanged because fixing it would silently redefine an archived
Block A column and break comparability between E9's replication rows and seeds
0–9. It is pinned by `test_archived_basis_whitened_defect_is_confined_to_iso_mpca`,
annotated in E9's `replication_summary.json`, and left to the desk. Round 9's
own metrics use the corrected `r9_semantics.fitted_basis_whitened`.

## Evidence audit and final state

`scripts/round9_evidence_audit.py`: **61/61 checks pass**, including that the
archived E1 digest is unchanged (`e9778eb2d663…`), exact seed sets
(E8 {0..9}, E9 {10..29}, E10 {0..9}), no duplicate keys, config hashes agreeing
within and differing between conditions, and every summary regenerating
byte-for-byte from its seed shards.

Final tests: 49 pass (18 existing NFPA correctness and leakage, 31 new Round 9).
`git diff --check` clean. `results/rps2026` is 1.1 MB. No Round 9 process, no
screen session, no stale PID file. Nothing under the TIDMAD raw-data tree was
modified. Checkpoints and denoised outputs remain outside Git.
