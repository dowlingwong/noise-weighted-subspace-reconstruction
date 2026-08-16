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
