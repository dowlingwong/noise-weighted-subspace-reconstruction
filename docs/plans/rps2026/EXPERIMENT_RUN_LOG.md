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
