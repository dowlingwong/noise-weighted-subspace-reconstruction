# Experiment checklist — R/PS 2026 workshop paper

**Date:** 2026-08-08
**Deadline:** 29 Aug 2026 AoE (= 30 Aug 2026 11:59 UTC). Results frozen by **25 Aug**.
**Scope:** every experiment the paper depends on — what is already archived, what must still run, and how to run it.

This is the single operational checklist. `EXPERIMENT_PLAN_2026-08-08.md` (NFPA protocol) and `PLAN_03_RUNG4_TIDMAD_2026-08-08.md` (rung-4 design) remain authoritative for *definitions*; this document is authoritative for *what to run, where, and in what order*.

---

## 0. Environment

All experiments run on the **remote server**, not the laptop. The laptop has no `/ceph` mount, no CUDA, and no raw data.

### 0.1 Dataset root — `/ceph/dwong/datasets`

**All datasets are downloaded to and read from `/ceph/dwong/datasets`.** Layout:

```
/ceph/dwong/datasets/
  tidmad/
    raw/                       abra_training_*.h5, abra_validation_00{00..19}.h5,
                               abra_science_*.h5           (~2 GB per file)
    denoised/                  abra_validation_denoised_<tag>_00NN.h5   (written by infer.py)
  gwosc/                       H1/L1 strain segments        (archived work; read-only)
  cresst/                      (archived work; read-only)
```

Set once per shell and never hard-code a path in a script:

```bash
# the one repository you need
export CODE_ROOT=~/noise-weighted-subspace-reconstruction

# data
export TIDMAD_DATA_ROOT=/ceph/dwong/datasets/tidmad/raw
export TIDMAD_DENOISED_DIR=/ceph/dwong/datasets/tidmad/denoised
export TIDMAD_UPSTREAM=~/src/TIDMAD          # unmodified release; benchmark.py lives here

# where results go — the one tracked results path
export RPS_RESULTS=$CODE_ROOT/results/rps2026
mkdir -p "$RPS_RESULTS"
```

> **Write results only to `$RPS_RESULTS`.** `.gitignore` excludes `results/*`
> generally, with an explicit exception for `results/rps2026/**`. Anything
> written elsewhere under `results/` is silently excluded and will **never sync
> back**. Large artifacts (checkpoints, denoised `.h5`) stay under
> `$CODE_ROOT/artifacts` or `/ceph` and are referenced by path, never copied.

> **Path-provenance warning.** Paper 1's archived run records reference `/ceph/dwong/paper1_dataset/tidmad/raw`. Moving to `/ceph/dwong/datasets` breaks the literal path in those archived JSON files. Do **one** of the following and record which in the run log:
> - keep `/ceph/dwong/paper1_dataset` as a symlink to the new location, so archived paths still resolve; **or**
> - add a `path_migration` note to each new run record stating the old and new roots and the date.
>
> Do not silently re-point. The archived numbers are cited in the manuscript and their provenance must stay checkable.

### 0.2 Integrity check before first use

```bash
ls -la $TIDMAD_DATA_ROOT | head
md5sum $TIDMAD_DATA_ROOT/abra_validation_0000.h5   # compare against the release manifest
python -c "import h5py;f=h5py.File('$TIDMAD_DATA_ROOT/abra_validation_0000.h5');print(list(f['timeseries']))"
```

Expect `channel0001` (SQUID readout) and `channel0002` (injected reference). If the channel names differ, stop — the data contract's convention no longer holds and every downstream number is suspect.

### 0.3 Compute

CUDA required. `uv sync` then `source .venv/bin/activate`. Runs go through
`./scripts/train_tidmad_local.sh`, which pins a single GPU, runs the GPU and data
preflight, applies `nice` and the wall-clock cap, and writes a PID file.

There is no batch system. `PYTHONPATH=src` is required for `python -m tidmad.*`
and is set by the driver script.

---

## 1. Already archived — do NOT re-run

These are cited in the manuscript and must not be recomputed. Sources are under `transfer_paper/data/`.

| ID | Experiment | Key result | Where |
|---|---|---|---|
| A1 | 30-seed synthetic metric reversal (rung 2) | PCA vs CW-PCA weighted residual 48.419 / 47.869; paired diff 0.550, CI [0.530, 0.571]; max principal angle 86.25° | Paper 1 §7 |
| A2 | 10-seed white-noise control | max PCA–CW-PCA angle 1.48×10⁻⁶ ° | Paper 1 §7 |
| A3 | TIDMAD ground-truth metric reversal (Phase 2) | amplitude RMSE 0.1209 → 0.0880. **Not replicated** at the confirmatory file (CI [−0.0793, 0.1779], gates false) | `data/tidmad/current/TIDMAD_P2_*` |
| A4 | TIDMAD PSD adequacy (Phase 4) | χ²/dof = 1.0002, dispersion 1.0142 (z = 0.64), all gates pass — **justifies frequency-diagonal Σ** | `TIDMAD_P4_real_*.json` |
| A5 | TIDMAD detector-facing score (Phase 5) | linear Wiener −1.8025 vs published none 1.10, PUNet 3.84, Transformer 4.18, FCNet 6.55 | `TIDMAD_P5_file_*.json` |
| A6 | G0 anchor reproduction | `none` = 1.1026 vs published 1.10 | `G0_baseline_reproduction.json` |
| A7 | GWOSC covariance adequacy | filter equivalence to 1.6×10⁻¹²; real H1/L1 score dispersion 9.42 / 8.83 vs synthetic 1.15 | `data/gwosc/` |
| A8 | NFPA single-split `C = k_c = 2` | appendix implementation check only | Paper 2 |

---

## 2. Blocking prerequisites — settle before any training

### P1 — Characterise injection frequencies for all 20 validation files

**Why blocking.** `n_bands_used = 328` covers 0–800 kHz of a 0–5 MHz band. The data contract characterises only files 0000, 0005, 0010 plus a sweep note for 0015, yet its coupling scan runs out to **3.4 MHz** — implying injections above the modelled band. For any such file the model receives no signal at all, and a band-limited model is not performing the same task as the published baselines, so the Table-1 comparison would not be valid.

**How.** Cheap FFT scan per file, respecting the contract's warning that frequency is sample-range specific and sweeps within a file:

```bash
cd $CODE_ROOT && PYTHONPATH=src timeout 30m python -m tidmad.characterise_injections \
  --data-dir $TIDMAD_DATA_ROOT \
  --glob 'abra_validation_*.h5' \
  --out    $RPS_RESULTS/P1_injection_survey.json
```

Exit status 0 means every probed peak is in band; **exit 2 means at least one file
is out of band** and the decision below is forced. The module probes three
positions per file (start, middle, end) because the contract records that the
tone sweeps within a file.

For each file record: peak frequency at start / middle / end of the file, prominence over the median, and the reference-to-readout coupling ratio.

**Gate and decision.**

- All 20 below 800 kHz → coverage confirmed, proceed unchanged; extend `tidmad_data_contract.json` with the survey.
- Any above 800 kHz → choose explicitly and record: widen coverage (coarser `n_fft` covers the full band at lower frequency resolution — state the trade-off against resolving a narrow tone), **or** keep band-limiting and **drop the published-baseline column from Table 1**, presenting only the paired contrast.

The paired `T-mse` / `T-chi2` contrast survives either way; only the external comparison is at risk.

### P2 — Reproduce the `none` anchor in the scoring environment

```bash
cd $TIDMAD_UPSTREAM && python benchmark.py -d $TIDMAD_DATA_ROOT -m none -c
```

**Gate:** 1.1026 (archived A6) or a documented equivalent. If it does not reproduce here, stop — environment or channel semantics differ from Paper 1's and every downstream number is suspect. Do **not** use `savgol` as a check; the contract records it as not bit-reproducible.

### P3 — Architecture decisions (see §5)

Resolve the two items in §5 before freezing the config. They change the frozen record.

### P4 — Training-loop fixes — **LANDED 2026-08-08**

Five gaps were verified and have now been fixed in `src/tidmad/train.py`:

| # | Gap | Fix |
|---|---|---|
| 1 | No checkpointing | `_checkpoint()` every `checkpoint_period` steps and at exit; writes `checkpoint_<step>.pt` and `latest.pt` |
| 2 | No batching | `train_step` accumulates gradients over `device_batch_size` windows, so the frozen config now describes the run |
| 3 | No resume | `--resume <path>` or `--resume auto`; restores model, both optimisers, both schedulers and the step counter |
| 4 | No held-out evaluation | `evaluate()` on the chronological split with guard gap; **both** metrics reported for both arms at every eval |
| 5 | No throughput logging | steps/second, elapsed and ETA logged at every eval |

Also added: a **SIGTERM/SIGINT handler**, so a run stopped by the `timeout` cap
checkpoints cleanly instead of losing the segment; optional `--wandb`; and
`grad_clip` (default 1.0) plus `eval_num_windows` (default 32) in the config.

**Gate — still to be run on the server:** a smoke run writes a checkpoint, is
killed mid-run, resumes from it, and reports steps/second.

```bash
./scripts/train_tidmad_local.sh smoke              # confirm a checkpoint appears
# interrupt it, then:
TIDMAD_RESUME=auto ./scripts/train_tidmad_local.sh smoke
```

`src/tidmad/tests/test_train_loop.py` covers the mechanical half: exact
checkpoint round-trip of model/optimisers/schedulers/step, and that the
chronological split leaves a guard gap with validation strictly later than fit.

---

## 3. Experiments to run

### E1 — NFPA Block A (rung 3 consolidation)

**Claim supported:** metric reversal at fixed architecture, with the channel axis genuinely compressed. Retires the `C = k_c = 2` degeneracy a referee will otherwise find.

**Protocol:** `EXPERIMENT_PLAN_2026-08-08.md`, as amended — **10 seeds**, `C = 8 > k_c = 2`, `T = 256`, `(k_c,k_t) = (2,3)`, matched total rank 6, 800/400/2000 train/val/test, 10 ALS restarts, SNR 10, Brownian 1/f² with channel correlation, oracle covariance, α = 0.1.

**Methods at matched rank:** CW-PCA, NFPA, Iso-MPCA/IsoAE, IsoPCA, OF.

**Primary contrasts (predeclared):** NFPA − Iso-MPCA under normalised χ²; the same under raw MSE; NFPA − CW-PCA under normalised χ².

**Gates:** all seeds complete without post-result config change; `C > k_c` with both planted projectors non-trivial; fixed-architecture reversal intervals exclude zero in the predicted directions; leakage audit passes; table and figure generated from archived seed-level results.

**Command.** The confirmatory harness is `experiments/synthetic/nfpa_confirmatory/run_blocks.py`:

```bash
cd $CODE_ROOT
python -m experiments.synthetic.nfpa_confirmatory.run_blocks --smoke          # 2 seeds first
python -m experiments.synthetic.nfpa_confirmatory.run_blocks --block A        # then the real run
```

CPU-bound, so it does not compete with the GPU arms and needs no `screen` session.
Copy the tidy outputs to `$RPS_RESULTS/E1_nfpa_blockA/`.

**Outputs:** one tidy row per (seed, method, split); paired bootstrap CIs at seed level.

### E2 — NFPA Block B (white-noise control)

**Claim supported:** covariance geometry, not rank or optimisation, causes the separation.

Same clean signals, `Σ = σ²I`, 20 seeds.

```bash
cd $CODE_ROOT && python -m experiments.synthetic.nfpa_confirmatory.run_blocks --block B
```

**Gate:** covariance-aware and Euclidean projectors agree to numerical tolerance; reversal intervals include zero or fall inside the predeclared equivalence margin. **Non-negotiable — do not drop this to save time.** It is small and carries disproportionate evidential weight. Copy outputs to `$RPS_RESULTS/E2_nfpa_blockB/`.

### E3 — Rung 4, arm `T-mse`

**Claim supported:** together with E4, the metric contrast at fixed architecture in a nonlinear model — the paper's headline rung-4 evidence.

```bash
GPU_INDEX=0 TIDMAD_TIMEOUT=6h ./scripts/train_tidmad_local.sh mse
```

or directly:

```bash
cd $CODE_ROOT && PYTHONPATH=src python -m tidmad.train --config src/tidmad/configs/t_mse.yaml \
  --data-dir $TIDMAD_DATA_ROOT --out artifacts/tidmad/t_mse
```

### E4 — Rung 4, arm `T-chi2`

```bash
GPU_INDEX=0 TIDMAD_TIMEOUT=6h ./scripts/train_tidmad_local.sh chi2
```

Interrupted by the wall-clock cap? Resume exactly where it stopped:

```bash
TIDMAD_RESUME=auto GPU_INDEX=0 TIDMAD_TIMEOUT=6h ./scripts/train_tidmad_local.sh chi2
```

**Gate for E3/E4 jointly — the load-bearing check.** The two `run_config.json` files must differ **only** in `train.loss`. The config-equality assertion is already implemented; verify its output appears in the log. Identical architecture, data, budget, seed handling. Any other difference invalidates the contrast.

### E5 — Inference and scoring

```bash
cd $CODE_ROOT && export PYTHONPATH=src
for ARM in t_mse t_chi2; do
  python -m tidmad.infer \
    --config     src/tidmad/configs/${ARM#t_}.yaml \
    --checkpoint artifacts/tidmad/$ARM/latest.pt \
    --data-dir   $TIDMAD_DATA_ROOT \
    --out        $TIDMAD_DENOISED_DIR \
    --model-tag  $ARM
  python -m tidmad.score \
    --upstream  $TIDMAD_UPSTREAM \
    --data-dir  $TIDMAD_DENOISED_DIR \
    --model-tag $ARM --coarse | tee $RPS_RESULTS/E5_${ARM}_benchmark_stdout.txt
done
```

`latest.pt` is written by every checkpoint; there is no `best.pt` because
checkpoint selection is by held-out `val_chi2` from the training log, not by an
automatic best-tracker.

**Note the split of outputs.** Checkpoints go to `artifacts/`, which is
gitignored and stays on the server. Only the tidy records go to `results/`,
which syncs back — see §5.

**Gate:** every score produced by the **unmodified** upstream `benchmark.py`. Capture command, returncode and raw stdout into the run record.

### E6 — Paired statistics and figure

Reuse Paper 1's TIDMAD protocol: chronological fit/eval split with guard gap, block bootstrap at block size 16, 300 resamples, zero window-hash and window-index overlap reported explicitly.

Report for **both** arms: raw MSE, whitened residual, truth-referenced amplitude error against `channel0002`, and the coarse denoising score. Figure per `PLAN_03_RUNG4_TIDMAD` §7.

### E7 — Secondary: `T-chi2+OF`

Optimal-filter feature injection. **Run only if E1–E6 are complete.** Attributed to prior work in the text per `PLAN_03_RUNG4_TIDMAD` §5.

---

## 4. Order, ownership and gates

| # | Item | Blocks | Gate |
|---|---|---|---|
| 1 | P1 band characterisation | E3–E7 | Survey archived; coverage decision recorded |
| 2 | P2 `none` anchor | E5 | 1.1026 or documented equivalent |
| 3 | P3 architecture decisions (§5) | E3, E4 | Frozen config reflects the decision |
| 4 | **Freeze config, timestamped commit** | E3, E4 | Freeze precedes any aggregate inspection |
| 5 | E1, E2 (NFPA) | — | Acceptance gates in `EXPERIMENT_PLAN` §4 |
| 6 | E3, E4 (rung-4 arms) | E5 | Configs differ only in `train.loss` |
| 7 | E5 scoring | E6 | Unmodified benchmark script |
| 8 | E6 statistics and figure | manuscript | Generated by script from archived results |
| 9 | E7 secondary | — | Only if time remains |

E1/E2 and E3/E4 are independent and can run in parallel.

**If a gate fails, report it — do not adjust the configuration to pass.** A null rung-4 contrast is a publishable finding (capacity can absorb a misspecified metric). Only one outcome triggers escalation: metric reversal failing to appear at the *linear* rungs, which would contradict a theorem and means a bug.

---

## 5. Architecture decisions still open

Both change the frozen config, so settle them before step 4.

### D1 — Stack real/imaginary as features, not as bands (recommended)

Currently real and imaginary parts are stacked along the band axis, giving `C = 656` tokens. Consequences: cross-band attention costs `656² ≈ 4.3×10⁵` pairs per patch position, and the model must *learn* that band `b` and band `b+328` are two components of one frequency.

Feeding real/imaginary as a **feature pair into the patch embedding** instead — `nn.Linear(patch_len * 2, d_model)` — gives `C = 328`, cuts cross-band attention cost **4×**, makes one token equal one physical frequency, and makes the learned band embedding index frequency directly. It also matches the manuscript's description of "cross-band attention" more honestly.

No downside identified. Recommend adopting before the freeze.

### D2 — The temporal axis is barely exercised

With `M = 96` frames and `patch_len = 8`, the temporal transformer attends over **12 positions** while the band stage attends over 656. The paper frames rung 4 as *"intra-band temporal attention followed by cross-band attention"* — the nonlinear counterpart of the Kronecker factorization. A 12-position temporal axis makes that factor close to vestigial, and a referee comparing §3's claim against the config will see it.

Options: reduce `patch_len` to 4 (N = 24), or lengthen the window (M = 192 → N = 24 at `patch_len = 8`). Either restores a meaningful temporal sequence at modest cost. Pick one, state the reason in the config docstring.

### Noted, not blocking

The model is small — ~611k parameters, `d_model = 128`, one temporal and one band layer. This is adequate for the paired contrast, which is the paper's actual claim, but makes it unlikely to approach FCNet (6.55) or the published Transformer (4.18). Combined with the P1 band-coverage risk, there are **two independent reasons the external scoreboard comparison may not land**. That is acceptable and honest — the contrast is the finding — but do not discover it late. If Table 1's baseline column is dropped per P1, note that this capacity limitation points the same way.

---

## 6. Archival contract

Every run writes, before any aggregation:

- one tidy row per (seed or arm, method, condition, split);
- `run_config.json`, git commit, config hash, seed, realised SNR, array shapes, data checksums;
- for rung 4: the exact scoring command, returncode and raw stdout;
- the resolved dataset root, and the `path_migration` note from §0.1.

Tables and figures are generated from those files by a deterministic script. **Never transcribe a number by hand into LaTeX.**

Log to `plan/EXPERIMENT_RUN_LOG.md`: one section per experiment, gate results, and a plain statement of what each run does and does not support.
