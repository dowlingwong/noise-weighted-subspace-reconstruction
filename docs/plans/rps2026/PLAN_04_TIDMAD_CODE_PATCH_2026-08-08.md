# PLAN 04 — TIDMAD code patch / adapt plan

**Date:** 2026-08-08
**Repository:** the `transformer/` working copy (DELight reconstruction project, 4 commits, HEAD `48642ee`)
**Governing:** `PLAN_00_MASTER_RESTRUCTURE_2026-08-08.md`; experiment design in `PLAN_03_RUNG4_TIDMAD_2026-08-08.md`
**Status of existing work: none.** `grep -ril "tidmad|stft|abra"` over the repository returns zero hits. Nothing has been adapted.

---

## 1. Architecture decision — build a parallel package, do not modify `train.py`

`reconstruction_model/train.py` is 988 lines carrying DDP init, env-var config resolution, W&B artifact publication, AMP/GradScaler selection, gradient accumulation, and a four-tuple dataset contract `(input, spatial_target, energy_target, recoil_type)` tuned for a three-task supervised DELight problem. Retrofitting a single-output reconstruction task into it means touching the loss, the dataset contract, `eval_model`, `unpack_model_outputs`, and `select_spatial_targets`, with a real chance of breaking the existing path.

**Build `tidmad/` as a sibling package instead.** It imports the reusable pieces and owns everything task-specific:

```
tidmad/
  __init__.py
  data.py        # TIDMAD h5 → windows → STFT → tensors
  psd.py         # noise PSD estimation (thin wrapper, see §3.2)
  model.py       # backbone import + reconstruction head
  loss.py        # the two objectives
  train.py       # minimal loop (~200 lines)
  infer.py       # inverse STFT → denoised h5 in release format
  score.py       # wrapper around upstream benchmark.py, unmodified
  configs/
    t_mse.yaml
    t_chi2.yaml
    t_chi2_of.yaml
  tests/
```

**Reuse by import, do not copy:** `reconstruction_model.model` (`TransformerConfig`, `TransformerBlock`, `MultiHeadAttention`, `FFN`, `precompute_rope_angles`, `patch_input_sequence`, `normalise_input_sequence`), `reconstruction_model.muon.Muon`, `reconstruction_model.schedulers`, `reconstruction_model.checkpoints`.

**Why this is also the right call for publication:** `PLAN_03_RUNG4_TIDMAD` §2A.5 requires the anonymized repository to be a clean subtree with no DELight material. `tidmad/` plus the imported backbone modules *is* that subtree. Building it separately means the publishable artifact exists by construction rather than by pruning.

---

## 2. The trap that would silently destroy the experiment

**Read this before writing any code.**

The intended contrast is: identical architecture, data, budget and seeds; **only the residual metric in the objective differs.**

`normalise_input_sequence` (`reconstruction_model/model.py:28`) z-scores each row of the `(B*C, T)` input independently. When `C` indexes **frequency bands**, per-row standardisation divides each band by its own standard deviation — which, for noise-dominated bands, *is approximately whitening*. Two consequences:

1. **Do not additionally whiten the input.** It is already approximately standardised per band. Doing it twice is not a bug you will notice; it will just make both arms identical.
2. **The loss must be computed in a fixed coordinate system that is not the standardised one.** If the model's output is compared to a target that has been per-band standardised, then plain MSE on that representation *is already* the χ²-weighted residual in measurement coordinates, and the two arms collapse to the same objective. You would get a null result that means nothing.

**Required implementation:**

- The model consumes the standardised representation. Both arms, identically.
- The reconstruction head's output is **mapped back to measurement (unstandardised) STFT coordinates** using the per-band scale factors recorded during normalisation, before the loss is computed.
- Both losses are then evaluated in those measurement coordinates:

```python
# resid: (B, F, M) complex or (B, 2F, M) real, in MEASUREMENT coordinates
# J:     (F,) noise PSD per band, broadcast over frames
loss_mse  = (resid ** 2).mean()
loss_chi2 = ((resid ** 2) / J).mean()
```

- Write a unit test that fails if the two losses are proportional on random input with a non-flat `J`. If they are proportional, the coordinate handling is wrong.

This is the single highest-risk item in the patch. Treat the unit test as a gate, not a nicety.

---

## 3. Components

### 3.1 `data.py` — TIDMAD → band–frame tensors

Data contract is already verified: `transfer_paper/data/tidmad/current/tidmad_data_contract.json` (`verified: true`). Use it as the source of truth; do not re-derive.

- `timeseries/channel0001/timeseries` = SQUID readout; `timeseries/channel0002/timeseries` = injected reference/target. Release convention, hardcoded across upstream `train.py`, `inference.py`, `benchmark.py`.
- Sampling 10 MHz, `int8`. Twenty validation files of roughly 200 s each.
- **Injection frequency is file- and sample-range-specific and sweeps within a file** (file 0010 drifts ~75 000 → ~89 000 Hz over 190 s). Read it per file from the contract. Never reuse a frequency across files — the contract records this as a bug that has already occurred once.
- Reference-to-readout coupling falls steeply with frequency (7.13 at 5.7 kHz to 0.0021 at 3.4 MHz).

**Follow the release's own train/target convention exactly.** Read upstream `~/src/TIDMAD/train.py` and mirror how its baselines construct inputs and targets. Do not invent a convention — comparability with the published scores is the entire point of using this benchmark.

**Predeclare and freeze before any training run:** `n_fft`, hop length, window function, complex representation, band range, and the chronological split with guard gap (mirror Paper 1: 70/30 chronological, four-window guard, zero hash and index overlap).

**Complex handling — default choice:** stack real and imaginary parts as separate bands, giving model input `(B, 2F, M)`. Cross-band attention then sees both; no architectural change is needed. Magnitude-only discards phase and cannot reconstruct — do not use it.

### 3.2 `psd.py` — noise PSD

**Do not write a new estimator.** Paper 1's Phase 4 already fits and validates a global stationary PSD on this data: median-averaged, boxcar window, 512 calibration windows, `χ²/dof = 1.0002`, score dispersion 1.0142 (`z = 0.64`), all gates passed. Import or port that estimator so the `J(f)` used here is the same object the paper cites.

`J(f)` must be estimated from **noise-only science files**, never from evaluation data.

### 3.3 `model.py` — backbone plus reconstruction head

Use `current_compact` (`reconstruction_model/model.py`). It is the only variant with `requires_pairwise_cache=False`; the other five depend on `models/cache/edge_feats.npy` and `pos_diff.npy`, which are 3D sensor geometry and unusable here.

Keep the backbone unchanged through the spatial layers and `final_proj_norm`. Then:

- **Delete** the mean-pool over `(patches, channels)` and both scalar heads (`spatial_head`, `energy_head`).
- **Add** a per-patch linear projection `(B, C, N, d_model) → (B, C, N, P)`, folded back to `(B, C, T)`. Roughly 30 lines. This is the only architectural change.
- Register the variant as `tidmad_stft` in `models/registry.py` with `requires_pairwise_cache=False`.

**Config:** `max_seq_len` → number of frames `M`; `patch_len` → frame patch size; `C` → number of bands. Note `max_patches = max_seq_len // patch_len` drives the RoPE cache. With `M` far smaller than 65 536, `patch_len = 64` may leave too few patches for temporal attention to be meaningful — choose deliberately and record the choice in the frozen config.

**Positional embeddings:** leave as they are. Temporal layers use RoPE; cross-band layers use the learned `AbsolutePositionalEmbedding`. Do not switch the band axis to RoPE without a reason — the manuscript is being corrected to describe the model as it exists (`MANUSCRIPT_CORRECTION` C4), and changing the code now would desynchronise them again. If you do change it, say so loudly so the paper is updated.

### 3.4 `train.py` — minimal loop

Reuse `Muon`, the cosine schedulers with warmup, `checkpoints`, and `set_seed`. Keep gradient accumulation. Drop DDP unless a run demands it. Write `run_config.json` in the same shape as `save_resolved_config` so the two arms can be diffed mechanically.

**Config-equality assertion (required gate).** Before training arm two, load both `run_config.json` files and assert they differ **only** in the loss flag. Fail loudly otherwise. The contrast is worthless if anything else drifts.

### 3.5 `infer.py` and `score.py`

- Inverse STFT, then write denoised h5 **in the release's format**. Paper 1 already produced `abra_validation_denoised_wiener_0000.h5`, so this writer exists — port it rather than rewriting.
- `score.py` invokes upstream `benchmark.py -c` **unmodified**, as Paper 1's Phase 5 did, and captures returncode, stdout and the command line into the run record.
- **Gate:** reproduce the `none` anchor at 1.1026 (published 1.10) in this environment before scoring any arm. If it does not reproduce, stop — the environment or channel semantics are wrong and every downstream number is suspect.

### 3.6 `tests/`

1. The **non-proportional-loss test** of §2. Gate.
2. Round-trip STFT → inverse STFT reconstructs the input within tolerance.
3. Reconstruction head output shape equals input shape.
4. `J(f)` is strictly positive; whitening identity error below tolerance.
5. Leakage: zero window-hash and zero window-index overlap between fit and evaluation splits.
6. Backbone parity — `tidmad_stft` and `current_compact` produce identical activations through `final_proj_norm` on the same input, confirming the head swap changed nothing upstream.

---

## 4. Repository hygiene and the migration into the experiment repo

### 4.1 Current exposure status — resolved

`github.com/dowlingwong/transformer.git` is **private** and owned by the author. `src/scripts/TraceSimulator.py` is tracked in its history, but a private repository is not publication. **No exposure has occurred. No history rewriting is required.**

The `artifacts/` (2.6 GB), `wandb/`, `*.pt`, `*.h5`, `*.npy` and `*.zst` paths are correctly gitignored and were never tracked. Only 122 files are tracked in total.

### 4.2 The planned migration, and the trap in it

The author intends to remove `transformer/.git` and fold the directory into the parent experiment repository `noise-weighted-subspace-reconstruction`, then "remove the DELight things later once the framework runs."

**Do not do it in that order.** If the directory is `git add`-ed as it stands, every detector-collaboration file enters the **parent** repository's history, and deleting them afterwards does not remove them. The problem is not solved, it is relocated one level up — into the repository most likely to become this paper's public or anonymized code link (desk task D5).

**Correct order: curate first, commit once.**

1. Create the target directory inside the parent as an empty path.
2. Copy in only the keep-list below. **Copy, do not move, and do not `git add` anything outside the list.**
3. Make the first commit from that curated state.
4. Keep the existing private `transformer` repository as-is, archived, as the provenance record. Nothing is lost by keeping it; a great deal is risked by rewriting it.

### 4.3 Keep / drop list

**Keep — generic infrastructure with no detector coupling:**

```
reconstruction_model/__init__.py
reconstruction_model/model.py          # current_compact backbone
reconstruction_model/muon.py
reconstruction_model/schedulers.py
reconstruction_model/checkpoints.py
reconstruction_model/utils.py
reconstruction_model/models/__init__.py
reconstruction_model/models/registry.py    # trim to current_compact + tidmad_stft
pyproject.toml, uv.lock, .python-version, .gitignore
scripts/smoke_test_training.py             # if it survives the head swap
```

**Drop — detector-coupled, third-party, or dead weight:**

| Path | Reason |
|---|---|
| `src/scripts/TraceSimulator.py`, `make_traces.py` | Collaboration simulator |
| `src/scripts/precompute_pairwise_features.py`, `compute_position_stats.py`, `run_integration_classification.py` | Sensor geometry and detector tasks |
| `reconstruction_model/models/cache/edge_feats.npy`, `pos_diff.npy` | 3D sensor geometry |
| `reconstruction_model/models/{original,pairwise,pairwise_channel_masking,triangular_pairwise,cnn_transformer}.py` | All require the geometry cache |
| `reconstruction_model/models/integration_classifier.py` | Detector XGBoost baseline |
| `reconstruction_model/dataset.py` | ER/NR, energy bins, `.zst` traces |
| `reconstruction_model/legacy/` (10 files) | Detector evaluation/visualisation utilities |
| `submit_ref/` (13 files) | Institution-specific cluster templates; includes a third-party wiki PDF that should not be redistributed regardless |
| `src/finetuning_references/` (~10 files) | Unrelated third-party training notes, some non-English, no bearing on this work |
| `notebooks/`, `containers/`, `run.sh` | Detector-run tooling; re-add only if actually needed |
| `README.md` | Rewrite from scratch; the current one is detector-branded throughout |

That is roughly 10 files kept out of 122 tracked. The backbone is small; almost everything else is coupling.

### 4.4 Parent-repository requirements

- Confirm `noise-weighted-subspace-reconstruction` is **private** before the first commit.
- Ensure its `.gitignore` covers `artifacts/`, `wandb/`, `*.pt`, `*.h5`, `*.npy`, `*.zst`, `training_data/`, `cache/`. The transformer repo's rules protected it; the parent may not have equivalents, and 2.6 GB of checkpoints will be committed silently if it does not.
- The anonymized artifact for task D5 must **still** be built as a fresh repository containing only `tidmad/`, the kept backbone modules, configs and tests. Curating the migration does not remove that requirement; it only means the curation happens once instead of twice.
- Migration does not affect attribution. The obligations in `PLAN_03_RUNG4_TIDMAD` §5 are unchanged by which repository the code sits in.

### 4.5 "Remove DELight things later"

The intent is right; the timing is the whole point. Anything removed after its first commit stays in history. **Everything on the drop list must never be committed to the parent at all.** After that, later cleanup is cosmetic rather than load-bearing — which is the position you want to be in.

---

## 5. Order of work

| # | Item | Gate |
|---|---|---|
| 1 | Scaffold `tidmad/`; register `tidmad_stft`; reconstruction head | Backbone parity test passes; smoke test runs on synthetic `(B, 2F, M)` |
| 2 | `psd.py` ported from Paper 1's Phase 4 estimator | `J(f)` positive; whitening identity within tolerance |
| 3 | `data.py`; read upstream `train.py` and mirror its convention | Round-trip and leakage tests pass |
| 4 | Reproduce the `none` anchor through unmodified `benchmark.py` | **1.1026 (or documented equivalent). Blocking.** |
| 5 | `loss.py` + measurement-coordinate handling | **Non-proportional-loss test passes. Blocking.** |
| 6 | `train.py`; config-equality assertion | Two configs differ only in the loss flag |
| 7 | `infer.py` / `score.py`; port Paper 1's h5 writer | An untrained model scores end-to-end without error |
| 8 | **Freeze config, timestamped commit** | Freeze precedes any aggregate inspection |
| 9 | Train `T-mse`, `T-chi2` | Both complete; scored through the unmodified script |
| 10 | `T-chi2+OF` if schedule allows | Secondary only |

Steps 1–8 are the patch. Steps 9–10 are the experiment and are out of scope for a patching agent.

**Checkpoint 18 Aug:** if step 9 has not started, escalate per `PLAN_03_RUNG4_TIDMAD` §6. The fallback is to present rung 4 as a structural observation with no new results, which still improves the paper.

---

## 6. Deliverable

`plan/TIDMAD_PATCH_LOG.md`, recording:

1. Each step, its gate result, and the files touched.
2. The frozen STFT parameters, split protocol, and the commit and timestamp of the freeze.
3. The `none` anchor reproduction with the exact command and captured output.
4. The non-proportional-loss test output, showing the two losses are genuinely different objectives.
5. Confirmation that no DELight file is present in the clean subtree, with the `git log --all --name-only` check.
6. Any deviation from the release's own train/target convention, with justification. There should be none.

**Do not train anything before every gate in steps 1–8 has passed.** A run that starts on an unverified data path costs more than the day it saves.
