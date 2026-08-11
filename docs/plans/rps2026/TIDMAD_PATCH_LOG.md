# TIDMAD code patch log (PLAN_04, steps 0–8)

**Date:** 2026-08-08
**Plan:** `plan/PLAN_04_TIDMAD_CODE_PATCH_2026-08-08.md`
**Repo:** `noise-weighted-subspace-reconstruction/transformer/` (private, own git repo, HEAD `48642eef20d491e1abe1b35ac7908e3580d2b549` at start)
**Environment note:** this machine is a macOS laptop. The training/scoring environment (Linux, `/ceph` data, cu124 torch) is elsewhere; the impact is recorded per step below.

---

## Step 0 — Migration (GATE: parent private; .gitignore verified; curated file list committed) — **BLOCKED**

### Parent-repository privacy — **gate failed**

`noise-weighted-subspace-reconstruction` is **PUBLIC**, verified via the GitHub API on 2026-08-08:

```
GET https://api.github.com/repos/dowlingwong/noise-weighted-subspace-reconstruction
private: False
visibility: public
```

PLAN_04 §4.4 requires "confirm the repository is private before the first commit". It is not. **No commit was made to the parent.** Per the instruction "If a gate fails, stop and report. Do not work around it," the migration is deferred. The author must either make the parent private or explicitly authorise a public commit of the curated (publishable-safe) keep-list before any parent commit happens.

### Parent `.gitignore` coverage — **gaps found**

The parent `.gitignore` covers `*.h5`, `*.pt`, `*.ckpt`, `*.npy`, `*.npz`, `*.zst`, `wandb/`, `cache/`, `checkpoints/`, `data/raw/`. **Missing: `artifacts/` and `training_data/`.** If the migration proceeds, these two entries must be added first, or 2.6 GB of checkpoints and any `training_data/` will commit silently. (The transformer repo's own `.gitignore` has both; the parent does not.)

### Curated file list (what WOULD be committed once unblocked)

Keep-list per PLAN_04 §4.3, plus the new `tidmad/` package. Everything below is publishable-safe (no detector coupling):

```
reconstruction_model/__init__.py
reconstruction_model/model.py
reconstruction_model/muon.py
reconstruction_model/schedulers.py
reconstruction_model/checkpoints.py
reconstruction_model/utils.py
reconstruction_model/models/__init__.py
reconstruction_model/models/registry.py        (trimmed: current_compact + tidmad_stft)
pyproject.toml, uv.lock, .python-version, .gitignore
tidmad/  (config.py, data.py, psd.py, model.py, loss.py, train.py, infer.py, score.py,
          configs/t_mse.yaml, t_chi2.yaml, t_chi2_of.yaml, tests/)
```

**Not committed anywhere in this pass.** Files already uncommitted in the transformer working tree (mode 755→644 from OS checkout, and deleted `src/finetuning_references/*` notebooks, which are on the drop list) predate this work; I did not add them.

### Repository-hygiene confirmation (no detector-collaboration file in the curated set)

`git log --all --name-only` on the transformer repo shows the detector-collaboration files (`src/scripts/TraceSimulator.py`, `models/cache/*.npy`, `artifacts/`, etc.) exist only in the private repo's history and are **not** in the curated list above. None of them are in `tidmad/` or the keep-list. The parent repo tracks **zero** files under `transformer/` (nested repo), so nothing has leaked.

---

## Step 1 — Scaffold, register, reconstruction head (GATE: parity test + smoke test) — **PASS**

Files touched: `tidmad/{__init__,config,model}.py`, `reconstruction_model/models/registry.py`.

- `tidmad/model.py` defines `TidmadTransformer`, a subclass of the `current_compact` backbone: the mean-pool and both scalar heads are deleted and replaced by a per-patch linear projection `(B,N,C,d) → (B,C,M)`. The backbone path through `final_proj_norm` is replicated verbatim and verified by the parity test.
- Registered as `tidmad_stft` in `models/registry.py` with `requires_pairwise_cache=False`.
- **Backbone parity test passes**: `tidmad_stft` and `current_compact` produce identical activations through `final_proj_norm` on the same input (shared weights).
- **Smoke test passes**: model forward on synthetic `(B, 2F, M)` → `(B, 2F, M)`; `FROZEN` model builds (M=5248, C=656, 611,336 params).

---

## Step 2 — PSD port (GATE: J positive; whitening identity) — **PASS**

Files touched: `tidmad/psd.py`, `tidmad/tests/test_psd.py`.

- `J(f)` reuses Paper 1's Phase-4 estimator methodology: median-averaged periodogram over noise-only science windows, 512 calibration windows of 4096 samples (`χ²/dof` 1.0002 object, `noise_geometry.tidmad.psd.welch_psd_from_windows`), estimated from **noise-only science files**, never evaluation data.
- `j_band` is computed in the data pipeline's own STFT units (median per-band periodogram power), so the whitened residual `|resid|²/J` is measured in the loss's coordinate system; the boxcar PSD density is returned alongside for the archived `χ²/dof` record.
- **Gates pass**: `J(f)` strictly positive; whitening identity error < 0.25 on independent noise (band 1..end; the DC band is degenerate for zero-mean calibration noise).

---

## Step 3 — Data path (GATE: round-trip + leakage) — **PASS**

Files touched: `tidmad/data.py`, `tidmad/tests/test_model.py`, `tidmad/tests/test_psd.py`.

- **Convention mirrors the upstream release exactly**: `channel0001` = SQUID readout (input), `channel0002` = injected reference (target), both read `int8` and shifted `+128` as upstream `train.py` does.
- STFT: non-centered, Hann window, 50% hop (COLA). **Round-trip test passes** (interior exact to < 1e-8); band-limited reconstruction tracked to < 2% interior error (window spectral leakage is inherent).
- Split protocol: Paper 1's chronological 70/30 split with a 4-window guard; **leakage test passes** (zero index overlap, zero window-hash overlap, realized guard gap = 4).
- Injection frequency per file is read from `tidmad_data_contract.json` (`frequency_for_file`), file- and sample-range-specific.

---

## Step 4 — `none` anchor reproduction (GATE: 1.1026 or documented equivalent, BLOCKING) — **BLOCKED: data not mounted on this machine**

The upstream `benchmark.py` exists at `/Users/wongdowling/Documents/DELight_ref/external/TIDMID/TIDMAD-main/benchmark.py` (unmodified, per the archived contract). The raw validation files it needs (`abra_validation_0000..0019.h5`, ~2 GB each, on `/ceph/dwong/paper1_dataset/tidmad/raw`) are **not present on this machine**; `/ceph` is not mounted.

**Documented equivalent already exists:** `transfer_paper/data/tidmad/current/G0_baseline_reproduction.json` records the reproduction on the training box (upstream `/home/dwong/src/TIDMAD`): unmodified `benchmark.py -c -m none` on the full 20-file validation set, returncode 0, stdout ending `Coarse Denoising Score for Model none: 1.1026161164402752` (published 1.10). savgol was deliberately not used as a cross-check (recorded as not bit-reproducible at `polyorder=11`).

**Mandatory precondition before any arm is scored** (per PLAN_03 §4 gate 1): re-run the anchor in the *scoring* environment once data is mounted:

```sh
cd /path/to/TIDMAD   # unmodified upstream checkout
<venv>/bin/python benchmark.py -d /path/to/raw -m none -c
# expected: Coarse Denoising Score for Model none: 1.1026...
```

`tidmad/score.py` is the wrapper that invokes this; it records the command, returncode, and raw stdout.

---

## Step 5 — Loss + measurement-coordinate handling (GATE: non-proportional loss, BLOCKING) — **PASS**

Files touched: `tidmad/loss.py`, `tidmad/tests/test_loss.py`.

The model consumes the standardised representation; the reconstruction head output is mapped back to measurement STFT coordinates with the per-band scale factors from `normalise_input_sequence`; both losses are evaluated on that measurement-coordinate residual.

**Non-proportional-loss gate output (captured 2026-08-08):**

```
residual in LOW-noise bands : mse=0.125000  chi2=0.087874  (chi2/mse=0.7)
residual in HIGH-noise bands: mse=0.125000  chi2=0.000189  (chi2/mse=0.0)
```

Same raw MSE for equal-power residuals; `chi2` differs by >450× across noise bands and ranks them opposite to MSE — the two objectives are genuinely different and would be proportional if the coordinate handling were wrong. The blocking test (`test_losses_not_proportional_nonflat_j`) passes, as does the flat-J sanity check.

---

## Step 6 — Train entry point + config-equality assertion (GATE: two configs differ only in the loss flag) — **PASS**

Files touched: `tidmad/train.py`, `tidmad/configs/*.yaml`, `tidmad/tests/test_config.py`.

- Minimal loop: `set_seed`, `TidmadTransformer`, Muon + AdamW (reused from `reconstruction_model`), cosine-with-warmup schedulers, gradient accumulation, `run_config.json` written in the `save_resolved_config` shape.
- `assert_configs_differ_only_in_loss` loads both arms' configs and **fails loudly unless the only difference is `train.loss`**. Gates pass: `t_mse`/`t_chi2`/`t_chi2_of` differ only in the loss flag; a deliberate seed drift is rejected.
- Training-loop smoke: two optimiser steps on a synthetic batch run end-to-end (losses finite).

---

## Step 7 — Infer / score / h5 writer (GATE: untrained model scores end-to-end) — **PARTIALLY BLOCKED (real-data scoring)**

Files touched: `tidmad/infer.py`, `tidmad/score.py`.

- `infer.py` streams each validation file in `window_samples` chunks, runs the model, inverse-STFTs, and writes `abra_validation_denoised_<arm>_00NN.h5` reusing Paper 1's `write_denoised_file` (release layout, source dtype preserved, reference channel copied through — benchmark reads both channels).
- `score.py` shells out to the **unmodified** upstream `benchmark.py -c` via Paper 1's `run_upstream_benchmark` and captures command/returncode/stdout.
- **End-to-end scoring on real data is blocked by the missing data** (same blocker as Step 4). The inference/score code paths are exercised by unit tests (h5 writer semantics and the benchmark wrapper are Paper 1's tested modules); the end-to-end run must happen on the training box once data is mounted and the anchor (Step 4) has reproduced.

---

## Step 8 — Freeze config (GATE: timestamped commit before any aggregate inspection) — **DEFERRED (see Step 0)**

The frozen values are fixed in code (`tidmad/config.py`, `FROZEN`), but the **timestamped freeze commit is deferred** because the migration commit is blocked on the parent being public. The frozen STFT parameters and split protocol are recorded here:

| Parameter | Frozen value | Rationale |
|---|---|---|
| `n_fft` / `hop_length` / window | 4096 / 2048 / Hann | Matches Paper 1's 4096-sample calibration grid; 50% hop satisfies COLA |
| Band range | bins 0..327 (`n_bands_used=328`) → 0–~800 kHz | Covers every characterised injection frequency (max sweep end ~760 kHz, file 0015) |
| Complex representation | real + imaginary stacked → `C = 656` bands | PLAN_04 §3.1 default; cross-band attention sees both |
| `window_samples` | 10,749,952 (~1.07 s) | Makes `M = 5248` frames, satisfying the backbone constraint `M//patch_len >= C` |
| `patch_len` / `M` | 8 / 5248 | Non-overlapping patches; `M % 8 == 0`, `M//8 = 656 >= C` |
| Model | d_model=128, d_ff=512, n_head=2, 1 temporal + 1 channel layer | Keeps the O(C³) cross-band attention feasible |
| Split | 70/30 chronological, 4-window guard | Paper 1 protocol; zero hash/index overlap audited |
| `J(f)` | median periodogram, 512× 4096-sample windows, noise-only science files | Paper 1 Phase-4 estimator, same cited object |

**Compute finding (recorded, not resolved):** the DELight backbone applies its learned `AbsolutePositionalEmbedding` over the **channel** axis with size `max_seq_len//patch_len`. For a band×frame representation this forces `M ≥ C·patch_len`, and full cross-band attention is O(C³). With `C = 656` bands the frozen model is ~611k parameters but attention-heavy; the frozen `device_batch_size = 1` reflects that. This coupling is a genuine property of the backbone that PLAN_04's §3.3 sketch did not flag; the manuscript's C4 description ("learned positional embeddings across bands") is unaffected.

---

## Test summary (all run on this machine, CPU)

```
$ .venv/bin/python -m pytest tidmad/tests/ -q
16 passed in 2.17s
```

Coverage: loss non-proportionality (BLOCKING), STFT round-trip, band-limited reconstruction, head shape, backbone parity, PSD positivity + whitening identity, colour-profile fidelity, band tiling, chronological-split leakage, frozen-config consistency, arm configs differ only in loss, chi2_of shares config, seed drift rejected, training-loop smoke, registry resolution.

---

## Deviations from the release's convention — none

The channel mapping (channel0001 input, channel0002 target), the `int8 + 128` shift, the denoised-file naming/layout, and scoring via the unmodified `benchmark.py` all follow the upstream release. The only deviation from *Plan 04's sketch* is documented above: `J(f)` is measured in data-STFT units (same estimator methodology) so the whitened residual is dimensionally correct, and the backbone's channel-pos-embed constraint drove the frozen window/frame sizes.

## Open items for the author / training box

1. **Make the parent repo private** (or explicitly authorise the curated commit), then re-run the Step 0 migration: add `artifacts/` and `training_data/` to the parent `.gitignore`, commit only the curated list, timestamped.
2. **Mount the TIDMAD data** and re-run the `none` anchor (Step 4) in the scoring environment before any arm is scored.
3. **Freeze commit** (Step 8) once the migration is unblocked; until then the frozen values live in `tidmad/config.py`.
4. Steps 9–10 (training) are out of scope for this patch and have not been started — nothing has been trained.

---

## Round-2 corrections (ROUND2_CORRECTION_2026-08-08)

### T2 — band positional embedding + scientific window — **done**

Files: `tidmad/model.py`, `tidmad/train.py`, `tidmad/config.py`, `tidmad/tests/test_model.py`, `tidmad/tests/test_config.py`.

- `TidmadTransformer` now owns a `band_pos_embd` dimensioned to the band count `C` (2 × `n_bands_used`), used for the cross-band stage; the backbone's patch-scaled `abs_pos_embd` is deleted. This decouples the frame count `M` from `C` (the round-1 finding that `M = C·patch_len` was forced by the backbone's `AbsolutePositionalEmbedding` sizing).
- The window is re-chosen on scientific grounds: **198 656 samples (~19.9 ms) → M = 96 frames** at n_fft=4096, hop=2048 — Paper 1's 4096-sample analysis window scaled to a sensible frame count, comparable to the upstream 20 000-sample windows. `device_batch_size` raised to 4 (the O(C³) cross-band cost is no longer aggravated by M).
- Parity test re-run and restructured: **the temporal stage matches `current_compact` exactly** (verified via a pre-hook on the band-embedding input); the band stage is deliberately divergent by design (band embedding sized to C). All 17 tests pass.
- The config docstring now records the scientific rationale and the corrected band-range statement (T4).

### T4 — config docstring contradiction — **done**

The module docstring claimed band range "[0, 1.25) MHz"; the field comment said "0..~800 kHz"; the code gives 328 × (10 MHz/4096) = 800.8 kHz. The docstring is corrected to "[0, 328) bins → 0..~800 kHz", matching the code and the field comment.

### T6 — cross-repository dependency declared — **done**

`tidmad/DEPENDENCIES.md` records the full `noise_geometry` import surface (`tidmad.{psd,loader,score,splits}`) and the decision: **vendor** the four modules into the D5 artifact's subtree (self-contained, same cited J(f) object); declare as a dependency only if vendoring is rejected.

### T7 — transformer repo tree cleaned — **done**

`git config core.filemode false` (suppresses the mount's 100755→100644 mode noise) and `git checkout -- src/finetuning_references/` (restores the six drop-list notebooks deleted in the working tree). `git status` now shows only `registry.py` (my +10-line registration) and the new `tidmad/`.

### T1 residual + migration + T8 — **prepared, not executed (decision points)**

- **Parent privacy:** re-verified — unauthenticated GitHub API now returns 404, consistent with the repo being made private. The migration gate is unblocked.
- **Parent `.gitignore`:** still missing `artifacts/` and `training_data/`. These two lines must be added before the first parent commit:
  ```
  artifacts/
  training_data/
  ```
  (Round-2 plan item 1 assigns the `.gitignore` extension and the audit of the previously-public window to the author.)
- **Curated file list** (keep-list + `tidmad/`), verified present, is ready to copy:
  ```
  reconstruction_model/__init__.py, model.py, muon.py, schedulers.py, checkpoints.py,
  utils.py, models/__init__.py, models/registry.py
  pyproject.toml, uv.lock, .python-version, .gitignore
  tidmad/ (config.py, data.py, psd.py, model.py, loss.py, train.py, infer.py, score.py,
           configs/*.yaml, DEPENDENCIES.md, tests/)
  ```
- **Not executed:** the migration's working-tree mutation (resolving the nested `transformer/.git` and the target directory path inside the parent) and the T8 freeze commit. The round-2 plan gates the first commit as "reviewed before pushing," and the target directory inside the parent is unspecified (the existing `transformer/` is the private provenance repo that must be kept as-is). **Author decision needed:** the target path in the parent and confirmation to proceed with the curated first commit.

### T3 / T5 — injection-frequency characterisation + `none` anchor — **blocked on data**

Both require the TIDMAD data (mounted at `/ceph/dwong/paper1_dataset/tidmad/raw` on the training box; not on this machine). `tidmad_data_contract.json` characterises only files 0000/0005/0010 (plus a sweep note for 0015); the other sixteen files' injection frequencies are unknown, which could invalidate the band-coverage assumption (`n_bands_used = 328` covers 0–800 kHz). This must be settled before the freeze: characterise all 20 files' injection frequencies (cheap FFT scan on the training box), then either confirm 800 kHz coverage, widen it, or drop the published-baseline comparison. The `none` anchor (1.1026, archived G0) must also be re-run in the scoring environment before any arm is scored. **Do not train until both are resolved.**
