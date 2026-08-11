# Round-2 correction plan — manuscript and TIDMAD patch

**Date:** 2026-08-08
**Reviews:** the TIDMAD patch (`plan/TIDMAD_PATCH_LOG.md`, `transformer/tidmad/`) and the current manuscript state.

---

## 0. Headline: the manuscript correction round never ran

**`MANUSCRIPT_CORRECTION_2026-08-08.md` has not been executed.** Verified:

| Check | Expected after the round | Actual |
|---|---|---|
| Section file timestamps | after 11:04 (when the plan was written) | all `02:35`–`02:45` |
| `references.bib` PENDING strings | 0 | **3** |
| Abstract placeholder/hedge | present | **0** |
| "All four rungs share…" claim in §3 | removed | **still present** |
| "Correction round" section in impl log | present | **absent** |

C1–C8 are all outstanding. Only the TIDMAD patch ran. **Dispatch Agent A — the prompt and task list are unchanged and still correct.** Nothing below supersedes them.

---

## 1. TIDMAD patch — assessment

The work is good and unusually disciplined. The package is 1,338 lines across a clean `tidmad/` sibling; 16 tests pass; both blocking gates are genuinely met.

**The loss-coordinate gate is correctly solved.** `loss.py` records the per-band statistics from the input, maps the head output back to measurement coordinates, and evaluates both objectives there. The test output is convincing: equal-power residuals give identical MSE (0.125) while `chi2` differs by >450× across noise bands and *ranks them opposite to MSE*. That is the trap avoided, not merely tested around.

**Discipline was maintained where it mattered.** Three gates were blocked and reported rather than worked around. `git diff --numstat` confirms `model.py`, `train.py` and `dataset.py` are content-unchanged (`0 0`); the 38 "modified" files are file-mode noise (`100755→100644`) from the mount. The only real edit to existing code is +10 lines registering `tidmad_stft` in `registry.py`, which was required.

Eight items follow. **T2 and T3 are the ones that matter.**

---

## T1 — Parent repository visibility — **RESOLVED 2026-08-08**

The agent verified via the GitHub API that `noise-weighted-subspace-reconstruction` was **public** (`private: False`) and correctly refused to commit. **The author has since made it private.** The migration gate is unblocked; proceed on path 1 (migrate into the now-private parent).

### Why this was blocking

Nothing had leaked. The `transformer` repository was already private, so the detector material in it had never been published. The risk was entirely **prospective**, and specific to the planned merge:

- `git add` of the transformer directory into a *public* parent would have published `src/scripts/TraceSimulator.py` — a collaboration simulator — along with the sensor-geometry caches and detector-coupled model variants.
- Git history is not undoable by deletion. A later "remove the DELight things" commit would have left every one of those files reachable in history, on a public remote, permanently.
- That is precisely the exposure the whole DELight → TIDMAD pivot exists to avoid, and in a worse form than the original concern: not *using* collaboration material, but *redistributing* it without clearance.
- The parent is also the natural candidate for this paper's public or anonymized code link (desk task D5). Detector material in its history would be both a clearance problem and a deanonymization vector — the simulator names the experiment outright.

### What remains, despite the fix

1. **Making a repository private does not retroactively unpublish it.** Audit what was in that repository during the period it was public, and how long that was. If detector-derived material was present, it may have been cloned or indexed, and that is a live question independent of today's change. This is a five-minute check and worth doing.
2. **The curation discipline in `PLAN_04` §4.3 still applies unchanged.** Privacy is reversible; history is not. The parent remains the likely source of the D5 public artifact, so anything committed there may still be published later. Curate the keep-list explicitly, file by file — never a directory add.
3. **`.gitignore` gaps are unaffected by visibility.** Add `artifacts/`, `training_data/`, `wandb/`, `*.pt`, `*.h5`, `*.npy`, `*.zst`, `cache/` before the first commit, or 2.6 GB of checkpoints will commit silently.
4. **The D5 artifact is still a fresh repository**, per `PLAN_04` §4.4 — not an export from the parent. A private parent does not change that.

---

## T2 — The window size is an architectural artefact, not a scientific choice

`window_samples = 10,749,952` (~1.07 s at 10 MHz) exists for exactly one reason: to satisfy `M ≥ C·patch_len = 656 × 8 = 5248`.

That constraint is a **latent defect in the DELight backbone**, correctly diagnosed in the log. `AbsolutePositionalEmbedding` is sized `max_seq_len // patch_len` — the *patch* axis — but `model.py:258` applies it to the *channel* axis. At `C = 56` and 1024 patches it silently worked. At `C = 656` it forces the window to inflate by a factor of ~2,600 over Paper 1's 4,096-sample protocol, and drives `device_batch_size = 1`.

**Fix the embedding, do not size the experiment around the bug.** Add a separate band positional embedding in `tidmad/model.py`, dimensioned to `C` rather than to `max_patches`:

```python
self.band_pos_embd = AbsolutePositionalEmbedding(n_bands_stacked, config.d_model)
```

and use it for the cross-band stage instead of the inherited `abs_pos_embd`. Roughly three lines. It touches only `tidmad/model.py`, leaving the backbone untouched, and it preserves the manuscript's C4 description ("learned positional embeddings across bands") exactly.

**Then re-choose the window on scientific grounds.** Candidate anchor: Paper 1's frozen TIDMAD protocol (4,096-sample windows, chronological split, guard gap), scaled to whatever gives a sensible frame count. State the reason in the config docstring. A window chosen because a positional embedding was mis-sized is not a predeclared choice, and a referee who reads the config will see that.

Re-run the backbone-parity test after the change: parity should still hold through `final_proj_norm` for the temporal stage, with the band stage now explicitly divergent by design.

---

## T3 — Band coverage may invalidate the benchmark comparison

`n_bands_used = 328` covers **0–800 kHz of a 0–5 MHz band** — about 16% of the spectrum. The config justifies this as covering "every characterised injection frequency (max sweep end ~760 kHz, file 0015)."

**"Characterised" is carrying too much weight.** `tidmad_data_contract.json` gives explicit injection frequencies for **three** of twenty validation files (0000, 0005, 0010) plus a sweep note for 0015. The other sixteen are uncharacterised. And the same contract reports the coupling scan running "from 7.13 at 5.7 kHz down to 0.0021 at **3.4 MHz**" — which means scanned files carry injections far above 800 kHz.

Two consequences:

1. For any validation file with an injection above 800 kHz, the model sees **no signal at all**. It cannot reconstruct what it never receives.
2. The published baselines (FCNet, Transformer, PUNet) denoise the full band. A band-limited model is **not performing the same task**, so Table 1's comparison against the published scoreboard — a headline element of rung-4's evidence — may not be a valid comparison.

The paired contrast between `T-mse` and `T-chi2` survives either way, since both arms are band-limited identically. It is the external comparison that is at risk.

**Required, before the freeze:**

1. **Characterise the injection frequency of all twenty validation files.** This is a cheap FFT scan and it settles the question. Extend `tidmad_data_contract.json` with the results and follow the contract's own warning about sample-range specificity.
2. Then choose deliberately:
   - If all twenty fall below 800 kHz — the concern evaporates; record the evidence and proceed.
   - If not — either widen coverage (a coarser `n_fft` covers the full band at lower frequency resolution; note the trade-off against resolving a narrow tone), or keep band-limiting and **drop the published-baseline column from Table 1**, presenting only the paired contrast.

Do not proceed to training until this is settled. Training two arms on a representation that cannot be compared to the scoreboard wastes the run.

---

## T4 — The frozen config contradicts itself

`config.py`'s module docstring states the band range is "**[0, 1.25) MHz**". The `n_bands_used` field comment states "**0..~800 kHz**". The code gives 328 × (10 MHz / 4096) = 800.8 kHz.

The docstring is wrong. In a file whose entire purpose is to be the predeclared record, an internal contradiction is a real defect. Fix it as part of T3's resolution so the config is corrected once.

---

## T5 — The `none` anchor is still unreproduced

Correctly deferred: `/ceph` is unmounted on the working machine, and the archived `G0_baseline_reproduction.json` records 1.1026 from the training box.

**This stays blocking.** Re-run `benchmark.py -c -m none` in the *training* environment before any arm is scored. If it does not reproduce there, stop — the environment or channel semantics differ from Paper 1's and every downstream number is suspect.

---

## T6 — Cross-repository dependency needs declaring

`tidmad/score.py` imports `noise_geometry.tidmad.score`, and `psd.py` similarly depends on Paper 1's estimator. Correct — reuse was the instruction — but it means `tidmad/` is not self-contained.

For the working setup this is fine. For the **D5 anonymized artifact** it is not: the published subtree must either vendor the required `noise_geometry` modules or declare the dependency explicitly. Record which, now, while the import surface is small.

---

## T7 — Clean the mode-change noise before migrating

38 files show as modified; almost all are `100755 → 100644` mode flips from the mount, with zero content change. Six third-party `finetuning_references` notebooks were deleted.

Left alone, this noise pollutes the first commit into the parent and makes the curated diff unreadable. **Before migrating:** `git checkout -- .` in the transformer repo to discard mode noise and restore the deleted notebooks, so the provenance record is pristine, then copy the keep-list out of a clean tree.

(The deletions were on the drop list anyway, so nothing is lost either way — but the instruction was to leave that repository untouched as the provenance record, and it is cheap to honour.)

---

## T8 — Freeze commit

Correctly deferred behind the migration. Once T1–T4 are resolved, freeze in a timestamped commit **before** any aggregate is inspected, per `PLAN_04` step 8.

---

## Execution order

**Agent A — dispatch now, independent of everything below.** Run `MANUSCRIPT_CORRECTION_2026-08-08.md` as originally written. It has not started.

**Agent B / author — in this order:**

| # | Item | Owner | Gate |
|---|---|---|---|
| — | ~~**T1** — repository visibility~~ | — | **Resolved: parent is now private. Migration unblocked.** |
| 1 | **T1 residual** — audit the previously-public window; extend the parent `.gitignore` | **Author** | Audit recorded; `.gitignore` covers all eight patterns |
| 2 | **T3** — characterise all 20 validation files' injection frequencies | Agent | Contract extended; band-coverage decision recorded with evidence |
| 3 | **T2** — add the band positional embedding; re-choose the window scientifically | Agent | Parity test re-run; window justified in the config docstring |
| 4 | **T4** — reconcile the config docstring | Agent | No internal contradiction |
| 5 | **T7** — restore the transformer repo to a clean tree | Agent | `git status` clean |
| 6 | **Migrate** the curated keep-list into the parent | Agent | Explicit file list; no directory add; first commit reviewed before pushing |
| 7 | **T6** — declare or vendor the `noise_geometry` dependency | Agent | Import surface documented |
| 8 | **T8** — freeze commit | Agent | Freeze precedes any aggregate inspection |
| 9 | **T5** — reproduce the `none` anchor in the training environment | Agent | 1.1026 or documented equivalent |

Steps 2 and 3 are the substantive ones. **Do not train until step 9 passes.**

If T3 shows injections above 800 kHz and widening coverage proves infeasible, that is a scope decision for the author, not a workaround for the agent: the paper would then present the paired contrast without the published-baseline comparison. Say so in the log rather than proceeding on an assumption.
