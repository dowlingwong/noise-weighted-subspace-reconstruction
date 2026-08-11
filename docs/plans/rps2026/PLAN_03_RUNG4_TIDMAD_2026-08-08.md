# Track C — rung 4 on TIDMAD

**Date:** 2026-08-08
**Supersedes:** `PLAN_03_DELIGHT_INTEGRATION_2026-08-08.md` — **that document is void. No DELight data, simulation, or physics appears anywhere in this paper.**
**Governing document:** `PLAN_00_MASTER_RESTRUCTURE_2026-08-08.md`
**Dataset:** TIDMAD (ABRACADABRA-10cm), public, CC-BY-4.0, `fry2024tidmad` — already cited in `references.bib`.

---

## 0. Why TIDMAD, in one paragraph

Paper 1 already ran a noise-aware linear estimator through TIDMAD's own unmodified benchmark script and scored **−1.8025**, against no-denoising at 1.10 and every published deep baseline. Its own archived record states the claim boundary: *"This measures how much of the gap between no denoising and a trained deep denoiser is explained by using the correct noise geometry with a linear model. It is not a claim to win the benchmark."* Rung 4 answers the question that record leaves open — **does the likelihood geometry still matter once the architecture is expressive enough to compete?** — on public data, with published baselines, against a real injected reference channel.

Everything needed is already in place: the data, the contract, the frozen configs, the reproduced calibration anchor, and the upstream scoring script.

---

## 1. What is already established (use it, do not redo it)

All from `transfer_paper/data/tidmad/current/`. **These are archived facts. Cite them; do not re-derive.**

**Data contract** (`tidmad_data_contract.json`, `verified: true`):

- `timeseries/channel0001/timeseries` = SQUID readout; `timeseries/channel0002/timeseries` = injected reference/target. This is the release's own hardcoded convention across `train.py`, `inference.py`, `benchmark.py`.
- Sampling 10 MHz, `int8`, 20 validation files of roughly 200 s each.
- **The injected tone sweeps within each file.** File 0010 sweeps ~75 000 → ~89 000 Hz over 190 s; file 0015 sweeps ~620 000 → ~760 000 Hz. Injection frequency is file- *and* sample-range-specific.
- File 0000 carries a genuine **two-tone** structure at ~69 510 / ~69 630 Hz, comparable power, confirmed across the full file.
- Reference-to-SQUID coupling falls steeply with frequency: 7.13 at 5.7 kHz down to 0.0021 at 3.4 MHz — a low-pass-like transfer function, a genuine property of the release.

**Published coarse denoising scores** (higher is better), plus your own:

| Method | Coarse score | Source |
|---|---:|---|
| FCNet | 6.55 | published |
| **Transformer** | **4.18** | published |
| PUNet | 3.84 | published |
| none | 1.10 (reproduced 1.1026) | published / G0 anchor |
| moving average | 0.64 | published |
| Paper 1 linear Wiener | **−1.8025** | archived, Phase 5 |
| savgol | −2.35 | published, **not bit-reproducible** — ill-conditioned at `polyorder=11`; do not use as a comparison point |

**Phase 4 — this is the load-bearing justification for the STFT choice.** A global stationary PSD model is empirically *adequate* for real TIDMAD SQUID noise: `χ²/dof = 1.0002`, score dispersion 1.0142 (`z = 0.64`), σ-inflation from model mismatch 1.0002, all gates passed. Unlike GWOSC, where the same test failed. **Therefore treating `Σ` as diagonal in frequency is justified on this dataset by measurement, not assumption.** Say this in the paper — it is the difference between a modelling convenience and an empirical result.

**Phase 2 — read the caveat carefully.** The headline metric-reversal result (amplitude RMSE 0.1209 → 0.0880, 27.2 %, paired CI [0.0045, 0.0321]) is from `TIDMAD_P2_real`. The **confirmatory** replication on file 0005 returned a null: `all_gates_passed: false`, truth-favours-EMPCA 0.0402 with CI [−0.0793, 0.1779]. Direction was correct, significance was not. **Do not describe TIDMAD rung-2 evidence as replicated.** The manuscript's current wording is accurate; keep it that way.

**Phase 3 — this is the motivation for rung 4, and it is a gift.** The analytic rank-2 prediction failed on all three real files, for diagnosed non-implementation reasons: a two-tone reference structure "a richer manifold than the single-frequency model assumes," weak reference-to-readout coupling, and insufficient power at 150 evaluation windows.

**Write this as the paper's transition into rung 4.** The linear rungs assume a fixed, low-dimensional signal subspace. TIDMAD's own diagnostics show the real signal is not that — it sweeps in frequency and is sometimes multi-tone. So the architecture constraint, not the metric, is what binds. Rung 4 lifts the architecture constraint and asks whether the metric still matters. **That transition is grounded entirely in your own archived evidence, which is the strongest possible way to motivate an experiment.**

---

## 2. The representation — STFT band × frame

Take the STFT of the readout channel, giving `Z ∈ C^{F×M}` over `F` frequency bins and `M` time frames. Whitening divides each bin by `sqrt(PSD(f))`, which is exactly `Σ^{-1/2}` when `Σ` is diagonal in frequency — justified by Phase 4 above.

This makes the two axes genuine:

| | Rung 3 (NFPA) | Rung 4 (attention) |
|---|---|---|
| axis 1 | frequency-subspace `U_f ∈ R^{F×k_f}` | cross-band attention |
| axis 2 | frame-subspace `U_m ∈ R^{M×k_m}` | intra-band temporal attention over frames |
| restriction | `range(U_f) ⊗ range(U_m)` | factored, learned, nonlinear |

**This is a better instantiation than a sensor axis would have been**, because the metric and the factorization now live in the same coordinates: `Σ⁻¹` is diagonal in exactly the basis the first factor indexes. Make that point explicitly — it is the cleanest statement of the paper's thesis anywhere in the draft.

It also predicts why the linear rung struggles: a sweeping tone is a *slanted ridge* in the spectrogram. A fixed rank-`k` subspace cannot represent it; a band×frame model can.

**Design notes for the port:**

- The DELight pairwise/edge-feature machinery encoded 3D sensor geometry. Frequency bins have a natural 1D ordering, so **RoPE on both axes replaces it.** Drop the pairwise-update block. This substantially simplifies the port — probably the single biggest time saving available.
- Choose `n_fft`/hop so that `F` and `M` are both large enough for two meaningful attention stages, and record the choice in the config before any run.
- Complex spectrogram: use real/imaginary as two feature planes, or magnitude+phase. Predeclare which; do not switch after seeing results.

---

## 2A. Port audit — completed 2026-08-08

The transformer repository has been read. **The port is much smaller than assumed.** Findings below are authoritative; do not re-derive them.

### 2A.1 Use `current_compact` (`reconstruction_model/model.py`), not the pairwise variants

`models/registry.py` lists seven variants. **Five** carry `requires_pairwise_cache=True` (`original`, `pairwise`, `pairwise_channel_masking`, `triangular_pairwise`, `cnn_transformer`) and depend on `models/cache/edge_feats.npy` and `pos_diff.npy` — precomputed **3D sensor geometry**. Those are unusable off-detector and must not be ported. A sixth, `integration_classifier`, is an XGBoost baseline rather than a transformer.

`current_compact` has `requires_pairwise_cache=False` and **no geometry dependency at all**. Its backbone is fully generic:

```
(B, C, T)
  → per-channel z-score normalise      # normalise_input_sequence, model.py:28
  → unfold into patches (patch_len=64) # patch_input_sequence, model.py:34
  → linear patch embedding             # model.py:157
  → temporal blocks, RoPE              # model.py:158, use_rope=True
  → reshape (B*N, C, d_model)          # model.py:257
  → absolute positional embedding over channels
  → spatial (cross-channel) blocks     # model.py:164, use_rope=False
  → RMSNorm → mean-pool over (patches, channels)
  → spatial_head (3), energy_head (1)
```

Blocks are pre-norm RMSNorm + MHA + SwiGLU-style FFN, muP spectral init (`model.py:182`), Muon for matrix params and AdamW for the rest (`model.py:193`).

**Nothing in the backbone knows it is looking at a detector.** `(B, C, T)` with `C` = frequency bands and `T` = frames works as-is.

This choice has a useful side effect: the pairwise/triangular machinery is where the prior work's more elaborate contribution lives. Dropping it is technically necessary *and* further separates this port from that contribution — which is consistent with the attribution position in §5.

### 2A.2 What is DELight-coupled, and what to do about it

| Item | Coupling | Action |
|---|---|---|
| `spatial_head` (3), `energy_head` (1) | task-specific | **Replace** with a reconstruction head, §2A.3 |
| `dataset.py` | `.zst` traces, `.h5` metadata, `ER`/`NR` dirs, energy bins, `max_seq_len=65536` | **Replace** with a TIDMAD STFT dataset |
| `models/cache/*.npy` | 3D sensor geometry | Not used by `current_compact`. Ignore. |
| `src/scripts/TraceSimulator.py` | **DELight simulator** | **Do not use. Do not ship.** See §2A.5 |
| `artifacts/` (2.6 GB) | checkpoints trained on DELight data | **Do not ship.** See §2A.5 |

### 2A.3 What must be built — this is the whole job

1. **Reconstruction head.** Replace the mean-pool and two scalar heads with a per-patch projection `(B,C,N,d_model) → (B,C,N,P)`, folded back to `(B,C,T)`. Standard PatchTST-style decoder, roughly 30 lines. This is the only architectural change.
2. **TIDMAD STFT dataset.** STFT of `channel0001`, target per the release convention, windowed under Paper 1's frozen protocol (4096-sample windows, chronological split, guard gap).
3. **The weighted loss.** See §2A.4.
4. **Inverse STFT and h5 writer** in the release's format so `benchmark.py` can score the output. **Paper 1 already produced `abra_validation_denoised_wiener_0000.h5`, so this writer exists** — reuse it rather than rewriting.

**Config-only changes:** `max_seq_len` → number of frames, `patch_len` → frame patch size, `C` → number of bands. `max_patches = max_seq_len // patch_len` drives the RoPE cache and follows automatically.

### 2A.4 The metric intervention — read this twice

**The intervention is in the loss, and only in the loss.**

```python
# T-mse    : ((z_hat - z) ** 2).mean()
# T-chi2   : (((z_hat - z) ** 2) / J_f).mean()      # J_f = noise PSD per band
```

`train.py` uses `F.mse_loss` at lines 505, 635 and 640; that is the intervention site.

> **Do not whiten the input.** If the input is whitened, MSE in whitened coordinates *is already* `χ²`, the two arms become mathematically identical, and the experiment is destroyed. Leave `normalise_input_sequence` (per-channel z-score, `model.py:28`) unchanged and identical in both arms.

Both arms must share: architecture, initialisation seed, data, batch schedule, optimiser settings, step count, and input normalisation. **The loss is the only difference.** Assert this in code — compare the two run configs field by field and fail if anything other than the loss flag differs.

### 2A.5 Repository hygiene — a real risk

Task D5 proposes an anonymized code link. If that repository is published, `src/scripts/TraceSimulator.py` and the 2.6 GB `artifacts/` tree must not be in it. Both are DELight property, and shipping either reintroduces exactly the exposure this whole redesign exists to avoid.

**Build the anonymized repo from a clean subtree** — the ported backbone, the TIDMAD dataset, the loss, the configs, the scoring wrapper — not by pruning a copy of this one. `.gitignore` is not sufficient protection when the files are already tracked; check `git log --all --name-only` before publishing.

### 2A.6 Compute

`artifacts/` shows completed runs at 150 000 steps on L40S, and `submit_ref/` holds working HTCondor templates. Infrastructure is proven. Two training runs is not a compute problem; the data path is the schedule risk.

---

## 3. The experiments

### 3.1 Primary contrast — the training metric (this is the paper's claim)

**Same architecture, same data, same everything. Only the training objective differs.**

| Arm | Objective |
|---|---|
| `T-mse` | raw MSE on the unwhitened spectrogram |
| `T-chi2` | `Σ⁻¹`-weighted residual — MSE on the whitened spectrogram |

This is the direct nonlinear analogue of NFPA vs IsoAE at rung 3, and of CW-PCA vs PCA at rung 2. **It is the experiment the paper exists to run, one rung higher than it has ever been run.**

Predicted signature, consistent with rungs 2–3: `T-mse` wins on raw MSE, `T-chi2` wins on the whitened residual and on truth-referenced amplitude error against `channel0002`. Report both metrics for both arms, always.

### 3.2 Secondary contrast — likelihood-feature injection

| Arm | Description |
|---|---|
| `T-chi2+OF` | `T-chi2` plus per-band/frame OF amplitude `â` and `χ²` at `â` injected into the forward pass |

This is the design demonstrated in prior work (see §5). It is **secondary**, reported as an additional data point, and attributed. Run it only after 3.1 is complete.

### 3.3 Benchmark comparison

Score every arm through TIDMAD's **unmodified** `benchmark.py -c`, exactly as Phase 5 did, so results are directly comparable to the published table. The `none` anchor at 1.1026 must reproduce in your environment before any arm is scored.

**Do not tune toward the leaderboard.** The claim is the paired contrast in 3.1. If `T-chi2` lands between the published Transformer (4.18) and FCNet (6.55), that is a complete and honest result. If it lands below 4.18, report that too — the contrast in 3.1 is still the finding, and a rung-4 arm that underperforms the published transformer while still showing the metric effect is scientifically informative and referees will respect it.

### 3.4 Statistics

Paper 1's TIDMAD protocol is already predeclared and should be reused wherever it applies: chronological fit/eval splits with a guard gap, block bootstrap at block size 16, 300 resamples, explicit window-hash and window-index overlap audits reported as zero. **Reuse it. Do not invent a new protocol.**

Report the paired contrast between arms, not just per-arm scores.

---

## 4. Acceptance gates

1. The `none` anchor reproduces at 1.1026 (or documented equivalent) in the training environment before any arm is scored.
2. Leakage audit reports zero window-hash and zero window-index overlap between fit and evaluation, as Phase 2 did.
3. Both arms of 3.1 trained with identical data, identical architecture, identical budget, identical seed handling. **Only the objective differs.** Any other difference invalidates the contrast.
4. Both raw-MSE and whitened-residual metrics reported for both arms.
5. Every score produced by the unmodified upstream script.
6. Injection frequency and sample range taken from `tidmad_data_contract.json` per file. **Never reuse a frequency across files or sample ranges** — the contract flags this as a real bug risk that has already bitten once.

**If the primary contrast in 3.1 comes back null,** report it. A null at rung 4 with confirmed effects at rungs 2–3 is a genuine, publishable finding: it would say that sufficient architectural capacity can absorb a misspecified metric. That is interesting and it bounds the paper's claim honestly. Do not retune to rescue it.

---

## 5. Attribution — the condition on which this plan depends

The decision is **acknowledgement**, not co-authorship. That is defensible, but only if the paper holds to all three of the following. Track C is responsible for enforcing them.

1. **The paper must not claim the architecture or the OF-injection design as a contribution.** Rung 4's contribution is the *metric contrast* (§3.1) and its place in the ladder. The architecture is a vehicle.
2. **The prior work must be cited where the design is described**, not only acknowledged. Acknowledgement plus citation is a defensible position; acknowledgement while presenting the design as new is not.
3. **The `T-chi2+OF` arm (§3.2), which is closest to the prior design, must be explicitly attributed in the text** where it is introduced.

Suggested wording for §3, to be refined: *"Feature injection of the optimal-filter statistic into an attention model follows a design demonstrated in prior work on calorimetric event reconstruction [cite]; here it serves as a secondary contrast."*

**Practical note.** The relevant prior work is an unpublished MSci thesis, which is awkward to cite but not impossible — thesis citations are standard. If it is uncomfortable to cite, that discomfort is itself a signal that the contribution is doing more work in the paper than acknowledgement covers. In that case reconsider co-authorship: with no DELight data shown there is nothing to clear, and physics is a small field.

Anonymity: acknowledgements go in the **camera-ready only**. The double-blind submission carries neither names nor thanks.

---

## 5A. Limitations — state all five

*(Added 2026-08-08. The rewrite from the DELight plan lost this section; the manuscript agent reconstructed four of these correctly from context. Item 5 was missed.)*

Write these into §5. Omitting any is the fastest way to lose a careful referee.

1. **Two runs on one public dataset.** The claim is the paired contrast between arms, not the leaderboard rank.
2. **A null contrast would itself be a finding** — sufficient architectural capacity can absorb a misspecified metric — and must be reported as such rather than retuned away.
3. **The linear-rung TIDMAD metric reversal is not replicated.** The confirmatory file returned the correct direction with an interval including zero (`all_gates_passed: false`, CI [−0.0793, 0.1779]). Do not describe it as replicated.
4. **Stationary-PSD adequacy is per-instrument, measured not assumed.** Adequate on the TIDMAD readout (`χ²/dof = 1.0002`); fails its calibration check on gravitational-wave strain. This is the synthesis sentence that ties three studies together.
5. **Single training run per arm — no training-seed variance.** Block-bootstrap intervals cover evaluation-window sampling only. If compute allows a second seed per arm, run it; if not, state plainly that seed variance is not characterized.

---

## 6. Scope and schedule

**Do not chase the leaderboard.** Two training runs for §3.1 is the deliverable. §3.2 is a third run if time allows.

| Dates | Work |
|---|---|
| 8–9 Aug | Reconstruction head on `current_compact` (§2A.3 item 1). Smoke-test via `scripts/smoke_test_training.py` on synthetic `(B, F, M)` input. |
| 10–12 Aug | TIDMAD STFT dataset. Reproduce the `none` anchor at 1.1026. Reuse Paper 1's h5 writer for the scoring path. Predeclare `n_fft`, hop, complex representation, split protocol. |
| 13 Aug | Weighted loss + the config-equality assertion of §2A.4. |
| 14 Aug | **Freeze config.** Timestamped commit before any aggregate is viewed. |
| 15–19 Aug | Train `T-mse` and `T-chi2`. Score both through the unmodified script. |
| 20–21 Aug | `T-chi2+OF` if on schedule. Paired statistics. Figure. |
| 22–23 Aug | Draft rung 4 of §3 and the §4 evidence paragraph. |
| 25 Aug | **Freeze.** |

The port audit is already done (§2A), so the schedule starts at implementation rather than investigation.

**Slack: two days.** If the port is not training by 15 Aug, drop §3.2 entirely and run only the primary contrast. If it is not training by 18 Aug, escalate — at that point the fallback is to present rung 4 as a structural observation only (factored attention as the nonlinear counterpart of the Kronecker restriction) with no new results, which still improves the paper's framing.

---

## 7. Deliverables

1. The ported `current_compact` backbone with a reconstruction head, in a **clean subtree** (§2A.5).
2. STFT data path, in the existing TIDMAD pipeline, with the frozen config.
3. Archived per-arm results and paired contrasts, seed-level, written before aggregation.
4. Rung 4 content in `sections/03_architecture_ladder.tex`.
5. §4 evidence paragraph and table, generated by script.
6. `plan/RUNG4_LOG.md` — anchor reproduction, leakage audit, frozen-config commit and timestamp, every score with the command that produced it, and a plain statement of what the result does and does not support.

**New bibliography entries** — verify each against arXiv or the publisher before adding; `fry2024tidmad` is already present:

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- Su et al., *RoFormer* (RoPE)
- Nie et al., *A Time Series is Worth 64 Words* (PatchTST)
- Gatti & Manfredi, Riv. Nuovo Cimento 9, 1 (1986) — the optimal-filter construction
- The prior-work thesis, per §5

Roughly five new references. If architecture internals push past that, describe the model as "intra-band temporal attention followed by cross-band attention" and cite only the load-bearing works.

**Do not mark complete** if any arm differs from another in more than the objective, if any score bypassed the upstream script, or if §5's three attribution conditions are not all satisfied in the text.
