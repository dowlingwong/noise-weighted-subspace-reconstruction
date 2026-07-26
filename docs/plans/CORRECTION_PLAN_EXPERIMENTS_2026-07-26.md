# Experiment-Repo Correction Plan (Agent Roadmap)

Date: 2026-07-26. Audience: an autonomous agent (or contributor) executing
corrections in `noise-weighted-subspace-reconstruction`. Derived from the
independent audit (`docs/EVALUATION.md`). Execute work
packages in order; each has scope, steps, and an acceptance gate. Never
overwrite frozen artifacts — corrections always create new, versioned runs.

Global evidence rules (apply to every WP):

- Frozen artifacts under `results/metrics/`, `results/sweeps/`,
  `evidence/paper_bundle/*`, and `transfer_paper/data/*` are immutable.
- Every new run writes: config, seeds, git commit, environment record, JSON
  metrics, and a SHA-256 manifest entry.
- A number may enter a paper only via an archived JSON referenced in the
  evidence map; "x times" wording only from a `ratio` block.

---

## WP-E1 (stop-ship): unify the two competing S5 30-seed artifacts

Problem. Two different "30-seed held-out S5" runs exist:

- Run A — seeds 0–29, archived only in the *latex* repo
  (`arXiv-.../transfer_paper/data/revision_20260726/s5/S5_sweep_30seeds.json`);
  diff 0.5405, CI [0.523, 0.560]; this is what the current PDF prints.
  Its seed range overlaps the frozen 0–9 block, so it violates this repo's own
  predeclaration ("seeds disjoint from the frozen block", see
  `scripts/run_s5_confirmatory_sweep.py`).
- Run B — seeds 1000–1029 (predeclared, disjoint), archived here under
  `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/`;
  diff 0.5503, CI [0.530, 0.571].

Steps:

1. Declare Run B (seeds 1000–1029) the primary confirmatory artifact — it is
   the one satisfying the predeclaration. Record this decision in
   `docs/CURRENT_STATUS.md` and in a short `docs/S5_EVIDENCE_DECISION.md`
   naming both runs, their checksums, and why B is primary.
2. Keep Run A archived, relabeled as a non-predeclared sensitivity replicate
   (its agreement with Run B in sign and magnitude is itself useful evidence).
3. Copy Run B's JSON/CSV into the latex repo's
   `transfer_paper/data/revision_20260726/s5/` under a distinct filename and
   update that repo's manifest (see the latex-repo plan, WP-P1).
4. Verify no other paper-facing S5 number is sourced from Run A
   (grep the latex repo for `48.418`, `47.878`, `0.540`, `2367.99`,
   `5388.75`, `86.07`).

Acceptance gate: one primary S5 run ID appears in `CURRENT_STATUS.md`, the
evidence manifest, and the manuscript; the other is labeled sensitivity;
checksums verified by `scripts/verify_paper_bundle.py`.

## WP-E2: fix the `OptimumFilter.sliding_fit` latent crash

Problem. In `src/canonical/OptimumFilter.py`, the `chisq_mode='none'` branch
calls `_slide_block_hop1_amponly` with 12 arguments against a 13-parameter
numba kernel (`self._F_imag` is missing after `self._F_real`). First call
raises a numba `TypingError`. No test covers this path.

Steps:

1. Insert `self._F_imag` into the call (line ~295), matching the kernel
   signature `(x, fs, F_re, F_im, E_re, E_im, X, start_sample, steps, N,
   scale, amps, out_offset)`.
2. Add a regression test: `sliding_fit(trace, hop=1, chisq_mode='none')`
   must (a) run, (b) return amplitudes equal to the `chisq_mode='all'` path
   within 1e-10, with NaN chisqs.
3. While there: add a test pinning `fit()` against `sliding_fit()[0]` on a
   window-aligned trace, and document that `fastmath=True` may cost a few ULPs
   versus the non-numba path (fine for amplitudes; keep tolerance-based tests).

Acceptance gate: new tests pass; no existing test changes.

## WP-E3: provenance hygiene for the v1 in-sample synthetic configs

Problem. `results/metrics/S4/S5/S6_*_seed*.json` (v1) are in-sample
(`test_frac=0`); held-out claims must never cite them. This is already
documented but not mechanically enforced.

Steps:

1. Add a `heldout: false` marker (or a sidecar deprecation note) to the v1
   README entries in `results/README.md`; do not modify the frozen JSONs.
2. Make `scripts/make_tables.py` / `make_all_figures.py` refuse (or loudly
   warn) when asked to render S4/S5/S6 values from a record whose config has
   `test_frac == 0`, unless an `--allow-insample` flag is passed.
3. Emit a single consolidated `results/EVIDENCE_MANIFEST.json` mapping every
   paper-facing table/figure to run IDs + checksums (the audit found evidence
   scattered across sweeps, standalone configs, and bundles).

Acceptance gate: manifest exists; table/figure scripts source only manifest
entries; CI or a test asserts the guard.

## WP-E4: Study B unblock (conditional sensitivity attribution)

Problem. Study B is implemented and cross-checked (<0.02% vs independent OF
variance) but the paper reports no numbers because template and acquisition
metadata are unresolved.

Steps:

1. Obtain and record: signal template provenance/calibration for the
   `signal.dat` pulse, the acquisition anti-alias response (or formally declare
   the two bounding branches as the only available assumption set), and the
   analyzed frequency support.
2. Freeze a Study B config (components, branches, scaling grid 1×/0.5×/0.2×,
   leave-one-out) and register a run ID before running.
3. Run both acquisition branches; archive per-branch JSON with the manifest;
   every reported number carries its branch label.
4. Update `docs/CURRENT_STATUS.md` and hand the run ID to the latex plan.

Acceptance gate: an accepted Study B run ID exists with branch-labeled
outputs, or Study B is formally removed from Paper 1 scope. No middle state.

## WP-E5: CRESST follow-up campaign (next replication, not a blocker)

These strengthen external validity; none block the current paper because the
manuscript already scopes its claims correctly.

1. Shared cross-stream cutoff: rerun the split with one common acquisition
   timestamp cutoff per channel (removes the 9 evaluation-pulses-precede-
   calibration-noise wrinkle); report alongside the frozen result.
2. Split-fraction sensitivity: 60/40 and 80/20 variants, same pipeline.
3. Trigger efficiency redesign: predeclare an amplitude grid bracketing the
   5%-FP threshold and 90% efficiency per channel (the frozen grid was
   underpowered); separate signal-injection and background protocol, frozen
   before evaluation.
4. If more public channels/runs become available, extend beyond 7 clusters
   and promote the hierarchical channel-then-event interval to primary.

Acceptance gate: each variant is a new predeclared config + archived run;
frozen results untouched.

## WP-E6: noise-module refinements

1. Replace `CalibrationPreset.sample_parameters()` (independent per-bin
   uniform log-PSD draws → jagged spectra) with one of: whole-spectrum
   bootstrap draws, a smooth log-spectral GP, or a low-dimensional component-
   parameter posterior. Preserve frequency correlation; for multichannel,
   preserve PSD-consistent CSD structure.
2. `calibration.py` gates: add chronological / acquisition-group splitting and
   block bootstrap options; keep the current random-record split as a labeled
   legacy mode; predeclare gate thresholds from downstream tolerance instead
   of the current 50% coverage / 0.75 CSD heuristics.
3. Fix the documentation mismatch (docs say custom mode `multiply`, config
   uses `scale`).
4. Add a test asserting `normalize_channel_variance=False` in every
   likelihood-calibration example/tutorial path (it breaks Gaussianity).
5. Optional consolidation: route the S-suite's `generate_colored_noise`
   through `noise_module` so one convention module serves both (currently the
   S-suite has its own generator in `src/noise_geometry/noise/`; they agree,
   but one source of truth is better).

Acceptance gate: module test suite green; new sampler validated by a
frequency-correlation check on drawn spectra.

## WP-E7: Paper 2 experiment repairs + noise-module wiring

Repairs (blocking any Paper 2 inference; from `paper2/`):

1. Seed Python/NumPy/Torch (+CUDA) before model construction in
   `paper2/trainers/train_reconstruction.py`; seeded loader generators;
   record split indices and seed state in each result folder.
2. Test-set discipline: save best-validation checkpoint, restore, evaluate
   test exactly once; keep `best_validation` and `final_test` as separate
   dictionaries.
3. Fix the timing metric coordinate mismatch in `paper2/analysis/reporting.py`
   (template origin, sign, units); add injected-shift unit tests.
4. Transformer: add temporal positional encoding, channel identity/geometry
   encoding, and an explicit cross-channel attention scheme; predeclare the
   ablation of each.
5. Match parameter count, compute budget, bottleneck/rank, and sample counts
   across architecture comparisons; report failed seeds.

Noise-module wiring (the new experiments, per the Paper 2 integration plan in
`paper2/PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md`):

6. NFPA separability sweep v2: generate multichannel noise from explicit
   Kronecker-separable CSDs, then interpolate toward non-separable targets
   (target-CSD path) with a declared separability index; ≥20 seeds.
7. Noise-mismatch axis: train under PSD/CSD Σ_train, evaluate under perturbed
   Σ_eval (scale, tilt, added lines, CSD-rank change) on a log severity grid.
8. Coverage experiment (§8.1 of Paper 2) with generator latents
   (A, t0, p, γ) and equal-n restricted-support controls.
9. Coverage-corrected training experiment (§8.2) with controlled Stage-I
   degradation and colored MC-SURE Stage II.

Acceptance gate: reruns from a frozen protocol with ≥20 seeds; current
exploratory result folders remain labeled exploratory and uncited.

---

## Suggested execution order

WP-E1 and WP-E2 first (small, stop-ship / correctness). Then WP-E3 (hygiene),
WP-E4 (unblocks the last Paper 1 appendix), WP-E6 (module refinements), and
WP-E7 (Paper 2 program). WP-E5 is a follow-up campaign and can run whenever
compute/data allow.
