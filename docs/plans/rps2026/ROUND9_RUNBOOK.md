# Round 9 runbook — the submission-level evidence the external audit requires

**Date:** 2026-08-19. **Deadline:** 29 Aug AoE. **Standing rules:**
`AGENT_PROMPT_REMOTE_EXPERIMENTS.md` applies in full.
**Relation to Round 8:** Task B (E7 second attention seed) continues unchanged.
This runbook adds the CPU-side experiments; all are independent of E7.

Priority order if time runs short: **E8 > E9 > E10**. E8 closes a predeclared
hypothesis the paper now explicitly lists as untested; E9 and E10 strengthen
but do not unblock.

## Step 0

```bash
export CODE_ROOT=~/noise-weighted-subspace-reconstruction
cd $CODE_ROOT && git pull && export PYTHONPATH=src:experiments/synthetic
export PYTHON=$CODE_ROOT/.venv/bin/python
export RPS_RESULTS=$CODE_ROOT/results/rps2026
# Also sync back, if not already: results/rps2026/E6_whitening_range.json
# and verify its excluding-DC cv value matches the paper's 0.69.
```

All E8–E10 are CPU-only; use `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 nice -n 10`
with an explicit `timeout` on each command.

## E8 — the predeclared H3 semantic evaluation (~30 min)

**Why:** the confirmatory protocol predeclared H3 (planted channel/temporal
subspace recovery) and a truth-referenced amplitude decoding; the executed
runner never invoked `semantic.py`, and the paper's appendix now says so
plainly. This run closes that gap with the code that already exists.

Write and run a small driver (commit it as
`experiments/synthetic/nfpa_confirmatory/run_semantics.py`):

- for each archived seed 0–9, regenerate the colored dataset
  (`make_dataset(seed, "colored", PRIMARY)` — deterministic), fit the same
  methods via `FIT_FACTORY` (NFPA, Iso-MPCA, CW-PCA, PCA),
- compute `evaluate_semantics(fit, ds)` — planted-projector errors
  (channel, temporal) and held-out amplitude/parameter decoding metrics,
- **gate:** each seed's refit must reproduce the archived
  `test_normalized_chi2` for that seed/method to ≤1e-9 relative
  (from `E1_nfpa_blockA/block_A_colored_raw.csv`) — proves the refit is the
  archived fit; stop on any mismatch,
- write `E8_semantics_colored.csv` (one row per seed × method), plus a JSON
  summary with per-method means and seed-level 95% bootstrap intervals
  (10,000 resamples, seed 20260819).

**Predicted signature (declared now):** NFPA's planted-projector errors are
small and its amplitude decoding is at least as good as Iso-MPCA's. If NFPA's
planted-subspace errors are NOT small, H3 fails — report it exactly as
measured; the paper's §4 would then state that the metric helps reconstruction
without recovering the planted factors, which is a finding.

## E9 — frozen 20-seed replication extension (~1–2 h)

**Why:** 10 seeds was the frozen amendment; the audit asks for 20–30. Extending
must not silently blend into the original: this is an explicitly labelled
**post-hoc replication** of the frozen protocol.

- Run Block A colored for seeds **10–29** (new seeds only), identical frozen
  config; write to `E9_blockA_replication_seeds10_29.csv` — do NOT append to
  the archived Block A file.
- Report the paired NFPA−Iso contrasts for the replication set alone, and for
  the pooled 30 only alongside both subset results.
- **Gate:** the replication interval must be computed before pooling; if the
  replication contradicts the original in sign, report both, pool nothing.

## E10 — estimated-covariance condition (~1 h)

**Why:** every synthetic result uses the oracle Σ; the audit (and §5) note this
bounds what a correctly specified Σ buys. One estimated condition measures the
price of estimation.

- For seeds 0–9: draw `N_cal = 2048` noise-only events from the same colored
  model, estimate Σ̂_c, Σ̂_t per the paper-1 estimators (per-axis, regularized;
  use the existing `noise_geometry` estimators — do not invent new ones),
  refit NFPA and Iso-MPCA whitening with Σ̂, evaluate held-out under the
  **oracle** metric (the truth does not change because the model's whitening
  is estimated).
- Write `E10_blockA_estimated_cov.csv`; report the paired contrast under
  estimated whitening beside the oracle result.
- **Predicted signature:** the reversal survives with a reduced margin. If it
  does not survive, that is a headline honesty point for §5, not a failure.

## Sync back

```
E6_whitening_range.json (if still missing)
E8_semantics_colored.csv + E8_summary.json
E9_blockA_replication_seeds10_29.csv + E9_summary.json
E10_blockA_estimated_cov.csv + E10_summary.json
run_semantics.py and any driver committed
```

Append a Round 9 section to `EXPERIMENT_RUN_LOG.md` with commands, gates,
numbers as measured, and machine state. **No tuning, no threshold adjustment
after seeing numbers, no pooling before subset reporting. Do not edit the
manuscript — the desk does the writing.**

## What the paper does with each outcome (decided now)

| Result | Paper action |
|---|---|
| E8 recovery good | H3 moves from "not executed" to a one-line §4 result + appendix table; a recovery panel can replace Figure 1B |
| E8 recovery poor | Reported as measured: metric helps reconstruction without factor recovery |
| E9 signs agree | "ten seeds" → "thirty seeds (ten frozen + twenty replication)" |
| E9 signs disagree | Report both; the frozen result stands alone with the replication as a caveat |
| E10 reversal survives | One §4 sentence: the reversal survives estimated whitening (margin quoted) |
| E10 reversal dies | §5 oracle caveat becomes a measured limitation with the number |
