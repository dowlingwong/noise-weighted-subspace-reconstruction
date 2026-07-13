# CRESST post-freeze sensitivity runbook

Status: prepared and executed 13 July 2026 (232 reproduction checks, 0
failures; output at
`transfer_paper/data/cresst/sensitivity/20260713_channel_robustness_v1/`).
See `docs/CRESST_RESULT.md` for the post-hoc pulse/noise cross-stream
ordering correction this run's leakage audit surfaced.

This run is a **post-freeze sensitivity analysis**. It does not tune, rerun
model selection, change the frozen validation, or replace the archived pooled
300-replicate event bootstrap. The frozen source is commit
`9c9f696ff6bca6e6b9729cdb90e41ee92d576787`; its synchronized JSON remains
unchanged at
`transfer_paper/data/cresst/current/cresst/p2_cresst_validation_seed22.json`.

## What was added

- `scripts/run_cresst_postfreeze_sensitivity.py`: fail-closed end-to-end runner.
- `configs/cresst/cresst_postfreeze_sensitivity.yaml`: fixed sensitivity seed,
  10,000 hierarchical replicates, source hashes, and versioned destination.
- `src/noise_geometry/cresst/sensitivity.py`: independent reproduction checks,
  estimands, normalized channel effects, and hierarchical bootstrap.
- A metadata-only event hook in the original release analysis. The OF, PCA,
  EMPCA, template-bank, negative-control, and bootstrap calculations are still
  executed by the same frozen numerical path.

## Remote preflight

From the repository root:

```bash
uv sync --extra dev
uv run pytest -q

test -f /ceph/dwong/paper1_dataset/cresst/raw/X_test.npy
test -f /ceph/dwong/paper1_dataset/cresst/raw/features_test.csv
test -f /ceph/dwong/paper1_dataset/cresst/raw/resolutions_test.csv

sha256sum \
  configs/cresst/cresst_validation.yaml \
  transfer_paper/data/cresst/current/cresst/p2_cresst_validation_seed22.json
```

The expected hashes are:

```text
0cb11a0e60bc9e9073923b342c6f18e70b48f328654e311c16f966be78f9514a  configs/cresst/cresst_validation.yaml
2e794345140fe291d77778cf8362981f7a863173ac3e01ef92dad07298d29b8f  transfer_paper/data/cresst/current/cresst/p2_cresst_validation_seed22.json
```

If a raw release file is missing, follow `scripts/download/download_cresst.md`.
A missing input is a setup failure. Do not synthesize event distributions from
the archived channel means or intervals.

## Execute once

```bash
uv run python scripts/run_cresst_postfreeze_sensitivity.py \
  --config configs/cresst/cresst_postfreeze_sensitivity.yaml \
  --data-root /ceph/dwong/paper1_dataset
```

The runner refuses to overwrite its destination:

```text
transfer_paper/data/cresst/sensitivity/20260713_channel_robustness_v1/
```

If that directory already contains a completed run, verify `SHA256SUMS` rather
than rerunning. If it contains a failed/incomplete run, preserve it and choose
a new versioned directory in a reviewed config change; do not delete evidence
or silently use `--force` (there is no force option).

## Fail-closed order of operations

1. Verify frozen config and archived-JSON SHA-256 digests and frozen parameters.
2. Regenerate the frozen validation and event residual arrays together.
3. Require exactly 19,335 evaluation pulses across seven channels, unique
   event IDs and release-trace hashes, chronological disjoint splits, and no
   calibration/evaluation row or release-trace-hash overlap.
4. Recompute all archived per-channel means, pooled means, and original
   per-channel/pooled 300-replicate paired-bootstrap intervals from the event
   export and compare within the configured floating-point tolerance.
5. Only if every check passes, run the fixed-seed 10,000-replicate hierarchical
   bootstrap and create paper-facing summaries.

Any failure stops before sensitivity inference. Inspect `console.log`,
`run_status.json`, and `reproduction_audit.json` (if it was reached). Do not
change rank, shrinkage, covariance model, split, templates, time shifts,
exclusions, thresholds, tolerance, or event count in response to a failure.

## Expected complete output

The versioned directory contains:

- `event_residuals_lossless.npz`: structured, one-row-per-event lossless table;
- `event_residuals_compact.csv`: rounded human-readable view;
- `event_residuals_metadata.json`: schema and units;
- `regenerated_primary_metrics.json` and `reproduction_audit.json`;
- `leakage_audit.json` and `reproduction_and_leakage_audit.md`;
- `summary.json`, `channel_summary.csv`, and `estimand_summary.csv`;
- `hierarchical_bootstrap_draws.npz`;
- `paper_facing_table.csv`, `paper_figure.png`, and `paper_figure.pdf`;
- copied `analysis_script.py` and `analysis_config.yaml`;
- `environment.json`, `console.log`, `manifest.json`, and `SHA256SUMS`;
- `WRITING_AGENT_HANDOFF.md` with the claim recommendation and key checksums.

The hierarchical result must always be described with the warning that seven
clusters make the cluster-level interval coarse and potentially unstable. The
figure contains reconstruction residual contrasts only and does not portray
trigger efficiency as a positive result.

## Handoff boundaries

The strongest permitted conclusion is about held-out within-channel linear
reconstruction on the seven released channels. Do not claim cross-channel or
cross-detector transfer, detector sensitivity or threshold improvement,
19,335 detector-level replications, nonlinear-autoencoder evidence, a universal
covariance-weighting advantage when `d_PCA` is heterogeneous, or replacement
of the frozen primary gate.
