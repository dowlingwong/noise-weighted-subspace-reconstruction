# HCP Rnoise D0–D6 dev-only extension

New additive package on `codex/hcp-rnoise-dev-20261019`, based on accepted EXP
394146e. HCP/R1/original kits remain byte-identical. Accepted AD2 addendum changes
only the native amplitude baseline to each selected channel's whole-scan median.
`CONFIG_DELTA.json` is the reviewed JSON Merge Patch, applied by `access.config`.
Exact signed-sheet and addendum hashes are pinned in `access.py`. All scientific
thresholds, roles, windows, dimensions, seeds, signal recipes and 24 claims come
from the inherited accepted CONFIG. No future learning, prediction, pilot,
registration or confirmation command is enabled.

Before any recording access, `authorize` checks all dates against the frozen dev
list, requires exactly 12 distinct dev dates, validates the exact signed sheet,
accepted addendum and independent implementation review binding the current kit
identity. Missing/unsigned/changed decisions, unknown dates and every confirmation
date refuse nonzero. `read_native` repeats authorization before the inherited
dev-header reader. It decodes only pinned selected magnetometers. No raw directory
traversal, recording hashing or confirmation header check occurs.

Run from EXP root using `/ceph/dwong/meg_env/bin/python`, PYTHONPATH=src,
PYTHONDONTWRITEBYTECODE=1, unset PYTHONPYCACHEPREFIX, all BLAS/OpenMP/NumExpr
threads 1. Long jobs use `tmux -L paper1hcp`.

    python -m experiments.mlst2026.hcp_rnoise_dev_20261019.runner verify-kit
    python -m pytest -p no:cacheprovider -q experiments/mlst2026/hcp_rnoise_dev_20261019/tests

All stage commands require `--dates <the 12 frozen dev dates> --ms <MS clone>
--review <accepted dev-kit review REPORT.md> --out <run directory>`. Exact dates
are obtained from frozen metadata, never by listing the recording directory.

1. `decode-dev` (alias `gates`): new run directory; read all 12 dates, native screen,
   unchanged FIR and guarded cut; G1/G2, G4/G5, lag/bootstrap/third descriptions,
   budgets and reasons in QUALIFICATION.json. Local decimated `.npy` files are
   never committed. E1 failure writes STOP and exits 2. Commit the qualification
   JSON in MS before continuing. A run directory is never overwritten/restarted.
2. `seal`: requires E1 and all 12 local source hashes; computes Sref and ref-active
   spatial/lag0..19 moments, writes SEALED_MOMENTS.json. Hash it, copy exact bytes
   into MS reviews/hcp_rnoise_dev_20261019/SEALED_MOMENTS.json, commit and push.
3. `fit-generator --seal-commit <MS commit>`: proves the seal bytes exist at that
   commit before fitting. AR10, innovation and SPD failures stop without repair;
   all 12 shape diagnostics are written to GENERATOR.json.
4. `cost`: five complete control probes per real shape and two full64-start
   population timing rows per shape (128s per row), timing namespace99. Projects
   worst observed timings at4/16 workers, ideal and illustrative25% overhead.
   Checks accepted6h/4-worker controls and8h/16-worker D3-S projections. These
   population calls are synthetic generator timing only, not learning/prediction
   or a D3-S campaign. A timeout or projection failure writes STOP.
5. `controls --workers 16`: current task's explicit16-worker instruction controls
   execution; the accepted6h controls wall cap and6h/4-worker projection check are
   retained.168000 replicates, exact pinned seeds/laws/cells; each replicate uses
   the inherited complete function, with all descriptions. A once-only launch
   marker prevents restart. The21600s deadline covers workers, shutdown, complete
   ledger reduction and publication; it terminates workers and persists STOP.
   Worker/infrastructure failure also stops without restart. CONTROLS.jsonl retains
   every outcome; CONTROLS_SUMMARY.json has24 exact CP claims, descriptions,
   denominators and causes. Missing/duplicate records or any failed claim stop.
   CAMPAIGN_COMPLETED.json binds ledger/generator/summary hashes, kit and elapsed
   time. `summarize-controls` refuses a persisted STOP or missing/mismatched
   completion record and repeats only the completed immutable ledger summary.

D6 is an MS report written by the implementation agent, then STOP. Preserve local
arrays only on ceph. No `.npy/.npz/.h5/.fif`, raw4D file or credential enters git.
If an output exceeds50MB, preserve it byte-for-byte under the route-R returns
directory, compare SHA256 and publish EXTERNAL_FILES.json with path/bytes/hash/
ceph location/generating command; no compression, splitting, LFS or rerun.

Tests use fabricated values and mocked recording readers exclusively. The guards,
AD2 boundaries/nonfinite semantics, unchanged FIR input, complete E1 outcomes,
committed seal, no-retune generator rejection, exact CP and no-restart behavior
are tested separately from the unchanged67 HCP and359 inherited tests.
