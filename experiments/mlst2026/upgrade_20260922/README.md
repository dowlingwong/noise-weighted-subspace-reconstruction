# Bounded MLST upgrade

Entry point: `PYTHONPATH=src python -m experiments.mlst2026.upgrade_20260922 --help`, from EXP. JSON-serialized `.yaml` protocols are closed/versioned. Unknown fields and deviations require a new reviewed implementation, not an ignored knob.

Scientific scope, exact remote commands, budgets and stop rules: MS/reviews/upgrade_20260922/REMOTE_HANDOFF.md. Only synthetic fixtures were run locally. Do not run production protocols locally under the implementation authorization.

Files: `c1/` covariance prediction; `cresst.py` strict adapter/support/math/roles; `protocol.py` identities/config/enumeration; `__main__.py` bounded process and ledger; `validation.py` receiving numerical checks; `reporting.py` derived source tables/figures. Source closure includes all EXP/src Python plus this package/configs, both commits and relevant MS sources; installed distribution versions bind runtime. `prepare` uses `MLST_MS_ROOT`; `freeze/run` take `--ms-root`. Runs must live outside both repositories.

`diagnose` is read-only, never repairs interrupted attempts. Zero retries; existing directories are never resumed. Scientific export rejects fixtures, pilots and incomplete campaigns. Receiving validation recomputes map/endpoint/contrast/interval summaries and rejects missing methods. Results are not signed against an adversary controlling all files; the return SHA manifest must be retained independently with the transfer record.

CRESST qualification requires real data, acquisition terms, sample spacing, one verified time axis, and project exposure history. Unknown exposure is rejected. Default release-train selection does not assert unseen-channel validation. Histories/exposed channels receive replication scope. A noiseless injection target is constructed, not a claim about the purity of the empirical mean template.
