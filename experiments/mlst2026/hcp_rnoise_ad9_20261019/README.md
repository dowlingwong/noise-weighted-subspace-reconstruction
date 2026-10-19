# Prospective HCP Rnoise AD8a/AD9 freeze

This additive package leaves every earlier kit and result unchanged. `power.py`
recomputes precision over n=8..60 from committed nine-date R50 pilot JSON, with
the original family34 and four AD8a cells. There is no recording reader,
resampling command or confirmation recording launcher in this package.

`guard.py` is the executable registration preflight and array-fit order guard.
It requires exact registered bundle bytes, the local and inherited kit identities,
OSF URL, tag ancestry, OTS receipt and signed author authorization on origin/main.
The caller must fetch origin/main and the named tag before the preflight. The
guard does not create a tag, verify Bitcoin attestation cryptography or contact
OSF. Author verification/maturation of the timestamp remains required.

`protocol.py` binds all 60 qualification outcomes, excludes without replacement,
and returns STOP/no claim below 18. `readings.py` uses R50, 18..60 included dates
and family34. It retains the earlier zero-SE/undecided and PZ1–PZ5 rules.
`build_bundle.py` publishes only new MS files, checks committed source bytes,
and writes sorted, indented JSON with a final newline and no runtime timestamps.
Every earlier file is checked against MS e58f315 and retained byte-for-byte.

Tests use fabricated arrays, JSON and mock git objects. Run with meg_env Python,
PYTHONPATH=src:. and all BLAS thread counts set to one. Confirmation prediction
and learning kernels remain the inherited ones; the future separately authorized
recording launcher must satisfy CONFIRMATION_RUN_CLI.md. Confirmation seed indices
are 12 plus the index in the complete sorted60 metadata list (12..71), disjoint
from dev indices 0..11. This indexing does not select dates or set analysis order.
