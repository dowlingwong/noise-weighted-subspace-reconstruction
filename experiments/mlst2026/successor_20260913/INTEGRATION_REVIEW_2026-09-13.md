# Integration review — 13 September 2026

Merged `local-build-20260913` into main at `1289100`. The companion manuscript's `reviews/2026-09-13-INTEGRATION-REVIEW.md` is the consolidated paper/experiment checklist.

Corrections: reject ledger protocol-hash mismatches before resume; use the actual frozen protocol hashes in ledger tests; test the T20 algebraic zero within 1e-12 rather than requiring an exact floating-point zero. Added a protocol-mismatch regression.

Validation on the local Python 3.12.7 environment: all six `tests/test_successor_20260913_*.py` suites passed, **127 passed, 0 skipped, 45.76 seconds**. Command:

```sh
CRESST_S_ARCHIVE_ROOT=/Users/wongdowling/Documents/arXiv-v1-polished-extended-preprint/evidence/data .venv/bin/python -m pytest tests/test_successor_20260913_common.py tests/test_successor_20260913_t19s.py tests/test_successor_20260913_t20s.py tests/test_successor_20260913_t21s.py tests/test_successor_20260913_t22s.py tests/test_successor_20260913_cresst_s.py -q
```

This is local development validation, not production evidence. No remote jobs or production matrices were executed. Rebuild and reverify the handoff from the final commits; previous archives predate the ledger correction. Full CRESST robustness needs raw pulse inputs, while T22's production matrix is simulation calibration, not the unfinished real-data matched comparison. Stored T22 timing projects about 105 single-worker hours; calibrate on the target server before assigning wall time. Check failed-overrun fit accounting and remote filesystem locking before relying on strict cap/concurrency guarantees. Existing noise-module and notebook edits were not part of this integration commit.
