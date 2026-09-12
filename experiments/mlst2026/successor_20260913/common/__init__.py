"""Shared infrastructure for the 13 September 2026 local-build round (L1–L5).

Every package in ``experiments/mlst2026/successor_20260913/<pkg>/`` uses this
module for the parts that must behave identically across packages:

* protocol loading and hashing (:mod:`.protocol`)
* run-matrix enumeration with per-cell fit/solve accounting (:mod:`.matrix`)
* the execution ledger that charges every attempt, including failures,
  retries and interruptions (:mod:`.ledger`)
* atomic result writing, hashing, schema checks and expected-versus-observed
  cell reconciliation (:mod:`.results`)
* the common CLI skeleton whose ``--help``, ``enumerate`` and ``validate``
  paths never compute anything (:mod:`.cli`)

Nothing here launches work on import.
"""
