# RESTART NOTE — close-out: transfer, scientific audit, evidence table

Production programme is COMPLETE (see /home/dwong/mlst-successor-audit-20260916-envloss/CONSOLIDATED_REPORT.md).
This stage launches NO production experiments and changes NO protocol, threshold or outcome.
All writes go under /home/dwong/mlst-successor-closeout-20260917 (never /ceph: entry quota).

## Phases
A. [x] Checksummed transfer bundle (run roots, preflights, manifests, freeze artifacts,
       both validator runs + correction, environment records, T19 amendment git bundle)
B. [x] Scientific audit of each selection against its frozen protocol criteria
C. [x] Diagnose reported exceptions from existing outputs only
D. [x] Follow-up proposals (question, inputs, cost) — proposals only, nothing launched
E. [x] Claim-by-claim manuscript evidence table (manuscript NOT edited)

## Update log
- 2026-09-17T01:39:40+02:00: close-out started.
- 2026-09-17T01:45:02+02:00: A progress — staged: environment-records (NFS silly-rename mechanism confirmed; 365 early cells from same-version /ceph-path interpreter disclosed), t19-amendment git bundle (verify + restore test passed, reproduces patch 4a5b968c), freeze-artifacts (package-specific binding; ordering anomalies resolved: 0 completed confirmation cells before any freeze; superseded wrong-method index retained & labelled), validator-runs (run1, run2, correction diff). NEXT: build tar + verify.
- 2026-09-17T01:48:05+02:00: PHASE A DONE — MLST_SUCCESSOR_TRANSFER_20260917.tar sha256 fac5498b81aa9150ce136228548fbb447c70d9c816692e3d5261ee229657c5d8, 4,758 members, every member == source, 4,636 == return inventory, 0 mismatches (TRANSFER_VERIFICATION.json, TRANSFER_CONTENTS.sha256). This tar is immutable; later reports go in a second archive. PHASE B in progress: T22 audited (H1 13/65 non-drift outside; H2 pass; H3 as expected; H5 random-effects omitted from reduction, computed read-only 24/65 in band).
- 2026-09-17T01:53:35+02:00: PHASES B+C computations DONE for all six selections (audit/*_computation.txt, diagnostics/*). Corrections to prior reports identified: T22 H1 scope 13/65 non-drift (not 15/69); T19 S_full/beyond outside H3 scope; T21 H1 fails overall -> loadings step untested despite reducer's correlated 'supported' label; T20 H3 bracket predictions scored for first time; T22 H5 random-effects omitted from reduction; CRESST more channels worsen coverage. NEXT: write SCIENTIFIC_AUDIT.md, then D, then E.
- 2026-09-17T01:56:56+02:00: PHASE D DONE — followup/FOLLOWUP_PROPOSALS.md (F1–F11, measured-cost basis followup/MEASURED_CELL_TIMES.txt). NEXT: E evidence table (read manuscript claims; do not edit).
- 2026-09-17T02:03:23+02:00: PHASE E DONE — evidence-table (28 claims, 45 citation ranges verified). CLOSEOUT_REPORT.md written. ALL PHASES COMPLETE. Next action: none required; close-out reports archived in MLST_SUCCESSOR_CLOSEOUT_REPORTS_20260917.tar (verification in CLOSEOUT_REPORTS_VERIFICATION.json, written outside the archive).
