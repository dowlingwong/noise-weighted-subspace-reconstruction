# RESTART NOTE — MLST 13 Sep successor programme, final stage

**PROGRAMME COMPLETE (2026-09-17): all six authorized selections executed and validated; T22 final validation passed 35/35.** Scientific review pending (reviewer task). No restart needed.
Update log at bottom; the last entry is the current state.

## Paths

| What | Path |
|---|---|
| T22 run root | `/ceph/dwong/mlst-successor-runs-20260913/t22s` |
| T22 launcher stdout | `/ceph/dwong/mlst-successor-runs-20260913/t22s.stdout` |
| T22 invocation logs | `/ceph/dwong/mlst-successor-runs-20260913/t22s/logs/20260913T133412-2933400/` |
| T22 receipt (never replace) | `/ceph/dwong/mlst-successor-runs-20260913/preflight/PREFLIGHT.json` (sha256 `f2d03b8f…`) |
| Live experiment repo (T22 source; never edit) | `/ceph/dwong/noise-weighted-subspace-reconstruction` @ `1d29658` |
| Interpreter (rebuilt 2026-09-16) | `/home/dwong/mlst-successor-py312-20260913/bin/python` |
| Damaged interpreter (quarantined evidence) | `/home/dwong/mlst-successor-py312-20260913.damaged-20260916` |
| Local copy of T22 source + manuscript | `/var/tmp/mlst-t22-envrebuild-preflight-20260916/{exp,ms}` |
| Rebuilt-env preflight (136/0/0/0) | `/var/tmp/mlst-t22-envrebuild-preflight-20260916/preflight-rebuilt-env/` |
| Records for this stage | `/home/dwong/mlst-successor-audit-20260916-envloss/` |
| Earlier audit records (archived by someone else, do not overwrite) | `/home/dwong/mlst-successor-audit-20260913.tar.gz` |
| T19 amended snapshot (commit `56de23e`) | `/var/tmp/mlst-t19-amendment-20260913/exp` |
| Completed roots | `.../mlst-successor-runs-20260913/{t19s-continuation,t20s,t21s,cresst-archive,cresst-calibration}` and original `t19s` |

## Steps

1. [x] T22 launcher exits; inspect `t22s.stdout` for the success line
       `Execution/reduction finished: ...` and absence of a traceback; inspect
       `logs/.../03-reduce.log` contents and `03-REDUCED.json`.
2. [x] Run terminal validation:
       `/home/dwong/mlst-successor-py312-20260913/bin/python /home/dwong/mlst-successor-audit-20260916-envloss/validate_t22_terminal.py`
       → read `T22_TERMINAL_MANUAL.json` `status` and `failed_checks`.
3. [x] Verify completed roots by per-file sha256 against recorded inventories.
4. [x] Cross-environment compatibility check (re-reduce a completed root under
       the rebuilt env, compare with its original-env REDUCED.json).
5. [x] Extract T22 scientific findings from `t22s/REDUCED.json`.
6. [x] Consolidated report + checksummed return-artifact inventory.

## Never

Do not replace T22's receipt, edit the live repo, rerun completed selections,
retry a failed stage without investigation, write large trees to `/ceph/dwong`,
push/publish, edit manuscript claims, or expand into full CRESST robustness or
real-data T22.

## Update log
- 2026-09-16T23:32:57+02:00: note created. T22 at 725/727, launcher alive, 0 failed. Rebuilt env + preflight done (see INCIDENT_2026-09-16_env_deletion.md). Steps 1–6 pending.
- 2026-09-16T23:36:13+02:00: step 3 DONE — all 6 completed roots intact by per-file sha256 (3,768 files; COMPLETED_ROOTS_CHECKSUM_VERIFICATION.txt). Step 4 DONE — T19 re-reduced under rebuilt env reproduces its 13-Sep original-env REDUCED.json identically (CROSS_ENV_COMPATIBILITY_T19.txt). Validator gained protocol/matrix identity checks. T22 at 726/727, launcher alive. Pending: 1, 2, 5, 6.
- 2026-09-17T01:25:50+02:00: step 1 DONE — launcher exited; stdout success line present, 0 tracebacks; 03-reduce.log partial=false, reconciliation 727/727 complete; 03-REDUCED.json byte-identical to root REDUCED.json. Stage attribution from ledger pids: pilot/control pid 2933410 and all 690 confirmation cells pid 3050746 (both started under ORIGINAL env); final reduction 23:43:39–23:44:45 spawned from REBUILT env. Launcher numeric exit code not observable (not a child); command() raises on any non-zero phase. Step 2 (validator) STARTED detached: output /home/dwong/mlst-successor-audit-20260916-envloss/validate_run.log, result /home/dwong/mlst-successor-audit-20260916-envloss/T22_TERMINAL_MANUAL.json.
- 2026-09-17T01:28:21+02:00: step 2 RUN 1 = STOP_REVIEW_REQUIRED. 31/32 checks passed; failed 'package_scientific_checks_passed' (detail null). Cause: VALIDATOR DEFECT — check copied from T19, whose REDUCED.json has 'scientific_checks'; T22's has none and reports 'controls' (5, all passed; protocol matrix.production.controls declares the same 5; protocol failure_interpretation says a control failure makes the run irreducible, exit 4, and the final reduce exited 0). Run-1 result preserved as T22_TERMINAL_MANUAL.run1_STOP_REVIEW_REQUIRED.json, script as validate_t22_terminal.run1.py. Fix: replace with protocol-declared control-set check; add per-cell freeze-binding check. Re-running as RUN 2. Programme NOT complete.
- 2026-09-17T01:30:11+02:00: step 2 RUN 2 = VALIDATED_COPY_REDUCTION_SCIENTIFIC_REVIEW_PENDING, 35/35 checks (run1's 32 minus the defective check, plus control-set match, controls passed, per-cell freeze binding 690/690, reduced frozen_sha match). Result T22_TERMINAL_MANUAL.json; run-1 retained. Step 5 DONE (findings extracted). Step 6 STARTED: build_return_inventory.py detached -> RETURN_INVENTORY.sha256/.json; then consolidated report. Programme NOT yet marked complete.
- 2026-09-17T01:33:30+02:00: step 6 DONE — RETURN_INVENTORY.sha256 (4,634 files; sha256 0fa65ff9…) cross-checked 4,485 files vs recorded inventories, 0 mismatches; CONSOLIDATED_REPORT.md written; addendum covers files written after the inventory. PROGRAMME COMPLETE. Nothing pending except reviewer scientific review.
