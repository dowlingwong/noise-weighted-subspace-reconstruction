# Follow-up round — round-2 corrections (17 September 2026)

Response to `output/implementation-review-20260917/REVIEW.md` and
`CORRECTION_PROMPT.md`. **No production pilot or confirmation run was launched;
no archived protocol, matrix, freeze, cell, ledger or reduction was modified; no
manuscript was touched; nothing was pushed or published.** Every number below is
from local fixtures, synthetic records or matrix enumeration.

## Finding-by-finding correction table

| ID | Finding | Fix (code / protocol) | Regression test |
|---|---|---|---|
| **F1** P1 | `reduce_run` recursively deleted `copy_to`; source==destination deleted the source | `r0_t22_h5/reducer.py`: added `guard_copy_destination`/`guard_output_destination` (resolve symlinks; refuse equality, ancestor, descendant, existing destination; never delete) and `reduce_run(..., copy_to=...)` requires a fresh copy unless `--allow-in-place` (test-only). CLI requires `--copy-to`. | `test_f1_r0_refuses_same_and_overlapping_destinations_without_touching_source`, `..._symlink_alias...`, `..._existing_destination_and_never_deletes`, `..._output_inside_input_tree` |
| **F2** P1 | baseline replaced by point ± z·SE and normal p | `r1_t22_calibration/candidates.py`: `candidate_intervals` carries the stored `ci95_lo`/`ci95_hi`/`p_two_sided` through exactly; a normal approximation is the separately named `normal_z` candidate; runner passes the stored endpoints/p. | `test_f2_archived_baseline_endpoints_and_p_are_carried_exactly` |
| **F3** P1 | `_claim_decisions` ignored `withheld` | `r1.../runner.py`: a withheld replicate returns all-False; explicit withheld status preserved. | `test_f3_withheld_replicates_are_never_supported`, `test_f3_support_never_exceeds_one_minus_withholding` |
| **F4** P1 | target-mismatched candidates primary-eligible; bootstrap draws used as dof; two type-I predicates | `candidates.py`: `PRIMARY_ELIGIBLE` (equal-file target) vs `SECONDARY_DIAGNOSTIC`; `welch_fe_t` dof = n_files−1 (sampling design), not bootstrap draws; one executable predicate `type1_accepts_upper_bound` used by rule doc, pilot selection, confirmation and receipts, with the archived overlap rule kept as `historical_type1_overlap`; `R1-v2` protocol records the rule. | `test_f4_primary_eligibility_excludes_target_mismatched_candidates`, `..._welch_dof_is_the_file_count...`, `..._unified_type1_predicate_rejects_5_of_100` |
| **F5** P1 | empty receipt returned `pass`; unresolved retained large cells; CLI bypassed the gate; partial R1 issued acceptance | `r4.../r1_gate.py` rewritten fail-closed (schema + explicit Booleans + complete confirmation + candidate identity + protocol/matrix identity; statuses `pass`/`cancel`/`blocked`/`unresolved`); `effective_eval_scenarios` keeps large cells only on `pass`; R4 `main` refuses production `run` when unresolved/blocked; R1 acceptance requires a complete, non-partial confirmation and refuses a non-primary-eligible candidate. | `test_f5_empty_acceptance_is_blocked_not_passed`, `test_f5_unresolved_gate_removes_large_file_cells`, `test_f5_run_refuses_when_the_gate_is_unresolved`, `test_f9_r1_acceptance_requires_a_complete_confirmation`, R4 gate tests |
| **F6** P1 | R4 scored archived intervals after the R1 gate | `r4.../streams.py`: `run_replicate_streams(..., inference=...)` computes the frozen R1 candidate family from the same per-file summaries, applies the selected candidate's interval/coherent p to coverage, exclusion and the Holm family, and records the inference version and R1 receipt hash; the archived baseline is retained under `baseline_contrasts`/`baseline_verdicts`. | `test_f6_r4_applies_the_frozen_r1_candidate` |
| **F7** P1 | development `n_cal` changed evaluation arrays; dev table overwrote rows; familywise error ignored | `streams.py`: `simulate_scenario_streams(..., n_cal_dev=...)` uses a separate development scenario object and named stream; `runner.py`: `dev_design_pairs()` keeps the evaluation calibration fixed at `CAL_EVAL=64` and the dev table keys retain the full scenario identity; `analysis.feasibility_row` includes the any-null-family rate and completeness. | `test_f7_development_design_does_not_change_the_evaluation_realization`, `..._table_preserves_scenario_identity`, `..._feasibility_row_includes_any_null_family_and_completeness` |
| **F8** P2 | R5 production took one independent seed per m, so prefixes did not nest | `r5.../runner.py`: one cell per seed group over the **full m ladder** (`n_groups`), exact fit accounting `n_seeds·len(m_grid)`, parent/derived seed identities per record; `R5-v2` protocol records the nested rule. | `test_runner_fixture_path_nests_across_m`, `test_validate_and_enumerate_compute_nothing` |
| **F9** P1 | freeze/source binding incomplete or late; R3 expected counts tautological | R1 `_read_frozen` binds candidate family, eligibility split, adequacy rule and pilot cells; R2 `_confirmation_precheck` recomputes the pilot digest **before** the draw; R3 expected seed counts come from the frozen/declared matrix (`confirmation_seeds`), not `len(cs)`, and an incomplete configuration cannot label itself complete; R3/R4 source lists include the transitively imported scientific modules and the CLI. | `test_f9_r1_freeze_must_bind_pilot_content`, `test_f9_r2_confirmation_precheck_checks_the_pilot_digest`, R3 `score_legacy` changes + suite |
| **F10** P2 | R0 trusted stored coverage flags; no inventory/hash validation; finite-sample displacement read as established mismatch | `r0.../reducer.py`: recomputes two-stage/t coverage from stored bounds + declared truth; validates unique cell ids, scenario set, freeze binding, complete reconciliation, source hash; provenance carries input-cell and reducer-source hashes; the displacement statistic is labelled exploratory and states it establishes neither a population mismatch nor the cause of undercoverage. | `test_f10_r0_recomputes_coverage_from_bounds_not_stored_flags`, `..._target_statistic_is_not_an_established_mismatch`, `..._inventory_detects_duplicate_cell_ids` |

## Versioned protocols retained

| Protocol | v1 (retained, unchanged) | v2 (new) | Change |
|---|---|---|---|
| R1 | `PROTOCOL_R1_T22_CALIBRATION_v1.json` | `..._v2.json` | exact baseline carry-through; `normal_z`; primary-eligibility split; effective-df fix; unified type-I predicate; acceptance requires a complete confirmation |
| R4 | `PROTOCOL_R4_T22_FEASIBILITY_v1.json` | `..._v2.json` | separated dev/eval designs; dev-table identity; any-null-family + completeness; frozen R1 inference applied; fail-closed gate at the run boundary |
| R5 | `PROTOCOL_R5_T19_REACH_v1.json` | `..._v2.json` | nested production path; exact fit accounting; seed identities |

R0's overlay is versioned `R0-v2` with a machine-readable `change_log`; `R0-v1`
outputs are retained unchanged.

## Executable cell / fit counts (from `PREFLIGHT.json`, actual enumeration)

| Package | production cells | production fits | fixture cells | fixture fits | cap |
|---|---|---|---|---|---|
| R1 | 726 | 0 (simulator) | 738 | 0 | 16 |
| R2 | 966 | 4800 | 69 | 192 | — |
| R3 | 3460 | 3456 | 516 | 512 | — |
| R4 pass-gate | 563 (320 eval + 240 dev + 3 ctrl) | 0 | 11 | 0 | — |
| R4 unresolved/blocked | 443 (200 eval + 240 dev + 3 ctrl) | 0 | 11 | 0 | — |
| R5 | 4 (1 reach + 3 ctrl) | 80 | 4 | 6 | 200 |
| R6 | readiness only (data-blocked) | — | — | — | — |

R4's large-file cells (n=32/64; 6 scenarios × 20 chunks = 120 cells) are present
**only** on a clean R1 pass. The unresolved count is what a launch would use
while R1 is not resolved.

## Test commands and outputs

Authoritative pre-transfer gate (re-run after the round-3 blocker remediation):

```
.venv/bin/python -m pytest tests/test_followup_20260917_r0.py \
  tests/test_followup_20260917_r1.py tests/test_followup_20260917_r2.py \
  tests/test_followup_20260917_r3.py tests/test_followup_20260917_r4.py \
  tests/test_followup_20260917_r5.py tests/test_followup_20260917_r6.py \
  tests/test_followup_20260917_namespaces.py \
  tests/test_followup_20260917_execution_modes.py \
  tests/test_followup_20260917_corrections.py -q -p no:cacheprovider
# 138 passed

.venv/bin/python -m pytest tests/test_followup_20260917_corrections.py -q
# 28 passed
```

Historical counts at the time this correction table was first written: 132 full /
26 corrections. The full suite is now 138 (the correction file grew to 28 tests,
and the round-3 remediation added the R5 superseded-v1 refusal test).

## Negative gate-test evidence

* R0 with `copy_to == run_root` (or a symlink to it, or a parent/child, or an
  existing unrelated directory) raises `ValueError`; the source tree's file
  hashes are unchanged after every refusal.
* An empty `{}` R1 acceptance object returns `status="blocked"`, `cancel=False`,
  `ready_to_transfer=False` (previously `pass`).
* An unresolved/blocked gate removes the n=32/64 eval cells from the R4 matrix
  and makes the production `run` command refuse (exit 6) before any draw/charge.
* A development-only or incomplete R1 freeze/acceptance is `blocked`, never
  `pass`; a target-mismatched frozen candidate is `blocked`.
* A withheld record returns all-False claim decisions; the observed supported
  count never exceeds `N − withheld`.
* Welch dof is identical (3.0) for `n_bootstrap` 40 and 4000.
* The 5/100 Wilson case (upper bound 0.11175) is rejected by the unified
  predicate and accepted only by the archived overlap rule, reported separately.

## Remaining scientific / data blockers

* No production run has been performed; the corrected outputs are not yet
  evidence. R1's frozen candidate still needs scientific review before launch.
* R4's feasibility verdict needs R1 resolved; a matched real-data comparison
  remains data-blocked (independent unexposed instrument files).
* R6 CRESST robustness needs raw pulse/noise/time arrays; R6 TIDMAD needs
  independent unexposed files. Interfaces fail closed; no data was manufactured.
* Only the R1 candidate family, the R1→R4 gate and the R5 nested path received
  the review; R2/R3/R6 were inspected selectively and their remaining design
  questions are unchanged by this round.
