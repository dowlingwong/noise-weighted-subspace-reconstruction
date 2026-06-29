# Documentation Map

This directory has been consolidated. The current documentation is the small
canonical set below; older files are preserved under
`docs/archive/2026-06-29_pre_consolidation/` for provenance.

## Read These First

| File | Purpose |
| --- | --- |
| `CURRENT_STATUS.md` | Current project status, supported claims, unsupported claims, and next steps. |
| `PAPER_REVISION_GUIDE.md` | Manuscript-safe wording and claim boundaries for paper revision. |
| `GWOSC_RESULT.md` | Final interpretation of the GWOSC global/local PSD and shared-FIR follow-ups. |
| `EXPERIMENT_PROTOCOLS.md` | Experiment registry, data access, metrics, preprocessing contract, and acceptance standards. |
| `REMOTE_EXECUTION.md` | Remote execution commands, Stage 0 gate, and evidence packaging rules. |

## Canonical Status

The current validation state is:

- controlled synthetic S1-S9 evidence: verified positive;
- Stage 0 remote reproducibility: verified positive;
- GWOSC PSD/reference and shared-FIR implementation checks: verified positive;
- GWOSC global-PSD and time-local real-noise calibration: verified negative;
- GW150914 event/significance and SNR-eight sensitivity: not validated;
- CRESST public-data validation: not completed.

Use `CURRENT_STATUS.md` and `PAPER_REVISION_GUIDE.md` for paper-facing claims.
Use `GWOSC_RESULT.md` for the real-noise boundary result.

## Archived Docs

The archived folder contains the pre-consolidation files, including the old
roadmap, GWOSC run records, agent handoff notes, dataset notes, metrics notes,
and compatibility stubs. They are retained because some contain useful
provenance and historical reasoning.

`docs/archive/plan/` holds the consolidated 2026-05 manuscript-planning material
moved out of the former top-level `plan/` directory: `PAPER1_MANUSCRIPT_PLAN.md`,
`REVIEW_AND_HANDOFF.md`, and the source `EMPCA_improvement.pdf`. The Paper 2 /
NPML planning notes were relocated to `paper2/plans/`.

When an archived file conflicts with a top-level canonical file, the top-level
canonical file governs.

## Cleanup Rules Going Forward

1. Add current status only to `CURRENT_STATUS.md`.
2. Add paper claim guidance only to `PAPER_REVISION_GUIDE.md`.
3. Add experiment commands, metrics, and acceptance rules only to
   `EXPERIMENT_PROTOCOLS.md`.
4. Add remote run/evidence workflow only to `REMOTE_EXECUTION.md`.
5. Add GWOSC interpretation only to `GWOSC_RESULT.md`.
6. Move superseded notes to `docs/archive/` with a short pointer to the
   canonical replacement.

`transfer_paper/` remains the paper-revision evidence bundle and figure/table
pipeline. It should consume current evidence, not become the canonical project
status document.
