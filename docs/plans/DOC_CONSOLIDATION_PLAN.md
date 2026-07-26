# Documentation Audit and Consolidation Plan

Date: 2026-07-26. Covers both repos: `noise-weighted-subspace-reconstruction`
(experiments) and `arXiv-v1-polished-extended-preprint` (manuscript).

> **Status: executed 2026-07-26.** All seven steps in §5 were carried out. See
> the `2026-07-26 — documentation consolidation` entries in both repositories'
> `CHANGELOG.md`, and [`../README.md`](../README.md) for the resulting layout.
> This document is retained as the record of what was decided and why; file
> paths below describe the *pre-consolidation* state.

Method: every non-archived `.md` in both repos was inventoried by modification
date and size; suspected duplicates were compared by MD5. Classification is
**stale** (contradicts current state or superseded), **drifted** (multiple
divergent copies of the same document), **redundant** (content now covered
better elsewhere), **misplaced** (correct content, wrong location), or
**current**.

---

## 1. The headline finding: three divergent copies of the status documents

This is the only issue in the documentation that can actually cause a wrong
number to reach the paper, so it comes first.

| Document | `exp-repo/docs/` | `exp-repo/transfer_paper/data/source_documents/` | `latex-repo/transfer_paper/data/source_documents/` |
| --- | --- | --- | --- |
| `CURRENT_STATUS.md` | 13,173 B · `ed4763b1` | 10,148 B · `5f6bc29d` | 6,939 B · `4efd16d6` |
| `EXPERIMENT_PROTOCOLS.md` | 12,602 B · `363a467b` | 11,439 B · `7c209ed8` | 9,975 B · `7ae4181c` |
| `PAPER_REVISION_GUIDE.md` | 6,592 B · `01ad4da4` | 6,089 B · `441764770` | 5,229 B · `2a0dcaf` |

Three different contents, three different sizes, three different hashes, for
each of the three documents that define *what the project currently claims*.
The manuscript repo is reading the oldest copy. `GWOSC_RESULT.md` and
`REMOTE_EXECUTION.md` are duplicated the same way (those two happen to still
match, which is luck, not process).

**Fix — make the copies generated, not maintained.**

1. `exp-repo/docs/` is the single source of truth. Nothing else is hand-edited.
2. Add `scripts/sync_source_documents.py`: copies the canonical files into
   `transfer_paper/data/source_documents/` in both repos, writes a
   `SOURCE_DOCUMENTS_MANIFEST.json` with SHA-256 and source commit, and stamps
   each copy with a header line: *"Generated copy — do not edit. Source:
   docs/<name>.md @ <commit>."*
3. Add a check to `scripts/verify_paper_bundle.py`: fail if any
   `source_documents/` copy's hash disagrees with the manifest.
4. Run the sync once now, so the manuscript repo stops reading a 13-day-old
   status document.

Do this before anything else in this plan; it is ~1 hour and it removes a real
risk.

---

## 2. Experiment repo — file-by-file classification

### 2.1 Stale (superseded; archive)

| File | Date | Why | Action |
| --- | --- | --- | --- |
| `NPML/plan.md` (24.6 KB) | 06-09 | Pre-restructure NPML roadmap; the NPML experiments are now documented in `paper2/plans/` and superseded by the Paper 2 integration plan. Its content contradicts the current "exploratory only" status. | move to `docs/archive/2026-07-26_pre_paper23/` |
| `paper2/MISSING_FOR_D_E.md` | 06-09 | Lists what was missing for NPML-D/E; those experiments are now classified invalid pending repairs, so the list is answering an obsolete question. | archive |
| `paper2/IMPLEMENTATION_SPEC.md` | 06-09 | Predates the seeding/test-set/timing/Transformer defects being identified. | archive or rewrite as part of WP-E7 |
| `paper2/plans/hybrid_ml_pipeline.md` | 06-30 | Superseded by `PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md`. | archive |
| `paper2/plans/NPML_experiments.md` | 06-30 | Same. Keep only if it holds run identifiers not recorded elsewhere — check before archiving. | archive after check |
| `docs/README.md` | 07-26 | **Stale despite the date.** It advertises a "small canonical set" of five documents; `docs/` now holds fifteen, including three evaluations and three correction plans that it does not mention. Its "Canonical Status" section also still lists Study B as "verified positive", which the manuscript no longer claims. | **rewrite** (see §4) |
| `CHANGELOG.md` | 07-13 | Stops before the entire 2026-07-26 correction pass. | append the 07-26 entries |

### 2.2 Redundant / overlapping (merge)

| Files | Problem | Action |
| --- | --- | --- |
| `docs/ACADEMIC_EXPERIMENT_AND_NOISE_MODULE_EVALUATION.md` (40.9 KB) **and** `docs/INDEPENDENT_EVALUATION_2026-07-26.md` (18.7 KB) | Two audits of the same object, dated the same day, ~60% overlapping. A reader cannot tell which is authoritative. The first is broader (includes Paper 2 defect list, action lists); the second is a verification pass with new findings (S5 provenance mismatch, `sliding_fit` bug, generator numerics). | **Merge into one `docs/EVALUATION.md`**: keep the first's structure, fold in the second's verified findings as a "second-pass verification" section, and archive both originals with their dates. One audit document, one verdict. |
| `docs/EXPERIMENTS_OVERVIEW.md` **and** `docs/EXPERIMENT_PROTOCOLS.md` | Overview = what each experiment is and what it targets. Protocols = how to run it and the acceptance rules. Genuinely complementary, but currently neither cross-references the other and both restate the S1–S9 list. | Keep both; add reciprocal links; delete the duplicated S1–S9 table from whichever restates it second (keep the descriptive one in `EXPERIMENTS_OVERVIEW.md`, the command/gate one in `EXPERIMENT_PROTOCOLS.md`). |
| `evolution_plan.md` (25 KB, repo root) | Correct content, wrong place and wrong tense: it is the noise-module roadmap, mostly *completed*, sitting at the top level of the experiments repo. | **Move to `noise_module/EVOLUTION_PLAN.md`**, mark completed items done, and leave the open items (calibration sampler, chronological gates) pointing at WP-E6. |
| `docs/plans/REVISION_PLAN_2026-07-26.md` (26.5 KB) and `latex-repo/plan/REVISION_PLAN_2026-07-26.md` (26.3 KB) | Same document in both repos, slightly different sizes. | Decide ownership — this is a *manuscript* plan, so the latex repo owns it; the experiments repo keeps a link, not a copy. |

### 2.3 Stubs (harmless but noisy)

`EXPERIMENT_PLAN.md` and `TODO.md` at the repo root are two-line redirects to
`docs/`. They are honest and intentional. Options: leave them (zero cost), or
fold both into a single `docs/`-pointing line in `README.md` and delete. Low
priority — prefer leaving them so old links keep working.

### 2.4 Current (no action)

`README.md`, `docs/CURRENT_STATUS.md`, `docs/results/CRESST_RESULT.md`,
`docs/results/CRESST_PREDECLARED_GATES.md`, `docs/runbooks/CRESST_POSTFREEZE_SENSITIVITY_RUNBOOK.md`,
`docs/runbooks/CRESST_REMOTE_AGENT_RUNBOOK.md`, `docs/results/GWOSC_RESULT.md`,
`docs/REMOTE_EXECUTION.md`, `docs/background/EMPCA_TIME_DOMAIN_BACKGROUND.md`,
`docs/PAPER_REVISION_GUIDE.md`, `evidence/README.md`,
`noise_module/*.md` (all current, dated 07-26; `PAPER_CONVENTION.md` is
correctly mirrored into the evidence bundles by the export script, which is the
right pattern — the source-documents copies should work the same way), and the
six documents produced in this audit cycle.

Note `docs/archive/2026-06-29_pre_consolidation/` is doing its job — a previous
consolidation was done properly. This plan continues that practice rather than
inventing a new one.

---

## 3. Manuscript repo — file-by-file classification

### 3.1 Stale (move to `plan/stale/`, which already exists for this purpose)

All dated 06-20 and superseded by `plan/REVISION_PLAN_2026-07-26.md` plus
`plan/CORRECTION_PLAN_2026-07-26_AUDIT.md`:

- `plan/revision1.md` (63 KB) and `plan/revision1_checklist.md`
- `plan/experiment.md` (32 KB) — superseded by the experiments repo's
  `EXPERIMENTS_OVERVIEW.md`
- `plan/build_experiment_repo.md` (26 KB) — the experiments repo exists now
- `plan/paper1_scope_revision.md` — scope decisions are now in the manuscript
- `plan/paper2_draft.md`, `plan/paper2_checklist.md` — Paper 2 lives in the
  other repo; keep a pointer, not a stale draft

`plan/stale/` already holds six files, so the convention is established; these
six join them. Add a one-line `plan/stale/README.md` explaining that nothing in
that directory is authoritative.

### 3.2 Current (no action)

`OPEN_ISSUES.md`, `ARXIV_CHECKLIST.md`, `CHANGELOG.md`, `AGENTS.md`,
`plan/REVISION_PLAN_2026-07-26.md`, `plan/CORRECTION_PLAN_2026-07-26_AUDIT.md`,
`transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md`, and the three 07-26 receipts
(`IMPORT_RECEIPT`, `REVISION_BASELINE`, `FINAL_BUILD`).

### 3.3 Decide-and-delete

`figures/figure_placement.md` and the unused `img/` files are already flagged in
`OPEN_ISSUES.md` #10. Resolve them as part of the arXiv upload checklist rather
than here.

### 3.4 Handoff documents that have served their purpose

`transfer_paper/REVISION_INSTRUCTION_2026-07-12.md`,
`WRITING_AGENT_BRIEF.md`, `PAPER_WRITING_HANDOFF.md`,
`EXPERIMENT_AGENT_INSTRUCTION_2026-07-13.md`, `CRESST_PAPER_UPDATE_PLAN.md`,
`PENDING_RESULT_PLACEHOLDERS.md` — these are agent instructions from completed
work cycles. They are not wrong, but they read as current instructions to
anyone who opens them. Move to `transfer_paper/handoffs/completed/` with a
README noting the cycle each belongs to. Exception: check
`PENDING_RESULT_PLACEHOLDERS.md` for placeholders that are *still* pending
before filing it away.

---

## 4. Target organization

### Experiment repo `docs/`

```
docs/
  README.md                  <- rewritten index; every file below listed with one line
  CURRENT_STATUS.md          <- the single living status document (source of truth)
  EVALUATION.md              <- merged audit (was ACADEMIC_… + INDEPENDENT_…)
  EXPERIMENTS_OVERVIEW.md    <- what each experiment is, method, target
  EXPERIMENT_PROTOCOLS.md    <- how to run, acceptance gates, data access
  PAPER_REVISION_GUIDE.md    <- manuscript-safe wording
  REMOTE_EXECUTION.md
  results/
    CRESST_RESULT.md
    GWOSC_RESULT.md
    CRESST_PREDECLARED_GATES.md
  plans/
    CORRECTION_PLAN_EXPERIMENTS_2026-07-26.md
    PAPER1_SECOND_REALDATA_ANCHOR.md
    TIDMAD_IMPLEMENTATION_ROADMAP.md
    DOC_CONSOLIDATION_PLAN.md          <- this file
  runbooks/
    CRESST_REMOTE_AGENT_RUNBOOK.md
    CRESST_POSTFREEZE_SENSITIVITY_RUNBOOK.md
  background/
    EMPCA_TIME_DOMAIN_BACKGROUND.md
  archive/
    2026-06-29_pre_consolidation/
    2026-07-26_pre_paper23/            <- new: NPML/plan.md, paper2 stale specs, both audit originals
```

Paper-specific plans stay with their paper: `paper2/`, `paper3/`,
`noise_module/`. Only cross-cutting documents live in `docs/`.

### Manuscript repo

```
plan/
  REVISION_PLAN_2026-07-26.md
  CORRECTION_PLAN_2026-07-26_AUDIT.md
  stale/                      <- + the six 06-20 files, + README.md
transfer_paper/
  MANUSCRIPT_EVIDENCE_MAP.md
  data/source_documents/      <- GENERATED, never hand-edited, hash-verified
  handoffs/completed/         <- retired agent instructions
```

### Three standing rules

1. **One living status document.** `docs/CURRENT_STATUS.md`. Everything else
   either points at it or is generated from it.
2. **Plans live with their subject.** Paper 2 plans in `paper2/`, Paper 3 in
   `paper3/`, noise module in `noise_module/`, manuscript in the latex repo's
   `plan/`. `docs/` holds only what spans them.
3. **Dated audits are immutable; status documents are living.** An evaluation
   dated 2026-07-26 is never edited — it is superseded by a later one and both
   are kept. A status document is always current and never dated in its
   filename. The current confusion comes from mixing these two kinds.

---

## 5. Execution order

| Step | Effort | Why this order |
| --- | --- | --- |
| 1. Source-document sync script + hash gate (§1) | ~1 h | Only item with a correctness risk attached |
| 2. Rewrite `docs/README.md` as a true index (§2.1) | ~30 min | Makes everything else discoverable |
| 3. Merge the two audits into `docs/EVALUATION.md` (§2.2) | ~1 h | Removes "which audit is authoritative?" |
| 4. Move stale files to archive dirs in both repos (§2.1, §3.1) | ~30 min | Mechanical |
| 5. Move `evolution_plan.md` → `noise_module/` (§2.2) | ~10 min | Mechanical |
| 6. Reorganize `docs/` into the subfolder layout (§4) | ~1 h | Do last; it breaks links, so fix links in the same commit |
| 7. Append 07-26 entries to both CHANGELOGs | ~20 min | Closes the record |

Total ≈ half a day. Steps 1–3 carry the value; 4–7 are tidying and can be
deferred if the TIDMAD smoke test is the priority.
