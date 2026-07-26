# Documentation Map

Last reorganized: 2026-07-26. Every current document in this directory is
listed below. If a document is not on this list, it is either archived or does
not exist.

## Three standing rules

1. **One living status document.** [`CURRENT_STATUS.md`](CURRENT_STATUS.md) is
   the single source of truth for what the project currently claims. Everything
   else points at it or is generated from it.
2. **Plans live with their subject.** Paper 2 plans in `../paper2/`, Paper 3 in
   `../paper3/`, the noise module in `../noise_module/`, the manuscript in the
   LaTeX repository's `plan/`. `docs/` holds only what spans them.
3. **Dated audits are immutable; status documents are living.** An evaluation
   dated in its filename is never edited — it is superseded by a later one and
   both are kept. A status document carries no date and is always current.

## Start here

| File | Purpose |
| --- | --- |
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | What is validated, what is not, and the claim boundaries. Read first. |
| [`EVALUATION.md`](EVALUATION.md) | The consolidated academic audit of the experiments and noise module, with the prioritized action list. |
| [`EXPERIMENTS_OVERVIEW.md`](EXPERIMENTS_OVERVIEW.md) | Every implemented experiment: description, method, target, status. |
| [`EXPERIMENT_PROTOCOLS.md`](EXPERIMENT_PROTOCOLS.md) | How to run each experiment, data access, acceptance gates. |
| [`PAPER_REVISION_GUIDE.md`](PAPER_REVISION_GUIDE.md) | Manuscript-safe wording and claim boundaries. |
| [`REMOTE_EXECUTION.md`](REMOTE_EXECUTION.md) | Remote commands, Stage 0 gate, evidence packaging. |

## `results/` — outcome documents

| File | Purpose |
| --- | --- |
| [`results/CRESST_RESULT.md`](results/CRESST_RESULT.md) | CRESST-A outcome: the positive public-data validation. |
| [`results/CRESST_PREDECLARED_GATES.md`](results/CRESST_PREDECLARED_GATES.md) | The CRESST predeclaration. Use as the template for any new stage. |
| [`results/TIDMAD_PREDECLARED_GATES.md`](results/TIDMAD_PREDECLARED_GATES.md) | The TIDMAD predeclaration. Frozen for the synthetic gates; data-dependent fields still open. |
| [`results/GWOSC_RESULT.md`](results/GWOSC_RESULT.md) | GWOSC outcome: implementation checks pass, calibration fails. Boundary result. |
| [`results/TIDMAD_PHASE1_RESULT.md`](results/TIDMAD_PHASE1_RESULT.md) | TIDMAD Phase 1: estimator identity and whitening convention verified (synthetic half). |
| [`results/TIDMAD_PHASES_2_TO_6_RESULT.md`](results/TIDMAD_PHASES_2_TO_6_RESULT.md) | TIDMAD Phases 2–6: metric reversal with truth, rank knee, calibration boundary, denoiser. |

## `plans/` — roadmaps and decisions

| File | Purpose |
| --- | --- |
| [`plans/REVISION_PLAN_2026-07-26.md`](plans/REVISION_PLAN_2026-07-26.md) | The evidence-constrained revision work packages (EXP-*). |
| [`plans/CORRECTION_PLAN_EXPERIMENTS_2026-07-26.md`](plans/CORRECTION_PLAN_EXPERIMENTS_2026-07-26.md) | Corrections from the audit (WP-E1…E7), in execution order. |
| [`plans/PAPER1_SECOND_REALDATA_ANCHOR.md`](plans/PAPER1_SECOND_REALDATA_ANCHOR.md) | Whether Paper 1 needs a second real-data anchor; candidate analysis; GWOSC decision. |
| [`plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`](plans/TIDMAD_IMPLEMENTATION_ROADMAP.md) | The TIDMAD stage: phases, gates, code layout, verification. |
| [`plans/DOC_CONSOLIDATION_PLAN.md`](plans/DOC_CONSOLIDATION_PLAN.md) | The documentation audit this layout implements. |

## `runbooks/` — operational procedures

| File | Purpose |
| --- | --- |
| [`runbooks/CRESST_REMOTE_AGENT_RUNBOOK.md`](runbooks/CRESST_REMOTE_AGENT_RUNBOOK.md) | Running the CRESST validation remotely. |
| [`runbooks/CRESST_POSTFREEZE_SENSITIVITY_RUNBOOK.md`](runbooks/CRESST_POSTFREEZE_SENSITIVITY_RUNBOOK.md) | Reproducing the channel-then-event hierarchical analysis. |
| [`runbooks/TIDMAD_STAGE_RUNBOOK.md`](runbooks/TIDMAD_STAGE_RUNBOOK.md) | Running the whole TIDMAD stage: download, phases 0–6, evidence archive. |

## `background/` — reference material

| File | Purpose |
| --- | --- |
| [`background/EMPCA_TIME_DOMAIN_BACKGROUND.md`](background/EMPCA_TIME_DOMAIN_BACKGROUND.md) | Time-domain EMPCA background and conventions. |

## Elsewhere in the repository

| Location | Contents |
| --- | --- |
| [`../paper2/PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md`](../paper2/PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md) | Paper 2: noise-module integration plan; why Paper 3 is standalone. |
| [`../paper3/PAPER3_PLAN.md`](../paper3/PAPER3_PLAN.md) | Paper 3 (with Junjie): thesis, claim set, NuBench + real-data two-track design. |
| [`../noise_module/`](../noise_module/) | Noise-module docs: `PAPER_CONVENTION.md`, `CONFIGURATION.md`, `CALIBRATION.md`, `CONSISTENCY.md`, `EVOLUTION_PLAN.md`. |
| [`../evidence/README.md`](../evidence/README.md) | Evidence bundles and their manifests. |
| [`../transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md`](../transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md) | Which archived run backs which manuscript number. |

## Generated copies — do not hand-edit

`../transfer_paper/data/source_documents/` and the same path in the manuscript
repository contain **generated** copies of the documents listed in
`scripts/sync_source_documents.py`. They carry a provenance header and a
`SOURCE_DOCUMENTS_MANIFEST.json`. Edit the source in `docs/` and re-run:

```bash
uv run python scripts/sync_source_documents.py --latex-repo ../arXiv-v1-polished-extended-preprint
uv run python scripts/sync_source_documents.py --check    # CI / pre-import gate
```

Before 2026-07-26 these were three independently maintained copies with three
different hashes, and the manuscript repository was reading the oldest one.

## Archive

- `archive/2026-06-29_pre_consolidation/` — pre-consolidation working documents.
- `archive/2026-07-26_pre_paper23/` — the two merged audits and the superseded
  Paper 2 / NPML plans. See that directory's `README.md` for what replaced each.
