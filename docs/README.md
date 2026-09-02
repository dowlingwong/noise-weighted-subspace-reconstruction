# Documentation Map

Last reorganized: 2026-07-26; stale-document sweep 2026-09-02. Every current document in this directory is
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

## Papers 1 and 3 — experiment program (added 2026-09-02)

| File | Purpose |
| --- | --- |
| [`EXPERIMENT_PLAN_PAPER1.md`](EXPERIMENT_PLAN_PAPER1.md) | What the merged Paper 1 still needs (P1-E1 … P1-E7), with acceptance rules. |
| [`EXPERIMENT_PLAN_PAPER3.md`](EXPERIMENT_PLAN_PAPER3.md) | ORACLE-Cov Tier-1 design (subjects, families, monitors, endpoints) and the Tier-2/3 runbooks. |
| [`IMPLEMENTATION_CHECKLIST.md`](IMPLEMENTATION_CHECKLIST.md) | Long-horizon checklist for both papers; tick in place. |
| [`plans/oracle/PREREGISTRATION.md`](plans/oracle/PREREGISTRATION.md) | Tier-1 pre-registration draft (endpoints, thresholds, mechanism predictions); frozen on commit. |
| [`plans/oracle/TIER2_RUNBOOK.md`](plans/oracle/TIER2_RUNBOOK.md), [`plans/oracle/TIER3_RUNBOOK.md`](plans/oracle/TIER3_RUNBOOK.md) | Cluster runbooks for ORACLE-Paired and TIDMAD. |
| [`reviews/2026-09-02_experiments_review.md`](reviews/2026-09-02_experiments_review.md) | Author's review of the implementation: fixed defects, scientific findings, open items. |
| [`REVIEW_PROMPT_2026-09.md`](REVIEW_PROMPT_2026-09.md) | Prompt for an independent reviewing agent (Paper 1, Paper 3, implementation). |

Code: `experiments/paper1/`, `experiments/oracle/` (package `oracle_cov`); results:
`results/paper1/<ID>/`, `results/oracle/<ID>/{dev,confirmatory}/`.

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
| [`plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`](plans/TIDMAD_IMPLEMENTATION_ROADMAP.md) | The TIDMAD stage: phases, gates, code layout, verification. |
| [`plans/rps2026/`](plans/rps2026/) | R/PS 2026 round runbooks, agent prompts, reports and the run log (`EXPERIMENT_RUN_LOG.md`). Execution record for the submitted workshop paper; provenance, not guidance. |
| [`archive/2026-07-28_executed_plans/`](archive/2026-07-28_executed_plans/) | The July revision, correction, second-anchor and doc-consolidation plans — executed 2026-07-28, archived 2026-09-02. |
| [`archive/2026-09-02_paper2_dissolution/`](archive/2026-09-02_paper2_dissolution/) | The Paper 2 / Paper 3 outline; Paper 2 was dissolved into Papers 1 and 3 on 2026-09-02 (see the manuscript repository's `plan/THREE_PAPER_STORYLINE_2026-09-02.md`). |

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
