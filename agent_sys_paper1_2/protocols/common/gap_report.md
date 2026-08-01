# Preflight inventory and gap report
# Phase 0 Gate · 2026-08-01
# Must be completed before any code edits.

## Repository boundaries

### noise-weighted-subspace-reconstruction
- Role: Research specifications, protocols, manuscript sources, paper-facing artifact contracts
- Protected paths (dirty/unrelated - do NOT modify):
  - Deleted files: paper1_preprint_arxiv.pdf, paper2_nonlinear_reconstruction.pdf
  - Modified: paper3/latex/paper3_proposal.pdf, paper3/latex/paper3_proposal.tex
  - Untracked new directories: agent_sys_paper1_2/ (our working directory), papers/, skills/, sources/
- Our working directory: agent_sys_paper1_2/
- LaTeX projects: agent_sys_paper1_2/latex/r1_dmlr/, agent_sys_paper1_2/latex/r2_tmlr/

### autoresearch_harness
- Role: Runtime implementation home
- Protected paths (dirty - do NOT modify):
  - M submodules: A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation, harness/claw-code
  - M files: paper_figures.ipynb, poster-KDD-ETAAI26/*
- Implementation surface: src/autoresearch/ (extend, do not fork)
- Guide: CLAUDE.md at harness root (authoritative)

## Compatibility map: reliability modules → harness interfaces

| Reliability module | Nearest harness interface | Extension strategy |
|---|---|---|
| schema/ | control_plane/events.py (CampaignEvent), memory/schemas.py (TrialRecord) | Add typed event payloads; extend TrialRecord with reliability fields via extra dict |
| observability/ | worker/base.py (WorkerResult) | Add state-difference adapters feeding into WorkerResult.extra |
| faults/ | nodes/spec.py (NodeSpec) | Add fault manifest configs; injector registry as new package |
| adjudication/ | evaluation/metrics.py (CampaignMetrics) | Add independent adjudication records; extend endpoint computation |
| benchmark/ | nodes/spec.py, configs/nodes/ | Add split resolver, task/fault manifests alongside node configs |
| warning/ | New analysis package | Consume event store; produce hazard/attribution predictions |
| control/ | New method package | Consume R1 belief outputs; emit policy decisions |
| training/ | New method package | Build immutable training views from verified intervention transitions |

## Gap report

### Blocking decisions (must be resolved before full implementation)
1. Task instances and model/provider versions are placeholder - real manifests needed before pilot
2. API/GPU/human-review budgets not authorized - required before confirmatory collection
3. Final fault mechanisms retained after pilot TBD - cannot freeze full injector set
4. Confirmatory sample size dependent on pilot variance - cannot commit trajectory count

### Reversible implementation choices (can proceed)
1. Protocol schemas and validation are independent of concrete task lists
2. Reliability scaffold interfaces are independent of model choices
3. Synthetic fixtures and golden tests use deterministic, self-contained examples
4. Manuscript prospectus sections use future/protocol language
5. LaTeX build infrastructure is template-pinned and independent of results

### What exists
- Governed harness with control plane, worker interface, append-only store, event stream
- 6 ML nodes with metric parsers and validity checks
- Campaign lifecycle with pending-guard, budget enforcement, keep/discard decisions
- Evaluation metrics, paper table/figure exports with provenance
- 20+ tests covering schema, state machine, permissions, memory, nodes, events

### What must be built
- src/autoresearch/reliability/ tree (all 8 subpackages)
- configs/reliability/ tree
- experiments/reliability/ tree
- tests/reliability/ tree
- protocol YAML files for R1 and R2
- Protocol validation and design-generation scripts
- Claim-artifact matrix (done - protocols/common/)
- Terminology and endpoint registries (done - protocols/common/)
- Full scientific prose for both manuscripts
- Project-level readiness command
