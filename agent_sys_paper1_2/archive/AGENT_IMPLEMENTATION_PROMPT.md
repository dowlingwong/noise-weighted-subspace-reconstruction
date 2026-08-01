# Implementation prompt for reliable scientific-agent Papers R1 and R2

_Copy-ready task brief for a coding/research agent · 2026-08-01_

---

## 🎯 Role and objective

You are the implementation and manuscript agent for a two-paper reliable scientific
agents program. Your goal is to make the existing research specification executable,
then refine both prospective manuscripts without fabricating evidence.

You must deliver three connected outcomes:

1. Refine the R1 and R2 experimental plans into machine-readable, validated,
   reproducible protocols.
2. Refine the R1 DMLR and R2 TMLR LaTeX projects into technically substantive
   prospective manuscripts whose claims trace to protocol and artifact identifiers.
3. Build the minimum tested software/research scaffold for R1 and R2 on top of the
   existing governed autoresearch harness.

Do not treat these as independent tasks. R1 freezes the trace, fault, split,
adjudication, and endpoint contract. R2 must consume that contract without changing
R1's held-out groups or outcome definitions.

## 📚 Required reading order

Read the following files completely before editing anything:

1. `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/agent_sys_paper1_2/README.md`
2. `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/agent_sys_paper1_2/AGENT_PRIORITY_AND_READINESS.md`
3. `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/agent_sys_paper1_2/10_reliable_scientific_agents_research_program.md`
4. `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/agent_sys_paper1_2/EXPERIMENT_PLAN.md`
5. `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/agent_sys_paper1_2/IMPLEMENTATION_PLAN.md`
6. `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/agent_sys_paper1_2/WRITING_AND_SUBMISSION_PLAN.md`
7. `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/agent_sys_paper1_2/PUBLICATION_PLAN_EVALUATION.md`
8. Both LaTeX project READMEs, main files, section files, template locks, and
   Makefiles under `agent_sys_paper1_2/latex/`
9. `/Users/wongdowling/Documents/autoresearch_harness/README.md`,
   `pyproject.toml`, current control-plane event types, memory/event stores, node
   contracts, evaluation/reporting modules, and relevant tests
10. Any `AGENTS.md`, `CLAUDE.md`, or equivalent instruction file governing a path
    before modifying that path. If work enters `harness/claw-code`, read and follow
    its `AGENTS.md` first.

If the experimental-design, scientific-writing, citation-management or
research-lookup, and Markdown/Mermaid skills are available, use them. Follow their
instructions, but the project documents above define the scientific scope.

## ⚠️ Non-negotiable boundaries

- Inspect `git status` in both repositories before editing. Preserve all unrelated
  and pre-existing changes; do not reset, discard, overwrite, or reformat them.
- Do not launch paid API calls, GPU training, long campaigns, human adjudication, or
  full pilot/confirmatory collection without explicit user authorization.
- Do not invent task instances, model versions, budget amounts, author metadata,
  redistribution rights, citations, numerical results, or effect directions.
- Do not write prose that implies a planned experiment has been completed.
- Do not count trace events, steps, checkpoints, or repeated seeds as independent
  task-level replicates.
- Keep fault truth sealed from the evaluated policy. Keep training reward, agent
  self-report, and final adjudication as separate channels.
- Do not duplicate the existing control plane, event store, node registry,
  evaluation, or reporting stack. Extend it through compatibility adapters.
- Do not modify the `harness/claw-code` submodule unless the vertical slice cannot
  be implemented through the existing worker interface. Explain and obtain any
  required authority before crossing that boundary.
- Make all schedules, manifests, schemas, derived tables, and figures reproducible
  from recorded seeds and content hashes.
- Treat R1's test outcomes and R2's unseen-composition outcomes as sealed during
  development.

## 📍 Phase 0 — Preflight and traceability

Before implementation:

1. Inventory the current files and tests in both repositories.
2. Record dirty-worktree paths that must not be disturbed.
3. Create a claim–artifact matrix for H1–H4. For each claim, record the experimental
   unit, intervention/predictor, comparator, primary outcome, analysis ID, required
   artifact, manuscript destination, and current status.
4. Map every proposed reliability module to the nearest existing harness interface.
5. Produce a short gap report. Distinguish blocking decisions from reversible
   implementation choices.

**Gate P0:** no code edit begins until the paper boundary, repository boundary,
protected user changes, and compatibility map are explicit.

## 🧪 Phase 1 — Refine the experiments

### R1 protocol

Convert R1-P, R1-C, and R1-E1 through R1-E4 into versioned protocol artifacts under
`agent_sys_paper1_2/protocols/r1/`. Add shared terminology, endpoint, and
claim–artifact registries under `protocols/common/`.

The machine-readable R1 protocol must specify:

- task/template grouping, experimental unit, block, treatment, repeat, and censoring
- factor names, level types, permitted values, pilot-only versus confirmatory status
- no-fault and benign controls
- fault assignment, severity, persistence, timing, observability, and injection
  fidelity rules
- workload/model/task/fault split groups and near-duplicate isolation
- primary and secondary endpoints with exact formulas and reaction windows
- invalid-run and missingness disposition
- randomization seed, run-order generation, balance checks, and design-matrix
  estimability diagnostics
- pilot engineering gates and rules for modifying/dropping an injector
- sample-size simulation assumptions and the rule for replacing assumptions with
  pilot estimates
- analysis identifiers for survival/hazard, warning/calibration, attribution, and
  recoverability

Implement deterministic protocol validation and design-generation scripts. Include
small synthetic/example inputs and golden outputs. The mixed-level design must report
rank, estimable terms, condition number or variance-inflation diagnostics, balance by
block, and any aliased or unsupported interaction. Never describe a design as
resolution-IV when it contains mixed two- and three-level factors unless the formal
alias analysis supports that terminology.

### R2 protocol

Convert R2-P, R2-C1, R2-C2, R2-T, and R2-A into versioned protocol artifacts under
`agent_sys_paper1_2/protocols/r2/`.

Specify:

- eligible decision points and observable policy state
- verify, continue, recover, rollback, abstain, and escalate actions
- action availability, costs, budgets, selection propensities, and postconditions
- never/always/fixed/random/threshold/adaptive/oracle policies
- blocked trajectory-level randomization or matched-episode fallback
- verified-intervention record and unsupported-counterfactual representation
- controller utility and success–cost frontier
- six post-training arms, data/token/step/compute matching, checkpoint selection,
  training seeds, and clean-task non-inferiority margin
- seen, unseen-task, unseen-injector, unseen-composition, unknown-fault,
  verifier-substitution, clean-task, and environment-shift evaluation sets
- primary contrasts, multiplicity families, missingness, and negative-result fallback

**Gate P1:** protocol validation, deterministic schedule regeneration, leakage checks,
and synthetic power/design diagnostics pass. Do not run real trajectories yet.

## 🔧 Phase 2 — Build the shared reliability scaffold

Implement the minimum vertical slice in
`/Users/wongdowling/Documents/autoresearch_harness` using the layout in
`IMPLEMENTATION_PLAN.md`:

```text
src/autoresearch/reliability/
├── schema/
├── observability/
├── faults/
├── adjudication/
├── benchmark/
├── warning/
├── control/
└── training/
```

Also create matching `configs/reliability/`, `experiments/reliability/`, and
`tests/reliability/` surfaces. Do not create empty directories merely to match the
diagram. Each package must contain either an implemented contract used by the
vertical slice or a README stating its deferred interface and dependency.

The first vertical slice must support:

1. Load and validate an immutable run manifest.
2. Resolve a group-isolated benchmark split.
3. Seal a deterministic fault assignment while exposing only policy-visible state.
4. Emit typed proposal, action, result, snapshot, state-difference, fault,
   verification, belief, recovery, terminal, and adjudication records through an
   adapter to the existing append-only event store.
5. Validate integrity, parent links, visibility classes, and typed absence.
6. Derive one synthetic R1 endpoint from raw events.
7. Export one manifest-bound LaTeX table or scalar macro with provenance.

Required golden tests include manifest immutability, event round-trip,
append-only behavior, fault-truth noninterference, failed-injection disposition,
state-difference reconciliation, task/template split leakage, adjudicator/reward
separation, deterministic replay fixtures, and provenance propagation.

**Gate P2:** the synthetic vertical slice and targeted tests pass from a clean
environment. Existing harness tests touched by the change also pass.

## ✍️ Phase 3 — Refine Paper R1

Work in `agent_sys_paper1_2/latex/r1_dmlr/` and preserve the pinned official DMLR
style. Use the section contract in `WRITING_AND_SUBMISSION_PLAN.md`.

1. Build a closest-work matrix from verified primary sources and add validated
   BibTeX records. Never infer bibliographic metadata from memory when it can be
   verified.
2. Replace bracketed prompts in Introduction, Related Work, Benchmark, Methods,
   Discussion/Limitations, Artifact Use, and Broader Impact with full scientific
   paragraphs.
3. Describe only implemented or explicitly prospective methods. Use future/protocol
   language for unexecuted work.
4. Keep Results connected to generated inputs. If results are unavailable, present a
   clearly marked analysis/display shell; do not insert illustrative numbers.
5. Bind every headline sentence to a claim card and protocol/analysis ID.
6. Add accessible text-source diagrams for the governed trace architecture and R1
   study design. Use those sources to generate a graphical abstract and methods
   schematic if the relevant visualization skill is available.
7. Add artifact-access, licensing, maintenance, intended-use, and prohibited-use
   sections required by the DMLR plan.
8. Compile from a clean temporary copy using the pinned template and inspect the log
   for fatal errors, undefined references, missing citations, and material overfull
   boxes.

**Gate P3:** R1 is a coherent prospective DMLR manuscript with no fabricated result,
no unresolved bracket prompt outside explicitly generated Results, verified
citations, and a clean build.

## ⚙️ Phase 4 — Complete R1 pilot readiness

Connect the R1 protocol to the reliability scaffold using synthetic tasks and fault
fixtures only. Demonstrate:

- one no-fault control and one injected-fault counterfactual
- event and required-field completeness reporting
- failed-injection classification without relabeling it as a successful fault
- blinded adjudication input that excludes sealed truth
- warning/calibration metric computation on a toy trace
- standardized recoverability probe plumbing
- disposition, cost, and provenance export into R1's generated manuscript inputs

Do not claim pilot gates have passed from synthetic fixtures. Report only that the
instrumentation is ready for a separately authorized pilot.

**Gate P4:** a single documented command runs the synthetic R1 readiness workflow and
reproduces its validation report and example table.

## 🔄 Phase 5 — Build R2 on the frozen R1 contract

Extend the scaffold with typed controller and training-view interfaces. R2 must use
the same R1 event, split, endpoint, and adjudication versions.

Implement synthetic support for:

- policy eligibility and action masks
- verification/recovery cost accounting
- selection propensity logging
- matched-cost baseline policies
- independently adjudicated intervention transitions
- explicit unknown values for unsupported counterfactual actions
- immutable training-view construction with split/deduplication isolation
- matched-data/compute ledger and clean-task regression gate
- reward-channel and verifier-substitution audit hooks

Do not implement a large learned controller or perform post-training before the
interfaces and baselines are testable. A deterministic adaptive-policy stub may be
used to exercise the contract, but label it as a fixture rather than a result.

**Gate P5:** the synthetic controller/data/training-view path passes tests, and no R2
code can access sealed R1 test outcomes or fault truth.

## 📚 Phase 6 — Refine Paper R2

Work in `agent_sys_paper1_2/latex/r2_tmlr/` and preserve TMLR review anonymity.

1. Expand the verified literature matrix and BibTeX database.
2. Replace bracketed prompts in Introduction, Related Work, Problem Formulation,
   Method, Dataset, Experimental Design, Discussion/Limitations, and Broader Impact
   with full scientific prose.
3. Separate the randomized controller claim from the matched post-training claim.
4. Define utility, budget, propensities, postconditions, non-inferiority, and all
   transfer regimes precisely.
5. Keep Results as generated shells until real evidence exists.
6. Add accessible architecture/study-design sources, a graphical abstract, and a
   methods schematic when visualization tooling is available.
7. Run the LaTeX build and `make anonymity-check`; also inspect source, PDF metadata,
   filenames, URLs, supplement, and artifact references for identity leakage.

**Gate P6:** R2 is a coherent prospective anonymous TMLR manuscript, uses the frozen
R1 contract, contains no fabricated result, and passes build/anonymity checks.

## ✅ Phase 7 — Unified verification and handoff

Add a documented project-level readiness command or script that performs only
safe, local checks:

1. Validate common, R1, and R2 protocols.
2. Regenerate deterministic example schedules and compare their hashes.
3. Run targeted reliability tests in the harness.
4. Run the synthetic R1 and R2 vertical slices.
5. Verify claim–artifact traceability and missing generated inputs.
6. Build both manuscripts against their pinned official templates.
7. Run the TMLR anonymity audit.
8. Emit a machine-readable and human-readable readiness report.

The readiness report must classify every H1–H4 claim as `planned`, `instrumented`,
`pilot-ready`, `pilot-observed`, `confirmatory-observed`, `failed`, or `blocked`.
Only real, audited evidence may advance a claim beyond `pilot-ready`.

## 📦 Required final deliverables

At handoff, report:

- every file created or changed, grouped by repository and phase
- the claim–artifact matrix and remaining user decisions
- tests and builds run, exact commands, and outcomes
- deferred work and why it was deferred
- any dirty or unrelated files intentionally left untouched
- whether real experiments were run; the expected answer is **no** unless separately
  authorized
- the next single recommended action, normally approval of the R1-P pilot budget and
  concrete task/model manifest

Do not mark the task complete merely because directories and placeholder files exist.
Completion requires validated protocols, a tested synthetic vertical slice for both
papers, substantive prospective LaTeX sections, clean builds, and an auditable
readiness report.
