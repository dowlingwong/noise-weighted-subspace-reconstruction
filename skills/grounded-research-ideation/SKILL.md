---
name: grounded-research-ideation
description: Turn a broad research direction, seed papers, or literature bundle into one evidence-grounded, falsifiable, and auditable research idea. Use when Codex needs to generate, sharpen, differentiate, stress-test, or review an early-stage research idea; identify a literature bottleneck; select and compose reusable research moves; check a novelty claim against prior art; or produce a reviewer-facing idea card. Designed for ML/AI and adaptable to literature-rich technical fields; not for writing a full paper, predicting acceptance, or claiming implementation success.
---

# Grounded Research Ideation

Produce exactly one defensible research idea or an honest stop report. Treat the pattern library as reasoning vocabulary, never as an acceptance predictor. Keep retrieved evidence, user-provided facts, and model inference visibly separate.

## Load the references progressively

- Read [references/patterns.md](references/patterns.md) during pattern selection and candidate construction.
- Read [references/audit.md](references/audit.md) before collision checking, critique, or revision.
- Read [references/schemas.md](references/schemas.md) before emitting a final card, stop report, or JSON artifact.

## Apply the workflow

### 0. Establish the contract

Capture the research direction, seed literature, contribution type, constraints, resources, and intended audience. Infer conservative defaults when omissions do not change the scientific direction. State material assumptions. Ask only when a missing choice would produce meaningfully different research programs.

Do not promise novelty, feasibility, acceptance, or impact. Aim to make one proposal inspectable enough for the researcher to decide whether it merits implementation.

### 1. Ground the literature

Search current literature unless the user explicitly forbids external retrieval. Prefer primary sources and full text. Use several complementary sources when available, deduplicate versions, and retain stable identifiers or links.

Build a focused evidence bundle:

1. Retrieve broadly around the problem, neighboring formulations, and recent mechanisms.
2. Rank by conceptual relevance, not keyword overlap alone.
3. Deep-read the closest works, including methods, limitations, and evaluation conditions.
4. Record each source's solved problem, mechanism, evidence, and unresolved remainder.
5. Label model-memory items `awareness-only`; never use them to support novelty or quantitative claims.

When retrieval is unavailable, proceed only from user-provided sources and mark the result `partial evidence`. Stop if the evidence cannot support a concrete bottleneck.

### 2. Diagnose one bottleneck

Arrange the closest methods into a small lineage: what each inherits, changes, solves, and leaves open. Look for:

- an additive gap: a capability no current leaf delivers;
- a subtractive gap: a shared, load-bearing assumption that may be unnecessary;
- a measurement gap: current evidence cannot isolate the claimed cause;
- a representation or optimization gap: the formulation makes the target unnecessarily hard;
- a boundary gap: a known limit is not characterized tightly or surpassed.

Accept a bottleneck only when retrieved evidence shows it exists, it changes an important outcome, no inspected work closes it, and it is testable under the stated constraints. Otherwise emit a `do-not-generate` report using the schema reference.

### 3. Select research moves by structural fit

Read the pattern library. Select one to three moves, defaulting to two only when they form a coherent contribution. For each move, record:

- its role: diagnostic, primary construction, supporting construction, or verification;
- the bottleneck feature that triggers it;
- the concrete artifact it must produce;
- the nearest failure mode to guard against.

Select by gap closure, not frequency, venue preference, or apparent acceptance rate. Pattern names describe the process and must never become the novelty claim.

### 4. Instantiate one candidate

Construct a mechanism, not a theme. Specify:

- a concise title and one-sentence thesis;
- the literature-grounded motivation;
- ordered method steps with inputs, operations, and outputs;
- how each selected move closes a named part of the bottleneck;
- the substantive delta from each closest prior work;
- the smallest credible compute and data envelope;
- a core claim and explicit assumptions;
- one falsification prediction.

The falsification prediction must name a load-bearing variable, predict a downstream outcome, and include a non-tautological negative control. Preserve the compute budget and falsification fields as kill-switch commitments through bounded revision. If they need to change, regenerate the candidate rather than silently softening it.

### 5. Search for collisions and audit

Run a second, mechanism-specific search using the candidate's operations, mathematical objects, claimed delta, and application. Compare each close work on problem framing, core mechanism, key insight, and application domain.

Apply every check in the audit reference. Return `advance`, `revise`, or `abandon` with evidence. Exact-mechanism collision, a clearly matched fatal failure mode, or an unmitigable incoherent composition requires abandonment or fresh generation; cosmetic renaming is not a repair.

Allow at most one bounded revision. Change only the named weak fields. Preserve the bottleneck and kill-switch commitments. Re-run the affected audit checks after revision.

### 6. Render and validate

Lead with the idea, not the process log. Return:

1. title and thesis;
2. motivation and bottleneck;
3. method flow;
4. differentiation from the closest work;
5. falsification test and feasibility;
6. audit verdict, remaining risks, and evidence status.

When writing files, emit both a readable idea card and the canonical JSON from the schema reference. Validate the JSON with:

```bash
python scripts/validate_candidate.py final_candidate.json
python scripts/validate_candidate.py final_candidate.json --initial initial_candidate.json
```

Fix validation failures before delivery. The validator checks structure and commitment preservation, not scientific truth.

## Maintain epistemic discipline

- Cite only sources actually retrieved or supplied by the user.
- Attach each differentiation claim to a specific source and result.
- Mark unsupported design choices as author decisions.
- Prefer directional falsification predictions unless same-setting evidence justifies numbers.
- Treat missing full text, ambiguous precedence, and narrow evidence as explicit risks.
- Do not turn descriptive conference patterns into causal claims or proposal-ranking priors.
- Leave experiment matrices, manuscript writing, and downstream scientific claims to a separately authorized task.
