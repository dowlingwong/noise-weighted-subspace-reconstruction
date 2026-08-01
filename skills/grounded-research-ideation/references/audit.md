# Candidate audit

Run the audit after mechanism-specific collision retrieval. Judge the candidate; do not edit it during the judgment pass.

## Contents

1. Evidence preparation
2. Required checks
3. Collision analysis
4. Verdict rules
5. Bounded revision
6. Recurring anti-patterns

## 1. Evidence preparation

Assemble the canonical candidate, diagnosed bottleneck, closest full-text papers, mechanism-specific search hits, selected pattern entries, and user constraints. Mark evidence as retrieved, user-supplied, or inferred.

Do not manufacture a threat when none was found. Report the search scope and residual uncertainty instead.

## 2. Required checks

### A. Gap-closure failure scan

For each proposed gap closure, compare the candidate with the selected pattern's nearest failure mode. Cite the exact candidate component and supporting evidence. Classify `clear`, `borderline`, or `matched`.

Questions:

- Is the gap real and load-bearing?
- Does the proposed artifact change the achievable outcome?
- Has the candidate reintroduced the assumption it claims to remove?
- Is the mechanism distinguishable from extra scale, capacity, data, or tuning?

### B. Recipe-application check

Verify that the mechanism performs each selected move's distinctive operation and produces its promised artifact. A pattern label or generic parent move is insufficient.

Classify `applied`, `bypassed`, or `misclassified`. A bypass requires revision or reselection.

### C. Composition integrity check

Verify that all moves close parts of one bottleneck and form one causal story. Check that each component has a unique role and that the result is more than the sum of familiar parts.

Classify `coherent`, `repairable`, or `method-salad`.

### D. Paper-pointed threat check

Name the closest prior work and state the specific result that overlaps. Explain whether it subsumes the problem, mechanism, insight, or application. Distinguish internal proof notation from an externally meaningful contribution.

Classify `no-specific-threat-found`, `adjacent`, `partial-collision`, or `exact-collision`.

### E. Falsification and feasibility check

Verify that the load-bearing variable is observable, the negative control changes placement or mechanism rather than its definition, and the predicted effect is on an independent downstream outcome. Check data access, compute, dependencies, and unresolved author decisions.

Classify `credible`, `underspecified`, or `infeasible`.

### F. Problem-method fit check

Judge three dimensions without converting them into a false precision score:

- **Problem position:** important, non-trivial, and evidenced rather than a soft target.
- **Method quality:** deep enough, internally sound, and executable under the constraints.
- **Fit:** the method directly closes the stated bottleneck rather than solving a nearby easier task.

Record one strongest point and one most serious concern.

## 3. Collision analysis

Compare against each close paper on four axes:

1. problem framing;
2. core mechanism;
3. key insight;
4. application domain.

Treat mechanism plus key-insight overlap as the most serious combination. Similar domain alone is not a collision. Similar wording without operational overlap is not evidence. Use the worst credible close paper, not an average over unrelated results.

## 4. Verdict rules

Return one verdict with cited findings:

- **advance:** No hard-floor finding; all moves are applied; composition is coherent; falsification is credible; remaining risks are bounded.
- **revise:** No exact collision or fatal failure, but one or more specific fields can be repaired without changing the bottleneck or kill-switch commitments.
- **abandon:** Exact-mechanism collision, clearly matched fatal failure mode, infeasibility at the load-bearing step, or incoherent composition with no bounded repair.

Hard-floor findings override an otherwise fluent proposal. Never use an overall score to average them away.

## 5. Bounded revision

Separate judgment from repair. Emit named revision targets with affected fields, evidence, and success criteria. Permit one revision that:

- changes only targeted fields;
- preserves the selected bottleneck;
- preserves compute budget and falsification prediction byte-for-byte;
- may swap a sub-move only within the same high-level role;
- triggers re-audit of every affected check.

If repair requires a new core claim, different bottleneck, or softened falsification condition, regenerate from candidate construction.

## 6. Recurring anti-patterns

- **Novel-but-empty:** Vagueness avoids prior-art collision because nothing operational is committed.
- **Internal-object novelty:** A renamed lemma, variable, or proof quantity is presented as a new external contribution.
- **Proxy substitution:** The method optimizes a convenient signal without demonstrating that it tracks the target.
- **Method salad:** Familiar components are combined without one load-bearing interaction or unifying principle.
- **Assumption laundering:** A visible assumption is replaced by an equally strong, untestable, or unavailable one.
- **Uncontrolled comparison:** Gains can come from compute, capacity, data, or tuning rather than the claimed mechanism.
- **Benchmark leakage:** The diagnostic, pretext task, or retrieval process exposes target information.
- **Static-adversary comfort:** A security or robustness claim is tested only against an uninformed or outdated adversary.
- **Loose-bound victory:** The candidate beats a non-tight theoretical or empirical boundary without characterizing the true limit.
- **Agent theater:** Multiple solvers add orchestration and rhetoric but no measurable specialization or verification gain.
- **Cosmetic differentiation:** New terminology, domain transfer, or formatting masks the same mechanism and insight.
- **Kill-switch erosion:** Revision weakens the prediction, control, or compute commitment after criticism.
