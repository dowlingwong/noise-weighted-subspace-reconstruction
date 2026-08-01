# Output contracts

Use one of three outcomes: an idea card, a do-not-generate report, or an abandon report. Keep detailed provenance in the canonical JSON even when the readable card is concise.

## Contents

1. Canonical candidate JSON
2. Readable idea card
3. Do-not-generate report
4. Abandon report

## 1. Canonical candidate JSON

Emit valid JSON with this shape. Add fields when useful; do not remove required fields.

```json
{
  "version": "1.0",
  "direction": "Research direction supplied by the user",
  "evidence_status": "grounded",
  "assumptions": ["Material assumption or inferred constraint"],
  "bottleneck": {
    "statement": "One concrete unresolved structural gap",
    "why_load_bearing": "Why closing it changes an important outcome",
    "lineage": [
      {
        "work": "Stable paper identifier or title",
        "inherits": "Prior assumption or mechanism",
        "changes": "What this work changes",
        "leaves_open": "Evidence-supported remainder",
        "source": "Verified URL, DOI, arXiv id, or user-provided citation"
      }
    ],
    "closest_work": [
      {
        "work": "Stable paper identifier or title",
        "result": "Specific relevant result",
        "remaining_gap": "What it does not close",
        "source": "Verified source"
      }
    ]
  },
  "pattern_composition": [
    {
      "pattern": "Pattern name from the library",
      "role": "primary construction",
      "trigger_fit": "Bottleneck feature that calls for this move",
      "output_artifact": "Concrete artifact this move must produce",
      "guarded_failure": "Nearest failure mode"
    }
  ],
  "title": "Candidate title",
  "thesis": "One-sentence contribution thesis",
  "motivation": "Literature-grounded motivation",
  "mechanism": [
    {
      "step": 1,
      "name": "Step name",
      "inputs": "Inputs and prerequisites",
      "operation": "Concrete operation",
      "outputs": "Observable artifact",
      "why": "How this closes the bottleneck"
    }
  ],
  "core_claim": "Testable core claim",
  "differentiation": [
    {
      "prior_work": "Closest work",
      "shared": "What is inherited or held constant",
      "delta": "Substantive external difference",
      "source": "Verified source"
    }
  ],
  "compute_budget": "Smallest credible compute and data envelope",
  "falsification": {
    "load_bearing_variable": "Observable variable carrying the mechanism claim",
    "prediction": "Directional downstream prediction",
    "negative_control": "Non-tautological intervention preserving obvious confounds",
    "failure_condition": "Observation that would reject the core mechanism"
  },
  "feasibility": {
    "verdict": "credible",
    "requirements": ["Data, system, or tooling requirement"],
    "unknowns": ["Open author decision or unverified dependency"]
  },
  "audit": {
    "verdict": "advance",
    "checks": [
      {
        "name": "paper-pointed threat",
        "status": "adjacent",
        "finding": "Evidence-backed finding"
      }
    ],
    "threats": ["Closest residual prior-art or validity threat"],
    "revision_targets": []
  },
  "provenance": {
    "retrieved_sources": ["Verified source"],
    "user_sources": [],
    "awareness_only": [],
    "retrieval_date": "YYYY-MM-DD"
  }
}
```

Use `evidence_status` values `grounded` or `partial`. Use audit verdicts `advance`, `revise`, or `abandon`.

## 2. Readable idea card

```markdown
# <Title>

**Thesis:** <One sentence>

## Motivation and bottleneck
<What the closest work achieves, where it stops, and why the remainder matters.>

## Method
1. **<Step>:** <input -> operation -> output -> reason>.
2. **<Step>:** <input -> operation -> output -> reason>.

## Differentiation
- **Against <work>:** <shared substrate>; <substantive delta>; <source>.

## Falsification
- **Load-bearing variable:** <variable>.
- **Prediction:** <downstream directional effect>.
- **Negative control:** <intervention>.
- **Failure condition:** <what would reject the mechanism>.

## Feasibility and audit
- **Compute/data envelope:** <budget>.
- **Verdict:** <advance/revise/abandon>.
- **Open decisions:** <unknowns>.
- **Evidence status:** <grounded/partial>.
```

## 3. Do-not-generate report

Use before candidate construction when no defensible bottleneck can be grounded.

```markdown
# Do not generate

**Reason:** <direction too broad, evidence too sparse, no unresolved gap, or incompatible constraints>.

## What was established
- <Evidence-backed finding and source>.

## What remains missing
- <Specific evidence or decision needed>.

## Safe next step
- <Narrowing question, retrieval target, or source request>.
```

## 4. Abandon report

Use after candidate construction when the audit hits a hard floor.

```markdown
# Candidate abandoned: <title>

**Hard-floor finding:** <exact collision, fatal failure, infeasibility, or incoherent composition>.

## Evidence
- **Threat or failure:** <specific paper/result or pattern failure>.
- **Subsumption argument:** <why the contribution is already occupied or invalid>.

## Why bounded revision cannot fix it
<Explain which load-bearing field would need to change.>

## Viable pivots
- <Distinct residual direction requiring fresh grounding>.
```
