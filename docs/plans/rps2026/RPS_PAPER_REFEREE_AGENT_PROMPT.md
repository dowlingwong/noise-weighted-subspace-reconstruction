# Dispatch prompt — orient, then referee the R/PS mini-paper

_Copy everything below the rule to a fresh agent session on Dow's laptop.
Read-only task: it produces a report, not a commit._

---

## Role

You are refereeing a four-page NeurIPS 2026 R/PS workshop submission. Two jobs,
strictly in order: **orient yourself until you can state the paper's thesis and
evidence structure in your own words**, then **review it adversarially**.

Do not skip the orientation. This manuscript is distilled from two longer papers
and sits on eleven rounds of archived experiments; a reviewer who has not loaded
that context produces findings the author resolved weeks ago, which is worse
than no review. But do not let orientation soften you either — the deadline is
**29 Aug 2026 AoE** and a comfortable review is worthless this close in.

You are read-only. Do not edit any file, do not commit, do not push, do not run
any experiment. Report findings; the author decides what to change.

## Step 1 — find the repository and prove you are in it

```bash
cd ~/Documents/noise-weighted-subspace-reconstruction
git log --oneline -1        # expect 6b07df4
git rev-parse HEAD          # expect 6b07df4eeac2a33259419837f3171748ce813f40
ls paper_rps2026/paper.tex  # must exist
```

If that path is wrong, find it by content rather than guessing:

```bash
find ~ -type d -name paper_rps2026 -not -path '*/.git/*' 2>/dev/null
```

Verify by SHA, not by path. If HEAD is **not** `6b07df4`, say so in your report
before reviewing — you are then looking at a different state than this prompt
describes, and every line number and page reference below may be stale.

The manuscript is `paper_rps2026/`. The built PDF is `paper_rps2026/paper.pdf`:
11 pages, main text 1–4, references 5–6, appendix 7–11. Rebuild with `make` from
`paper_rps2026/` if you need to; that is the one write this task permits, and
only into build artifacts.

### A trap, stated up front

There is a **`paper-review` skill installed on this machine, and it is scoped to
a different document** — the long arXiv preprint *"Noise Covariance Defines
Reconstruction Geometry"* in the sibling folder
`arXiv-v1-polished-extended-preprint`. It will happily trigger on "referee this
paper" and send you to `02_unified_objective.tex`, Theorems 4.1/5.1 and
`appH_symbols.tex`, none of which exist in the mini-paper.

You may read that skill for its four-pass *structure*, which is good. Do not
follow its file table, its section names, or its symbol-grading step. If you
find yourself reading `main.tex`, you are in the wrong repository.

The long preprint is still useful as a **source**: the mini-paper is distilled
from it, and `paper_rps2026/notes/source_map.md` maps each mini-paper section
back to its origin. Consult it to check whether a compressed claim survived
compression intact.

## Step 2 — orientation reading, in this order

| Read | For |
|---|---|
| `paper_rps2026/README.md` | The one-paragraph thesis, venue, and build instructions |
| `paper_rps2026/paper.pdf` | The whole thing, cover to cover, before any source file |
| `paper_rps2026/notes/claim_audit.md` | Every claim, its status, its evidence file, whether it is in the main text, and the control on it. **This is the spine of the review.** |
| `paper_rps2026/notes/source_map.md` | What was kept and what was deliberately cut from the source papers |
| `docs/plans/rps2026/ROUND10_PAPER_REVISION_QA.md` | The record of what has already been audited and settled, with evidence |
| `paper_rps2026/notes/openreview_metadata.md` | Submission fields; the abstract there must match the PDF |
| `results/rps2026/README.md` | What each archived experiment directory is |

Then, and only then, the sources: `paper_rps2026/sections/00_abstract.tex`
through `05_discussion.tex`, `appendix/appendix_proofs.tex`,
`appendix/appendix_experiment_details.tex`, `tables/`.

## Step 3 — the idea the paper is trying to prove

State this back in your own words before you critique it. If your restatement
and the list below disagree, the paper has a clarity defect and that is itself a
finding — say which reading the text actually supports.

**The thesis.** A reconstruction method is the pair $(\Sigma^{-1}, \mathcal{F})$.
The measurement likelihood fixes the residual geometry; the architecture fixes
the admissible representation class. These are separate choices, and whether the
metric choice *bites* is a measurable property of the instrument and the
representation — not a matter of taste. For Gaussian noise, inverse covariance
is the likelihood-consistent metric, so measured observations can supply their
own reconstruction supervision without clean targets.

**What is deliberately not claimed.** No new estimator is proposed. NFPA is
positioned as a covariance-aware member of the established MPCA/Tucker family,
not a new factorization. Nothing claims NFPA beats CW-PCA intrinsically.

**Four representation classes, and the thing reviewers get wrong.** Rungs 1–4
are *labels, not a capacity ordering*. Rung 3 **restricts** rung 2. Rung 1 is
rung 3's degenerate fixed-template case. Rung 4 is not comparable to any of
them. If you catch yourself reading it as a ladder of increasing power, re-read
§3 — and if the text invites that misreading anywhere, that is a finding.

**The weak claim versus the substantive one.** That a model trained under a
metric leads under that metric is close to tautological; the paper says so. The
substantive result is **truth-referenced**: the covariance-weighted factors
recover the *planted* channel subspace almost exactly, and decode held-out
planted signal parameters better, while temporal recovery stays weak. Check that
the paper keeps this ordering everywhere and never leans on the tautological
version to carry weight.

**Two experimental conditions that must never be conflated.** The primary
reversal experiment runs at nonseparability $\alpha = 0.1$. The truth-referenced
recovery is evaluated separately at the exactly separable $\alpha = 0$. They
live in `results/rps2026/E8_semantics/primary_alpha_0p1/` and
`.../separable_alpha_0/`. An earlier draft conflated them and it took an
external reviewer to catch. **Verify the current text does not** — check §4, the
figure caption, both panel titles, and the conclusion.

**Evidence classes, which the paper is strict about.** Frozen confirmatory /
post-hoc replication / delayed predeclared / post-hoc exploratory. The
estimated-covariance result (E10b) is **one frozen sensitivity condition at one
calibration size**, never a robustness sweep. E11 is a post-hoc diagnostic
correction, appendix-only. Rung 4 is a **two-seed pilot** reported as an open
observation. Flag any sentence that upgrades an evidence class.

## Step 4 — the review

Four passes. Run all four unless something in pass 1 is severe enough to stop on.

**Pass 1 — math and correctness.** Follow every derivation in §2, §3 and
Appendix A. Check Proposition 1 (bridge), Proposition 2 (Kronecker-restricted
bridge) and Corollary 1: do the stated hypotheses suffice, is every hypothesis
used, is the proof complete or does it gesture?

*Verify algebra programmatically — numpy or sympy — not by eye.* State in your
report which results you machine-checked. This is not boilerplate: a Kronecker
factor-order error survived roughly ten rounds of human reading in this
manuscript and was only caught by running it. Specifically, re-derive the
vec identity in Appendix A.3 yourself and confirm the projector order in §3, the
appendix, and Figure 2 all agree. Commits `a665f8e` and `6b07df4` claim to have
fixed exactly this — treat that claim as unverified and check it.

Cross-check **every** number in the main text against the archived file behind
it. Trace each to a result file; never verify a number by re-deriving it from
other numbers in the paper. An unsourced number is a blocking finding.
`paper_rps2026/tables/round10_sensitivity.tex` is machine-generated by
`scripts/make_rps_round10_table.py` from digest-pinned inputs — confirm it
regenerates identically rather than trusting it.

**Pass 2 — referee critique.** Write as a reviewer for this workshop. Be direct.
For each contribution claimed in §1, name the specific evidence and say whether
it carries the claim. Where is the paper vulnerable to "this is a re-derivation
of known linear algebra"? Is the novelty boundary drawn honestly? What is the
strongest rejection argument available to a hostile reviewer, and does the paper
already answer it?

**Pass 3 — clarity.** Four pages is the binding constraint. Which paragraphs
earn their space and which do not? Is §3 followable by someone who has not read
the source papers? The paper carries one main figure (Figure 1, two panels) and
one appendix schematic (Figure 2) — is the visual budget spent well, and is
anything currently in prose that should be a figure?

**Pass 4 — submission readiness.** Double-blind anonymity in the *built PDF*,
not the sources. NeurIPS style conformance and the four-page limit (references
and appendix excluded). Every claim in `claim_audit.md`. Abstract in
`openreview_metadata.md` matching the PDF word for word.

## On re-raising settled items

`ROUND10_PAPER_REVISION_QA.md` records issues already investigated and closed —
some with compile-level evidence, such as the workshop footer, which was raised
three times and is the template's designed behaviour.

You are **not** forbidden from disagreeing with a settled decision. You are
required to engage with the recorded reasoning rather than restating the
original complaint as though it were new. "The footer is wrong" is noise.
"The QA record argues X from `neurips_2026.sty:397`, but that reasoning fails
because Y" is a finding. Same standard for everything in that document.

## Deliverable

Write your report to `docs/plans/rps2026/REFEREE_REPORT_<yyyy-mm-dd>.md`. That
is the only file you create.

Structure it as:

1. **Orientation check** — the thesis, the four classes, the weak-vs-substantive
   distinction and the two α conditions, in your own words. If your
   understanding diverged from Step 3 anywhere, say where and why.
2. **Blocking findings** — things that should stop submission. Each one: the
   claim, the location, the evidence you checked, why it fails.
3. **Non-blocking findings** — ordered by how much a reviewer would care.
4. **What you machine-checked**, with the code or commands you ran.
5. **What you could not verify**, and what access you would have needed.
6. **Deadline call** — given 29 Aug AoE, what you would fix and what you would
   leave, with reasoning about cost.

Separate what you *verified* from what you *suspect*. A finding you did not
check is a question, and should be labelled as one. Say plainly when something
is fine — a review that manufactures findings to look thorough wastes the
author's remaining days.
