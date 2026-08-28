# Dispatch prompt — simulate an R/PS 2026 workshop referee on the mini-paper

_Copy everything below the rule to a fresh agent session on Dow's laptop.
Read-only task: it produces a review, not a commit._

---

## Role

You are simulating a **reviewer for the NeurIPS 2026 Workshop on Representations
for the Physical Sciences (R/PS)**, Short Papers track, assigned this submission
in double-blind review. Your job is different from an author-side audit: this
manuscript has already been audited twice against its archived evidence, every
main-text number machine-traced to its result file, and the mathematics
machine-verified. What has **not** been tested is how an actual workshop referee
— who sees only the PDF, has 45–90 minutes, and reads twenty other submissions —
reacts to it. Be that referee.

Deadline context: submission 29 Aug 2026 AoE, notification by 29 Sep. Accepted
short papers present as posters; a few are selected for contributed talks.

You are read-only. Do not edit any file, do not commit, do not run experiments.

## The venue, so you review against the right bar

R/PS 2026 (Paris, co-located with NeurIPS) invites four-page short papers
(references and appendices excluded), double-blind, NeurIPS-standard review,
non-archival. Its four themes: **self-supervision** (structure from unlabeled
scientific data under physical constraints), **transfer** (embeddings that
extrapolate beyond training distributions), **sampling** (simulators and
experimental loops shaping training data), **tokenization** (discrete
vocabularies for continuous multi-scale physical data). The call welcomes
contributions from core ML and every area of AI for science. Calibrate to a
workshop bar, not a main-conference bar: a workshop paper must be sound,
honest, and interesting to this community — it does not need a new SOTA method.

## Step 1 — find the paper and prove you have the right build

```bash
cd ~/Documents/noise-weighted-subspace-reconstruction/paper_rps2026
pdfinfo paper.pdf | grep Pages        # expect 11
```

Then fingerprint the build by content, not by commit (the fix round of
2026-08-26 may or may not be committed when you run):

```bash
pdftotext paper.pdf /tmp/rps.txt
grep -c "held-out ranking reverses" <(tr -s ' \n' ' ' < /tmp/rps.txt)   # expect 1 (allow hyphen-break: also try "ranking reverses")
grep -c "48.419" /tmp/rps.txt                                          # expect 0
grep -c "T-mse) vs. 0.0231" /tmp/rps.txt                               # expect 1
```

If any fingerprint fails, say so at the top of your review — you are reading a
different build than this prompt describes.

### A trap, stated up front

A `paper-review` skill installed on this machine is scoped to a **different
document** — the long preprint in `arXiv-v1-polished-extended-preprint/`. Do not
follow it. If you find yourself reading `main.tex` or `02_unified_objective.tex`,
you are in the wrong manuscript. This task needs no skill.

## Step 2 — Phase 1: the blind read (do this FIRST, and alone)

Read **only** `paper_rps2026/paper.pdf`: main text pages 1–4, then references
and the appendix as a referee actually would — to check something, not
cover-to-cover. **Do not open anything else yet.** In particular do NOT read,
until Phase 3:

- `docs/plans/rps2026/REFEREE_REPORT_2026-08-26.md` (the author-side audit)
- `docs/plans/rps2026/ROUND10_PAPER_REVISION_QA.md`
- `paper_rps2026/notes/` (claim audit, metadata) and `paper_rps2026/reviews/`
- anything under `results/`, `evidence/`, `transfer_paper/`

The entire value of your run is independence. An opinion formed after reading
the author's own audit is worthless as a rehearsal of peer review.

Write the complete review of Step 4 from this read alone.

## Step 3 — Phase 2: spot-checks you would actually make (optional, bounded)

A real referee occasionally checks one thing. If your Phase-1 draft contains a
claim you doubt, you may verify **at most three** specific items against the
archived files under `results/rps2026/` (each experiment directory is named in
the appendix's own terms; `results/rps2026/README.md` orients). Record what you
checked and whether it changed your draft. Do not turn this into an audit — the
audit exists and you will see it in Phase 3.

## Step 4 — the review, in OpenReview form

Write a review a NeurIPS-workshop area chair would recognise:

1. **Summary** (neutral, 4–8 sentences — what the paper claims and shows, in
   your words; getting this wrong is itself a data point about the paper's
   clarity).
2. **Strengths**, each tied to something specific in the PDF.
3. **Weaknesses**, each stated as a referee would — including anything that
   *reads* badly even if you suspect the authors could rebut it. Do not
   pre-rebut on their behalf.
4. **Questions to the authors** — what you would need answered in the response
   to raise your score.
5. **Limitations** — adequately acknowledged by the paper or not.
6. **Theme fit** — score the paper against each of the four R/PS themes
   (central / partial / absent) and say whether the fit claims in the abstract
   and §3 feel earned.
7. **Scores**: overall recommendation on the NeurIPS scale (1–10) with the
   decision phrase (reject / borderline reject / borderline accept / accept /
   strong accept), confidence (1–5), and a separate yes/no + one sentence:
   *would you nominate this for a contributed talk over a poster, and why?*
8. **The decisive paragraph** — if the committee asked you to defend your
   recommendation in three sentences, what are they?
9. **What would flip you** — the single author-response or change that would
   move your score one step in each direction.

Style: direct, specific, unsentimental. Quote the PDF's own sentences when a
weakness is about wording. A review that manufactures findings to look thorough
is as useless as one that praises to be safe.

## Step 5 — Phase 3: calibration (only after Step 4 is written and frozen)

Now read `docs/plans/rps2026/REFEREE_REPORT_2026-08-26.md` and
`paper_rps2026/reviews/2026-08-21-paper-review.md`. Append a short **Calibration
appendix** to your file: where your independent read agreed with the prior
reviews, where it diverged, and — most usefully — anything you flagged that no
prior pass caught, or anything you *missed* that a prior pass caught (say why a
blind reader missed it; that is exactly the risk profile of the real review).
Do not revise your Phase-1 scores after reading these; the un-revised score is
the datum.

## Deliverable

Write everything to `docs/plans/rps2026/REFEREE_SIMULATION_<yyyy-mm-dd>.md`.
That is the only file you create. Structure: build fingerprint result → the
Step-4 review → Phase-2 spot-check log (if any) → the calibration appendix.
