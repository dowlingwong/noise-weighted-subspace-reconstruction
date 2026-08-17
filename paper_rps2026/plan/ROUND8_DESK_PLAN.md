# Round 8 desk plan — workshop-fit revision while Task B trains

**Date:** 2026-08-18. **Deadline:** 29 Aug 2026 AoE.
**Runs in parallel with:** the remote second-seed replication (ROUND8_RUNBOOK.md Task B).
Nothing here depends on its outcome; C2 prepares for both outcomes.

## Requirements verified against the workshop (2026-08-18)

Checked the live CFP and diffed our style file against the official template zip:
4 pages of main content (we are at 4); references excluded (ours start p5);
appendix unlimited but **"reviewers are not obliged to read appendices, so keep the
main paper self-contained"**; single PDF, no supplementary files; double-blind;
no NeurIPS checklist; non-archival; anonymized code/data links allowed.
`neurips_2026.sty` is byte-identical to the template zip. The one compliance-adjacent
problem: our evidence figure lives in the appendix, and page 4 is six unbroken text
paragraphs. Workshop themes: **self-supervision, transfer, sampling, tokenization**.

## Work items

### A — Fit and framing

- **A1 Name the tokenization contribution.** The paper already chooses the STFT
  band×frame plane, pairs real/imaginary as 2 features of 328 physical-frequency
  tokens (not 656 unrelated ones), and patches over frames. State the principle:
  *choose the token axis in which Σ⁻¹ is diagonal, so the likelihood metric is a
  per-token weight rather than a dense coupling.* ~3 sentences in §3 rung 4; one
  clause where the tuple is introduced. Directly answers the theme's "naive patching
  destroys geometric priors".
- **A2 SSL framing into the abstract opening.** The workshop's self-supervision
  question is "what objectives preserve physical constraints"; our answer — the
  measurement likelihood is the objective; augmentation-based SSL presupposes
  simulable nuisance — currently first appears in §1 ¶2. Move one sentence of it
  into the abstract's opening.
- **A3 GWOSC as transfer.** Retitle "Covariance adequacy on real data" so it reads
  as a transfer/falsifiability result: a Σ validated in-distribution fails on real
  strain. One retitle plus one clause; the numbers do not change.
- **A4 Title options.** Draft alternatives (below); **no change without the author's
  decision.** Current title is honest but signals signal-processing.

### B — Evidence presentation

- **B3 Delete the stale FIGURE SLOT comment** in §4 (references an internal plan
  file by name). Do first; costs nothing.
- **B2 Prototype the 3-row contrast table** (rung / χ̄² contrast / raw-MSE contrast /
  verdict) as a replacement for §4's densest number-prose. Measure its page cost.
- **B1 Move the evidence figure into §4.** The A-vs-C panel contrast is the paper's
  argument and reviewers are not obliged to see it where it is. Requires cutting
  ~500–700 chars; donors: the "Why the architecture binds next" bridge paragraph and
  the superseded linear-rung recap. B1 and B2 compete for the same space —
  prototype both, keep what fits 4 pages, demote the loser to the appendix.

### C — Referee-proofing

- **C1 Autocorrelation caveat.** The rung-4 bootstrap resamples 50 evaluations along
  one training trajectory; adjacent evaluations are autocorrelated, so the intervals
  are narrower than independence would warrant. Say so where the intervals are
  reported. This is the weakest methodological spot in the paper.
- **C2 Contingency paragraphs.** Draft both Task-B replacement paragraphs now —
  "seeds agree in sign" and "signs disagree" — in `notes/round8_contingency.md`, so
  the result drops in the day it lands.
- **C3 Anonymized code link.** Allowed by the CFP. Prepare the checklist of what an
  anonymized snapshot must exclude; hosting/whether is the author's decision.

### D — New analysis, no server needed

- **D1 Cross-rung whitening dispersion.** Compute the dispersion of inverse
  whitening weights for the synthetic rung-3 covariance (eigenvalues of the oracle
  Σ = Σ_c ⊗ Σ_t, since whitening is diagonal in its eigenbasis) and put it beside
  rung 4's measured CV(1/J) = 0.69. Gives the paper a quantitative axis: reversal
  observed/absent vs. metric dispersion. Script `scripts/rung3_whitening_range.py`,
  JSON to `results/rps2026/E6b_rung3_whitening_range.json`, one sentence into §4.
- **D2 Proposition 2's checkable consequence.** One sentence: how to test, on a
  fitted model, whether the Kronecker restriction was lossless.

## Order and gates

1. D1 (produces the number the §4 edit needs) → 2. B3 → 3. B2 prototype →
4. B1, decide B1-vs-B2 by measured page cost → 5. A1–A3 → 6. C1, D2 →
7. C2, C3, A4 (documents only) → 8. verification → 9. report.

**Verification gate (after every step that touches .tex):** `make` succeeds; main
content ≤ 4 pages with references starting on the following page; 0 overfull boxes;
0 undefined citations/references. **Final gate additionally:** render changed pages
to PNG and inspect; anonymity sweep of the PDF text (dowling, wong, karlsruhe,
kit.edu, delight, acknowledg); commit with a report.

## A4 — title options (decision deferred to author)

1. *(current)* Noise Covariance as Representation Geometry: One Metric, Four
   Architecture Classes
2. Whose Error Is It? Noise Covariance as the Metric of Label-Free Scientific
   Representation
3. The Likelihood Fixes the Metric, the Architecture Fixes the Class: A Ladder for
   Label-Free Reconstruction
4. Noise-Weighted Representation Learning Across an Architecture Ladder: When Does
   the Metric Bite?

Recommendation: keep (1) or adopt (4); (4) names the empirical question the paper
actually answers and its last clause matches the abstract's last sentence.
