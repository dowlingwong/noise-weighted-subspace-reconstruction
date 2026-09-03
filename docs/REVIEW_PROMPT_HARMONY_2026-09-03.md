# Review prompt — does the 3 September update keep the two papers coherent?

You are an independent reviewer with read access to three repositories:

- Paper 1 manuscript: `~/Documents/arXiv-v1-polished-extended-preprint` (branch `main`; build
  with `latexmk -pdf main.tex`; 67 pages).
- Paper 3 / ORACLE proposal: `~/Documents/ORACLE` (branch `dev`; `latex/paper3_proposal.tex`,
  6 pages; build twice with `pdflatex` from `latex/`).
- Experiment repository: `~/Documents/noise-weighted-subspace-reconstruction` (branch `main`).

Do not edit any file. Produce one report, `docs/reviews/2026-09-0X_harmony_review.md` in the
experiment repository (create the file content; the author will commit it).

## What changed and why you are being asked

On 2026-09-03 the author adopted a set of methodological standards from two papers by Jan
Kieseler (GravNet/GarNet, arXiv:1902.07987; object condensation, arXiv:2002.03605) — notes in
`docs/background/KIESELER_METHOD_NOTES_2026-09.md` — and made these additions:

Paper 1 (commits after `056af14`):
- `06b_related_work.tex`: new paragraph "Learned geometry versus measured geometry"
  (GravNet/GarNet, object condensation, Panda Diplomacy).
- `05b_separable_subspaces.tex`: new paragraph "Parameter count and cost" (Grassmannian
  dimensions 12 252 / 771 / 0; ALS cost vs full diagonalisation).
- `02_unified_objective.tex` + `figures/fig_pipeline.tex`: new Figure 1, the (Σ⁻¹, F) pipeline.
- `TODO.md`: decisions (NFPA main text; R/PS camera-ready does not cite Paper 1; pursue the
  noise-Laplacian PE uniqueness corollary) and the P1-E8 experiment design.

ORACLE (commits after `97e0989`):
- §1 new paragraph "Position relative to learned-geometry reconstruction" (GravNet, object
  condensation, Panda, the bias-aware physics-FM benchmark arXiv:2605.29283).
- §5 new paragraph "Reporting standard" (parameter-matched arms, resource row, inclusive +
  stratum endpoints, threshold curves, extrapolation arm, hyperparameter table).
- E5 row: bridge metrics named.
- New Figure 1: monitor stack (TikZ).
- Page count 5 → 6.

Experiment repository: `docs/EXPERIMENT_PLAN_PAPER1.md` §P1-E8, `docs/TODO.md` (P1-E8 and a
Kieseler-standard section), `docs/background/KIESELER_METHOD_NOTES_2026-09.md`,
`docs/NOVELTY_CHECK_2026-09-03.md` (copy also in ORACLE `docs/`).

The author's question is exact: **does this update break the harmony of the papers as they
were, and does it make each paper stronger or messier?** Answer that, not a general review.

## What to check, in this order

1. **Read the pre-update storyline first.** `plan/THREE_PAPER_STORYLINE_2026-09-02.md` (Paper 1
   repo) and `plan/STORYLINE_AFTER_UPDATE_2026-09-03.md`. Then read each new passage in its
   surrounding section. For each new passage answer: does it say something the paper did not
   already say, in the paper's own voice, at the right place, without a new claim?
2. **Voice and register.** Paper 1 is written as derivation + boundary statements; ORACLE as
   contract + falsifiable prediction. Flag any sentence in the new passages that reads as
   advocacy, comparison-for-its-own-sake, or a claim about transformers, GNNs, or foundation
   models that the paper does not test. The related-work paragraph must draw a line, not
   pick a fight; quote any phrase you would soften.
3. **Claims ledger.** List every claim in the new passages (e.g. "12 252 free parameters", "a
   learned coordinate space must rediscover Σ⁻¹ or pay in the likelihood", "object
   condensation's β is the nearest relative of the Fisher rank") and mark each as: follows from
   an existing theorem / arithmetic (verify it: Gr(k, n) has dimension k(n − k)); supported by a
   cited paper (check the citation says that); or new and unsupported. Anything in the third
   class is a harmony break.
4. **Figure 1 placement (Paper 1).** The author's stated intent is "physics in Figure 1" (the
   planted-vs-learned factor picture, after 29 Sep). Today the pipeline schematic is Figure 1.
   Say whether the schematic earns a figure at all, whether it belongs in §2.4 or §7, and whether
   its caption over-promises ("the loss is the unique residual functional…" — is that proved
   before §7 is reached? if not, is a forward reference acceptable?).
5. **Figure 1 (ORACLE).** Does the monitor-stack figure match the text of §3 exactly (same
   symbols a_{ℓ,k}, v_{ℓ,h}, J_h^⊤Σ⁻¹J_h, rank I(z); same four endpoints)? Any arrow the text
   does not justify?
6. **Reporting standard vs the rest of the proposal.** Does any earlier sentence in §4–§5 now
   contradict it (e.g. a single operating point stated as an endpoint; a baseline that is not
   parameter-matched; a claim C1–C4 whose endpoint is not a curve)? List each.
7. **Cross-paper consistency.** The two papers must agree on: what Paper 1 owns (Σ known,
   (Σ⁻¹, F)); what ORACLE owns (Σ̂ ≠ Σ, support shift); the pullback metric J^⊤Σ⁻¹J as
   ORACLE's consequence measure taken from Paper 1; and the description of GravNet / object
   condensation / Panda in both. Quote any sentence where the two papers describe the same
   thing differently.
8. **Length and mess.** Paper 1 is 67 pages; ORACLE grew 5 → 6. For each addition, say whether
   it could be cut to half without losing the point, and whether the proposal should return to
   five pages (what you would drop instead of the additions, if anything).
9. **The PE uniqueness plan (Paper 1 `TODO.md`, P1-E8).** Check the mathematics as stated:
   (a) T-invariance ⇒ block-diagonal on isotypic components; for the cyclic group are the
   isotypic components really the individual Fourier modes over ℂ, and what happens over ℝ
   (conjugate pairs)? (b) does the Schur argument on K = ΦΦ† go through with a rank
   constraint m < d, or only for spectral truncations as claimed? (c) is the CRB statement
   for the γ-sweep exactly the OF theorem or does it need the readout to be least squares on
   whitened features? State what would make (a)–(d) fail; do not prove them.
10. **Verdict.** One paragraph per paper: stronger, messier, or neutral, and the three edits
    you would make first (cut, move, or reword — with the exact sentence).

## Rules

- Quote before you judge. Every finding cites file and line (or section and paragraph).
- Distinguish "this is wrong" from "this is not in the paper's voice" from "this is fine but
  long".
- No new experiments, no new theorems, no restructuring proposals beyond moving or cutting
  the passages added on 2026-09-03.
- Do not read or cite the R/PS submission (`paper_rps2026/`) — it is under double-blind review
  until 29 September 2026.
- Report length: two pages. Findings first, verdicts last.
