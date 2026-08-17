# Desk revision plan — R/PS 2026 short paper

**Date:** 2026-08-08
**Target venue:** Representations for the Physical Sciences (R/PS), NeurIPS 2026, Paris
**Deadline:** 29 Aug 2026 AoE (= 30 Aug 2026 11:59 UTC)
**Working directory:** `rps_noise_covariance_representation/`
**Scope:** manuscript-only. No new experiments. Everything here is doable without running code, except tasks D4 and D7 which need a re-export from existing archived results.

---

> ## ⚠ AMENDMENT — 2026-08-08, read before starting
>
> The paper is being restructured around an **architecture ladder**. See
> `PLAN_00_MASTER_RESTRUCTURE_2026-08-08.md`, which governs.
>
> **All tasks D1–D7 below remain valid and unchanged.** They are, if anything,
> more important under the restructure: D1–D3 anchor the paper in the ML
> literature, and a paper that now spans optimal filters to attention needs that
> anchoring more, not less.
>
> **Two changes:**
>
> 1. **§0.3 (interface with the experiment plan) is superseded** by §3 of the
>    master plan. There are now *three* tracks, not two. You own §1, §2, §5,
>    `references.bib`, and the GWOSC paragraph in §4. You do **not** own §3, which
>    becomes `sections/03_architecture_ladder.tex` under Track C.
> 2. **The cut order in §0.3 and in `notes/cut_list.md` is superseded** by §2 of
>    the master plan. Do not cut Eq. (1) — the `(Σ⁻¹, F)` tuple is now the paper's
>    organizing device and is load-bearing.
>
> **One addition to D4.** When writing the GWOSC paragraph, note that Track C's
> rung 4 also assumes a stationary noise PSD, and that assumption degrades in the
> same way. §5 should connect the two in one deliberate sentence — it is a genuine
> synthesis across three separate studies. Coordinate the wording with Track C.
>
> **Title.** The current title's "Self-Supervision" becomes inaccurate under the
> restructure, because the headline evidence is supervised. Master plan §2 lists
> replacement candidates. You own the final choice; make it last, after §4 is settled.

---

## 0. Read this first

### 0.1 What this plan fixes

An external review of `paper.pdf` (2026-08-08) identified seven defects. Ranked by referee risk:

| ID | Defect | Severity |
|---|---|---|
| D1 | The generalized-least-squares / matched-filter lineage is uncited, so the paper does not preempt "this is GLS, known since 1936" | **High** |
| D2 | PPCA and factor analysis are uncited, so the novelty boundary is not stated in ML-native terms | **High** |
| D3 | No self-supervised-learning or scientific-representation citations, so the title's framing is unanchored in the reviewers' literature | **High** |
| D4 | Paper 3 (GWOSC calibration) is entirely cut, forfeiting the workshop's Transfer theme and the only real-data falsification evidence | Medium-high |
| D5 | Section 4 cites results that cannot be traced (Papers 1 and 2 are unciteable under double-blind; "described separately" points at nothing) | Medium |
| D6 | Zero figures in four pages, at a poster-first workshop | Medium |
| D7 | `< 10^{-5}` appears in the results table | Low, but reads as an unfinished draft |

### 0.2 Hard constraints — do not violate

1. **Four pages of main content maximum.** References and appendices are excluded from the limit. The `\clearpage` before `\bibliography` in `paper.tex` enforces the split; do not remove it.
2. **Double-blind.** No author names, no affiliations, no non-anonymous URLs, no PDF metadata author field. Do not cite Papers 1, 2, or 3 as "our prior work" or by any identifying title.
3. **Single PDF submission.** No supplementary files. Anonymized repository links are permitted.
4. **Do not add any claim not already supported.** `notes/claim_audit.md` is authoritative. Anything marked `cut` there stays cut. If a revision seems to require a new claim, stop and escalate rather than writing it.
5. **Do not manually transcribe numbers into LaTeX.** Any number that comes from archived results must be traced to its source file in the commit message.

### 0.3 Interface with the experiment plan — read this before touching Section 4

A parallel agent is executing `plan/EXPERIMENT_PLAN_2026-08-08.md`, which will **replace most of Section 4**. To avoid wasted work and merge conflicts:

- **Do not rewrite or polish** `sections/04_experiments_discussion.tex` lines 6–13 (the `Factored proof-of-concept` paragraph, the table, and the two interpretation paragraphs). The experiment agent owns those.
- **You do own** line 4 (the `Controlled evidence for metric reversal` paragraph — this comes from Paper 1 and is not being re-run), and lines 15–19 (Limitations, Takeaway).
- Task **D4** inserts a new paragraph into Section 4. Place it **after** line 4 and **before** line 6, so it does not collide with the experiment agent's edits.
- Task **D6** reserves a figure slot; the experiment agent fills it. Create the slot, do not create the figure.
- **Page budget:** the draft currently fills four pages exactly. Your additions (D1–D4) cost roughly 0.45 pages. The experiment agent's rewrite of §4 will *free* roughly 0.3 pages, because the single-split hedging in lines 11 and 13 collapses once there are confidence intervals. Do not perform emergency cuts on the assumption that you must absorb the full overflow alone. If you overflow before the experiment lands, follow the cut order in `notes/cut_list.md` and record what you cut in the handoff note — do not invent a new cut order.

### 0.4 Definition of done

- `make` builds cleanly with no LaTeX warnings about overfull boxes in the main text.
- Main content is ≤ 4 pages, verified by `pdfinfo paper.pdf` plus visual check that the bibliography starts on page 5.
- Every task below is marked complete or explicitly deferred with a reason in `plan/DESK_REVISION_LOG.md`.
- No claim in the final PDF contradicts `notes/claim_audit.md`.

---

## D1 — Add the generalized-least-squares and matched-filter lineage

**Why.** The paper's core move is "use `Σ⁻¹` as the residual inner product." That is generalized least squares (Aitken, 1936) and, in its rank-one signal-processing form, the matched/optimal filter. A referee who knows this literature and finds it uncited will conclude the authors do not know it either. Paper 1's `06b_related_work.tex` handles this correctly; the distillation dropped it.

**This is cheap.** Every bib key you need is already in `references.bib`. Nothing new to look up.

**Where.** `sections/01_introduction.tex`, insert as a new paragraph after line 7 (the Noise2Self/SSDU/Noisier2Inverse paragraph), before line 9 (the "We ask" paragraph).

**What to write.** A 3–4 sentence paragraph that does three things in order: (i) name the lineage, (ii) say precisely what it already establishes, (iii) say precisely what it does not address, which is the gap this paper occupies. Use approximately this content — rewrite in the paper's voice, do not paste verbatim:

> Inverse-covariance weighting is classical. Generalized least squares establishes it as the efficient estimator under correlated errors [aitken1936least], and its signal-processing form is the matched or optimal filter [turin1960introduction, kay1993fundamentals, wiener1949extrapolation], which is the standard amplitude estimator in cryogenic calorimetry [szymkowiak1993signal, fixsen1993pulse]. That literature fixes the weighting but assumes a *known*, one-dimensional signal template. The question here is what the same weighting implies once the signal model is *learned* and multidimensional.

**Bib keys to use, all already present:** `aitken1936least`, `turin1960introduction`, `kay1993fundamentals`, `wiener1949extrapolation`, `szymkowiak1993signal`, `fixsen1993pulse`. Optionally `kurinsky2021`.

**Acceptance test.** A reader who believes "this is just GLS" can find, within the first page, one sentence that concedes the point and one that states the delta. If you cannot point to those two sentences, the task is not done.

---

## D2 — State the novelty boundary against PPCA and factor analysis

**Why.** This is the single most efficient novelty statement available to you, because it is a strict containment chain a NeurIPS reviewer parses in one reading:

| Model | Noise covariance assumed |
|---|---|
| PCA / linear autoencoder under MSE | `Σ = σ²I`, implicitly |
| Probabilistic PCA [tipping1999probabilistic] | `Σ = σ²I`, explicitly |
| Factor analysis [rubin1982algorithms, roweis1998algorithms] | `Σ` diagonal |
| CW-PCA / this paper | `Σ` full, specified |
| NFPA | `Σ` full, plus Kronecker-restricted representation class |

**Where.** Two edits.

1. `sections/01_introduction.tex`, appended to the D1 paragraph or as its own two sentences immediately following. State the chain in prose — do **not** add the table above to the paper, it costs too much space for what it says.
2. `sections/02_covariance_geometry.tex`, one clause where CW-PCA is first defined, cross-referencing the introduction.

**What to write.** Approximately:

> The probabilistic latent-variable literature already relaxes this assumption partway: probabilistic PCA assumes isotropic noise [tipping1999probabilistic] and factor analysis permits a diagonal covariance [rubin1982algorithms, roweis1998algorithms]. Neither accommodates a full, instrument-characterized covariance, and neither identifies the resulting rank-one degenerate case with the optimal filter.

**Bib keys, all already present:** `tipping1999probabilistic`, `rubin1982algorithms`, `roweis1998algorithms`. Optionally `golub1979generalized` if you mention the generalized eigenproblem form.

**Do not** claim that full-covariance weighted PCA is itself novel — `bailey2012principal` and `allen2014gls` are already cited at line 9 and establish it. The novelty claim is the *placement* of OF, CW-PCA and the tied autoencoder in one metric, plus the Kronecker restriction. Keep that boundary exactly where `notes/claim_audit.md` puts it.

---

## D3 — Anchor the self-supervision framing in the SSL literature

**Why.** The title is "Likelihood-Aware Self-Supervision" and the abstract says "self-supervised setting," but the only self-supervised citations are three *denoising* methods. Denoising is not representation learning, and R/PS reviewers are drawn substantially from the joint-embedding / masked-modelling / scientific-foundation-model community. As written, the framing reads as retrofitted onto a linear-algebra paper. Two or three citations fix the perception at negligible page cost.

**Where.** `sections/01_introduction.tex`, line 7 — extend the existing paragraph rather than adding a new one.

**What to add.** One sentence positioning this work against the dominant SSL paradigm, approximately:

> Most self-supervised representation learning defines its training signal through augmentation invariance or masked prediction, which presuppose that the nuisance structure is known and can be simulated. Measurement noise supplies that structure directly, and in the Gaussian case it fixes the residual geometry rather than the pretext task.

**Bib entries you must ADD to `references.bib`** — these are not present. Add 2–3, no more:

- One masked-modelling reference (He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022, is the safe default).
- One joint-embedding reference (Chen et al. SimCLR, ICML 2020; or Bardes et al. VICReg, ICLR 2022).
- Optionally one scientific-representation reference if you can verify it accurately — e.g. a multiple-physics pretraining or astronomical contrastive-pretraining paper. **Only include this if you can verify author list, venue and year against a primary source.** A hallucinated citation in a four-page paper is disproportionately damaging. If you cannot verify, omit it and keep two.

**Verification requirement.** For every new bib entry, confirm title, authors, venue and year against arXiv or the publisher. Record the URL you checked in `plan/DESK_REVISION_LOG.md`. Do not generate a bib entry from memory.

---

## D4 — Restore one Paper 3 paragraph (GWOSC calibration boundary)

**Why.** R/PS names **Transfer** as one of four themes and its call states that learned embeddings must "remain falsifiable and actionable rather than just empirically successful." Paper 3 contains exactly that: a verified negative result on real detector data. It is currently excluded by `notes/cut_list.md` ("all Paper 3 monitoring terminology"). That exclusion was a scope decision made before the venue fit was assessed; it is now reversed **for this one paragraph only**. Everything else in the Paper 3 cut list stays cut.

**The evidence, with exact provenance.** All values below are already archived. Source of truth:

- `transfer_paper/tables/gwosc_filter_equivalence_summary.csv`
- `transfer_paper/tables/gwosc_time_local_psd_summary.csv`

The story in three facts:

1. **Implementation is verifiably correct.** Shared-statistic filter equivalence passes on synthetic, H1 and L1 at `max_abs_score_difference` of 5.0e-15, 1.6e-12 and 1.4e-12 respectively, against an acceptance threshold of 1e-10. All three rows have `accepted = True`.
2. **The synthetic control passes.** Under a stationary synthetic noise model, the whitened score statistic has `score_std` ≈ 1.15 for the predeclared primary model (`local_radius_64s`), consistent with the unit-variance expectation.
3. **Real detector data fails, under both global and time-local PSD.** On H1 and L1, `local_radius_64s` has `accepted = False`, with `score_std` of 9.42 (H1) and 8.83 (L1) — roughly seven to nine times over-dispersed. The global leave-one-out variant does not rescue it (H1 1.77, L1 7.29). Coverage is 1.0 in every case, so this is not a data-availability artifact.

**The claim you may make.** Precisely this and nothing more: *a covariance model can be implemented correctly and validated on synthetic data, and still fail its calibration check on real instrument data; implementation identity is not calibration.* This directly supports the paper's own limitation sentence ("a wrong `Σ` defines a wrong geometry") by making it empirical rather than hypothetical.

**Claims you may NOT make.** Do not say the method fails, do not compare detector sensitivities, do not name a physics analysis, do not use Paper 3's monitoring terminology, do not imply this evaluates NFPA. Add a row to `notes/claim_audit.md` recording this claim, its status (`experiment`, real data, archived), and its controls.

**Where.** `sections/04_experiments_discussion.tex`, new paragraph inserted after line 4, before line 6. Suggested heading: `\paragraph{Covariance adequacy on real data.}`

**Length.** Three to four sentences. Report the over-dispersion factor and the fact that the synthetic control passed; that is the whole argument. Do not include a table.

**Citations.** `gwoscgw150914` and `macleod2021gwpy` are already in `references.bib`. Describe the data as a public gravitational-wave strain release without identifying the authors' own analysis.

**Also update.** `sections/00_abstract.tex` — one clause noting that covariance adequacy is checked against real instrument data. Only if the page budget allows; the abstract is already dense.

---

## D5 — Fix untraceable provenance

**Why.** Section 4 reports specific numbers (`48.419`, `47.869`, `0.550`, `[0.530, 0.571]`, `86.25°`, `0.1209 → 0.0880`) whose source is Paper 1, which cannot be cited. Line 16 then says claims "require the predeclared multi-seed experiments described separately" — a reference to a document a reviewer cannot see. A skeptical referee reads this as unsupported.

**Three fixes, in order of preference.**

1. **Anonymized repository link.** R/PS explicitly permits these. Create an anonymized repo (anonymous.4open.science or equivalent) containing the seed-level result files and the generating scripts for every number in Section 4. Add a single footnote on page 1: "Code and seed-level results: <anonymized URL>." **Verify the link resolves before submission.** A dead anonymized link is worse than none.
2. **Move the protocol into the appendix.** The predeclared experiment protocol currently lives in `notes/nfpa_submission_experiment_plan.md`, outside the PDF. Appendices are unlimited and excluded from the page count. Add a condensed subsection to `appendix/appendix_experiment_details.tex` stating the predeclared hypotheses, the acceptance gates, and the primary configuration. Then change line 16's "described separately" to a concrete `\ref{}` into that appendix.
3. **Rewrite the phrase regardless.** Even with fixes 1 and 2, "described separately" is weak. Replace with a specific forward reference.

**Do fix 2 unconditionally** — it is free, uses appendix space that costs nothing, and converts a hand-wave into a checkable artifact. Do fix 1 if the infrastructure is available.

---

## D6 — Reserve a figure slot

**Why.** Four pages, zero figures, one four-row table. At a workshop whose accepted papers are presented as posters, reviewers partly assess whether the work will make a good poster. The contour figure was correctly removed (`notes/presentation_figure_audit.md` — its curved-manifold language contradicts the linear analysis) but nothing replaced it.

**What to do — slot only.** Do **not** create the figure; the experiment agent produces it from Block A results.

1. Add to `sections/04_experiments_discussion.tex`, positioned after the results table:

```latex
% FIGURE SLOT — filled by EXPERIMENT_PLAN_2026-08-08.md Block A.
% Two panels: (A) paired relative difference NFPA vs Iso-MPCA under raw MSE and
% normalized chi-square, with 95% CIs and a zero line; (B) seed-level principal
% angle to CW-PCA for NFPA and Iso-MPCA. Caption must state matched total rank,
% shared factored architecture, and n = 30 seeds.
% \begin{figure}[t] ... \end{figure}
```

2. Confirm `\usepackage{graphicx}` is loaded in `paper.tex` — it is, line 8.
3. Reserve approximately 0.35 pages in your budget arithmetic.

**Contingency.** If the experiment does not complete by 25 Aug, a schematic figure is still better than none: a two-panel diagram contrasting the Euclidean and `Σ⁻¹` residual balls with the same two candidate reconstructions, showing the ranking flip geometrically. `figures/geometry_schematic.tex` already exists and was retained as an unused alternative — evaluate whether it can be adapted. It must not use curved-manifold language.

---

## D7 — Re-export the sub-threshold raw-MSE values

**Why.** `< 10^{-5}` in two cells of Table 1 is honest but signals an unfinished pipeline. The underlying values exist; the source draft's display precision truncated them.

**What to do.** Re-export the IsoAE and IsoPCA held-out raw MSE from the archived Paper 2 run at full precision and substitute the actual numbers into `tables/results_summary.tex` lines 13–14.

**If the experiment agent's Block A supersedes this table entirely, skip D7** and note it as superseded in the log. Coordinate before spending time here.

**If the archived values are genuinely unrecoverable,** keep `< 10^{-5}` and add four words to the caption explaining it ("below the archived export precision"). Do not fabricate a value.

---

## Task order and estimated effort

| Order | Task | Effort | Depends on |
|---|---|---|---|
| 1 | D1 — GLS/matched filter | 45 min | nothing; all bib keys present |
| 2 | D2 — PPCA/FA boundary | 30 min | D1 (same paragraph region) |
| 3 | D3 — SSL anchoring | 90 min | requires verifying new bib entries |
| 4 | D5 — provenance, fix 2 | 60 min | nothing |
| 5 | D4 — Paper 3 paragraph | 90 min | reading the two CSVs |
| 6 | D6 — figure slot | 15 min | nothing |
| 7 | D7 — raw-MSE re-export | 30 min | coordinate with experiment agent |
| 8 | Page-budget pass and build | 60 min | all of the above |

Total: roughly one focused working day.

---

## Handoff back for review

When complete, produce `plan/DESK_REVISION_LOG.md` containing:

1. One row per task: ID, status (done / deferred / superseded), files touched, and for D3 the verification URLs for each new bib entry.
2. Page count before and after, and any cuts made, with justification against `notes/cut_list.md`.
3. Any point where you believed a task required a claim not supported by `notes/claim_audit.md`, and what you did instead.
4. A diff summary (`git diff --stat`) and the rebuilt `paper.pdf`.

Do not mark the revision complete if the build has errors, the main text exceeds four pages, or any new citation is unverified.
