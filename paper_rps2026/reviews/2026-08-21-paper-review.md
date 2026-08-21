# Paper Review — 2026-08-21

**Scope reviewed:** full main text (§1–§5), appendix skim, all quantitative claims traced to archived result files
**Build:** pass — clean `latexmk -C && make`, 10 pages, 4 main-text pages, references from page 5, 0 undefined references, 0 undefined citations, 0 overfull boxes
**Branch:** `codex/rps-round10-paper-revision` @ `644a516`
**Verdict:** **minor revision** — submit after the two judgement calls below are decided

## Summary

Every quantitative claim in the main text traces to an archived result file; I
re-derived the ones that could be recomputed rather than reading them back. One
number had silently lost its denominator during the four-page compression and is
now fixed. The evidence discipline — frozen versus replication cohorts kept
apart, digest-pinned generators, retained negative results — is well above what a
workshop normally sees, and it is the paper's real asset. The exposure is not
correctness but *ambition*: the paper states plainly that it introduces no new
estimator, and a referee who wants a method will have nothing to argue with and
will score it down anyway. The two open judgement calls are whether the
rung-2-versus-rung-3 comparison belongs back in the main text, and what to do
about an unverifiable citation.

## Scores

| Dimension | Score | Basis |
|---|---|---|
| **Novelty** | **3.0 / 5** | The contribution is organizational: a `(Σ⁻¹, F)` framing, a ladder, and matched tests. §1 says "We introduce no new estimator" in italics. Every component — GLS, matched filter, weighted PCA, the linear-AE/PCA correspondence, MPCA — is classical and cited as such. What *is* new: the controlled metric-versus-capacity separation, the exact lossless condition for the Kronecker restriction, and a genuine negative result at rung 4. That is a real but modest contribution, and honestly labelled. |
| **Solidness** | **4.5 / 5** | Preregistered hypotheses with an amendment trail; frozen ten-seed cohort never pooled with the twenty-seed replication; a white-noise identity control that returns *exactly* zero (verified: max \|paired difference\| = 0.000e+00 across 20 seeds); seed-level bootstrap intervals with stated resampling units; a covariance-estimation condition; digest-pinned, fail-closed table generation. Loses half a point only because the headline effect is synthetic. |
| **Completeness** | **3.5 / 5** | Each rung has exactly one evidential setting. Rung 4 has two seeds. E10b is one calibration size. No sweep over rank, SNR, noise colour, or nonseparability — and the appendix says so explicitly. Nothing is missing that the paper claims to have; several things are missing that a referee will want. |
| **Math** | **4.0 / 5** | Correct and now properly hypothesised: Prop. 1 states the Σ⁻¹-orthonormal basis condition, Prop. 2 states `Σ = Σ₂ ⊗ Σ₁`, and §2's eigenspace argument is qualified "at the population level" with the finite-$N$ perturbation noted. Proof sketches in the main text with full proofs in Appendix A is the right split for four pages. Deductions: the propositions are re-derivations of known linear algebra in new notation — true and admitted — and a referee cannot check them without the appendix. |
| **RPS workshop fit** | **4.5 / 5** | Hits three of the four themes directly: **self-supervision** (the measurement likelihood *is* the objective, no augmentation policy), **tokenization** (choose the axis in which Σ⁻¹ is diagonal — 328 physical-frequency tokens, not 656 unrelated rows), **transfer** (a covariance validated synthetically and falsified on real strain). **Sampling** appears only as future work. The label-free, instrument-noise framing is squarely what this workshop exists for. |
| **Readability** | **4.0 / 5** | The abstract is 182 words and every clause is delivered on. Section logic is clean: likelihood geometry → whitening → bridge → restriction → attention. Cost of the four-page budget: §4 is four dense paragraphs of numbers with little connective tissue, and §5 is a single 8-line paragraph of stacked limitations. Readable by a referee in the field; hard work for anyone else. |
| **Consistency** | **4.5 / 5** | The `00_abstract.tex` text and the OpenReview metadata abstract are **word-identical** (182 words each, machine-diffed). All `\ref` targets resolve; all `\cite` keys resolve. Cohort language ("frozen", "replication", "never pooled") is used identically in §4, the appendix, the figure caption and the claim audit. Two small documentation staleness items below. |
| **Overall** | **4.0 / 5** | A strong workshop submission whose ceiling is set by its own honesty. |

## Blocking findings

None. The paper builds clean and no claim is unsupported.

## Major findings

### 1. The rung-2 comparison a referee will ask for first is now in the appendix

**Where:** `04_experiments_discussion.tex:18` (removed), `appendix_experiment_details.tex:22` (added).

The moment a reader sees rung 3 (Kronecker-restricted) evaluated at matched rank
against rung 2 (unrestricted), the obvious question is: how does the restricted
family compare to the unrestricted optimum? The answer is surprising — NFPA is
*better* on held-out likelihood residual by `−0.0026` (I reproduced the bootstrap
independently: mean `−0.002558`, 95% `[−0.002592, −0.002520]`) — and the paper's
explanation, that this is finite-sample generalization rather than evidence the
restriction helps, is exactly right.

That paragraph was moved to the appendix during the last compression round to
hold four pages. It is now the only place a referee's most natural objection is
answered, and they may not get there.

**Fix, in order of preference:** (a) return one sentence to §4 —
*"Against unrestricted CW-PCA the residual differs by $-0.0026$ --- finite-sample,
not evidence the Kronecker restriction is superior (Appendix~B.2)"* — and take
the line back from the §5 rung-4 sentence, which still restates §4; or (b) leave
it and accept the risk. This is a judgement call, not a defect.

### 2. No empirical comparison against the self-supervised methods §1 cites

**Where:** `01_introduction.tex:5` cites Noise2Self, SSDU and Noisier2Inverse as
the neighbouring literature; none appears in §4.

The paper's positioning is "measurement noise supplies the residual weighting,
where augmentation-based SSL constructs its own nuisance structure." A skeptic
will ask what happens if you simply run Noise2Self on the same synthetic data. The
honest answer — that these methods solve a different problem (denoising without
clean targets, not choosing a residual metric at fixed capacity) — is defensible
but is nowhere stated.

**Fix:** one clause in §1 saying why they are not comparators, not an experiment.
Cheap insurance against a "missing baseline" review.

### 3. The headline reversal is synthetic; the real-data instance is explicitly not replicated

**Where:** `05_discussion.tex:2`, "The linear-rung TIDMAD reversal is not
replicated at a confirmatory file."

Rungs 1–3 are evidenced entirely on planted synthetic data. The one real-data
linear-rung reversal failed its confirmatory file and the paper says so. A referee
can reasonably write: *the central phenomenon is demonstrated only where the
authors control the ground truth.*

There is no fix short of an experiment, and the paper's disclosure is the correct
behaviour. Flagged so the response letter is prepared rather than surprised.

## Minor findings

| File · location | Issue | Suggested fix |
|---|---|---|
| `04_experiments_discussion.tex:27` | **Fixed during this review.** "seven to nine times over-dispersed" lost its denominator in the last compression round; the ratio is against the synthetic control's dispersion (`9.424/1.148 = 8.2`, `8.827/1.148 = 7.7`), and without `1.15` in the sentence a referee computing `9.42/1` gets `9.4`. Now reads "calibrated on synthetic stationary noise (dispersion $1.15$) is about eight times as dispersed". | done |
| `04_experiments_discussion.tex:27` | **Fixed.** `1.77, 7.29` were labelled "leave-one-out"; the archived table (`docs/results/GWOSC_RESULT.md:48`) calls that column *Global std*, and both arms are leave-one-out. Now "global-PSD arm". | done |
| `03_architecture_ladder.tex:6` | Rung 1 says the metric question has a "closed-form answer" while §1 and the abstract say "exact geometric characterization". Both are correct — rung 1 genuinely *is* closed-form — but the claim audit records that "closed-form" was deliberately weakened for rungs 1–3 collectively. | Either leave (defensible) or say "an exact answer in closed form" to make the scoping visible. |
| `03_architecture_ladder.tex:23` | "$\chi^2/\mathrm{dof}=1.0031$, all gates passed" — verified against `docs/results/TIDMAD_PHASE4_6_REAL_RESULT.md:31` (`1.00312`). "All gates passed" refers to the Phase-4 calibration gates. Phase 6 separately reports 2 of 5 negative controls not degrading. | Scope is fine as written; be ready to say "Phase-4 calibration gates" if asked. |
| `notes/claim_audit.md:24` | Says the `χ²/dof = 1.0031` claim appears in "§3, abstract". It is in §3 and §5; the current abstract does not mention it. | Update the cell to "§3, §5". |
| `paper.log:621` | `pdfTeX warning: \pdfendlink ended up in different nesting level` at the page 2→3 break in §3 — a citation hyperlink split across pages. Cosmetic; `hidelinks` is set so nothing is visible. | Ignore, or add a `\nolinkurl`-free rephrase if it recurs at camera-ready. |
| `E10b` | 2 of 10 seeds show *negative* MSE degradation (estimated covariance beat oracle on those seeds); the reported mean is positive with an interval excluding zero. | No change needed. Worth knowing if a referee asks about per-seed spread. |
| `references.bib` | 27 entries present but never cited. | Harmless; prune at camera-ready if the file is released. |

## Writing notes

**§5, limitations.** The paragraph is eight lines of stacked semicolons and is
the hardest reading in the paper.

> "Rung 4 rests on two seeds per arm, and its band-limited arms are not comparable to the published full-band baselines; since the objectives there are measurably non-degenerate, the absence remains unexplained — capacity, optimization trajectory and implicit regularization all remain candidates."

This is correct and I would not cut it, but if a line is ever freed, breaking §5
into two paragraphs (evidence-strength limits, then scope limits) would repay it
more than any other edit in the paper.

**§4, semantic recovery.** The delayed-predeclared/post-hoc split now reads
precisely. Note that "no multiplicity correction" appears without saying whether
one *would* change the conclusion — with all seven contrasts at `fraction_negative
= 1.0` and every interval excluding zero, it would not. One clause ("all seven
would survive any standard correction") would close the door, if space appears.

## Verified clean

Machine-checked against archived result files, not read off the manuscript:

| Claim | Source | Status |
|---|---|---|
| Frozen χ̄² `−0.0192 [−0.0194,−0.0190]` | `E9/replication_summary.json` → `−0.019201 [−0.019411,−0.019003]` | ✓ |
| Frozen MSE `+0.0034 [+0.0034,+0.0034]` | same → `+0.0034143 [+0.0033941,+0.0034342]` | ✓ |
| Replication χ̄² `−0.0190 [−0.0192,−0.0189]` | same → `−0.019046 [−0.019167,−0.018924]` | ✓ |
| Replication MSE `+0.0034` | same → `+0.0034131` | ✓ |
| White-noise control "exact zero" | `E2/block_B_white_raw.csv`, recomputed | ✓ **max \|paired difference\| = 0.000e+00 over 20 seeds, both metrics** |
| E10b χ̄² `−0.0191 [−0.0193,−0.0189]`, MSE `+0.0034` | `E10b/summary.json` | ✓ |
| E10b `0.0078 %` / `0.0345 %` and denominators | recomputed from `E10b/raw.csv`: `7.905e-05 / 1.013904285` and `3.548e-06 / 0.010297200` | ✓ |
| "all ten seeds" preserve the reversal | `fraction_negative = 1.0` (χ̄²), `0.0` (MSE) | ✓ |
| Rung 4 `−1.15`, `−1.17 ×10⁻⁴`; `−57.2`, `−57.4` | `E7_rung4_seed_replication.json` → `−1.14794e-4`, `−1.16896e-4`; `−57.184`, `−57.408` | ✓ |
| Rung 4 "all four signs agree" | `sign_agreement.*.sign_agrees` all `True` | ✓ |
| Coarse score `−1.0×10⁻³`, intervals span zero | `E5_benchmark_contrast_interval.json` → `−0.0010185`, both bootstraps `excludes_zero: false` | ✓ |
| `J` spans a factor `441`; cv `0.69` | `E6_whitening_range.json` → `ratio_max_min 441.275`, `cv_inverse_J 0.69145` | ✓ |
| Rung-3 cv `1.10`; white control `0` | `E6b_rung3_whitening_range.json` → `1.09642`, `0.0` | ✓ |
| 328 frequency tokens | `E6.n_bands = 328` | ✓ |
| E8 `0.035 / 1.021`, `0.768 / 1.280` | `E8/separable_alpha_0/summary.json` → `0.0349 / 1.0207`, `0.7677 / 1.2796` | ✓ |
| "All seven paired contrasts favour NFPA, every interval excludes zero, all ten seeds agree in sign" | all 7 contrasts: `excludes_zero True`, `fraction_negative 1.0` | ✓ |
| `χ²/dof = 1.0031` | `TIDMAD_PHASE4_6_REAL_RESULT.md:31` → `1.00312` | ✓ |
| GWOSC `9.42`, `8.83`, `1.77`, `7.29`, control `1.15` | `GWOSC_RESULT.md:47-50` → `9.424`, `8.827`, `1.771`, `7.288`, `1.148` | ✓ |
| Appendix `−0.0026 [−0.0026,−0.0025]` | recomputed bootstrap from `E1/block_A_colored_raw.csv` → `−0.002558 [−0.002592,−0.002520]` | ✓ |
| Abstract vs OpenReview metadata | machine-diffed, 182 words each | ✓ **word-identical** |
| All `\ref` targets defined | 0 dangling | ✓ |
| All `\cite` keys in `references.bib` | 0 missing | ✓ |
| Anonymity | no author name, no ORCID, no repository URL in the PDF; empty PDF Title and Author metadata | ✓ |
| Archive integrity | 15/15 authoritative files byte-identical to the pre-revision baseline | ✓ |
| Focused test suite | R9 + R10 + hardening + generator: **101 passed** | ✓ |

Do not re-check these.

## Rejection risks, ranked

1. **"No new method."** (highest) The paper disarms this by admitting it in
   italics, which converts a fatal objection into a scoring penalty — but a
   referee looking for contribution will still mark it down. *Mitigation:* the
   rebuttal should lead with the rung-4 negative result and the reversal itself as
   findings, not with the framing.
2. **Synthetic-only headline.** The reversal is demonstrated where the authors
   control the ground truth, and the real-data instance failed replication.
3. **Underpowered rung 4.** Two seeds. A referee may read the negative result as
   "inconclusive" rather than "a finding", despite the non-degeneracy measurement
   that rules out the obvious confound.
4. **Small χ̄² effect.** `−0.019` against a base near `1.014` is a 1.9% change.
   The MSE side is ~50%, so the *reversal* is unambiguous, but a referee who fixes
   on the likelihood column alone may call it negligible. Figure 1A's relative
   axis is the defence — make sure the rebuttal points at it.
5. **Unverifiable citation.** `priorThesis` typesets as "[Sjostrom, 2026] … MSci
   project report, Imperial College London, 2026. Unpublished; available on
   request." Not checkable by a referee, and it places a named institution in the
   bibliography next to an unpublished source.
6. **No comparison to cited SSL neighbours** (finding 2 above).
7. **Sampling theme absent.** Three of four workshop themes, not four. Low risk.
8. **Density.** A tired reviewer meeting four pages of compressed §4 may
   under-rate work they did not fully read. Low risk, non-zero.

## Already known

These restate items already tracked and are listed so the new-findings list stays
honest:

- E10b is one condition, never a sweep — `notes/claim_audit.md:58`.
- E11 is post-hoc with sub-degree effect, appendix only — `notes/claim_audit.md:59`.
- `priorThesis` needs a verifiable source — `ROUND10_PAPER_REVISION_QA.md`, open
  author items.
- The anonymized code link remains no-go pending gates 3, 5, 6 — same file.
- The complete test suite is the server's gate; this desktop runs the focused
  suite — same file.
- One inherited trailing-whitespace line in an archived machine log, deliberately
  not fixed — same file.
