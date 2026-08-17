# Round 5 revision plan — to submittable

**Date:** 2026-08-16
**Deadline:** 29 Aug 2026 AoE
**State at entry:** `5f80386` plus this round's §1/abstract novelty boundary. Build clean: 0 overfull boxes, 0 undefined citations, 0 undefined refs. Main content **5 pages** against a 4-page limit.

---

## 0. What is already done — do not redo

| | |
|---|---|
| Related work (D1–D3) | GLS/matched-filter, PPCA/factor-analysis boundary, SSL anchoring |
| Paper 3 GWOSC (D4) | Real-data covariance adequacy restored |
| Architecture ladder | §3 rewritten as four rungs; both key sentences present |
| Block A / Block B | Archived numbers in §4; four placeholders retired |
| Title | "Self-Supervision" removed |
| **Novelty boundary** | **This round** — §1 paragraph and abstract clause naming every classical antecedent and stating "no new estimator" |
| Lineage citations | Bailey and Allen restored (a compression regression), all six now cited and resolving |

---

## 1. The three remaining placeholders

All rung-4, all blocked on one rescore. **None is a failed result.**

| # | Location | Placeholder | Resolved by |
|---|---|---|---|
| P-1 | `sections/00_abstract.tex` | `rung-4 paired contrast` | R1 below |
| P-2 | `sections/04_experiments_discussion.tex` | `rung-4 paired contrast` + appendix ref | R1 below |
| P-3 | `sections/05_discussion.tex` | `rung-4 verdict` | R1 below |

### R1 — the rescore (server, ~30 min, no retraining)

The upstream `benchmark.py:139` rounds its raw score to two decimals before the logarithm, which put both arms in the same 0.01 bin and produced a spurious exact-zero contrast. `tidmad.score_unrounded` recomputes the same quantity from the release's own SNR functions without the rounding.

```bash
cd <repo root> && export PYTHONPATH=src
for ARM in t_mse t_chi2; do
  python -m tidmad.score_unrounded --upstream ~/src/TIDMAD \
    --data-dir /ceph/dwong/datasets/tidmad/denoised --model-tag $ARM \
    --in-band-only results/rps2026/P1_injection_survey.json \
    --out results/rps2026/E5_${ARM}_inband_unrounded.json
done
```

**Then fill P-1 to P-3 with the paired difference**, and state in §4 that the upstream score is quantised to two decimals so the unrounded aggregation is reported instead — one sentence, because a reader who tries to reproduce the published number needs to know why it differs.

**If the unrounded contrast is still ~zero**, that is a genuine null and reportable as one. Say *"the benchmark cannot resolve the contrast at this operating point"*, not *"the metric made no difference"* — both arms sit ~6 orders of magnitude below FCNet, so the score is near its floor.

### R2 — the better rung-4 evidence (do this regardless)

Extract held-out χ² and raw MSE per evaluation window from `artifacts/tidmad/{t_mse,t_chi2}/training.log` and report the paired contrast with block-bootstrap intervals. Final values were 11519.3 / 0.0135078 and 11484.4 / 0.0134244.

**This is the direct analogue of Block A's contrast, it is the paper's actual claim, and it does not depend on the scorer.** The benchmark score becomes supporting context. If only one rung-4 number survives to the paper, it should be this one.

---

## 2. The page budget — needs a structural decision, not more trimming

Main content is 5 pages; **~1500 characters overflow**. Six paragraphs have already been compressed (intro SSL, linear-rung numbers, rung-3→4 transition, GWOSC, rung-4 architecture, covariance contract) and returns are diminishing: each further cut saves ~200 characters and the text reflows.

Two costs are still coming: the rung-4 figure (~0.3 page) and, offsetting slightly, the placeholders shrinking to numbers (~3 lines back).

**Decide one of these; do not keep nibbling.**

| Option | Cost | Recommendation |
|---|---|---|
| **A. Figure to the appendix** | Loses the poster-facing figure from the main text | **Take this.** Appendices are unlimited and excluded from the limit. A referenced appendix figure still counts for the poster. |
| **B. Merge §5 into §4** | Removes one section heading and its whitespace; ~0.2 page | Take this *with* A |
| **C. Cut the GWOSC paragraph** | Loses the Transfer-theme hook and the strongest real-data evidence | Only if A and B are insufficient |
| **D. Cut a proposition** | Guts the contribution | Never |

**Recommended: A + B**, then re-measure. If still over, compress §2's proof sketch — the appendix carries it in full.

**Protected, do not cut:** the `(Σ⁻¹, F)` tuple, Proposition 1, Proposition 2, metric reversal, the NFPA/factored-attention parallel sentence, the novelty-boundary paragraph, the Block A numbers.

---

## 3. The figure — the largest cheap gain

There is still **no figure** in a poster-first venue. The content already exists.

**Panel A:** Block A paired contrast — NFPA minus the isotropic baseline under both metrics, with 95% intervals and a zero line. It shows the reversal in one glance: −0.0192 [−0.0194, −0.0190] on χ², +0.0034 [+0.0034, +0.0034] on MSE.

**Panel B:** the Block B control at exactly zero, same axes.

Generated by script from `results/rps2026/E1_nfpa_blockA/block_A_colored_raw.csv` and the Block B CSV. Caption states matched total rank, shared architecture, 10 and 20 seeds. Colourblind-safe, vector output.

**This is the single highest-value remaining item after the rescore.** It needs no new experiment.

---

## 4. Order

| # | Item | Owner | Blocks |
|---|---|---|---|
| 1 | R1 rescore | server | P-1/2/3 |
| 2 | R2 held-out contrast extraction | server | rung-4 evidence |
| 3 | Fill P-1 to P-3 | desk | — |
| 4 | Figure from Block A/B CSVs | desk | page budget |
| 5 | Page decision A+B, re-measure | desk | submission |
| 6 | Final build, anonymity check | desk | submission |

Steps 1–2 are one server session. Steps 3–6 are one desk session.

---

## 5. Submission checks

- Main content ≤ 4 pages; references and appendices excluded.
- **Zero `\PENDING` anywhere**, including `references.bib`.
- No author names, affiliations, acknowledgements, or identifying URLs. Acknowledgements are camera-ready only.
- The `priorThesis` citation stays — it is required by the attribution conditions — but **note that it names a contributor**, which is a deanonymisation vector under double-blind. This is an author decision that has been open since round 1 and must be settled before upload.
- Single PDF; no supplementary files. Any code link must be anonymised.
- Every number traceable to an archived run.
