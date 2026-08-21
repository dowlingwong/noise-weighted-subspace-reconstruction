# OpenReview submission metadata — R/PS 2026

Venue: `NeurIPS.cc/2026/Workshop/RPS`, Short Papers track. Deadline 29 Aug 2026
AoE. Paste-ready fields below; the PDF is `paper_rps2026/paper.pdf` (single file,
no supplementary). Field names on the actual form may differ slightly — map by
meaning.

## Title

```
Noise Covariance as Representation Geometry: One Metric, Four Architecture Classes
```

## Abstract (plain text, LaTeX stripped)

```
Label-free scientific reconstruction learns from measured observations alone:
each observation supplies its own reconstruction supervision, while the
instrument-characterized noise covariance determines which residuals are
informative. For Gaussian noise, inverse covariance is the likelihood-consistent
metric. Rather than proposing a new estimator, we organize classical methods as
a pair of inverse covariance and representation class, separating measurement
geometry from admissible model capacity, and follow this metric through optimal
filtering, unrestricted linear projection, Kronecker-separable projection, and
factored attention. The linear classes admit an exact geometric
characterization. At the separable-projection rung, models sharing an
architecture but differing only in the training metric reverse their ranking
under raw MSE and the likelihood residual, in a frozen ten-seed experiment and
in an independent twenty-seed replication. The reversal persists when the
covariance is estimated from 2,048 noise-only calibration events in one fixed
sensitivity condition, while the covariance-weighted factors recover the planted
channel subspace almost exactly but the temporal subspace only weakly. At the
factored-attention rung the reversal is absent across two independent seeds,
despite non-degenerate weighting. Noise covariance therefore defines the
scientifically relevant geometry, while instrument and representation class can
determine whether that geometry measurably changes what is learned.
```

**Updated 2026-08-21 (external review response).** This is the current abstract,
transcribed from `sections/00_abstract.tex` with LaTeX stripped and the pair
notation spelled out. Re-verify it against the built `paper.pdf` before pasting;
the field must match the PDF.

Evidence status: all planned experiments are complete; nothing is pending from
the server. E10b (estimated separable covariance, `N_cal = 2048`) is **one
frozen sensitivity condition** — it earns the single abstract clause about
estimated covariance and must never be described as a robustness sweep. E11
(corrected principal angles) is a **post-hoc diagnostic correction** whose
practical effect is under one degree; it is confined to the appendix, is not
confirmatory, and does not appear in the abstract.

## TL;DR (one sentence)

```
Instrument noise covariance is the natural self-supervised metric for scientific
reconstruction; walking one fixed metric up four architecture classes shows
where the metric provably changes what is learned -- and measures where it stops
mattering.
```

## Keywords

```
noise covariance, self-supervised learning, representation learning,
scientific machine learning, whitening, weighted PCA, multilinear PCA,
transformers, tokenization, time series, dark matter detection,
gravitational waves
```

## Workshop theme fit (if a field asks)

Self-supervision (the measurement likelihood as the objective; no augmentation
policy), Tokenization (choose the token axis in which the inverse covariance is
diagonal; Section 3), Transfer (a covariance validated in-distribution fails
calibration on real strain; Section 4).

## Confirmed against the published call (2026-08-21)

Short Papers · **non-archival** · four pages of main content, **references and
appendices excluded** · **single PDF**, no additional files · **double-blind**,
NeurIPS main-track approach · NeurIPS paper checklist **not required** ·
concurrent submissions allowed, author responsible for the other venue's
dual-submission policy · anonymized code/data links permitted but optional.

Deadline **29 Aug 2026 AoE**; notification **by 29 Sep 2026 AoE**.
Submit at `https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/RPS`.

Two items to settle before uploading:

1. **Diff `neurips_2026.sty` against the workshop's `rps_2026_template.zip`.**
   Ours is the NeurIPS style with `dblblindworkshop`, which is what the call
   describes, but the template was not unpacked and compared here.
2. **Decide the companion-preprint timing.** The appendix's carried rung-2
   numbers also appear in the named companion preprint; posting it during the
   review window weakens the blind. See the anonymous-submission audit in
   `docs/plans/rps2026/ROUND10_PAPER_REVISION_QA.md`.

## Submission-day checklist

1. Rebuild from clean (`make distclean && make` equivalent: `latexmk -C; make`).
2. Verify: 4 main pages, references from p5, 0 overfull/undefined/PENDING,
   anonymity sweep clean, empty Author metadata.
3. Abstract field synced to the built PDF; Round 10 appendix table regenerated
   by `scripts/make_rps_round10_table.py` (digest check passes).
4. If the code link is live and gate-6-verified: it appears in the PDF; if not,
   no mention of a repository anywhere.
5. Upload single PDF; fill fields from this file; submit; save the receipt.
```
