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
Label-free scientific reconstruction learns representations from measured
observations alone -- self-supervision in which the instrument-characterized
noise covariance, not an augmentation policy, decides which residuals are
informative. For Gaussian noise, inverse covariance is the likelihood-consistent
reconstruction metric. Every ingredient here is classical -- generalized least
squares, the matched filter, weighted PCA, the linear-autoencoder/PCA
correspondence, multilinear PCA -- and we introduce no new estimator; we
organize them as a pair (inverse covariance, architecture), the likelihood
fixing the residual geometry and the architecture the admissible representation
class. Holding the metric fixed, we walk the architecture up four rungs: optimal
filtering, an unrestricted rank-k linear projection, a Kronecker-separable
two-factor projection, and factored attention. At the linear rungs the metric
question has a closed-form answer and reverses in practice: models sharing an
architecture and differing only in the training metric rank oppositely under raw
MSE and the likelihood residual (ten seeds, paired intervals excluding zero; a
twenty-seed white-noise control is exactly zero). At the attention rung the
reversal does not appear, and the obvious explanation -- a near-degenerate
whitening there -- is ruled out by measurement. The metric fixes the geometry;
whether that choice bites is a property of the instrument and the
representation.
```

**Updated 2026-08-19 (Round 10 evidence integration).** The abstract above is
superseded — regenerate it from `sections/00_abstract.tex`, which now states the
twenty-seed replication, the channel/temporal recovery asymmetry, and two
independent rung-4 seeds. The abstract field must match the PDF exactly; paste
from the built `paper.pdf`, not from this file.

Still pending before submission: E10b (estimated-covariance sensitivity) and
E11 (corrected angles) from the server; neither changes the abstract.

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

## Submission-day checklist

1. Rebuild from clean (`make distclean && make` equivalent: `latexmk -C; make`).
2. Verify: 4 main pages, references from p5, 0 overfull/undefined/PENDING,
   anonymity sweep clean, empty Author metadata.
3. If E7 landed: contingency applied, figure regenerated (with or without
   `--with-seed2 E7`), abstract field synced to the PDF.
4. If the code link is live and gate-6-verified: it appears in the PDF; if not,
   no mention of a repository anywhere.
5. Upload single PDF; fill fields from this file; submit; save the receipt.
```
