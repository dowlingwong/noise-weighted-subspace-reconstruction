# Paper Update Plan: Integrating the CRESST-A Positive Result

Created: 2026-07-04. Source evidence: `docs/CRESST_RESULT.md`,
`evidence/cresst/20260704T013758Z_9c9f696ff6bc/`,
`transfer_paper/data/derived/cresst_{pooled,per_channel}_summary.csv`.
Predeclaration commit: `9c9f696ff6bc`.

## Headline change

The paper gains its first positive public real-data validation. The claim
ladder becomes: theory → S1–S9 synthetic (positive) → GWOSC (negative
real-noise stress test, boundary) → CRESST (positive predeclared real-data
validation of the linear hierarchy, per-channel). Results should end on
CRESST so the narrative closes on the positive result GWOSC motivated.

## Section-by-section edits

### Abstract

After the GWOSC sentences, insert (condensed from the manuscript-safe
paragraph in `docs/CRESST_RESULT.md`):

> A subsequent predeclared validation on the public CRESST-II/III
> pulse-shape release passed: with each of seven detector channels split
> chronologically into calibration and evaluation segments, rank-6 EMPCA
> under a per-channel PSD-stationary noise metric beat the single-template
> optimal filter and remained competitive with a multi-template
> optimal-filter bank, with paired-bootstrap 95% intervals excluding zero
> and all predeclared negative controls degrading as required.

Keep the closing scope sentence ("not ... general real-detector
calibration") — it remains true: CRESST validates the *relative hierarchy*
per channel, not absolute residual calibration (see whitening-ratio
diagnostic below).

### 1.2 Contributions (item 6)

Extend the evidence list: "...negative held-out GWOSC calibration under
both global and predeclared local PSD models, **and a passed predeclared
CRESST-II/III public-pulse validation in which rank-6 EMPCA beat
single-template OF and remained competitive with a multi-template OF bank
in all seven test-release channels**."

### Section 7 — new subsection "CRESST public pulse-shape validation"

Place after the GWOSC subsections (7.8–7.12), before the placeholder
section. Question/Setup/Metric/Result/Interpretation format:

- **Setup**: DMDC CRESST-II/III pulse-shape release (arXiv:2508.03078);
  7 test-split detector channels; per-channel chronological 70/30
  calibration/evaluation split by `hours`; noise metric = inverse of a
  circulant covariance built from the calibration-segment median PSD
  (257 free parameters vs ~131k for a full sample covariance) with
  shrinkage 0.1; predeclaration frozen at commit `9c9f696ff6bc` before the
  single run; n = 19,335 pooled evaluation events.
- **Result table (pooled paired bootstrap, 300 reps)**:
  OF − EMPCA = 1012.3 [927.9, 1097.6];
  1.1×multi-OF − EMPCA = 352.2 [272.4, 424.0];
  all five negative controls degrade with CIs excluding zero
  (wrong_psd 1.57 [0.97, 2.08]; permuted_time_bins 59,813;
  random_weighted_basis 4,013; noise_only_train 489; mismatched_of_template
  6,912). EMPCA won in every channel individually.
- **Per-channel table**: from `cresst_per_channel_summary.csv`
  (PCA/EMPCA/OF/multi-OF weighted residuals, n_train/n_test/n_bg,
  sigma_OF source incl. the (35,56) CRB fallback).
- **Honest diagnostics in the same subsection** (Reporting Standard 6):
  1. wrong_psd sign reversal on channel 35_58.0 (mean −0.005, CI
     straddles zero) — the only reversal in 35 control×channel cells, in
     the weakest-margin control; connect explicitly to the S4 white-noise
     control: when noise is near-white the metric provably stops
     mattering, so this is the theory's own predicted failure mode of the
     control, not an anomaly. (35_58.0 also has the whitening ratio
     closest to 1, consistent with the most nearly white/stationary noise.)
  2. Residual whitening ratio ≪ 1 for most channels (medians 0.003–0.88):
     absolute residual calibration was NOT achieved; the noise model
     over-predicts residual power. Plausible cause: the noise bank uses
     all `noise`-flagged random triggers, ~5% of which fail quality cuts
     and can contain chance pulses/artifacts (per the release paper),
     inflating the PSD. Report as diagnostic; do not claim calibrated
     residuals on CRESST.
  3. Trigger efficiency inconclusive: at 5% FPR, 6/7 channels never rose
     off the background floor across the frozen [0–10]×σ_OF grid. Same
     plausible root cause (inflated background score thresholds from the
     contaminated noise bank) plus grid anchoring to full-resolution
     published σ_OF. Reported as limitation; grid frozen predeclared, not
     widened post hoc.

### 7.13 Placeholder section

Resolve and delete the CRESST placeholder (evidence is now synchronized in
`transfer_paper/`). Keep the confirmatory-GWOSC and calibrated-significance
placeholders. Remove "CRESST/SCRESST ... still needs independent review"
framing; replace "CRESST/SCRESST" with "CRESST" throughout (the run used
the CRESST DMDC release).

### Claim support state table/figure (Section 7 end)

Add row: "CRESST-A linear hierarchy (public real data) — verified positive
— Results and Methods." Wording per `claim_status.csv`.

### Section 8 Discussion

Add the two-sided boundary statement: GWOSC and CRESST together locate
what transfers to real data and what does not. The *relative* likelihood
geometry (hierarchy ordering, negative-control structure) validated on
real cryogenic pulses per channel; *absolute* noise-model calibration
failed on GWOSC and is diagnostically imperfect on CRESST (whitening ratio
< 1). The lesson "correct implementation ≠ calibrated noise model" now has
a positive counterpart: the comparative claims survive noise-model
imperfection because all methods share the same metric.

Also note the PCA-vs-EMPCA margins: on several channels ordinary PCA is
close to EMPCA (e.g. 7.94 vs 7.03; 2.50 vs 2.46) while both crush
single-template OF. State honestly that on this dataset the dominant real
gain is low-rank pulse-shape structure, with noise weighting contributing
a smaller but pooled-significant improvement — consistent with some
channels' noise being near-white, and exactly what the theory predicts for
that regime.

### Section 9 Limitations

Replace "positive real-data validation should come from ... CRESST/SCRESST
after schema review" (now done) with the actual residual limitations:

- per-channel design: no cross-channel/cross-detector transfer claim;
- test split only (7 channels, single frozen seed/split);
- trigger efficiency / detection sensitivity not validated (inconclusive
  at the frozen grid);
- absolute residual calibration not demonstrated (whitening ratio < 1;
  suspected noise-bank contamination from non-clean random triggers);
- σ_OF anchoring imperfect (published resolutions are full-resolution OF
  values; one channel required a CRB fallback);
- Stage C2 smoke-run contamination acknowledgment (values directional,
  gate not threshold-fitted; see `docs/CRESST_PREDECLARED_GATES.md`).

### Section 9/10 Future work

Update the future-work list: new predeclared CRESST stages — (a) noise
bank restricted to `noise & clean` traces plus wider/denser amplitude grid
to resolve the efficiency crossings; (b) extension to the 61-channel train
split; (c) an explicit cross-channel transfer study framed as domain
transfer. Keep GWOSC Stage 3b. Drop "CRESST validation after schema
review" (completed).

### Section 10 Conclusion

Add one sentence: the hierarchy's predictions were confirmed on public
cryogenic detector data under a predeclared protocol, making the linear
noise-geometry baseline validated in controlled synthetic settings and in
one public real-detector domain, with real-noise absolute calibration
remaining the open frontier.

### Methods / Appendices

- Describe the PSD-stationary circulant covariance and note its inverse is
  diagonal in the Fourier basis — i.e. identical to the paper's Section 2–3
  inverse-PSD weighting; cite the N≈p motivation (~510–540 calibration
  noise traces vs 512 features).
- Preprocessing contract items for CRESST (baseline fraction 0.2, native
  512 samples, chronological split unit, seed 22, CRB fallback rule).
- Data availability: DMDC GitLab repository URL; headerless `.npy` format
  note.
- Citations (required by DMDC): CRESST Collaboration, arXiv:2508.03078
  (2025); G. Angloher et al., Eur. Phys. J. Plus 138, 100 (2023),
  arXiv:2211.00564. Add DMDC/ORIGINS acknowledgement.

### Figures to add

1. Per-channel paired weighted-residual comparison (OF, multi-OF, PCA,
   EMPCA; log y), 7 channels — the headline figure.
2. Negative-control forest plot: pooled mean difference with 95% CI per
   control (log x), marking the wrong_psd/35_58.0 reversal.
3. (Appendix) residual whitening ratio vs frequency per channel — the
   honest calibration diagnostic.
4. (Appendix, optional) efficiency curves showing the inconclusive floor,
   labeled as limitation.

### Consistency sweep

- Update every instance of "CRESST remains the next candidate" /
  "planned" (README, Status, Section 7 intro, Discussion, Limitations,
  Future work) — grep for `CRESST` and `SCRESST` in the LaTeX source.
- Results ordering: S1–S9 → GWOSC checks → GWOSC negative → CRESST
  positive → limitations.
- Do not touch: GWOSC wording (frozen), internal K-α Experiment A framing,
  S1–S9 numbers.

## Wording to avoid (from docs/CRESST_RESULT.md)

No validated detection sensitivity; no cross-channel transfer; no "all
controls degraded in every channel"; no calibrated-residual claim on
CRESST; no quoting Stage C2 smoke metrics.
