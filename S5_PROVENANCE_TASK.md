# Task: resolve the S5 provenance discrepancy in Paper 1 §7.2

**Raised:** 2026-07-28, during an external review pass on the arXiv v1 preprint.
**Severity:** stop-ship for arXiv posting. The manuscript reports S5 values that
match no archived run, in a paper whose central credibility claim is that every
quantitative result traces to a checksummed record.
**Repos:** manuscript = `arXiv-v1-polished-extended-preprint`;
evidence/code = `noise-weighted-subspace-reconstruction`.

---

## 1. The discrepancy

`07_experiments.tex` (§7.2, "Controlled synthetic validation") reported the
following, described in the text as the **30-seed confirmatory** run:

| Quantity | Manuscript (as of 2026-07-27) |
|---|---|
| PCA held-out weighted residual | 48.418 |
| EMPCA held-out weighted residual | 47.878 |
| Paired PCA−EMPCA difference | 0.540 |
| 95% bootstrap interval | [0.523, 0.560] |
| Raw MSE, PCA | 2367.99 |
| Raw MSE, EMPCA | 5388.75 |
| Max whitened principal angle | 86.07° |

None of these seven values appears in any archived S5 JSON. A numeric sweep
over every `S5*.json` in the repo (tolerance 5e-4, all nested keys flattened)
returned exactly one match: `0.540` is within tolerance of
`absolute_difference.weighted_residual_pca_minus_empca.ci68[0] = 0.5402…` in the
30-seed file — i.e. the **lower bound of the 68% interval**, not a point
estimate.

## 2. What the archives actually contain

| Quantity | 10-seed | 20-seed | **30-seed confirmatory** |
|---|---|---|---|
| PCA weighted residual | 48.490 | 48.478 | **48.4192** |
| EMPCA weighted residual | 47.945 | 47.934 | **47.8690** |
| Paired difference | — | 0.5446 | **0.5503** |
| 95% CI | — | [0.5248, 0.5649] | **[0.5301, 0.5709]** |
| 68% CI | — | — | **[0.5402, 0.5605]** |
| Raw MSE, PCA | — | 2360.35 | **2330.27** |
| Raw MSE, EMPCA | — | 5392.24 | **7412.67** |
| Max principal angle | 85.442 | 86.423 | **86.2469** |
| Held-out NLL diff (PCA−EMPCA) | — | — | **11.6705, CI [11.2429, 12.1083]** |
| PCA NLL mean | — | 1028.17 | **1026.92** |

Files:

- `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`
- `results/sweeps/S5_sweep_20seeds.json`
- `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`
- config: `configs/synthetic/s5_metric_reversal_v2.yaml`
  (`n_traces: 512`, `n_features: 256`, `rank: 2`, `test_frac: 0.5`,
  `split_seed: 0`, `seed: 19`)

## 3. Why this is not a simple rounding error

Two independent reasons:

1. **The manuscript's values straddle two runs.** Its weighted residuals sit
   near the 30-seed run (48.418 vs 48.4192) while its raw-MSE values sit near
   the 20-seed run (2367.99 vs 2360.35; 5388.75 vs 5392.24) and are nowhere
   near the 30-seed run's EMPCA raw MSE of 7412.67. A single stale run cannot
   produce that pattern; it suggests values transcribed from more than one
   source, or from a run that predates both.
2. **The EMPCA raw-MSE gap is large, not marginal.** 5388.75 vs 7412.67 is a
   38% difference. That is a different computation, not a formatting artifact.

## 4. Interim action already taken (must be reviewed)

To keep the manuscript internally consistent and traceable, §7.2 was updated on
2026-07-28 to report the **30-seed confirmatory run throughout**, on the
grounds that the surrounding text already declares that run as its source. The
manuscript now reads 48.419 / 47.869 / 0.550 / [0.530, 0.571] / 2330.27 /
7412.67 / 86.25°, plus a newly reported held-out NLL contrast.

**This substitution is provisional and is exactly what needs confirming.** If
the intended source is a different run, §7.2 must be corrected again and that
run archived.

Also reverted on 2026-07-28: a sentence added earlier to §Code-and-data
claiming "every quantitative claim in this manuscript resolves to a checksummed
record in that archive." That claim cannot stand until this task closes.

## 5. What to do

### 5.1 Determine the true source of the original numbers

Search execution history and any unarchived output directories for a run
producing **EMPCA raw MSE ≈ 5388.75 together with PCA weighted residual ≈
48.418**. Candidate explanations, in rough order of likelihood:

- an earlier 30-seed run at a different `split_seed` or before the `v2` config
  change (`test_frac` was previously defaulting to 0.0, i.e. in-sample — see the
  comment block in `configs/synthetic/s5_metric_reversal_v2.yaml`);
- a run on the remote host (`/ceph/dwong/...`) whose JSON was never synced back,
  the same failure mode as the TIDMAD file-`0000` incident;
- hand-transcribed values from a notebook that was not saved.

If found: archive it with a checksum, add it to `BUNDLE_MANIFEST.json`, and
report which of §7.2's numbers it reproduces exactly.

If not found after a bounded search: report that, and the interim substitution
in §4 becomes the permanent fix.

### 5.2 Re-run the 30-seed confirmatory analysis for verification

Re-execute the frozen 30-seed S5 confirmatory sweep and confirm it reproduces
`S5_confirmatory_30seeds.json` to the precision reported in §2. This
distinguishes "the archive is right and the text was stale" from "the archive
itself is not reproducible", which would be a much larger problem.

Record: exact command, code revision, environment, wall-clock, and a checksum of
the regenerated JSON.

### 5.3 Audit the other synthetic experiments the same way

S5 was found only because its numbers were spot-checked. **S1–S4 and S6–S9 have
not been checked against their archives.** Run the same flattened numeric
comparison for every value §7.2 reports:

- S1: OF sd / predicted bound = 0.994, ten-seed CI [0.985, 1.002]
- S3: tied-AE relative gap = 1.95e−16
- S4: white-noise principal angle = 1.48e−6 degrees
- S6: mean best rank by clean-signal MSE = 4.6
- S7: reconstructed uncertainty ratio = 1.032
- S8: mean whitened residual per dof = 0.997
- S9: diagonal-vs-full uncertainty factor = 1.025

A reusable script for this is worth writing regardless — see §6.

### 5.4 Report back

Return a short markdown with: the source run if found; the re-run verification
result; a per-value table for §5.3 marked reproduced / discrepant / not found;
and any manifest updates made.

## 6. Suggested reusable check

The comparison that found this is mechanical and should become a repo test:
flatten every archived result JSON, extract every numeric literal from the
manuscript `.tex` sources, and report manuscript values with no archive match
within tolerance. That is a stronger guarantee than the current semantic
checker, and it would have caught this before the review did.

Note the sharpest signal here: the only near-match for the manuscript's stated
point estimate was a **68% interval endpoint**. A checker that matched only
point estimates against point estimates would have flagged it immediately.

## 7. Related open lead (separate task, low priority)

The withdrawn legacy Appendix F residual values were **1029.2 / 1028.9 /
1028.6**. The S5 30-seed run's `pca_nll_mean` is **1026.92** with 95% interval
**[1024.78, 1029.01]**, and the 20-seed run's is **1028.17** with interval
[1025.99, 1030.40]. All three legacy values fall inside or adjacent to those
intervals.

This suggests the legacy "residual" table may have been **held-out negative
log-likelihood values from the S5 family, mislabelled as residuals** — which
would explain both their near-identity to each other and their absence from any
residual record. Worth one hour of checking, because confirming it would let
Appendix F state what those numbers were rather than only that they could not
be traced. Do not restore them to the manuscript on this basis alone.
