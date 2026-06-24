# Draft figure captions

These captions are working text for the paper-writing agent. Exact figure
numbering should be assigned only after the manuscript figure sequence is
frozen.

## Synthetic validation overview

Representative multi-seed synthetic validation gates. **A**, The empirical
optimal-filter amplitude standard deviation divided by the Cramér–Rao bound
across ten seeds; the dashed line marks unity. **B**, The weighted principal
angle between the EMPCA direction and the optimal-filter direction. **C**, The
held-out weighted-residual difference between ordinary PCA and weighted PCA in
colored noise; positive values favor weighted PCA. **D**, Mean whitened
residual chi-square per degree of freedom; the dashed line marks unity. Each
point is one deterministic seed. Complete 68% and 95% intervals are reported
in `data/derived/synthetic_gate_summary.csv`.

## GWOSC held-out null calibration

Held-out amplitude-spread calibration for 256 seconds of GW150914-centered H1
and L1 open data. **A**, Five deterministic random calibration/evaluation
splits. **B**, Five contiguous chronological evaluation blocks with nearby
calibration windows. The ordinate is the observed held-out amplitude standard
deviation divided by the PSD-derived prediction. The dashed line marks unity,
the darker green band marks the predeclared median interval `[0.8, 1.2]`, and
the lighter green band marks the per-split interval `[0.5, 1.5]`. The
predeclared gate failed for both detectors and both split designs.

## GWOSC reference comparison

Independent reference and score-path diagnostics. **A**, The repository
Hann-windowed, bias-corrected median PSD divided by GWpy’s PSD on identical
calibration windows. Both detector ratios equal unity at recorded precision.
**B**, Correlations among the repository generalized least-squares score, the
repository direct-rFFT-whitened score, and the earlier GWpy
inverse-spectrum-FIR score. Panel B compares non-identical effective filters
and is diagnostic rather than an implementation-equivalence acceptance test.

## GWOSC run-history audit

Archived remote-run audit trail. Rows are synchronized GWOSC evidence runs and
columns show Stage 0 reproducibility, the baseline GWOSC scientific gate, and
the presence or absence of the follow-up filtering and local-PSD evidence. This
figure is primarily a handoff/supplemental control plot: it prevents writing
agents from confusing operational success, scientific acceptance, negative
results, and pending evidence.

## Paper claim-support matrix

Claim-state matrix used as a writing-control figure. Each row is a paper claim
area and the marker indicates whether the current bundle supports it as
verified, verified-negative, implemented-but-pending, or not validated. This
plot is not a scientific result by itself; it is an audit figure that helps
separate supported manuscript statements from overclaims.

## Pending shared-FIR equivalence figure

For each stationary-synthetic, H1, and L1 dataset, the planned figure will show
the maximum absolute score difference between explicit FFT convolution and
GWpy convolution over the predeclared FIR-duration/edge-trim grid. A separate
panel will show the original PSD-domain GLS/shared-FIR correlation. The first
quantity is the software identity gate; the second is a methodological
sensitivity diagnostic.

## Pending time-local PSD figure

The planned figure will compare normalized held-out score standard deviations
for leave-one-out global PSDs and fixed 32-, 64-, and 96-second local PSDs. The
64-second model is primary. A second panel will display the primary model’s
five chronological blocks. Template-projected and fixed narrow-band power
ratios will be reported in supplementary panels or tables without defining
post hoc frequency cuts.
