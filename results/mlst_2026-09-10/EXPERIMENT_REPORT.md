# MLST revision — experiment report

**Run date** 10 September 2026 · **Host** `deepthought` (Linux 4.18, 2×10-core
Xeon E5-2630 v4 + HT = 40 threads, 503 GB RAM, 4× GTX TITAN X — GPUs unused,
every stage is CPU/NumPy) · **Scheduler** none used; local execution, never
exclusive, ≤16 of 40 threads.

Nothing was pushed, published or submitted. `\gatedevidencefalse` unchanged.
No historical output path was overwritten: every artefact is in the new
versioned tree `results/mlst_2026-09-10/`.


## 0. Summary of outcomes

| Task | Status | Headline outcome |
|---|---|---|
| **T13-a** CRESST channel robustness | **run** | θ = **21.5689 %**, hierarchical 95 % CI **[10.31, 38.83] %**; all 7 channel effects positive and individually excluding zero; temporal dependence negligible. Labelled a *sensitivity*, not a replication |
| **T10–T12** TIDMAD quadrature endpoint | **run** (frozen, executed once) | Combined EMPCA-vs-zero **+1.6e−05 [−8.9e−05, +1.2e−04]** and vs calibration-mean **+4.6e−04 [−3.8e−05, +9.6e−04]** — both include zero. The no-denoising bin read beats EMPCA (**−1.19e−02 [−1.31e−02, −1.08e−02]**). No "useful" verdict issued |
| **T15** covariance-error regime | **run** (80 fits) | Bound holds **80/80** but is **never informative** (≥ 4.27 > 1) for any non-oracle arm; ε alone does not predict task outcome |
| **T14** structural-prior tradeoff | **run** (1000 fits) | **Null crossover** — none anywhere in the declared grid at either SNR. Confirmation stage correctly not triggered; its seeds never spent. Claim stays conditional |
| **T16** AE optimisation diagnosis | **not run, by design** | Not needed to retain the empirical optimizer/tying claim (§9) |
| **T13-b / T13-c** | **not run** | Outside the authorised stage list — but both recorded *data* blockers are resolved (§10) |

**Three results the manuscript does not yet reflect:**

1. **TIDMAD gives no recovery result at this operating point.** Rank-2
   whole-window PCA/EMPCA is indistinguishable from predicting zero (mean RMSE
   0.655 and 0.661 vs 0.654) and ~2.7× worse than reading the injection bin
   (0.246). The mechanism is measured, not conjectured: `p/n = 187` puts the
   Marchenko–Pastur eigenvalue inflation at ≈216× the noise level against a
   tone at 25×.
2. **A stated protocol expectation is refuted.** High-frequency TIDMAD files
   were expected to be the uncoupled failure regime; they are the
   best-coupled ones (0018 at 1.80 MHz and 0019 at 3.40 MHz, readout line SNR
   ≈ 30). The low-frequency files fail instead.
3. **The covariance-error bound is valid but vacuous** in the regime the
   theory section addresses, and structured misspecification at ε = 0.557 is
   harmless where unstructured sampling error at ε = 0.575 is catastrophic.

---

## 1. Source state

| Role | Path | Required baseline | Verified |
|---|---|---|---|
| MS | `/ceph/dwong/arXiv-v1-polished-extended-preprint` | `1ad2ac98e366815f9b8a7f6bcf6596c22ff4732c` | HEAD **=** `origin/main` **=** baseline ✓ |
| EXP | `/ceph/dwong/noise-weighted-subspace-reconstruction-mlst` | `27220c0ed45b92cc76ab357c25f6fab590614ac9` | `origin/dev` **=** baseline; worktree started there ✓ |

`origin/dev` (`27220c0e`) was used, not `origin/main` (`daf51918`). Both
baseline objects were confirmed present with `git cat-file`. Neither repository
was modified outside the isolated work branch; MS is untouched (clean tree at
the baseline commit).

* EXP work branch **`mlst/exp-phase-20260910`**, started at `27220c0e`.
* EXP is a git worktree of `/ceph/dwong/noise-weighted-subspace-reconstruction`;
  the pre-existing `main` worktree (at `8ee9eb1`) was left untouched.
* Environment: `uv sync --extra dev` from the repository's `uv.lock` —
  Python 3.12.2, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.3, h5py 3.16.0.

### Commits produced (all local; nothing pushed)

| Commit | Content |
|---|---|
| `a52db10` | freeze(T13-a): bootstrap runner + 14 tests |
| `38b0a33` | freeze(T10–T12): TIDMAD endpoint, stage runner, 47 tests |
| `c8dab7a` | freeze(T11): predeclared operating point |
| `559f743` | evidence: T13-a, TIDMAD characterisation/development/freeze |
| `077e3d1` | evidence: TIDMAD validation + final test |
| `64dfe1e` | freeze(T15): generator G1 + 18 sanity checks |
| `f9714e9` | freeze(T14): runner, derived budget, crossover rules, 17 tests |

## 2. Data preflight

`PAPER1_DATA_ROOT` was **unset** in the environment; the CLI/config default
`/ceph/dwong/paper1_dataset` resolves correctly and was used.

| Dataset | Status | Used for |
|---|---|---|
| TIDMAD `abra_validation_0000–0019.h5` | **all 20 present** (1.9–2.4 GB each, int8, 2.01×10⁹ samples/channel, gzip, 10 MHz) | T10/T11/T12 |
| TIDMAD `abra_science_0000/0001.h5` | present (2.9 GB each, readout channel only) | noise bin powers |
| CRESST archived residuals | present in MS bundle | T13-a |
| CRESST raw traces / noise timestamps | not required for T13-a | T13-b/c remain blocked |

Metadata were inspected before any bulk read. No release download was performed.
Read throughput 22 Ms/s on the readout channel, so a 37 M-sample stretch costs
about 2 s and I/O was never the bottleneck.

---

## 3. Stage 1 — T13-a, CRESST channel robustness

Frozen at `a52db10`; executed once. 10 000 replicates, base seed 20260713,
percentile intervals, 52 s wall, 131 MiB peak RSS.

**This is a sensitivity analysis of an archived result, not an independent
replication.** Seven channels drawn from two runs are seven correlated
clusters, not seven independent detector replications, and no number below may
be relabelled as one.

Both archive point estimates reproduce exactly from
`event_residuals_lossless.npz`: θ = **21.5689 %** and θ_ratio = **22.0719 %**
(19 335 evaluation events, all weights 1, 7 channels, 2 runs).

### Headline and named sensitivities

| Variant | Estimand | Channels | Block | 95 % CI (%) | sd (%) |
|---|---|---|---|---|---|
| **headline** | mean-of-fractions | hierarchical | 1 | **[10.312, 38.827]** | 7.483 |
| channel-fixed | mean-of-fractions | fixed | 1 | [17.415, 27.129] | 2.487 |
| ratio-of-means | ratio-of-means | hierarchical | 1 | [12.137, 39.832] | 7.135 |
| ratio-of-means | ratio-of-means | fixed | 1 | [15.871, 29.050] | 3.373 |
| block 64 | mean-of-fractions | hierarchical | 64 | [10.283, 38.888] | 7.481 |
| block 64 | mean-of-fractions | fixed | 64 | [17.650, 26.975] | 2.385 |
| block 16 | mean-of-fractions | hierarchical | 16 | [10.193, 38.630] | 7.434 |
| block 256 | mean-of-fractions | hierarchical | 256 | [10.173, 38.654] | 7.349 |

The ratio-of-means row is a **baseline-scale-weighted** estimand and is
labelled as a sensitivity, never as the headline.

### All seven channel effects

| Channel | Run | n | R̄_PCA | R̄_EMPCA | fraction % | iid 95 % CI (%) | block-64 95 % CI (%) |
|---|---|---|---|---|---|---|---|
| 33_36.0 | 33 | 5323 | 41.0397 | 35.1811 | 14.276 | [10.52, 20.92] | [10.66, 20.46] |
| 33_37.0 | 33 | 5440 | 7.9398 | 7.0274 | 11.492 | [3.03, 19.92] | [2.78, 19.93] |
| 35_26.0 | 35 | 1369 | 12.6611 | 4.8066 | 62.036 | [37.66, 86.42] | [39.95, 85.99] |
| 35_27.0 | 35 | 1618 | 11.8491 | 10.3012 | 13.064 | [9.52, 17.31] | [9.60, 17.48] |
| 35_56.0 | 35 | 1522 | 2.3383 | 1.7472 | 25.278 | [16.89, 33.60] | [17.16, 33.79] |
| 35_57.0 | 35 | 1931 | 41.6219 | 31.9521 | 23.233 | [9.21, 43.29] | [10.03, 41.37] |
| 35_58.0 | 35 | 2132 | 2.4963 | 2.4563 | 1.605 | [1.22, 2.06] | [1.21, 2.08] |

All seven effects are positive and each individually excludes zero.

### Leave-one-out

| Dropped | θ (%) | shift (pp) | hierarchical 95 % CI (%) |
|---|---|---|---|
| channel 33_36.0 | 22.785 | +1.216 | [10.394, 40.402] |
| channel 33_37.0 | 23.248 | +1.680 | [11.119, 41.384] |
| channel 35_26.0 | 14.824 | **−6.745** | [8.726, 22.622] |
| channel 35_27.0 | 22.986 | +1.418 | [10.281, 41.015] |
| channel 35_56.0 | 20.951 | −0.618 | [8.865, 39.176] |
| channel 35_57.0 | 21.292 | −0.277 | [9.368, 39.406] |
| channel 35_58.0 | 24.896 | +3.327 | [14.229, 41.973] |
| **run 33** (keeps 5 ch) | 25.043 | +3.474 | [11.600, 43.869] |
| **run 35** (keeps 2 ch) | 12.884 | **−8.685** | [10.226, 16.104] |

### Findings

1. The effect direction is robust: every leave-one-out interval and all seven
   channel effects exclude zero.
2. Nearly all uncertainty is **channel-level**, not event-level: the
   hierarchical sd (7.48 pp) is 3× the channel-fixed sd (2.49 pp).
3. **Temporal dependence is negligible.** Block lengths 16/64/256 move the
   interval by less than 0.35 pp anywhere.
4. The hierarchical interval is wide and driven by 7 clusters; the archive's
   own `cluster_caution` about coarse cluster-level percentile intervals stands.
   The result supports "EMPCA reduces the weighted residual on this archive",
   not a precise effect size.

---

## 4. Stage 2 — T10/T11/T12, TIDMAD

Frozen at `38b0a33`/`c8dab7a`. Predeclaration: `PREDECLARED_T11.md`.

### 4.1 The five items resolved before the operating point was selected

1. **Window-length normalisation.** Unitary DFT (`rfft/√w`) throughout. For a
   real tone at an interior bin `|c| = A√w/2`, so raw squared improvements are
   not comparable across `w`; the endpoint is the **physical amplitude**
   `a = 2q/√w` in readout ADC units, and labels are transformed identically.
   Boxcar window, single interior bin, `k_inj` fixed once per stretch from the
   calibration windows. The off-bin Dirichlet factor is **recorded, not
   corrected**: at the selected `w = 65536` file 0010 sits δ = −0.434 bins
   off-grid, D(δ) = 0.718. Estimate and label share the bin and δ, so the
   contrast is unaffected; only the absolute amplitude scale carries D(δ).
2. **Joint bootstrap.** Each replicate block-resamples the **paired**
   reference/readout calibration windows, refits `ĝ` and with it *every*
   calibration-dependent quantity in the target — the labels `t_i = ĝρ_i`,
   the calibration-mean baseline and the shrinkage weight — and independently
   block-resamples the evaluation windows. **Held conditional (declared):** the
   350 fit windows and the PCA/EMPCA bases and Wiener signal PSD estimated from
   them, the science-file noise bin powers, `k_inj`, and the block lengths.
   The calibration-only component is reported beside every interval. Shared-gain
   coverage was validated on known-transfer simulations (200 trials, nominal 95 %).
3. **Line SNR in compatible units.** Everything is native **bin power**
   `N_k = E|c_k|²`, with the density conversion `N_k = (f_s/2)·S₁(f_k)`
   documented and tested; no implicit bandwidth factor anywhere. The
   reference-SNR ≥ 100 rule is recorded as a **declared necessary condition**
   with its three errors-in-variables assumptions stated explicitly (additive
   reference error, uncorrelated with the true reference value, uncorrelated
   with the readout error) — *not* as a demonstrated bias bound. Empirical
   calibration checks are reported with it: split-half gain agreement (2.5 % at
   the operating point) and the predicted-vs-observed residual bin power.
   A recorded caveat: a strong tone's rectangular-window leakage inflates the
   off-line floor and so **biases line SNR downward** — conservative for the
   gate, and its size is reported as `reference_line_snr_leakage_sensitivity`
   (3.3–6.5× on the development stretches).
4. **Plateau/dependence rule.** The single adjacent variance-ratio test is
   replaced by: the smallest candidate that **no** larger candidate exceeds by
   15 %, with at least two larger candidates required so the test is a plateau
   and not an adjacent pair. The largest candidate is never accepted vacuously;
   a locally flat top is flagged `stabilised-at-endpoint`; otherwise
   `inconclusive` uses `b = b_max`, flags the interval and **withholds the
   "useful" verdict**. Aggregation over development files/contrasts takes the
   maximum normalised curve. Calibration candidates stop at 8 (64 windows give
   ≥ 5 blocks only to there).
5. **No-denoising quadrature and analytical baselines.** `raw-bin`
   (`q̂ = c_i`, no denoising) is included, and its **exact equality** to
   bin-aligned quadrature OF under diagonal Fourier weighting is computed
   through the generic weighted-LS path as a live check. Beyond zero and
   calibration-mean, the **Wiener** per-bin gain and an **analytical
   calibration-shrinkage** baseline `λc_i + (1−λ)q̄_cal`, λ = S/(S+N), are
   reported. Eligibility and the fewer-than-three-eligible-files case are
   resolved deterministically, including the empty case, which forbids a
   "useful" verdict outright.

### 4.2 Characterisation (all 20 files × 3 window lengths, 39 s)

Run on calibration windows of both channels only, before any method was fitted.
Outcomes at `w = 65536`: **11 eligible-coupled**, 7 uncoupled, 1 multi-tone
(0007), 1 chirped (0005); none underpowered.

**A stated protocol expectation is refuted.** The programme anticipated that
high-frequency files would come out *uncoupled* and form the failure regime.
They do not: at `w = 65536`, `0018` (1.80 MHz) and `0019` (3.40 MHz) are
eligible-coupled with readout line SNR 30.2 and 32.8. The uncoupled/chirped
regime is instead the **low**-frequency files (`0003`–`0006`, 5.6–13 kHz) and
the ≈69.6 kHz two-tone group (`0000`, `0001`, `0002`). The amplitude coupling
ratio does fall with frequency, but the readout noise floor falls with it, so
the line stays well above the floor. This was recorded in `PREDECLARED_T11.md`
before the method outcomes on those files were computed.

Line SNR grows ∝ `w` as the theory requires (e.g. file 0000: 5.3 → 52 → 124),
which is why the coupling verdict is window-length dependent.

### 4.3 Frozen operating point

`w = 65536`, rank 2 (fixed), `b_eval = 16`, `b_cal = 2`, 2000 replicates,
seed 20260910. Selection on eligible development stretches of
Δ(zero, EMPCA): −0.0095 at 4096, +0.0043 at 16384, **+0.0973 at 65536**.

**Dependence verdict: `inconclusive`.** The aggregated variance curve
(1.000, 1.773, 2.431, 3.290, 1.895) is non-monotone, so the large-block
estimates are themselves unstable at 9 blocks. This is **not** an aggregation
artefact: 7 of the 10 individual contrast curves are independently
inconclusive and 7 of 10 select b = 16. By the declared rule **no file in this
run may receive a "useful" verdict**.

**Recorded limitation:** only `abra_validation_0010.h5` was eligible-coupled
among the development files at any window length, so the selection rests on a
single development stretch per `w`, and the dependence rule's "aggregation
over development files" reduces to aggregation over contrasts of one file.

### 4.4 Validation (0008, 0009, 0011) — 100 s

| File | Δ(zero, EMPCA) | Δ(cal-mean, EMPCA) | Δ(PCA, EMPCA) |
|---|---|---|---|
| 0008 | +0.00159 [−0.0027, +0.0063] | +0.00181 [−0.0013, +0.0159] | +0.00438 [−0.0059, +0.0142] |
| 0009 | +0.01054 [+0.0067, +0.0141]* | +0.01064 [+0.0081, +0.0301]* | +0.01241 [+0.0067, +0.0188]* |
| 0011 | +0.02020 [+0.0142, +0.0262]* | +0.02073 [+0.0181, +0.1178]* | +0.02233 [+0.0068, +0.0384]* |

Combined θ_IV(zero, EMPCA) = +0.00927 [+0.0067, +0.0119], unweighted +0.01077,
Q = 24.3/2, I² = 0.918.

### 4.5 Final test — executed once, 14 files, 6 m 47 s

8 eligible-coupled, 5 uncoupled, 1 multi-tone. All outcomes evaluated and
reported; only eligible-coupled files enter the combined estimate.

| File | f_inj (kHz) | outcome | Δ(zero, EMPCA) | Δ(cal-mean, EMPCA) |
|---|---|---|---|---|
| 0001 | 69.6 | uncoupled | −3.97e−05 [−2.5e−04, +1.8e−04] | +1.16e−04 |
| 0002 | 69.6 | uncoupled | −2.10e−05 | −1.12e−05 |
| 0003 | 5.6 | uncoupled | −4.47e−06 [−6.6e−06, −2.4e−06]* | +4.03e−06 |
| 0004 | 7.2 | uncoupled | −5.31e−04 [−7.5e−04, −2.6e−04]* | −5.15e−04 |
| 0006 | 13.0 | uncoupled | −1.06e−02 [−1.6e−02, −3.8e−03]* | +8.36e−03 |
| 0007 | 28.1 | multi-tone | −4.91e−05 [−6.4e−05, −3.5e−05]* | −1.95e−05 |
| 0012 | 160.1 | eligible | +4.08e−03 [−5.2e−04, +9.0e−03] | +9.66e−03 * |
| 0013 | 310.1 | eligible | −3.66e−03 [−1.9e−02, +8.9e−03] | −3.43e−03 |
| 0014 | 460.1 | eligible | −8.81e−03 [−1.0e−02, −7.5e−03]* | −7.43e−03 |
| 0015 | 620.0 | eligible | +3.37e−02 [+2.3e−02, +4.5e−02]* | +3.38e−02 * |
| 0016 | 770.0 | eligible | +2.24e−02 [+1.7e−02, +2.9e−02]* | +2.24e−02 * |
| 0017 | 930.0 | eligible | +1.35e−03 [−4.5e−05, +2.8e−03] | +1.71e−03 * |
| 0018 | 1799.9 | eligible | +1.30e−02 [+1.0e−02, +1.6e−02]* | +1.31e−02 * |
| 0019 | 3400.0 | eligible | +2.91e−05 [−7.8e−05, +1.3e−04] | +3.55e−05 |

**Combined over the 8 files entering the estimate:**

| Contrast | θ_IV (precision-weighted) | 95 % CI | unweighted | Q / dof | I² |
|---|---|---|---|---|---|
| zero − EMPCA | +1.58e−05 | **[−8.9e−05, +1.2e−04]** | +7.76e−03 | 328.3/7 | 0.979 |
| cal-mean − EMPCA | +4.63e−04 | **[−3.8e−05, +9.6e−04]** | +8.73e−03 | 87.7/7 | 0.920 |
| PCA − EMPCA | −1.86e−05 | [−2.2e−04, +1.8e−04] | +1.50e−02 | 232.6/7 | 0.970 |
| zero − raw-bin | +1.16e−02 | [+1.05e−02, +1.28e−02] | +4.05e−01 | 1469.1/7 | 0.995 |
| **raw-bin − EMPCA** | **−1.19e−02** | **[−1.31e−02, −1.08e−02]** | −3.97e−01 | 1253.2/7 | 0.994 |

θ_IV is a **weighted** estimand, not equal-file physical performance; the
unweighted mean and every per-file value are given beside it. Heterogeneity is
extreme (I² ≈ 0.98), so the two differ by more than two orders of magnitude.

**The weighted estimand is one file.** Recomputing the inverse-variance weights
independently from the per-file rows (which reproduces θ_IV and Q exactly):

| file | Δ(zero, EMPCA) | bootstrap sd | weight share |
|---|---|---|---|
| **0019** (3.40 MHz) | +2.91e−05 | 5.4e−05 | **98.65 %** |
| 0014 | −8.81e−03 | 7.2e−04 | 0.56 % |
| 0017 | +1.35e−03 | 7.2e−04 | 0.56 % |
| 0018 | +1.30e−02 | 1.5e−03 | 0.13 % |
| 0012, 0016, 0015, 0013 | — | 2.5e−03 … 7.0e−03 | 0.10 % combined |

File 0019 carries **98.65 %** of the precision weight because its injected
amplitudes are the smallest in the set, so its bootstrap sd is the smallest —
not because it is the most informative about the physics. θ_IV is therefore
essentially file 0019's value, which is ≈ 0. This is precisely the failure
mode the protocol anticipated when it required θ_IV to be named a weighted
estimand and shown with the unweighted mean and every per-file value; on this
data the precision weighting should not be read as a summary of performance.

### 4.6 RMSE by method (readout physical amplitude units, 11 eligible stretches)

| | zero | cal-mean | PCA | EMPCA | raw-bin | quad-OF | Wiener | shrink-cal |
|---|---|---|---|---|---|---|---|---|
| mean | 0.6544 | 0.6578 | 0.6614 | **0.6550** | **0.2455** | 0.2455 | 0.2427 | **0.2408** |

### 4.7 Findings

1. **Rank-2 whole-window PCA/EMPCA is indistinguishable from predicting zero**
   at this operating point (mean RMSE 0.655 and 0.661 vs 0.654 for zero), and
   is ~2.7× worse than the trivial no-denoising bin read (0.246). The combined
   contrasts against both usefulness references include zero.
2. **The mechanism is the high-dimension/low-sample regime, and it is
   quantitative.** At `w = 65536` the real-view dimension is `p = 65536`
   against `n = 350` fit windows, so `p/n = 187` and the Marchenko–Pastur
   top-eigenvalue inflation is `(1 + √(p/n))² ≈ 216×` the noise level. The
   injected tone's bin power is only **25×** the noise floor (readout line
   SNR 25.1). The leading sample eigendirections are therefore noise-fitting
   artefacts roughly 8.6× stronger than the tone, and a rank-2 subspace
   cannot be the tone direction. Measured on the evaluation windows of file
   0010: the injection bin carries 1.65e−3 of the observed window energy
   (54× the uniform per-bin share of 3.05e−5, so the tone is plainly present
   in the spectrum), yet PCA's rank-2 reconstruction places only 2.13e−4 of
   its energy there — under-representing the tone bin by ~8×. This is a
   property of the operating point, not an implementation defect: the fits
   behave exactly as a rank-2 projection should (PCA captures 6.44 % of the
   centred evaluation variance; EMPCA, whose subspace is chosen in the
   whitened metric, captures 0.14 % of *raw* variance by construction).
3. EMPCA does beat PCA consistently (0.6550 vs 0.6614) — the predicted
   direction of the metric effect — but the margin is small next to the gap
   to the trivial baseline. The same diagnostic shows why the direction is
   right: EMPCA places 2.78e−3 of its reconstruction energy at the injection
   bin against PCA's 2.13e−4, i.e. the noise-weighted metric does concentrate
   an order of magnitude more on the tone bin. It simply does not concentrate
   enough to compete with reading the bin directly.
4. **`quadrature-OF` reproduces `raw-bin` to machine precision on all 11
   eligible stretches** (max relative difference 2.6e−13), confirming the
   diagonal-Fourier-weighting identity on real data.
5. **No "useful" verdict is issued anywhere**, because the frozen dependence
   assessment was inconclusive and the declared rule withholds it. **That
   withholding is not the only thing standing in the way**, and the report
   should not be read as "it would have passed but for a technicality":

   * On the **combined** final test the criterion fails on its own terms —
     both Δ(zero, EMPCA) and Δ(cal-mean, EMPCA) have 95 % intervals that
     include zero, so the verdict would be *ordering* at best even with a
     stabilised dependence assessment.
   * **Per file**, the picture is genuinely heterogeneous rather than a
     uniform null: had the dependence verdict stabilised, 3 of the 8 eligible
     final-test files (`0015`, `0016`, `0018`) and 2 of the 3 validation files
     (`0009`, `0011`) would each have met the criterion individually, while
     `0013` and `0014` go the other way (`0014` significantly negative).
   * Even on the files that would qualify, EMPCA remains far worse than the
     no-denoising bin read, so "useful against zero and the calibration mean"
     would still not mean "useful as a denoiser" at this operating point.
6. Design D (fixed duration 2 097 152 samples) would give 512 / 128 / 32 total
   windows at w = 4096 / 16384 / 65536 against Design F's 564 at every w — a
   16-fold loss of evaluation power across the grid and a ≈4× wider interval at
   the selected window. Reported only; never used for selection.

---

## 5. Stage 3a — T15, covariance-error regime

Frozen at `64dfe1e`; executed once, 80 fits, 19 s.

### 5.1 Generator mismatch, resolved

The programme names "the S5/S7 synthetic generator at its frozen
configuration". No such generator exists: **S5** uses PSD exponent −2 (1/f²)
with a *sinusoid pair*, **S7** uses AR(1) noise with a *single pulse*, and
neither is the combined exponent −1, rank-2 model the theorem section refers
to. Rather than silently adopting one, one new model **G1** was defined and
frozen:

| Element | Frozen value |
|---|---|
| Dimension | d = 256, unit sampling frequency |
| Noise PSD | one-sided `(√(f² + f_knee²))^(−1)` — **exponent −1** — `f_knee = 4/d` |
| True covariance | exact circulant Σ from that PSD (spectrum known in closed form) |
| **Scale** | `trace(Σ)/d = 1` **exactly** (unit average variance per sample) |
| Signal | rank-2 *generic* orthonormal basis, seed 20260915 (deliberately not sinusoids, so the result is not an artefact of the signal being a circulant eigenvector family) |
| Amplitudes | `a ~ N(0, diag(SNR², (SNR/2)²))`, SNR = 3 — unequal, giving a well-defined eigengap |
| Sizes | 512 training (signal+noise), m calibration (**noise only**), 2000 held-out test |
| Misspecification: slope | assumed exponent −1.5 / −0.5 (truth stays −1) |
| Misspecification: corr. length | assumed knee ×½ / ×2, i.e. correlation length ×2 / ×½ — the explicit translation of "AR(1) length doubled and halved" onto a power-law generator |
| Renormalisation | every misspecified Σ is renormalised to the same `trace/d = 1`, so misspecification changes **shape**, never scale |

**80-fit limit reconciled exactly**: 8 arms × 10 seeds = 80, with the two
misspecification arms splitting seeds 0–4 / 5–9 between their two sub-arms.
Each (arm, seed) is one fit. All eight arms at a given seed share the same
training/calibration/test draw, so **every comparison is paired**.

Resulting regime: cond(Σ) = 32.0, λ₁ = 19.44, Δλ = 5.20, λ₁/Δλ = 3.74.

### 5.2 A known-truth check corrected a real error

The theorem bounds the subspace shift *caused by handing the estimator a
perturbed covariance*. Comparing `6ελ₁/Δλ` against the distance to the
**planted** subspace is wrong, because that distance also contains finite
training-sample error which is present even at ε = 0 — the oracle arm has
ε exactly 0 yet sinΘ = 0.319 to the planted subspace at N = 512. The correct
target is the distance to the **paired oracle fit on the same training draw**.
This was found by the sanity checks, corrected before the run, and both
quantities are now reported. Oracle-arm subspace error follows the n^(−1/2)
rate to within 30 % across a 16× range in N.

### 5.3 Results (80 fits)

| arm | ε_operator | ε_Frobenius | sinΘ vs planted | sinΘ vs paired oracle | bound 6ελ₁/Δλ | amp RMSE | probe R² |
|---|---|---|---|---|---|---|---|
| oracle | 0.0000 | 0.0000 | 0.3194 | 0.0000 | 0.000 | 0.0879 | 0.9217 |
| estimated-64 | 1.3121 | 1.1926 | 0.9993 | 0.9990 | 29.42 | 9.2632 | 0.0633 |
| estimated-128 | 0.8976 | 0.8401 | 0.9991 | 0.9993 | 20.13 | 12.4429 | 0.0761 |
| estimated-256 | 0.5753 | 0.5951 | 0.9472 | 0.9419 | 12.90 | 0.2245 | 0.1123 |
| estimated-512 | 0.3842 | 0.4178 | 0.4151 | 0.2942 | 8.62 | 0.1144 | 0.8293 |
| estimated-2048 | 0.1905 | 0.2118 | 0.3339 | 0.0958 | 4.27 | 0.0920 | 0.9095 |
| misspecified-slope | 0.7111 | 0.5379 | 0.3511 | 0.1081 | 15.95 | 0.0982 | 0.9066 |
| misspecified-corrlength | 0.5572 | 0.3070 | 0.3213 | 0.0118 | 12.50 | 0.0886 | 0.9214 |

### 5.4 Findings

1. **The bound holds on all 80 fits**, as it must.
2. **The bound is never informative.** For every non-oracle arm
   `6ελ₁/Δλ ≥ 4.27`, i.e. above 1 and therefore vacuous for a quantity that
   cannot exceed 1. Only the oracle arm (ε = 0) has an informative bound.
   From the observed scaling ε_op ≈ 0.590·√(d/m), an informative bound would
   need ε < Δλ/(6λ₁) = 0.0446, i.e. **≈ 45 000 calibration traces at d = 256**.
   That figure is derived from the fitted scaling, not run — it is far outside
   the 80-fit budget and no extra fits were spent on it.
3. **ε does not determine the task outcome.** Structured misspecification with
   ε_op = 0.557 (wrong correlation length) is nearly harmless — R² 0.9214
   against the oracle's 0.9217, sinΘ to the paired oracle 0.0118 — while
   *unstructured* sampling error at a **smaller** ε_op = 0.575
   (`estimated-256`) is catastrophic: R² 0.112, sinΘ 0.942. A bound that is a
   function of ε alone cannot distinguish these two, and is loose by one to
   three orders of magnitude exactly in the benign cases. This is the main
   scientific content of T15 and it is a limitation of the bound, not a
   refutation of it.
4. The failure of `estimated-64/128` is the expected singular regime (m < d);
   `estimated-256` sits at m = d, and recovery is well underway by m = 2d
   (R² 0.83) and near-complete at m = 8d (R² 0.91).
5. **The operator and Frobenius errors differ materially** — most for
   `misspecified-corrlength` (0.557 vs 0.307) — so reporting the theorem's
   operator-norm quantity rather than the old normalised Frobenius error
   changes the numbers, as the audit required.

---

## 6. Stage 3b — T14, structural-prior tradeoff

Frozen at `f9714e9`; runner source SHA-256
`feb8b2f79b5d4c8a922cb96a9bc213a119dd20656df4a31610194421d0bfc106`.

### 6.1 Budget derived, not assumed

The protocol JSON carries `fits_per_cell: 40`. That number was **not** trusted:
the runner derives it from the seeds, methods and restarts it will actually
execute, and refuses to start if the total exceeds the cap.

| Quantity | Value |
|---|---|
| Repository `PRIMARY.n_restarts` | **10** |
| Protocol `nfpa_restarts` | **3** |
| Derived fits per cell | 10 seeds × (3 NFPA restarts + 1 CW-PCA) = **40** |
| Stage 1 | 15 cells → **600** fits |
| Stage 2 | 10 cells → **400** fits |
| Stage 3 (confirmation, max) | 6 cells → 240 fits |
| **Total planned** | **1240** ≤ 1240 planned cap < 1500 hard cap |

The config's 40 is reproduced **only because** the protocol's 3 restarts
override the repository's 10; at the repository value a cell would be 110 fits
and the full plan 3410, over the hard cap. Both numbers are recorded in the
output. The runner connects to the common configuration through
`dataclasses.replace(PRIMARY, …)`, so `C`, `T`, `k_c`, `k_t`, `N_test`,
`n_als_iter` and the convergence tolerance all come from the frozen config.

Comparator: `methods.fit_cw_pca`, the **thin-SVD** unrestricted fit at matched
rank `k_c·k_t = 6`.

### 6.2 Benchmark before scaling (required)

One small and one large representative cell, single BLAS thread:

| Cell | s / seed | projected s / cell |
|---|---|---|
| `N=200, α=0.4, SNR 10, oracle` | 26.0 | 260 |
| `N=3200, α=0.4, SNR 10, oracle` | 203.2 | 2032 |

Projected ≈6 h serial for stages 1–2; run at 16 worker processes (of 40
threads) with one BLAS thread each. Stage 1 took **1128 s** for 600 fits.

### 6.3 Stage 1 — exploration, SNR 10, oracle covariance (600 fits)

`d = χ²_test(NFPA) − χ²_test(CW-PCA)`, paired per seed; `d < 0` means the
structural prior helps. `*` marks a paired 95 % interval excluding zero.

| N | α = 0.0 | α = 0.4 | α = 0.8 |
|---|---|---|---|
| 200 | −0.006929 * | −0.001272 * | −0.000035 |
| 400 | −0.004488 * | −0.001485 * | −0.000077 * |
| 800 | −0.002751 * | −0.001387 * | −0.000290 * |
| 1600 | −0.001630 * | −0.000988 * | −0.000312 * |
| 3200 | −0.001041 * | −0.000693 * | −0.000345 * |

**Every cell is negative.** NFPA beats CW-PCA at every `N` and every `α`
tested at SNR 10. Resolved crossings in the predicted direction: **0**.
Ambiguous crossings: 0. Reverse crossings: 0.

### 6.4 Why there is no crossover

The crossover requires the unrestricted class's variance excess to fall below
the restricted class's approximation cost. On this configuration it cannot,
within the declared grid:

| Class | Free parameters |
|---|---|
| NFPA (separable, `k_c×k_t = 2×3`) | (8−2)·2 + (256−3)·3 = **771** |
| CW-PCA (unrestricted rank 6 in `CT = 2048`) | 6·(2048−6) = **12 252** |

CW-PCA carries **15.9×** more free parameters, and at the grid maximum
`N = 3200` it is still heavily over-parameterised (`N/p = 0.26`). Fitting
`d(N) = A − B/N`, with `A` the constant approximation cost and `B/N` the
variance excess, gives `A ≤ 0` on all three lines — i.e. **no crossover at any
N** under that model, not merely one beyond the grid:

| line | A | B | verdict |
|---|---|---|---|
| α = 0.0, SNR 10 | −9.65e−04 | +1.24e+00 | A ≤ 0 → no crossover |
| α = 0.4, SNR 10 | −9.85e−04 | +9.31e−02 | A ≤ 0 → no crossover |
| α = 0.8, SNR 10 | −3.48e−04 | −7.01e−02 | A ≤ 0, and B < 0 (model misfit) |

At α = 0 this is exactly right and needs no explanation: the truth *is*
separable, so the approximation cost is identically zero and `d → 0⁻` without
ever crossing. At α = 0.4 and 0.8 an approximation cost does exist, but it is
smaller than the variance the unrestricted class pays even at `N = 3200`.

### 6.5 Optimisation diagnostics

The protocol asks for the restart spread and the excess objective over the
best observed restart. The latter is **identically zero by construction** —
`als_fit` selects the minimum-training-loss restart, so its excess over the
best training loss is 0 in all 250 seed-runs. That is a tautology, not a
finding, so the informative version is reported instead: the **held-out**
excess of that selection.

| Diagnostic | Value |
|---|---|
| Held-out excess χ² of the selected restart | mean 1.65e−05, median 0, max 2.11e−04 |
| Seeds where training-loss selection did **not** pick the held-out-best restart | **34.7 %** |
| ALS runs converged within 150 iterations at tol 1e−10 | **89.3 %** |
| Selected-restart distribution (restarts 0/1/2) | 104 / 25 / 21 |

Restart spread against the effect being measured:

| α | restart held-out χ² sd (mean) | mean \|d\| | sd / \|d\| |
|---|---|---|---|
| 0.0 | 4.1e−17 | 3.37e−03 | 0.00 |
| 0.4 | 6.1e−05 | 1.17e−03 | 0.05 |
| 0.8 | 7.7e−05 | 2.23e−04 | **0.35** |

At α = 0 the ALS problem is effectively unimodal — all restarts reach the same
solution to 1e−17. At α = 0.8 the restart-to-restart spread is **35 % of the
effect size**, so the α = 0.8 line is optimisation-limited and its small
negative values should not be over-read.

### 6.6 Stage 2 — exploration, SNR 3, oracle covariance (400 fits, 797 s)

| N | α = 0.0 | α = 0.8 |
|---|---|---|
| 200 | −0.000825 * | +0.000016 |
| 400 | −0.001001 * | +0.000000 |
| 800 | −0.001006 * | +0.000025 |
| 1600 | −0.000851 * | +0.000003 |
| 3200 | −0.000618 * | +0.000000 |

At α = 0 the structural prior still wins at every `N`. At α = 0.8 the two
methods are **statistically indistinguishable**: all five point estimates are
within 2.5e−05 of zero and no interval excludes zero. Lowering the SNR from 10
to 3 does not produce a crossover either.

### 6.7 Stage 3 — confirmation correctly **not** run

Applying the declared rule to the merged 25 exploration cells:

| | count |
|---|---|
| Resolved crossings in the predicted direction | **0** |
| Ambiguous crossings (sign change, endpoint unresolved) | 0 |
| Reverse crossings (positive → negative) | 0 |
| Triggers confirmation | **False** |

Across all 25 cells, 20 have `d < 0` and 5 have `d > 0` (all five are the
α = 0.8, SNR 3 line, none resolved); **every cell with an interval excluding
zero is negative**. The declared no-crossing rule therefore applies: stage 3
does not run, the confirmation seeds 10–19 are never spent, and the claim
stays conditional.

| Budget | fits |
|---|---|
| Stage 1 | 600 |
| Stage 2 | 400 |
| Stage 3 | **0** (not triggered) |
| **Total used** | **1000** ≤ 1240 planned ≤ 1500 hard cap |

### 6.8 Findings

1. **Null crossover, reported as such.** No crossover between the
   finite-sample class contrast and the population approximation cost exists
   anywhere in the declared grid, at either SNR. The manuscript's
   structural-prior tradeoff claim therefore **stays conditional**, exactly as
   the protocol requires when no crossover is found.
2. The null is **explained, not merely observed**: the unrestricted class
   carries 15.9× more free parameters and is still over-parameterised at the
   largest `N` on the grid (`N/p = 0.26`), so its variance excess never falls
   below the restricted class's approximation cost. The fitted `A ≤ 0` on
   every line says this is not a crossover just beyond the grid.
3. At α = 0 no crossover should exist at any `N` — the truth is separable, the
   approximation cost is identically zero, and `d → 0⁻`. That the run
   reproduces this is a correctness check on the whole pipeline.
4. **Where the effect is smallest, the optimiser is the limiting factor**:
   at α = 0.8 the ALS restart spread is 35 % of |d|, and training-loss restart
   selection misses the held-out-best restart in 34.7 % of seeds. Any future
   attempt to resolve the α = 0.8 line needs more restarts or a better
   selection rule before more seeds.
5. Reaching a crossover would need `N` well beyond 3200 *and* a configuration
   where the unrestricted class is not over-parameterised — e.g. a smaller
   `C·T` or a larger rank gap. That is a design change, not more compute, and
   it was **not** attempted here: the grid, caps and stage structure were
   declared in advance and were not widened after seeing the null.

---

## 7. Protocol changes made *before* evaluation

Every item below was decided and frozen before the data it affects were
scored. None was made in response to a validation or final-test outcome.

| # | Where | Change | Why |
|---|---|---|---|
| 1 | T10 endpoint | Endpoint expressed in physical amplitude `a = 2q/√w` | Raw unitary-DFT coefficients scale as √w, so squared improvements are not comparable across window lengths |
| 2 | T10 dependence | Plateau rule with ≥2 larger candidates required; largest candidate never accepted vacuously; endpoint case flagged; otherwise inconclusive | The declared single adjacent variance-ratio rule left the grid endpoint undefined and could certify a plateau on one noisy ratio |
| 3 | T10 dependence | The `stabilised-at-endpoint` branch as first written was **unreachable** (if `V(16)/V(8) ≤ 1.15` then `b = 8` already qualifies). Rewritten so all three verdicts are reachable | Found by a unit test; a dead branch would have silently become "inconclusive" in a case the protocol meant to treat separately |
| 4 | T10 characterisation | Multi-tone neighbourhood made well-defined at coarse `w` (guard 1 bin, band `max(guard+1, ⌊500/Δf⌋)`), with `multi_tone_band_resolvable_at_declared_500hz` recorded | At `w = 4096` one bin spans 2441 Hz, so the declared ±500 Hz band contained **no** testable bin and the rule silently returned "no second line" |
| 5 | T10 characterisation | Sub-bin beat diagnostics (line-power CV, sub-bin peak scatter) added as **reported, non-gating** | The known ≈120 Hz two-tone case in file 0000 is sub-bin at every `w` used, so the declared ±500 Hz rule cannot see it; the gate is unchanged |
| 6 | T10 characterisation | `reference_line_snr_wide_guard` reported beside the gate value | A strong tone's rectangular-window leakage inflates the off-line floor, biasing line SNR **down**; conservative for the gate, but its size should be visible |
| 7 | T10 freeze | Inconclusive-branch message corrected (it described a non-monotone curve as "still rising") and per-curve selections added | Message-only + diagnostic; **no decision changed**. Done before validation ran |
| 8 | T10 seeds | `zlib.crc32` replaces the builtin `hash()` for per-stretch seed offsets | `hash()` on `str` is salted per process, so seeds would not reproduce across runs |
| 9 | T15 generator | One new frozen generator **G1** replacing the non-existent "S5/S7 generator" | S5 is exponent −2 + sinusoids, S7 is AR(1) + one pulse; neither is the named combined exponent −1 rank-2 model |
| 10 | T15 metric | Bound compared against the distance to the **paired oracle fit**, not to the planted subspace | The theorem bounds the shift due to covariance perturbation; the planted-subspace distance also contains training-sample error present at ε = 0 |
| 11 | T15 numerics | `SIN_THETA_TOL = 1e-6` on the bound comparison | `sinΘ = √(1−σ²)` has ≈√ε ≈ 1.5e−8 resolution near zero, so an exact zero reads as ~2e−8 |
| 12 | T14 budget | Fit counts **derived** from seeds × (restarts + comparator), not read from `fits_per_cell` | The config's 40 holds only because the protocol's `nfpa_restarts = 3` overrides the repository's `PRIMARY.n_restarts = 10`, which would give 110 |
| 13 | T14 evaluation | χ² computed directly in whitened coordinates; local BLAS whitener replacing a 3-operand `np.einsum` | `covariance.whiten_separable` has no `optimize=`, giving the naive O(n·C²·T²) path (~24 s per batch, 4 calls per evaluation). The archival function is **left untouched**; a test pins the two to agree to 5e−15. Evaluating in whitened coordinates also avoids the archival round trip's spurious `W·μ` term |

## 8. Resources and artefacts

| Stage | Wall time | Peak RSS | Threads |
|---|---|---|---|
| T13-a | 52 s | 131 MiB | BLAS default |
| TIDMAD characterise (20 files × 3 w) | 39 s | ~1 GB | BLAS default |
| TIDMAD development (9 stretches) | 2 m 28 s | ~2 GB | BLAS default |
| TIDMAD validation (3 files) | 1 m 40 s | ~2 GB | BLAS default |
| TIDMAD final test (14 files) | 6 m 47 s | ~2 GB | BLAS default |
| T15 (80 fits) | 19 s | ~0.5 GB | BLAS default |
| T14 benchmark | 3 m 51 s | ~0.5 GB | 1 per process |
| T14 stages 1–2 | see §6 | ~0.5 GB/worker | 16 workers × 1 thread |

Never exclusive: at most 16 of 40 hardware threads, peak load average ≈17.
No GPU used. Disk: 294 TB free on `/ceph`; total new output **≈2 MB**.

All outputs live under `results/mlst_2026-09-10/`, one versioned directory per
stage, each with `RESULTS_SHA256` and a `provenance` block recording the EXP
commit, tree-dirty flag, interpreter and library versions, input paths with
SHA-256 digests, seeds, argv, and wall/CPU/RSS. No raw data, caches,
credentials or large arrays were committed; the only binary artefact is a
99 KB bootstrap-draw archive.

**Provenance caveat, stated plainly.** The T13-a, TIDMAD and T15 runs executed
from a clean tree at the commit named in their own provenance block. The T14
exploration was launched from a working tree that was committed immediately
afterwards as `f9714e9`; its provenance block therefore records
`tree_dirty: true`, and the runner source is pinned instead by its SHA-256
(`feb8b2f79b5d4c8a922cb96a9bc213a119dd20656df4a31610194421d0bfc106`), recorded
in the same block.

---

## 9. T16 — left explicitly pending (not run)

The instruction makes T16 conditional: implement and run it only if retaining
the empirical optimizer/tying claim requires it. It does not, so T16 was **not
implemented and not run**, and it stays a design outline.

* The empirical optimizer claim in the manuscript is claim-table **row 2**
  ("independently trained tied linear AE reaches the EMPCA optimum", mean
  relative objective gap 1.95×10⁻¹⁶ on S3, 10 seeds). That claim already rests
  on an existing independently-trained-AE gate; nothing in this programme
  weakens it, and no result produced here bears on it.
* Claim-table **row 3** (CRESST tied-AE/EMPCA principal angle 1.91×10⁻⁶°) is
  already labelled in the table as a **closed-form parameterization consistency
  check**, not an optimizer result, so it does not need T16 either.
* T16's own blocker is unchanged: the two new untied arms
  (untied-in-whitened-coordinates, scale-balanced untied) and the diagnostic
  recording are runner changes that do not exist, and the tied runner's
  hard-coded `ok=True` has not been removed.

No T16 numbers are reported, and none should be inferred from this programme.

## 10. T13-b and T13-c — not run, but their recorded data blockers are resolved

Neither was run — both are outside the stage list this instruction authorises
(Stage 1 = T13-a, Stage 2 = TIDMAD, Stage 3 = T15/T14, T16 conditional).
Neither was reconstructed from summaries, and no substitute analysis is
labelled as either of them. But the **data inventory changes both dependency
statements**, and that is reported here because the programme records them as
blocked on data that is in fact present.

### T13-b — the recorded blocker is resolved

The programme states: "the archive carries pulse `chronological_time` but no
noise-stream timestamps. Dependency: the release's noise-trace timestamps, or
the leakage-audit listing of the 6 + 3 affected pulses."

The mounted release **does** carry them. `features_test.csv` /
`features_train.csv` have an `hours` column and a boolean `noise` column
(5 205 noise rows in the test file alone), so the last calibration-noise
timestamp per channel — the exact missing input — is derivable:

| channel | n noise | first noise (h) | **last noise (h)** | n pulse | last pulse (h) |
|---|---|---|---|---|---|
| 33_36.0 | 767 | 0.0129 | **51.8619** | 19 304 | 51.9106 |
| 33_37.0 | 767 | 0.0129 | **51.8619** | 19 304 | 51.9106 |
| 35_26.0 | 739 | 0.0337 | **43.4223** | 5 599 | 43.4182 |
| 35_27.0 | 739 | 0.0337 | **43.4223** | 5 599 | 43.4182 |
| 35_56.0 | 731 | 0.0338 | **43.4224** | 7 691 | 43.4154 |
| 35_57.0 | 731 | 0.0338 | **43.4224** | 7 691 | 43.4154 |
| 35_58.0 | 731 | 0.0338 | **43.4224** | 7 691 | 43.4154 |

In every run-35 channel the last noise trace comes *after* the last pulse, so a
common time cut removes nothing there; in run 33 it precedes the last pulse by
0.049 h, which is consistent with the programme's "6 + 3 affected pulses".
**T13-b is unblocked and ready to run**; it was not run because it is not in
this instruction's stage list.

### T13-c — waveforms are present; the blocker is the refit pipeline

The programme records T13-c as needing "the released traces". They are mounted:
`paper1_dataset/cresst/raw/X_train.npy` (979 446 × 512 float32) and
`X_test.npy` (78 084 × 512), with `y_train`/`y_test` labels. The `.npy`
extension is misleading — the manifest declares `"format": "raw_headerless"`,
so these are raw float32 buffers and `np.load` fails on them with an
unrelated pickle error; they read correctly with `np.memmap` at the declared
shape and dtype, and all file sizes match the manifest.

The remaining T13-c dependency is therefore **not data** but the refit itself:
rank {4,6,8} × shrinkage {0.05,0.1,0.2} at `9c9f696…` or a frozen successor,
with selection on calibration data and a single evaluation. That is a new
implementation outside the authorised stages and was not attempted.

---

## 11. Claim-to-evidence mapping

How each result bears on the manuscript's existing claim rows
(`MS:CLAIM_EVIDENCE_TABLE.md`). **Supports** = the row's stated limitation is
narrowed; **qualifies** = a new limitation must be recorded; **neutral** = no
bearing.

| Claim row | Result | Effect | What must change in the table |
|---|---|---|---|
| 8 — CRESST same-rank contrast, equal-channel mean 21.6 % | T13-a | **supports** | The row's limitation "**no interval for the fractional estimand itself** (T13-a)" is now satisfied: hierarchical 95 % CI **[10.31, 38.83] %**, channel-fixed [17.42, 27.13] %, ratio-of-means sensitivity [12.14, 39.83] %. "7 clusters from 2 runs" and "one operating point" stand unchanged, and the interval must be labelled a sensitivity, not a replication |
| 13 — TIDMAD metric reversal on file 0000 | T10/T11 | **qualifies** | The archived row is a MAE contrast on one burnt file. The new protocol's estimand, run once on 14 held-out files, gives a combined EMPCA-vs-zero contrast whose interval **includes zero**, and shows EMPCA is decisively worse than a no-denoising bin read. The "ordering only" wording stays and must be strengthened: on the new endpoint there is no recovery result at this operating point |
| 13b — archived clean-waveform contrast | T10/T11 | **qualifies** | The row already notes the absence of zero/mean baselines and a transfer model. Both now exist, and with them the contrast does not survive: the row should point at the new evidence rather than stand alone |
| 14 — zero-predictor level 0.0747 | T10/T11 | **supports** | The new run reproduces the phenomenon at a different operating point and generalises it: across 11 eligible stretches mean RMSE is 0.654 for zero against 0.655 for EMPCA |
| 15 — two secondary TIDMAD files, intervals include zero | T10/T11 | **supports** | Extends from 2 files to 14 evaluated once each, with every outcome class reported |
| 17 — rank-2 prediction not confirmed on real TIDMAD data | T10/T11 | **supports** | The mechanism is now quantified: p/n = 187 gives Marchenko–Pastur top-eigenvalue inflation ≈216× the noise level against a tone at 25×, so a rank-2 whole-window subspace cannot isolate the tone. This converts a "diagnoses are hypotheses" note into a measured statement |
| 20 — relative covariance-error bound sinΘ ≤ 6ελ₁/Δλ | T15 | **supports + qualifies** | The row's "no data experiment yet (T15)" is discharged: the bound **holds on all 80 fits**. But a new limitation is required — on this generator the bound is **never informative** (≥ 4.27, i.e. above 1) for any non-oracle arm, and ε alone does not predict task outcome (structured misspecification at ε = 0.557 is harmless; sampling error at ε = 0.575 is catastrophic) |
| T14 target — structural-prior tradeoff (NFPA vs unrestricted) | T14 | **qualifies** | **Null crossover.** No crossover exists anywhere in the declared grid at either SNR; 20 of 25 cells favour the restricted class and every resolved interval is negative. Any claim of a finite-sample/approximation crossover must stay **conditional**, and the reason is structural: the unrestricted class carries 15.9x more parameters and is over-parameterised at the largest N tested |
| 2, 3 — tied-AE claims | — | **neutral** | T16 not run; nothing here bears on them (§9) |
| 1, 4–7, 9–12, 16, 18, 19 | — | **neutral** | Not addressed by this programme |

### Claims this programme does **not** support

* No "useful recovery" claim on TIDMAD. The declared usefulness verdict is
  withheld everywhere, and the combined intervals include zero regardless (§4.7).
* No independent-replication claim from T13-a. It is a sensitivity analysis of
  one archive.
* No claim that the covariance-perturbation bound is practically tight. It is
  valid and vacuous in the regime tested.
* No crossover claim for the structural-prior tradeoff. The confirmation stage
  was not triggered and its seeds were never spent.
