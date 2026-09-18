# Scientific audit — MLST 13 September successor programme

**Scope:** each authorized selection is re-scored against its own frozen protocol hypotheses, using only its
existing `REDUCED.json` and stored cells. **No protocol, threshold, tolerance or outcome was changed and no
production experiment was launched.** Supplementary computations below are post hoc, read-only, and labelled
as such. Computation logs: `audit/*_computation.txt`, `diagnostics/*`. Verbatim hypotheses:
`audit/PROTOCOL_HYPOTHESES_VERBATIM.md`.

**Verdict vocabulary:** *supported* / *not supported* are the protocol's own outcomes. *Descriptive* hypotheses
are reported, not scored. *Not reachable* and *inconclusive* follow the protocols' failure-interpretation rules.
A validated reduction is not a positive result.

## 0. Corrections to earlier reports (including this agent's consolidated report)

| # | Earlier statement | Audited position |
|---|---|---|
| 1 | T22: coverage fails in "15 of 69" scenarios; "the worst are the drift scenarios" | H1 applies to **non-drift** scenarios only: **13 of 65 outside the band** (10 under, 3 over). The two drift failures lie outside H1's scope. The reducer's 15/69 uses the broader `tolerances.applies_to` scope, and both counts are reported. |
| 2 | T19: `S_full/beyond` at 0.917 is "a partial failure… a negative finding" | H3 is scored on **in-domain fits only**. Beyond-domain agreement is descriptive and outside H3's scope. H3 passes: every in-domain class is 1.000. |
| 3 | T21: "H1 holds in 296/298, so it isn't universal" | Correct but incomplete. H1 requires **both** arms, so it **fails overall**. Under the predeclared rule `predictor_fails_on_orthogonal_only`, the isolated-spike step fails here, so **the loadings step is untested**. The reducer's `verdict_by_geometry` label "correlated: supported…" must **not** be read as support for the loadings explanation. |
| 4 | T20: H3 not scored in earlier reports | Scored here (see §2). |
| 5 | T22: H5 treated as reported | The predeclared **random-effects interval is absent from the reduction**. It was computed per record but never summarised. Supplied here, read-only (see §1). |
| 6 | Freeze-binding index, first draft (this close-out) | It applied T19/T22's per-cell digest method to T20/T21 and reported them "not bound". That was a method error, now superseded (retained and labelled). By each package's own mechanism, **all four freezes bind and precede every completed confirmation cell**. |

## 1. T22-S — calibration of the archived T22 inference procedure (simulation)

| Hypothesis | Criterion (frozen) | Result | Verdict |
|---|---|---|---|
| H1 coverage | two-stage 95% interval within [0.925, 0.975] (Wilson not entirely outside) on **every non-drift** scenario | 52 / 65 within; **13 outside**: 10 under-cover (0.837–0.902), 3 over-cover (0.987–0.999) | **Not supported** |
| H2 type-I | P(any null claim supported) and per claim ≤ 0.07 on every null scenario | 0 / 30 scenarios and 0 per-claim violations; max any-claim 0.024; per-claim max 0.021 / 0.016 / 0.004 | **Supported** |
| H3 withholding | descriptive: rises with φ, approaches 1 under drift | non-drift medians by φ 0 / 0.3 / 0.6 / 0.85: 0.430 / 0.461 / 0.595 / 0.943; drift 0.935–0.943 | As expected (descriptive) |
| H4 power | descriptive, by effect × files | given not withheld: large 0.95–1.00, medium 0.30 / 0.74 / 0.76 at 4/8/16 files, small 0.03–0.06; unconditional power is capped by withholding | Descriptive |
| H5 secondary intervals | t-interval and random-effects interval reported beside primary | t-interval: median 0.932, ~49/65 in band. **Random-effects interval absent from the reduction.** Supplementary read-only from stored records: **24/65 in band**, median 0.876 (0.427–0.998), 36% shorter than two-stage | Partially reported (omission) |
| H6 bias sign | sign with MC SE | resolved in 2/69 (both `t3`, n16, negative) | Descriptive |
| Feasibility (part B) | frozen rule | **infeasible at 4, 8, 16 files**: coverage out of tolerance at each; medium-effect power 0.019 / 0.039 / 0.035 < 0.5; max withholding 0.94–0.96 > 0.5 | Feasibility failure (the correct part-B output) |
| Controls | 5 declared | 5 / 5 passed | — |

**Verification:** the reducer's Wilson bounds and tolerance flags reproduce exactly (0 disagreements), and
two-stage coverage recomputed from all stored Monte-Carlo records matches `REDUCED.json` in 65/65 scenarios.

**Diagnosis of the H1 exceptions** (`diagnostics/T22_H1_coverage_diagnosis.txt`)
* **Standard-error miscalibration explains the failures.** Mean bootstrap SD ÷ empirical SD of the point estimate
  correlates **0.79** with coverage. **All 10 under-covering scenarios have ratio < 1 (0.79–0.98). All 3 over-covering
  have ratio > 1.3 (1.35–1.67).**
* The two-stage interval does not scale with file count like a t-interval: length ÷ t-length is **0.73 at 4 files,
  1.03 at 8, 1.24 at 16**. Under-coverage concentrates at 4–8 files in alternatives: **9 of the 10 under-covering scenarios are
  alternatives, all at 4 or 8 files**; the tenth is `null-partial-n4-phi0.6-tau0.03`. Over-coverage concentrates at 16 files and in `dominant` null-partial scenarios.
* **Not caused by withholding:** coverage given not withheld ≈ unconditional (e.g. 0.838 vs 0.837).
  **Not caused by bias:** the bias sign is unresolved in 67/69.
* Per protocol: "the archived interval is not calibrated for that dependence/heterogeneity/file-count regime;
  reported as such… no re-tuning on confirmation seeds."

## 2. T20-S — class contrast on an analytic Gaussian family

| Hypothesis | Result | Verdict |
|---|---|---|
| H1 control θ = 0 | lower bound 0, best-found a −5.7e−14; profile all "−" | **Supported** |
| H2 invariants | every θ: eigenvalue deviation ≤ 1.1e−14, rank 256, trace/SNR/Δk fixed; k − cap and a-interval move with θ | **Supported** |
| H3 sign profile vs design bracket | θ=0.12: design **above** core grid (800–1600) → no sign change through N=3200 (**consistent; not confirmable within grid**). θ=0.25: design **inside** → bracket (1000, 1250) (**as designed**). θ=0.40: design **below** → bracket (400, 800) (**as designed**). No unresolved cells. | Descriptive: 2 of 3 as designed, 1 consistent-not-confirmable |
| H4 x_U scaling (frozen c/N + d/N²) | inside band 6/6 at all four θ (closest: θ=0.12, 2.82 vs 3.02) | **Supported at all θ** |
| H4 d − a_ref scaling (frozen c/N + d/N²) | θ=0, 0.25, 0.40: 6/6. **θ=0.12: 5/6, outside at N=1000 (3.370 vs critical 2.984)** | **Not supported at θ=0.12**; supported at the other three. Per protocol, a partial pattern is not partial support. |

**Diagnosis of the θ=0.12 exception** (`diagnostics/T20_theta012_diagnosis.txt`)
* The observed contrast lies **above the frozen prediction at every N** (residual ÷ half-width +0.19 … +1.17).
  The other three θ show mixed signs.
* The band narrows with N (0.22 → 0.02), so a small systematic offset first exceeds it mid-grid.
* It is not optimisation error: restart spread ~1e−14 and no unresolved cells.
* The pilot selected c/N + d/N² by a narrow CV margin: 0.2895 against 0.2907 for c/N^α and 0.2975 for none.
* Reading: consistent with mild misspecification of the frozen scaling model at θ=0.12, not with noise.

## 3. T19-S — covariance-error model on generator G1

| Hypothesis | Result | Verdict |
|---|---|---|
| H1 deterministic orders and controls | generic (2,2,2,2); pure coupling (3,2,3,3); 2-D example slopes graph 2.999, projector_matrix 2.000, projector_norm 2.999, angle 2.999; block-diagonal and zero controls ≤ 1.2e−14; reducer would exit 4 on failure (exit 0) | **Supported** |
| H2 theorem domain (≤ 44 η² for η ≤ 0.1) | C 300/300, F 300/300, S_toeplitz 188/188 in-domain fits: **0 violations**; max remainder/η² 0.0054 | **Supported where reachable** (necessary check only, per protocol) |
| — `S_full` | **theorem domain not reachable at the allocated m**: median η 9.53 / 4.12 / 1.91 / 0.93 / 0.45 / **0.23** at m = 512 … 524,288; 0 fits scored | **Not reachable** (predeclared finding; not support, not failure) |
| H3 first-order agreement (in-domain) | C, F, S_toeplitz in-domain: 1.000 each. Beyond-domain `S_full` 0.917 is descriptive, outside scope | **Supported** |
| H4 residual orders within 0.3 | all 8 (family × observable) within 0.3; max \|dev\| 0.015 | **Supported**; archived registered T19 prediction stays **NOT MET** (protocol) |

Freeze: 100/100 confirmation cells record the pilot-selection digest. The only pre-freeze confirmation attempt
is the refused, charged attempt from the defective launcher.

## 4. T21-S — isolated-spike × loadings recovery predictor

| Hypothesis | Result | Verdict |
|---|---|---|
| H1 primary (\|err\| ≤ 0.05 in **both** arms and **both** γ) | 296/298: correlated 106/106; **orthogonal 190/192** (both failures `orthogonal|hi|N128`: predicted 0.000 vs observed 0.051; predicted 0.439 vs observed 0.387) | **Not supported overall** → rule `predictor_fails_on_orthogonal_only`: isolated-spike overlap does not describe this learner at these sizes; **loadings step untested** |
| H1′ near-edge (≤ 0.10) | correlated 102/102; orthogonal 9/16. **All 7 failures at θ/√γ = 1 exactly**, predicted 0, observed 0.106–0.161 | Expected finite-d limitation — not support, not refutation (protocol) |
| H2 overlap confound | correlated, θj = 0: ordinary recovery **4–25× chance**, within 0.023 of prediction; unique recovery **0.29–0.98× chance**. Orthogonal: unique ≡ ordinary (3e−16) | **Supported** |
| H0 all-zero control | all 4 arms within z = 5 of chance (max \|dev\| 0.0023 vs tol 0.011) | **Passed** (pipeline check) |

**Diagnosis:** both primary failures and all near-edge failures sit where finite-size smoothing of the asymptotic
overlap is largest (high γ, N = 128, at or below the BBP threshold). Sub-threshold spikes predicted at 0 recover
0.028–0.051. The "primary" classification band (spikes ≥ 0.05 loading outside 0.7 ≤ θ/√γ ≤ 1.8) admits
θ/√γ = 0.5 configurations that still show finite-d excess recovery at N = 128.
**Descriptive:** the 95% CI of the observed mean covers the prediction in only 70–85% of rows by arm. The
absolute tolerance of 0.05 is looser than the sampling precision at 20–24 seeds.

Historical verdicts preserved: all five registered T21 predictions remain **NOT met**.

## 5. CRESST — archive analysis and calibration (**INTENTIONALLY PARTIAL**)

| Hypothesis | Result | Verdict |
|---|---|---|
| H1 reproduction | point estimates: abs deviation **0.0** (θ = 0.21569; ratio θ = 0.22072); intervals **10/10 bit-identical** | **Supported** |
| H2 cluster structure | runs 33 (2 channels) and 35 (5); between-run contrast −12.2 percentage points (1 df, descriptive); leave-one-run-out equals run-stratified | Descriptive |
| H3 non-identifiability | between-run component **refused**: 1 df, ≥ 4 runs required, naive moment estimate negative (−0.0071); between-channel share 0.887 | **Supported** |
| H4 calibration | nominal 0.95. sd_run = 0: C2 0.853 → 0.915 → 0.945 and C5 0.915 → 0.910 → 0.968 at 2/4/8 runs. **sd_run = 0.3: C2 0.610 → 0.713 → 0.820; C5 0.438 → 0.522 → 0.575** | **Supported as predicted** (under-covers; rises with runs) |
| H5 robustness feasibility | 350 robustness cells enumerated and costed, none executable without raw pulses | Prepared, unexecuted (**excluded**) |
| Controls | archive root 5/5; calibration root null-dependence control passed | — |

* **Partiality, exactly:** 45 of the 395 matrix cells are non-robustness; all 45 are complete across the two roots
  (one control cell ran in both by phase plan). The remaining **350 are robustness cells (315 confirm + 35 pilot),
  intentionally unexecuted**. No freeze artifact exists because `cresst_s` has no freeze mechanism.
* **Diagnostic note on H4:** under between-run variance, **more channels per run lowers coverage** (C5 below C2 at
  every run count). Channels narrow the interval without informing the run layer. The reducer's
  `coverage_increases_with_runs` flag compares endpoints only (C5, sd_run 0 is non-monotone: 0.915 → 0.910 → 0.968).
* **Not** replication, **not** new-run coverage, **not** executed robustness (package's own exclusions).

## 6. Environment and provenance disclosures bearing on results

* **365 completed cells** (T19 20, T20 14, T21 310, CRESST archive 21) ran on 13 Sep from a same-version interpreter
  at a `/ceph` path that no longer exists. Affected reductions reproduce on copies under later environments.
* **T22's final reduction** ran under the rebuilt environment. The T22 cells, and every stage of the other
  selections, ran under the original one. T19's original-environment reduction reproduces identically under the
  rebuilt environment.
* The 16 Sep deletion was **blocked, not interrupted**. NFS kept files open by T22 production, which held
  original-environment modules until exit (reproduced on this mount).
