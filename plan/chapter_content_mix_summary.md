# Chapter Content Mix Summary

This summary is based on `plan.md` and the eight `block_*.md` planning files in this directory. Percentages are editorial estimates of the intended chapter content, not measured word counts. Each row sums to 100%.

## Category Definitions

| Category | Meaning in this plan |
| --- | --- |
| Math / theory | Equations, constrained optimization, derivations, theorem statements, proofs, CRB/variance algebra, projector/subspace arguments. |
| Statistical modeling | Gaussian ML assumptions, covariance/noise modeling, whitening interpretation, estimator assumptions, finite-sample caveats, rank/model selection, PPCA/factor-analysis/VAE positioning. This is separate from math because it concerns probabilistic meaning and modeling assumptions, not just algebra. |
| Experimental verification | Numerical theorem checks, benchmark metrics, real-data characterization, ablations, rank-saturation plots, convergence plots, populated tables. |
| Argument / practical framing | Motivation, positioning, contribution claims, limitations, writing transitions, implementation guidance, paper organization, appendix cleanup, companion-paper scoping. |

## Estimated Content Mix by Chapter

| Planned chapter / file | Main block source | Chapter summary | Math / theory | Statistical modeling | Experimental verification | Argument / practical framing |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Abstract, `00_abstract.tex` | Block 01 | Four-beat abstract: shared reconstruction problem, unified ML family, theorem hierarchy, empirical verification, and restrained Paper 2 setup. | 20% | 25% | 20% | 35% |
| Introduction / related-work synthesis, `01_intro.tex` | Block 01 | Opens with the shared measurement problem, connects signal processing, statistics, and ML readers, states the missing bridges, lists theorem-level contributions, and scopes nonlinear methods to Paper 2. | 10% | 20% | 10% | 60% |
| Statistical model and unified objective, `02_unified_objective.tex` | Block 02 | Establishes the Gaussian ML objective, whitening transform, hierarchy table, estimator constraints, degrees of freedom, and multi-channel extension. This is the conceptual spine of the paper. | 45% | 40% | 0% | 15% |
| Optimal filtering as rank-1 ML, `03_optimal_filter.tex` | Block 03 | Derives basic OF, time-shift OF, and joint-channel OF as rank-1 constrained ML; adds CRB/variance and energy-resolution interpretation; states fixed-template limits. | 55% | 25% | 5% | 15% |
| EMPCA as rank-`k` ML, `04_empca.tex` | Block 04 | Presents EMPCA as the learned-subspace generalization of OF, with rank-1 equivalence theorem, `k > 1` residual improvement, geometry, gauge/amplitude handling, and finite-sample caveats. | 50% | 25% | 10% | 15% |
| Noise-aware linear autoencoder bridge, `05_linear_ae.tex` | Block 05 | Reframes the same objective in encoder-decoder language, proves the bridge theorem to EMPCA, interprets EM as coordinate descent, verifies numerical equivalence, and sets up Paper 2 by relaxing linearity. | 45% | 25% | 15% | 15% |
| Convergence, rank selection, and regularization, `06_convergence.tex` | Block 06 | Gives a practical training recipe: supported convergence claim, initialization, rank selection by weighted residual saturation, and covariance/solve regularization. | 25% | 35% | 15% | 25% |
| Experiments, verification, and tables, `07_experiments.tex` | Block 07 | Converts experiments into direct support for the claims: theorem verification, rank-`k` saturation, noise-aware versus isotropic loss, convergence behavior, K-alpha characterization, and filled Study A/B tables. | 5% | 20% | 65% | 10% |
| Discussion and conclusion, `08_discussion.tex` | Block 08 | Widens significance around the principle of putting noise in the loss, positions the work against PPCA/factor analysis/VAE, states limitations, previews the companion paper, and closes cleanly. | 5% | 25% | 5% | 65% |
| Appendix, `appendix.tex` | Block 08 | Carries proof overflow, convergence proof details, detector/simulation setup, K-alpha dataset characterization, structured-noise module details, and implementation specifics. | 35% | 20% | 25% | 20% |

## Explanation by Chapter

### Abstract and Introduction

The opening is mostly argument and positioning. Its job is not to prove the paper but to make the reader understand the shared problem, the cross-community gap, and the exact claims that later sections will prove or verify. It still needs some statistical content because the paper is built around known or estimated covariance, Gaussian ML, whitening, and noise-aware reconstruction.

### Unified Objective

This chapter should be balanced between math and statistics. The central equation is mathematical, but the importance of the chapter comes from its statistical interpretation: different estimators are not separate inventions, but constrained solutions of one Gaussian likelihood. The hierarchy table is the key narrative device, but it should support the objective rather than replace it.

### Optimal Filtering

The OF chapter is the most derivation-heavy rank-1 section. It needs amplitude estimation algebra, time-shift and joint-channel formulas, and the CRB/variance link. The statistical share is also substantial because the section must state the assumptions under which OF is optimal and explain why those assumptions produce detector-relevant variance or energy-resolution metrics.

### EMPCA

EMPCA is the main theoretical engine. Most of the chapter should be math: rank-`k` constraints, equivalence theorem, monotone `chi^2` improvement, and subspace geometry. The statistical content comes from finite-sample caveats, identifiability, gauge/amplitude interpretation, and assumptions behind subspace recovery.

### Linear Autoencoder

The AE chapter is theoretical but should read as the payoff of the earlier objective. Its math is the bridge theorem and projector/equivalence proof. Its statistical contribution is the claim that noise-aware loss changes the geometry of learning while the architecture can remain linear. Experimental content should be limited to the numerical equivalence and loss-ablation evidence needed to support the bridge.

### Convergence and Practicals

This chapter should not overclaim new theory. Its strongest role is operational: say what convergence is actually guaranteed, how to initialize, how to pick rank `k`, and how to regularize ill-conditioned covariance or coefficient solves. The statistical content is high because rank selection, whitening, and conditioning are model-assumption and estimation-quality issues.

### Experiments

The experiment chapter should be the empirical backbone, not a detector-background dump. It should verify the theorem claims first, then show rank saturation, noise-aware-loss gains, and convergence behavior. Math should be minimal here except for metric definitions and references back to the theorem claims.

### Discussion and Appendix

The discussion is mostly argument: it positions the paper in ML/ST terms, states limitations honestly, and previews Paper 2 without turning it into a second paper. Statistical framing remains important because PPCA, factor analysis, VAE, stationarity, covariance dependence, and finite-sample limits are all modeling claims. The appendix should absorb long proofs, simulation details, data characterization, and implementation specifics so the main paper remains readable.

## Overall Balance

The planned Paper 1 is math-centered but not math-only. A reasonable whole-paper target is approximately:

| Overall target | Share |
| --- | ---: |
| Math / theory | 35% |
| Statistical modeling | 25% |
| Experimental verification | 20% |
| Argument / practical framing | 20% |

This balance matches the plan's bottom-line recommendation: the manuscript should be mathematically centered, experimentally verified, explicit about its statistical assumptions, and disciplined about keeping nonlinear systems material in Paper 2.
