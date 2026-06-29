# Archived: Paper 1 Manuscript Plan (pre-consolidation)

> **Status: archived provenance.** These are the original 2026-05 paper-writing
> plans. They predate the GWOSC negative result and target LaTeX sources that do
> not live in this repository. The current, claim-accurate manuscript guidance is
> [`docs/PAPER_REVISION_GUIDE.md`](../../PAPER_REVISION_GUIDE.md); current status
> is [`docs/CURRENT_STATUS.md`](../../CURRENT_STATUS.md). Kept only for the
> two-paper-split rationale and the section-by-section structure. Source PDF:
> `EMPCA_improvement.pdf` in this folder.
>
> Consolidated from `plan/`: plan.md, chapter_content_mix_summary.md,
> block_01..08, noise_module_placement.md, revision_plan.md, experiment_checklist.md.


---

<!-- source: plan/plan.md -->

# MLST-Oriented Plan for Paper 1

## Executive decision

This repo should support a two-paper sequence, not one overloaded manuscript.

- Paper 1: linear theory paper. Core claim: OF, EMPCA, and a noise-aware linear autoencoder are constrained solutions of one Gaussian ML objective.
- Paper 2: nonlinear systems paper. Core claim: a hybrid learned trigger-and-verification pipeline improves detection and reconstruction on long waveform streams under structured, nonstationary noise.

This split matches the strongest recommendation in `EMPCA_improvement.pdf` and fits the current state of the repo. The first paper already has the right mathematical backbone; the second paper has a richer systems/ML story but should not dilute the linear paper.

## What Paper 1 must achieve

The first paper should read as a clean MLST paper, not as a detector-note dump and not as a half-finished nonlinear paper.

Non-negotiable upgrades from the PDF review:

- Add a clear CRB/variance/energy-resolution thread for OF and connect it to detector figures of merit.
- Make the hierarchy table central: OF, shifted OF, joint-channel OF, EMPCA, linear AE.
- Finish the joint-channel story and stop leaving it as a stub.
- Turn the AE section into the payoff of the paper, not a duplicate appendix.
- Add model-order selection, conditioning, and convergence guidance.
- Fill the experiment tables and characterize the real K-alpha data.
- Add PPCA / factor-analysis / VAE positioning in the discussion.
- Keep nonlinear ResNet / RL / trigger ideas scoped to the companion paper preview.

## Recommended target architecture

### Editorial target for MLST

If you are willing to renumber files later, the strongest final section order is:

1. Abstract
2. Introduction
3. Related Work
4. Statistical Model and Unified Objective
5. Equivalence Theorems
6. EMPCA as a Noise-Aware Linear Autoencoder
7. Convergence and Practical Considerations
8. Experimental Validation
9. Discussion
10. Conclusion
11. Appendices

### Repo-first implementation path

To keep momentum in the current repo, use the existing file skeleton and fold the MLST improvements into it:

- `00_abstract.tex`
- `01_intro.tex`
- `02_unified_objective.tex`
- `03_optimal_filter.tex`
- `04_empca.tex`
- `05_linear_ae.tex`
- `06_convergence.tex`
- `07_experiments.tex`
- `08_discussion.tex`
- `appendix.tex`

The practical compromise is:

- Put the strongest related-work synthesis at the end of `01_intro.tex` for now.
- Keep the current file layout while writing.
- Renumber into a dedicated Related Work section only after the content stabilizes.

## Section-by-section target

### `00_abstract.tex`

Four beats only:

1. Shared reconstruction problem under heteroskedastic noise.
2. Main contribution: OF, EMPCA, and noise-aware linear AE are one ML family.
3. Main theoretical outputs: equivalence hierarchy and bridge theorem.
4. Main empirical outputs: theorem verification, rank-k gains, and motivation for nonlinear follow-up work.

### `01_intro.tex`

Goal: make three reviewer types feel included immediately.

- Signal processing reader: OF / matched filtering / generalized least squares.
- statistics reader: PCA / PPCA / EMPCA / factor analysis.
- ML reader: linear AE / representation learning / noise-aware loss.

The intro must do four jobs:

- state the shared problem before naming methods;
- identify the three missing bridges;
- list theorem-level contributions explicitly;
- scope nonlinear methods to Paper 2.

### `02_unified_objective.tex`

This section is the conceptual spine of the paper.

It must contain:

- generative model;
- Gaussian ML objective;
- whitening transform;
- one hierarchy table with estimator, constraint, degrees of freedom, and solution form;
- multi-channel extension.

Everything else should feel downstream of this section.

### `03_optimal_filter.tex`

This section should become the clean rank-1 constrained ML case.

Must include:

- basic OF derivation;
- time-shift OF;
- joint-channel OF;
- CRB / estimator variance;
- link from variance to energy resolution;
- fixed-template limitation.

### `04_empca.tex`

This section should become the rank-k constrained ML case.

Must include:

- rank-k constraint;
- EMPCA algorithm and weighted/frequency-domain form;
- main theorem that rank-1 EMPCA recovers OF under matched assumptions;
- `chi^2` improvement for `k > 1`;
- geometric interpretation;
- gauge / amplitude reconstruction discussion;
- finite-sample caveat and model-rank caveat.

### `05_linear_ae.tex`

This is the theoretical dessert and should feel inevitable.

Must include:

- encoder-decoder language;
- noise-aware loss;
- formal bridge theorem;
- EM as coordinate descent;
- exact numerical equivalence / ablation;
- one short paragraph that sets up Paper 2 by relaxing the linear encoder.

### `06_convergence.tex`

This section should be practical, not over-ambitious.

Must include:

- what convergence guarantee is actually claimed;
- how to initialize;
- how to choose rank `k`;
- how to regularize whitening and ill-conditioned solves.

### `07_experiments.tex`

This section should benchmark the theory, not become detector-background overflow.

Must include:

- benchmark domain in three short paragraphs max;
- evaluation metrics tied directly to theorem claims;
- verification tables already mentioned in the repo;
- rank-`k` saturation result;
- isotropic MSE vs noise-aware loss ablation;
- convergence behavior;
- full Study A / Study B tables finally populated.

### `08_discussion.tex`

This section should widen the significance without bloating the paper.

Must include:

- transferable principle: encode noise in the loss, not the architecture;
- PPCA / factor-analysis / VAE relation;
- honest limitations;
- one restrained companion-paper preview;
- a clean conclusion.

### `appendix.tex`

Move detector-specific and simulation-heavy material here so the main paper stays focused.

Must include:

- detailed proofs;
- simulation details;
- K-alpha dataset characterization;
- structured-noise generator details;
- extra figures and implementation specifics.

## Content mapping from current repo

Use the current repo as follows.

- `02_unified_objective.tex`: already contains the core ML objective and whitening basis.
- `03_optimal_filter.tex`: already contains the basic OF, time-shift OF, and joint-channel OF algebra.
- `04_empca.tex`: already contains classic PCA, EMPCA, weighted frequency-domain updates, and several stubs that should become theorem-facing subsections.
- `05_linear_ae.tex`: already contains the core AE projector proof and the equivalence table from notebook results.
- `07_experiments.tex`: already contains the benchmark-study scaffolding and some figure/table placeholders.
- `noise.tex` and `appendix.tex`: already contain detector and simulator material that should be pushed to appendix-first use.

## Highest-priority missing writing

Write these in this order:

1. Hierarchy table in `02_unified_objective.tex`.
2. Main theorem sequence in `04_empca.tex`.
3. Formal bridge theorem in `05_linear_ae.tex`.
4. CRB / variance / energy-resolution link in `03_optimal_filter.tex`.
5. Rank-selection / conditioning / convergence material in `06_convergence.tex`.
6. Experiment tables and K-alpha characterization in `07_experiments.tex`.
7. PPCA / factor-analysis / VAE relation in `08_discussion.tex`.

## What to cut or downscope

- Do not let ResNet sections reappear inside Paper 1.
- Do not let Paper 1 become a survey of every detector pathology.
- Do not put nonstationary/artifact noise inside the theorem sections.
- Do not make the AE section a duplicate of EMPCA in different notation.
- Do not bury the main conceptual contribution under detector details.

## How to use the block files

The companion markdown files in this directory break the plan into implementable writing blocks:

- `block_01_positioning_and_intro.md`
- `block_02_unified_objective_and_hierarchy.md`
- `block_03_optimal_filter_rank1_case.md`
- `block_04_empca_rankk_case.md`
- `block_05_linear_ae_bridge.md`
- `block_06_convergence_and_practicals.md`
- `block_07_experiments_and_tables.md`
- `block_08_discussion_appendix_and_cleanup.md`

Also included:

- `paper2_hybrid_ml_pipeline.md`
- `noise_module_placement.md`

## Bottom-line recommendation

The strongest Paper 1 is:

- mathematically centered;
- experimentally verified but not detector-overloaded;
- explicit about what is linear and what is left for Paper 2;
- built around one table, one theorem sequence, one bridge theorem, and one rank-saturation experiment.

If you keep that discipline, the current repo can be turned into an MLST-shaped manuscript without rewriting the whole project architecture first.


---

<!-- source: plan/chapter_content_mix_summary.md -->

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


---

<!-- source: plan/block_01_positioning_and_intro.md -->

# Block 01: Positioning, Abstract, and Introduction

## Goal

Write the paper opening so the reader immediately understands:

- what problem is shared across OF, EMPCA, and AE;
- why this is an MLST paper instead of a detector-specific note;
- what theorems and experiments the paper will actually deliver;
- why nonlinear models belong to Paper 2, not this manuscript.

## Main files

- `00_abstract.tex`
- `01_intro.tex`

Optional later restructure:

- add a dedicated related-work section after `01_intro.tex`

## Source material to reuse

- `EMPCA_improvement.pdf`, especially the improved structure pages
- current comments and subsection headings in `01_intro.tex`
- existing paper title and current section names in `main.tex`

## Target structure

### Abstract

Write four compact beats:

1. shared reconstruction problem under known/estimated noise covariance;
2. theoretical unification of OF, EMPCA, and noise-aware linear AE;
3. key proofs and hierarchy;
4. empirical verification and setup for Paper 2.

### Introduction

Use this structure:

1. shared physics problem first;
2. one paragraph each for signal processing, statistics, and ML communities;
3. the core gap across these communities;
4. numbered contribution list tied to later theorem labels;
5. scope paragraph that explicitly excludes nonlinear trigger models from Paper 1;
6. paper organization.

## Content that must appear

- explicit statement that the paper is about constrained solutions of one Gaussian ML objective;
- contribution bullets that map to the actual theorem sequence;
- one sentence that says the linear AE section is not a side note but the endpoint of the linear theory;
- one short paragraph that says Paper 2 will relax the linear encoder while preserving the noise-aware training principle.

## Implementation instructions

1. Draft the abstract only after the hierarchy table and theorem list are stable.
2. In the intro, open with the measurement problem, not with acronyms.
3. Give each community one paragraph and end each paragraph with the same gap.
4. Keep detector specifics light in the introduction; heavy context belongs in the appendix and benchmark section.
5. If you do not want to renumber files yet, place related-work synthesis inside the intro after the contributions.
6. If you later create a separate Related Work section, move those paragraphs almost verbatim.

## Things to avoid

- Do not discuss ResNet, transformer, RL trigger, or SBI in detail here.
- Do not lead with LAMCAL implementation detail.
- Do not promise robustness claims that the experiments do not verify.

## Done when

- the abstract reads like a finished paper abstract, not a project summary;
- the intro states the gap cleanly for all three communities;
- the contribution bullets clearly point to specific theorem-level outputs;
- the nonlinear story is explicitly postponed to Paper 2.


---

<!-- source: plan/block_02_unified_objective_and_hierarchy.md -->

# Block 02: Unified Objective and Hierarchy Table

## Goal

Turn `02_unified_objective.tex` into the conceptual spine of the paper.

After this block, a reader should understand the single optimization problem and how each estimator appears as a constraint choice.

## Main file

- `02_unified_objective.tex`

## Source material to reuse

- current generative-model and whitening sections in `02_unified_objective.tex`
- joint-channel notation already present in `03_optimal_filter.tex` and `OF_EMPCA.tex`
- hierarchy advice from `EMPCA_improvement.pdf`

## Target structure inside the section

1. Generative model
2. Gaussian ML objective
3. Whitening transform
4. Three constraints, three estimators
5. Multi-channel extension

## Required deliverables

### One orbit equation

Make one equation visually central:

- `chi^2(\hat{s}) = (x - \hat{s})^\dagger Sigma^{-1} (x - \hat{s})`

Everything else should read as a consequence of this.

### One hierarchy table

The table should compare:

- basic OF
- OF with time shift
- joint-channel OF
- EMPCA
- noise-aware linear AE

For each row include:

- constraint on reconstruction;
- learnable degrees of freedom;
- solution form;
- where it is proved or used later.

### Multi-channel bridge

Explain that stacking channels with full covariance keeps the same objective and only enlarges the vector space.

## Implementation instructions

1. Keep the current math and notation wherever it is already clean.
2. Add a short practical whitening subsection that explains diagonal PSD whitening versus full covariance whitening.
3. Add a short regularization sentence here, but defer detailed thresholds and methods to `06_convergence.tex`.
4. Make the hierarchy table the visual center of this section.
5. End the section with a transition sentence into the rank-1 OF case and the rank-`k` EMPCA case.

## Things to avoid

- Do not start proving theorems here.
- Do not expand into detector-specific PSD estimation details here.
- Do not let the multi-channel extension become a stub again.

## Done when

- the reader can state the common ML objective in one sentence;
- the hierarchy table makes the entire paper legible at a glance;
- multi-channel OF is positioned as the same principle, not as a separate method family.


---

<!-- source: plan/block_03_optimal_filter_rank1_case.md -->

# Block 03: Optimal Filtering as the Rank-1 Case

## Goal

Turn `03_optimal_filter.tex` into a complete, self-sufficient rank-1 constrained-ML section.

It should not just derive OF; it should also explain why OF is the physically meaningful baseline for the rest of the paper.

## Main file

- `03_optimal_filter.tex`

## Source material to reuse

- existing OF derivations already in `03_optimal_filter.tex`
- joint-channel formula already present there
- equivalence notes in `OF_EMPCA.tex`
- experimental verification numbers later reused in `07_experiments.tex`

## Target structure

1. Rank-1 constraint and estimator
2. Time-shift OF
3. Joint-channel OF
4. Numerical verification pointer
5. Limitation: fixed-template assumption

## Required content additions

### CRB and variance

Add the missing detector-facing statistics:

- Fisher information / normalization term;
- `Var(\hat{A})`;
- connection to CRB;
- relation to energy resolution or calibrated detector metric.

### Practical assumptions

Briefly acknowledge:

- stationarity assumption;
- finite-record / PSD-estimation caveat;
- fixed-template limitation.

These should be short and controlled, not a literature dump.

## Implementation instructions

1. Preserve the current algebra for basic OF and time-shift OF.
2. Add the variance/CRB calculation immediately after the amplitude estimator so the physics relevance lands early.
3. Make joint-channel OF a full subsection with one worked explanation, not just a proposition pasted from elsewhere.
4. Add a short concluding paragraph that explains what OF cannot capture: timing manifolds, pulse-shape variation, and multi-mode structure.
5. Push long detector-specific details to the appendix.

## Things to avoid

- Do not let this section become a full review of matched filtering literature.
- Do not defer the fixed-template limitation entirely to later sections.
- Do not bury the energy-resolution connection in the discussion.

## Done when

- rank-1 OF is clearly defined as the baseline constrained estimator;
- the reader sees why OF is optimal only under the fixed-template assumption;
- the CRB / variance / resolution link is explicit enough for detector reviewers.


---

<!-- source: plan/block_04_empca_rankk_case.md -->

# Block 04: EMPCA as the Rank-k Case

## Goal

Make `04_empca.tex` the paper's main theoretical engine for moving from rank-1 to rank-`k`.

This section should explain why EMPCA is not "another algorithm section" but the learned-subspace generalization of OF.

## Main file

- `04_empca.tex`

## Source material to reuse

- classic PCA and EMPCA material already in `04_empca.tex`
- weighted frequency-domain formulation already present there
- equivalence theorems and geometry from `OF_EMPCA.tex`
- numerical verification metrics already referenced elsewhere in the repo

## Target structure

1. Rank-`k` constraint
2. EMPCA algorithm
3. Main theorem: rank-1 EMPCA equivalent to OF
4. Proposition: `chi^2` improvement for `k > 1`
5. Geometric interpretation
6. Numerical verification
7. Limitation: algorithmic definition motivates AE framing

## Required content additions

### Formal theorem flow

This section needs a clean theorem sequence:

- main rank-1 equivalence theorem;
- exact algebraic degeneracy statement;
- `chi^2` monotonic improvement with increasing rank;
- remark on finite-sample caveats and subspace identification.

### Gauge and amplitude reconstruction

Do not leave the coefficient/gauge issue as a stub. Add:

- what is identifiable and what is not;
- how amplitudes are compared across bases;
- where regularized solves may be needed.

### Finite-sample realism

Add one remark on:

- small-`N` regime;
- spike separation / subspace recovery caveat;
- why theorem verification remains empirical in finite data.

## Implementation instructions

1. Reuse the current PCA and weighted EMPCA material as the algorithmic background.
2. Put the rank-1 theorem after the algorithm, not before it.
3. Keep the geometric interpretation visual and concise; it should set up the rank-`k` residual improvement claim.
4. Add a short transition from the EMPCA algorithm to the AE bridge: once the objective is linear-subspace reconstruction in whitened space, the encoder-decoder reframing becomes natural.
5. Move any excessive implementation detail to appendix if it interrupts the theorem flow.

## Things to avoid

- Do not let the algorithm subsection dominate the theorem section.
- Do not mix the numerical-verification evidence into the proof text.
- Do not promise exact finite-sample equivalence beyond the stated assumptions.

## Done when

- the rank-1 equivalence theorem reads as the main theorem of the linear paper;
- the case for `k > 1` is mathematically and physically motivated;
- the AE section now feels like the natural next step.


---

<!-- source: plan/block_05_linear_ae_bridge.md -->

# Block 05: Linear Autoencoder Bridge

## Goal

Turn `05_linear_ae.tex` into the payoff section of the paper.

The reader should feel that the AE bridge is the inevitable conclusion of the previous sections, not a detached ML add-on.

## Main file

- `05_linear_ae.tex`

## Source material to reuse

- existing tied linear AE proof already in `05_linear_ae.tex`
- numerical equivalence table already present there
- EMPCA objective from `04_empca.tex`
- whitened-objective framing from `02_unified_objective.tex`

## Target structure

1. Encoder-decoder language
2. Noise-aware loss
3. Bridge theorem
4. EM as coordinate descent on the AE objective
5. Numerical verification: ablation
6. Connection to Paper 2

## Required content additions

### Formal bridge theorem

The theorem should say clearly:

- what the linear AE objective is;
- what constraints are assumed;
- why its minimizer spans the EMPCA subspace;
- in what sense the equivalence holds.

### Why noise-aware loss matters

Do not just replace MSE by a weighted norm and move on. Explain:

- standard MSE encodes isotropic noise;
- whitening changes geometry, not architecture;
- the paper's transferable principle is "noise in the loss, not in a more complicated model by default."

### Coordinate-descent interpretation

Explain the E-step and M-step as:

- latent-code solve;
- basis update;
- same objective, different parametrization.

## Implementation instructions

1. Keep the projector proof short and clean.
2. Add a formal theorem statement before or around the existing proof.
3. Put the numerical table immediately after the theorem logic, not pages later.
4. Add one ablation framing sentence: same linear architecture, different loss.
5. End with one paragraph only for Paper 2: the linear encoder is the part to relax next.

## Things to avoid

- Do not reintroduce duplicate AE content in another section.
- Do not let this section become a VAE or deep-AE survey.
- Do not oversell nonlinear implications that are not demonstrated yet.

## Done when

- the bridge theorem is explicit and defensible;
- the reader understands why AE enters the paper at all;
- the transition to Paper 2 is clean and restrained.


---

<!-- source: plan/block_06_convergence_and_practicals.md -->

# Block 06: Convergence, Rank Selection, and Regularization

## Goal

Make `06_convergence.tex` answer the practical questions reviewers will ask after reading the theory:

- does it converge;
- how do you initialize it;
- how do you choose `k`;
- what happens when whitening or coefficient solves are ill-conditioned.

## Main file

- `06_convergence.tex`

## Source material to reuse

- current stubs in `06_convergence.tex`
- convergence-related comments in `EMPCA_improvement.pdf`
- numerical behavior notes already mentioned in `07_experiments.tex`

## Target structure

1. Convergence guarantee
2. Practical initialization
3. Model order selection
4. Conditioning and regularization

## Required content

### Convergence claim

State only what you can support:

- monotone objective improvement per EM iteration under the stated setup;
- local/global caveat where necessary;
- stationarity and Gaussian assumptions as part of the practical setup.

### Initialization

Recommend one practical default:

- whitened SVD warm start or whitened subspace warm start;
- convergence criterion based on objective change or principal angle.

### Rank selection

Use one primary rule and one secondary rule:

- primary: weighted residual / `chi^2` improvement saturation;
- secondary: scree or information-criterion style view.

### Conditioning

Include:

- ill-conditioned covariance handling;
- ridge or shrinkage on whitening / normal equations;
- guidance on when to regularize.

## Implementation instructions

1. Keep the section practical and short.
2. Put rigorous theorem-level details in the appendix if they become too long.
3. Add direct forward references to the convergence and rank-selection plots in `07_experiments.tex`.
4. Make the model-order subsection operational: tell the reader what they would actually do.

## Things to avoid

- Do not promise a sweeping global-convergence theorem unless you really write it.
- Do not hide rank selection in the appendix; it is central for reviewers.
- Do not turn this section into a broad survey of nonconvex optimization.

## Done when

- the reviewer can see a credible training recipe;
- the choice of `k` no longer looks arbitrary;
- the paper has an explicit answer to conditioning and whitening edge cases.


---

<!-- source: plan/block_07_experiments_and_tables.md -->

# Block 07: Experiments, Verification, and Tables

## Goal

Turn `07_experiments.tex` into a benchmark section that verifies the paper's claims instead of merely collecting figures.

## Main file

- `07_experiments.tex`

Supporting sources:

- `noise.tex`
- figures already in `figures/`
- numerical tables already referenced in `05_linear_ae.tex` and the old equivalence notes

## Target structure

1. Benchmark domain
2. Evaluation metrics
3. Verification of equivalence theorems
4. Reconstruction quality versus rank `k`
5. Noise-aware loss versus isotropic MSE
6. Convergence behavior

## Required deliverables

### Fill the empty tables

Do not leave placeholder tables in the paper.

At minimum fill:

- Study A summary table;
- Study A crossover table;
- Study B scenario table;
- theorem-verification summary table;
- rank-`k` summary table if needed.

### Characterize the K-alpha data

Add enough information for a reader to interpret the real-data verification:

- what the data are;
- how many events;
- what preprocessing was applied;
- what metrics are being compared.

### Core experiments

The section should visibly support four claims:

1. rank-1 EMPCA matches OF under matched assumptions;
2. increasing `k` improves residual fit until saturation;
3. noise-aware loss outperforms isotropic MSE where the noise model matters;
4. the optimization is numerically well-behaved under the stated recipe.

## Structured-noise module usage

Use the nonlinear / nonstationary / correlated / artifact noise module here as a robustness stress test, not as the main theorem-verification dataset.

Recommended placement:

- brief mention in benchmark-domain subsection;
- direct use in the noise-aware-vs-isotropic ablation;
- optional robustness panel in rank-`k` saturation study.

## Implementation instructions

1. Keep the benchmark-domain subsection short and push long detector detail to the appendix.
2. Tie every metric to a theorem or design claim.
3. Put theorem-verification evidence before the more exploratory studies.
4. Use the structured-noise module to show practical relevance without undermining the clean Gaussian-baseline theorem tests.
5. Make one plot the visual centerpiece: whitened reconstruction error versus rank `k`.

## Things to avoid

- Do not let this section turn into a detector-noise survey.
- Do not mix proof language with empirical evidence.
- Do not let the structured-noise results replace the clean baseline equivalence tests.

## Done when

- every table referenced in the text is populated;
- the experiments directly support the paper's stated claims;
- the structured-noise module appears as a useful robustness stress test, not as a theory-breaking distraction.


---

<!-- source: plan/block_08_discussion_appendix_and_cleanup.md -->

# Block 08: Discussion, Appendix, and Final Cleanup

## Goal

Finish the manuscript so it ends like a coherent MLST paper and not like a partially merged note set.

## Main files

- `08_discussion.tex`
- `appendix.tex`
- `main.tex`

## Source material to reuse

- current discussion stubs in `08_discussion.tex`
- detector/simulation material already present in `appendix.tex`
- old noise-model content in `noise.tex`
- file-organization evidence from the repo and `EMPCA_improvement.pdf`

## Target structure

### Discussion

1. The noise-aware loss as a transferable principle
2. Relation to PPCA and VAE
3. Limitations
4. Companion paper preview
5. Conclusion

### Appendix

1. proof of main theorem
2. proof of bridge theorem
3. convergence proof details
4. simulation and dataset details
5. structured-noise module details
6. implementation specifics that would overload the main text

## Required content

### Discussion

Add the missing conceptual positioning:

- PPCA as isotropic special case;
- factor-analysis link for structured noise;
- VAE as orthogonal latent-variable extension, not the main story here.

### Limitations

Be direct about:

- stationarity assumption;
- known or estimated covariance dependence;
- finite-sample subspace limits;
- pile-up / nonlinear pulse-family limits.

### Appendix cleanup

The appendix should absorb:

- detector physics setup;
- simulator pipeline details;
- K-alpha characterization;
- noise-generator / structured-noise module specification;
- proof overflow.

## Implementation instructions

1. Make the discussion widen the significance but keep it brief.
2. Move detector-heavy exposition out of the main paper if it is not needed for a theorem or experiment interpretation.
3. Check that duplicate or removed nonlinear sections do not resurface.
4. Verify every cross-reference, table label, and theorem label after section growth.
5. End the conclusion with the design principle plus one restrained sentence about Paper 2.

## Things to avoid

- Do not introduce new major results here.
- Do not turn the companion-paper preview into a second abstract.
- Do not leave appendix material unlabeled or disconnected from the main text.

## Done when

- the discussion clearly positions the paper in ML/ST language;
- the appendix carries the heavy technical load cleanly;
- the paper closes on one design principle and one realistic next step.


---

<!-- source: plan/noise_module_placement.md -->

# Where to Place the Nonlinear / Nonstationary / Correlated / Artifact Noise Module

## Short answer

Yes: it should appear in Paper 1, but not in the core theorem sections.

Best placement:

- define it in the appendix / simulation-details material;
- mention it briefly in the benchmark-domain subsection;
- use it as a robustness stress test in experiments;
- discuss it again in the limitations section;
- make it central in Paper 2.

## Why it should not sit in the theory core

Paper 1's main theorems rely on a clean Gaussian ML story with known or estimated covariance structure.

Your structured-noise module includes effects that are outside the clean theorem assumptions:

- nonlinearity;
- nonstationarity;
- correlated artifacts;
- potentially non-Gaussian structure.

If you move that module into the theory core, the paper's clean linear claim becomes muddled.

## Best placement inside Paper 1

### 1. Appendix / simulation details

Primary home:

- `appendix.tex`

This is where you should define:

- what noise families are included;
- how the artifact process is injected;
- what parameters are sampled;
- how correlated and nonstationary components are generated;
- how this differs from the clean stationary-Gaussian benchmark.

This keeps the main text light while making the setup reproducible.

### 2. Benchmark domain subsection

Secondary home:

- `07_experiments.tex`, benchmark-domain subsection

Recommended wording structure:

- baseline benchmark: stationary Gaussian / PSD-described regime used for theorem verification;
- robustness benchmark: structured-noise regime used to test practical behavior beyond ideal assumptions.

That gives the reader two regimes without confusing the central claim.

### 3. Experiment usage

Best experimental uses in Paper 1:

- noise-aware loss versus isotropic MSE ablation;
- robustness panel in reconstruction-versus-rank experiment;
- optional residual-diagnostic comparison.

Best experiment not to use it for:

- the primary theorem-verification experiment.

The theorem-verification experiment should stay on the clean, matched-assumption setting.

### 4. Limitations section

Use the module again in `08_discussion.tex` to make a precise statement:

- the theory assumes stationary Gaussian structure;
- the structured-noise experiments probe out-of-model robustness;
- fully modeling such effects is deferred to the nonlinear companion paper.

## Best placement inside Paper 2

Paper 2 is where this module should become central.

There it supports the main claims:

- long-stream triggering under real detector pathologies;
- nonlinear candidate generation;
- uncertainty and confidence under domain shift;
- hybrid verification under artifact-rich conditions.

So the module should be:

- appendix-plus-robustness in Paper 1;
- core benchmark in Paper 2.

## Concrete recommendation by file

- `appendix.tex`: full definition and generation details
- `07_experiments.tex`: benchmark split plus robustness experiments
- `08_discussion.tex`: limitations and transition to Paper 2

Do not place it as a major subsection of:

- `02_unified_objective.tex`
- `03_optimal_filter.tex`
- `04_empca.tex`

Those sections should remain mathematically clean.

## Final recommendation

Use the structured-noise module in Paper 1 to show practical relevance without breaking the paper's clean linear identity.

Use the same module in Paper 2 as the main environment where the hybrid nonlinear pipeline earns its value.


---

<!-- source: plan/revision_plan.md -->

# Revision Plan — Remaining Blockers

Six issues to resolve, ordered from quickest to most involved.

---

## 1. Theorem naming fix (30 min) — `04_empca.tex` ~line 408

**The problem.** `thm:rank1-equivalence` is titled `[Rank-1 OF--EMPCA equivalence]` with no indication in the theorem block itself that the result is at the population level. The population caveat currently lives only in a downstream remark (line 476) and the finite-sample subsection (~line 643), so a skimming reviewer reads the theorem as stronger than it is.

**What to change.** Rename the theorem title and add a scoping clause to the conclusion:

```latex
% BEFORE
\begin{theorem}[Rank-1 OF--EMPCA equivalence]

% AFTER
\begin{theorem}[Rank-1 OF--EMPCA equivalence, population level]
```

Then in the conclusion block of the theorem (the three-item `enumerate`), prepend "At the population level," to item 1:

```latex
% BEFORE
\item the rank-1 EMPCA subspace equals $\mathrm{span}(\tilde{s})$ ...

% AFTER
\item At the population level, the rank-1 EMPCA subspace equals $\mathrm{span}(\tilde{s})$ ...
```

**What to proof-read after.** Grep every `\ref{thm:rank1-equivalence}` call site (01_intro.tex line 144, 03_optimal_filter.tex line 367, 05_linear_ae.tex lines 207 and 302, 08_discussion.tex line 198, OF_EMPCA.tex line 255, appendix.tex line 10) and check that the surrounding prose still reads correctly with the new title in a `\ref` cross-reference.

**How to improve further.** Consider adding a one-sentence preamble before the theorem:

> "The following theorem is a population-level result; finite-sample corrections are given in §\ref{subsec:empca-finite-sample}."

This makes the scoping explicit even when the theorem is read in isolation.

---

## 2. k > 1 containment caveat — `04_empca.tex` ~line 560–585

**The problem.** The proposition correctly conditions on $\mathcal{S}_1 \subseteq \mathcal{U}_k$ (line 562), but there is no sentence anywhere in the body that states learned EMPCA is *not* guaranteed to satisfy this containment unless the basis is constrained or initialized with the template. The reviewer specifically asked for this disclaimer, plus four validation metrics.

**What to change.**

After the discussion of eq. `\eqref{eq:equiv-k-chi2}` (the $\chi^2$ inequality), add a warning paragraph:

```latex
\paragraph{Caveat: containment is not automatic.}
The inequality $\chi^2_{\mathrm{EMPCA},k} \le \chi^2_{\mathrm{OF}}$ holds
only when $\mathcal{S}_1 \subseteq \mathcal{U}_k$, i.e.\ when the OF
template direction lies within the learned EMPCA subspace.
Standard unsupervised EMPCA minimizes the population reconstruction
error and will include $\mathcal{S}_1$ in $\mathcal{U}_k$ whenever the
signal component is the dominant variance direction---which is the
case when $\mathbb{E}[a^2]\,\|\tilde{s}\|^2$ exceeds the next-largest
noise eigenvalue.
If that SNR condition is not met, or if EMPCA is initialized without
reference to the template, the containment $\mathcal{S}_1 \subseteq \mathcal{U}_k$
may fail and the inequality can be violated in finite samples.
Constrained initialization (seeding $u_1 \leftarrow \tilde{s}/\|\tilde{s}\|$)
or a joint OF+EMPCA basis is then required to guarantee the bound.
```

**Four validation metrics (scaffold in §7 subsections).** These are currently only referenced in §7 TODOs. Add a forward-reference sentence immediately after the caveat paragraph:

```latex
Section~\ref{sec:experiments} verifies empirically that the containment
holds in our benchmark regime by reporting: (i)~weighted residual
$\chi^2_{\mathrm{EMPCA},k}$ versus $k$; (ii)~amplitude bias
$|\hat{a}_{\mathrm{EMPCA}} - \hat{a}_{\mathrm{OF}}|$ at $k=1$;
(iii)~energy resolution $\sigma_E(k)$; and (iv)~held-out log-likelihood
as a function of rank.
```

**What to proof-read after.** Re-read the entire §4 k>1 subsection end-to-end to verify the caveat paragraph doesn't contradict the SNR condition already stated in the finite-sample subsection (~line 643). Align the SNR notation ($\Delta_\lambda$ there vs. the energy-eigenvalue framing here).

---

## 3. Szymkowiak et al. 1993 citation — `references.bib` + body files

**The problem.** `grep` finds zero matches for "Szymkowiak" in `references.bib` and no `\cite` anywhere in the tex files. The reviewer called it out by name.

**What to add to `references.bib`:**

```bibtex
@article{Szymkowiak1993,
  author  = {Szymkowiak, A. E. and Kelley, R. L. and Moseley, S. H. and Stahle, C. K.},
  title   = {Signal processing for microcalorimeters},
  journal = {Journal of Low Temperature Physics},
  year    = {1993},
  volume  = {93},
  number  = {3--4},
  pages   = {281--285},
  doi     = {10.1007/BF00693433},
}
```

**Where to cite it in the body.** The natural home is `01_intro.tex` in the paragraph that introduces the optimal filter / matched filter in the microcalorimeter context — wherever the first mention of "optimal filter" or "matched filter for calorimetry" appears. A second cite is appropriate in `03_optimal_filter.tex` in the opening motivation paragraph. A third cite in the discussion (§8) when comparing to prior work is also standard.

**What to proof-read after.** Compile with `bibtex` / `biber` and confirm the entry resolves without error. Check the journal name renders correctly (some templates italicize, some abbreviate).

---

## 4. `07_experiments.tex` — six TODOs and the empty Study B table (days of writing)

This is the largest remaining blocker. The six TODO subsections and the empty table are listed below with specific guidance for each.

### 4a. Benchmark domain (~§7.1)

The stub says only "expand with concise experimental setup description." Write 2–3 paragraphs covering: detector type and operating temperature, digitizer sampling rate and trace length, number of events in the training set vs. held-out set (specify the train/held-out split the reviewer asked for), and the noise sources characterized in Study A. Cross-reference Appendix D for full simulation details.

### 4b. Evaluation metrics (~§7.2)

Write a compact enumeration (prose, not bullet-list) of the four metrics the reviewer requested:

1. **Weighted residual** $\chi^2_{\mathrm{EMPCA},k}$ and its ratio to $\chi^2_{\mathrm{OF}}$, as a function of $k$.
2. **Amplitude bias** $|\hat{a}_{\mathrm{EMPCA}} - \hat{a}_{\mathrm{OF}}|$ at $k=1$, verified to be $< 0.1\%$ (this is the parity table the reviewer asked for).
3. **Energy resolution** $\sigma_E(k)$ (FWHM in eV).
4. **Held-out log-likelihood** as a function of rank $k$, used for model-order selection.

State the train/test split explicitly here (e.g., 80/20 or a fixed held-out set of $N_{\rm test}$ events).

### 4c. Verification of equivalence theorems (~§7.3)

The comment already points to `tab:of-empca-verification` and `tab:empca_ae_primary`. Fill this subsection by:

1. Reporting the OF vs. rank-1 EMPCA amplitude parity table (the reviewer's first requested artifact). Show that estimated amplitudes agree to within numerical precision when whitening is matched.
2. Reporting the principal-angle vs. iteration plot (the reviewer's second artifact): plot the principal angle between the EMPCA subspace and the OF template direction as EM iterations proceed, showing convergence to zero.

If these tables/figures already exist as notebook outputs, export them and include them. If not, note exactly what script needs to run.

### 4d. Reconstruction quality vs. rank k (~§7.4)

Fill the stub with:

1. A table of $\chi^2_{\mathrm{EMPCA},k}$ and $\sigma_E(k)$ for $k = 1, 2, \ldots, k_{\max}$ (the reviewer's third artifact: weighted-residual/energy-resolution vs. $k$ table).
2. A brief narrative interpreting the elbow in the curve — where adding rank stops improving $\sigma_E$.

### 4e. Noise-aware loss vs. isotropic MSE (~§7.5)

This is the reviewer's fourth artifact: an ablation comparing PSD-weighted loss to plain $\ell_2$ loss. Write:

1. A description of the two training conditions (PSD-weighted EMPCA vs. unweighted PCA).
2. A table or figure comparing $\sigma_E$ and $\chi^2$ for both, on the held-out set.
3. A sentence explaining why the isotropic-MSE model is expected to underperform at low frequencies where the noise PSD is colored.

### 4f. Convergence behavior (~§7.6)

Expand the stub (which currently has only the cuBLAS enumeration) with:

1. A plot of the EMPCA objective vs. EM iteration number, for several values of $k$ (the reviewer's convergence-plot request).
2. Quantitative statement of the convergence criterion used and the number of iterations to reach it.

### Study B table (line 96)

The `% fill from notebook export` placeholder needs the actual scenario results. Export the leave-one-out and scale-down rows from the notebook and paste them in. The column headers (`Scenario`, `Relative sensitivity`, `Improvement vs baseline`) are already in place.

**What to proof-read after §7 is written.** Read §7 against the abstract and §1 introduction and verify every claim in the abstract is now backed by a number or figure in §7. In particular, check the abstract's wording about "experimental validation on cryogenic detector pulse data" and "energy resolution improvement" — both must be quantified somewhere in §7.

---

## 5. Appendix proof bodies — `appendix.tex` (major writing task)

The intro (§1, line 186) promises: "Technical proof details and detector-specific simulation material are collected in the appendix." Eight sections are currently TODO stubs. Priority order:

| Appendix section | Priority | Notes |
|---|---|---|
| Proof of Main Theorem (thm:rank1-equivalence) | **High** | Proof already exists inline in §4 (04_empca.tex ~line 440). The appendix version should be the extended proof — move or expand the inline proof here and add a pointer in §4: "Full proof in Appendix A." |
| Proof of Bridge Theorem (§5.3) | **High** | Write from scratch. Check 05_linear_ae.tex for the theorem statement. |
| Convergence proof details | **Medium** | Expand on §6 material; cross-ref app:convergence-proof. |
| LAMCAL and cryogenic detector response | **Medium** | Stub only. Write 1–2 pages of detector physics. |
| Quasi-particle signal formation | **Medium** | Stub only. Write 1 page; cross-ref the TraceSimulator description already present in §A.3. |
| NoiseGenerator implementation details | **Low** | The §A.3 summary paragraph exists; expand with pseudocode or equations. |
| OF implementation | **Low** | Write pseudocode block. |
| EMPCA implementation | **Low** | Write pseudocode block. |

**What to proof-read.** After filling the Main Theorem proof appendix section, compile and check that the `\label{app:convergence-proof}` label resolves from `\ref` calls in §6. Also verify the appendix section letters auto-assigned by LaTeX match any hard-coded cross-references (e.g., "Appendix A" written as prose in §1 or §4).

---

## 6. Cross-cutting: consistency check after all edits

Once issues 1–5 are addressed, do a final pass:

1. **Grep `TODO` and `fill from notebook export`** across all `.tex` files — confirm zero remaining hits in files that will be submitted.
2. **Grep `\ref{thm:rank1-equivalence}`** — verify all 7 call sites read naturally with the new "(population level)" title.
3. **Compile end-to-end** and check for undefined references (`?` in the PDF) and bibtex warnings.
4. **Read the abstract against §7** — every quantitative claim in the abstract must have a backing number in the paper.
5. **Read §1 introduction bullet list** (lines ~144–147) — the promise about Theorem 1 should mention the population-level scope after the rename.

---

## Summary priority table

| # | File | Location | Effort | Blocks submission? |
|---|---|---|---|---|
| 1 | `04_empca.tex` | line ~408 | 30 min | No, but creates review risk |
| 2 | `04_empca.tex` | line ~580 | 1–2 hr | Partially |
| 3 | `references.bib` + body | — | 30 min | Yes (reviewer called it out) |
| 4 | `07_experiments.tex` | all subsections | Days | **Yes** |
| 5 | `appendix.tex` | all TODO sections | Days | Yes (intro promises it) |
| 6 | All files | final grep/compile | 1 hr | — |


---

<!-- source: plan/experiment_checklist.md -->

# Experiment Verification Checklist

_Maps every theorem and claim in the paper to a concrete experiment.
Updated to reflect: (1) precise paper section labels, (2) explicit data source
classification, (3) input/output specification, (4) QPSimulator API as implemented._

**QPSimulator API (all four additions implemented ✅):**
- `generate(arrival_times, return_amplitude=False)` — single clean trace ± ground-truth amplitude
- `generate_family(n_events, tau_decay_range, t0_jitter_range, n_QP_range, rng)` → `(traces, params)`
- `get_template_at_shift(t0_shift_ns)` → shifted template `(trace_samples,)`
- `estimate_psd(noise_traces, sampling_frequency)` → `(freqs, J_k)` — static method

**Noise modules (no changes needed):**
`NoiseGenerator`, `TemporalNoiseWrapper`, `ArtifactInjector`, `MultiChannelNoiseGenerator`

**Data source classification used below:**
- **SIM-single** — one clean trace from `generate()` + repeated independent noise draws
- **SIM-batch** — traceset from `generate_family()` + noise applied externally
- **CAL-kalpha** — real K-alpha calibration traces from the detector

---

## E1 — Theorem 1: rank-1 EMPCA ≡ OF amplitude estimator

**Paper section:** §4 (`\label{thm:rank1-equivalence}`, `\label{subsec:equiv-rank1}`);
numerical summary populates `\label{tab:of-empca-verification}` in §3 (`\label{subsec:equiv-verification}`).

**Purpose:** Verify that under matched whitening, rank-1 EMPCA produces the identical
template direction and per-event amplitude estimates as optimal filtering — the numerical
confirmation of Theorem 1 (rank-1 equivalence).

**Data source:** SIM-batch

**Input:**
```python
sim = QPSimulator()   # tau_decay=3e6, tau_rise=50e3, trigger_time=default
traces, params = sim.generate_family(
    n_events=500, n_QP_range=(5000, 5000), rng=rng
)
ng = NoiseGenerator(dict(noise_type='pink', noise_power=1.0, sampling_frequency=2.5e5))
noisy = traces + np.stack([ng.generate_noise(sim.trace_samples) for _ in range(500)])
freqs, Jk = QPSimulator.estimate_psd(noise_cal_2000, sim.frequency)
```

**Expected output / pass thresholds:**

| Metric | Formula | Pass |
|---|---|---|
| Weighted subspace cosine | `ρ_w(U_EMPCA, s₀)` | > 0.9999 |
| Amplitude correlation | `corr(A_EMPCA, A_OF)` | > 0.999 |
| Median relative error | `median(|A_EMPCA − A_OF| / A_OF)` | < 1×10⁻³ |
| KS test on residuals | `p-value` | > 0.05 |

**Status:** Partially done (real-data result 0.9999999655 in `tab:of-empca-verification`).
Needs full simulation script producing all four metrics systematically.

---

## E2 — Theorem 2 (Bridge Theorem): noise-aware linear AE ≡ EMPCA

**Paper section:** §5 (`\label{thm:bridge}`, `\label{subsec:ae_pca_equiv}`);
numerical summary populates `\label{tab:empca_ae_primary}` in §5 (`\label{subsec:numerical_ae}`).

**Purpose:** Verify that at convergence, the rank-k noise-aware tied linear AE spans the same
subspace as rank-k EMPCA — numerical confirmation of the Bridge Theorem.

**Data source:** SIM-batch (same traceset as E1, reused)

**Input:**
```python
# Method A: run rank-k EMPCA with weight=1/Jk → basis U_EMPCA (d×k)
# Method B: SVD of whitened data matrix X_tilde = X @ inv_sqrt_Sigma → W_AE (d×k)
# Repeat for k = 1, 2, 3
```

**Expected output / pass thresholds:**

| Metric | Formula | Pass |
|---|---|---|
| Principal-angle cosine (k=1) | `cos θ₁(span(U), span(W))` | > 0.9999 |
| Principal-angle cosine (k=2) | `cos θ₁, cos θ₂` | > 0.9999 each |
| Principal-angle cosine (k=3) | `cos θ₁, cos θ₂, cos θ₃` | > 0.9999 each |
| Relative residual difference | `‖res_EMPCA − res_AE‖ / ‖res_EMPCA‖` | < 1×10⁻⁵ |
| KS test on residuals | `p-value` | > 0.2 |

**Status:** Partially done. Real-data result exists. Needs systematic simulation script for k=1,2,3.

---

## E3 — Proposition: χ²(k) monotone decrease with rank

**Paper section:** §4 (`\label{subsec:equiv-kgreater1}`).

**Purpose:** Verify that χ²_EMPCA(k) ≤ χ²_EMPCA(k−1) numerically, with strict inequality
when the signal family has dimension > 1.

**Data source:** SIM-batch (two separate tracesets)

**Input — Setup A (1D family):**
```python
traces_A, _ = sim.generate_family(
    n_events=1000, tau_decay_range=(3e6, 3e6), n_QP_range=(2000, 8000), rng=rng
)
# Add pink noise; run EMPCA for k=1…8; record χ²_test(k)
```

**Input — Setup B (multi-dimensional family):**
```python
traces_B, _ = sim.generate_family(
    n_events=1000, tau_decay_range=(1e6, 5e6),
    t0_jitter_range=(-1e5, 1e5), n_QP_range=(3000, 7000), rng=rng
)
# Add pink noise; run EMPCA for k=1…8; record χ²_test(k)
```

**Expected output / pass thresholds:**

| Setup | Metric | Expected |
|---|---|---|
| A | Δχ²(2)/χ²(1) | < 1% (plateau) |
| B | Δχ²(2)/χ²(1) | > 5% (strict improvement) |
| Both | χ²(k) sequence | Monotone non-increasing for k=1…8 |

**Deliverable:** Figure — χ²_test(k) vs k, two curves on the same axes.

---

## E4 — CRB: empirical Var(Â) = 1/N_Φ

**Paper section:** §3 (`\label{subsec:of-crb}`), equations `\eqref{eq:of-variance}` and `\eqref{eq:of-fisher}`.

**Purpose:** Verify the Cramér-Rao bound: OF amplitude estimator achieves `Var(Â) = 1/N_Φ`.

**Data source:** SIM-single

**Input:**
```python
trace_clean, A_true = sim.generate([sim.trigger_time], return_amplitude=True)
ng = NoiseGenerator(dict(noise_type=noise_type, noise_power=pw, sampling_frequency=2.5e5))
A_hat = [OF_amplitude(trace_clean + ng.generate_noise(N)) for _ in range(5000)]
# N_Phi = sum(|S_k|^2 / Jk);  predicted_var = 1/N_Phi
```
Repeat for `noise_type ∈ {white, pink, brownian}` with N_Φ ∈ [10, 100].

**Expected output / pass thresholds:**

| Noise type | Metric | Pass |
|---|---|---|
| white | `|σ²_emp − 1/N_Φ| / (1/N_Φ)` | < 5% |
| pink | same | < 5% |
| brownian | same | < 5% |

**Deliverable:** Table — three rows (noise type), columns: N_Φ, 1/N_Φ, σ²_emp, relative error.

---

## E5 — Energy resolution: σ_E = E₀/√N_Φ

**Paper section:** §3 (`\label{subsec:of-crb}`), equation `\eqref{eq:of-energy-resolution}`.

**Purpose:** Verify the ∝ 1/√noise_power scaling in simulation; ground the formula against
real K-alpha data.

**Data source:** BOTH — SIM-batch (scaling curve) and CAL-kalpha (absolute calibration)

**Input — Simulation:**
```python
for pw in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
    traces, _ = sim.generate_family(n_events=1000, n_QP_range=(5000, 5000), rng=rng)
    noisy = traces + noise_at_power(pw)
    sigma_E_emp[pw] = OF_amplitude_batch(noisy, Jk(pw)).std()
    sigma_E_pred[pw] = E0 / np.sqrt(N_Phi(pw))
```

**Input — Real data (CAL-kalpha):**
```python
freqs, Jk_real = QPSimulator.estimate_psd(baseline_traces, sampling_frequency)
# Compare sigma_E_pred = E_Kalpha/sqrt(N_Phi) vs sigma_E_obs = std(A_OF_Kalpha)
```

**Expected output / pass thresholds:**

| Component | Metric | Pass |
|---|---|---|
| Simulation | slope of log(σ_E) vs log(noise_power) | −0.5 ± 0.05 |
| Simulation | residuals from theory curve | < 10% per point |
| Real data | `|σ_E_pred − σ_E_obs| / σ_E_obs` | < 15% |

**Deliverable:** (1) log-log plot σ_E vs noise_power with theory line; (2) one-row real-data table.

---

## E6 — Noise-aware EMPCA vs isotropic PCA ablation

**Paper section:** §7 (`\subsection{Noise-aware loss versus isotropic MSE}`);
conclusion restated in §8.1 (`\label{subsec:noise-aware-principle}`).

**Purpose:** Main practical ablation. Shows Σ⁻¹-weighted EMPCA beats unweighted PCA under
colored noise, and that the gap grows with noise coloredness.

**Data source:** SIM-batch

**Input:**
```python
traces, _ = sim.generate_family(n_events=700, n_QP_range=(2000, 8000), rng=rng)
train_clean, test_clean = traces[:500], traces[500:]

for noise_type in ['white', 'pink', 'brownian']:
    Jk = QPSimulator.estimate_psd(noise_cal_500, sim.frequency)[1]
    U_empca = run_empca(train_clean + noise(500), Jk, k=3)
    U_pca   = run_pca(train_clean + noise(500), k=3)
    rel_improvement[noise_type] = (
        weighted_chi2(test_clean + noise(200), U_pca,   Jk) -
        weighted_chi2(test_clean + noise(200), U_empca, Jk)
    ) / weighted_chi2(test_clean + noise(200), U_pca, Jk)
```

**Expected output / pass thresholds:**

| Noise type | `(χ²_PCA − χ²_EMPCA) / χ²_PCA` | Expected |
|---|---|---|
| white | ≈ 0 | control |
| pink | > 5% | EMPCA wins |
| brownian | > 15% | EMPCA wins strongly |

**Deliverable:** Bar chart — three noise conditions × two methods. Primary figure for §7.

---

## E7 — Template mismatch: fixed-template OF bias and EMPCA recovery

**Paper section:** §3 (`\label{subsec:of-limitation}`), equations `\eqref{eq:of-bias}`.

**Purpose:** Verify the bias formula `E[Â_OF] = A · cos²θ_w` and show rank-2 EMPCA
recovers the unbiased amplitude when signal shapes vary.

**Data source:** SIM-batch

**Input:**
```python
traces_mis, params_mis = sim.generate_family(
    n_events=500, tau_decay_range=(1e6, 5e6),
    n_QP_range=(5000, 5000), rng=rng
)
noisy_mis = traces_mis + noise_batch('pink', 500)
Jk = QPSimulator.estimate_psd(noise_cal_200, sim.frequency)[1]
# Method A: OF with nominal template (tau_decay=3e6)
# Method B: rank-1 EMPCA
# Method C: rank-2 EMPCA
# Ground truth: params_mis['amplitude_ADC']
```

**Expected output / pass thresholds:**

| Metric | Method A (OF) | Method C (EMPCA k=2) |
|---|---|---|
| Mean bias `|E[Â] − A_true| / A_true` | > 10% | < 3% |
| Bias vs cos²θ_w | matches formula within 5% | — |

**Deliverable:** (1) Scatter Â vs A_true; (2) bias table; (3) formula-verification curve.

---

## E8 — Time-shift OF: arrival time and amplitude recovery

**Paper section:** §3 (`\label{subsec:equiv-shifted-of}`), equations
`\eqref{eq:equiv-shifted-of}` and `\eqref{eq:equiv-shifted-of-t0}`.

**Purpose:** Verify that time-shift OF recovers both t̂₀ and amplitude correctly;
confirm SNR degradation of fixed-template OF on jittered data.

**Data source:** SIM-batch

**Input:**
```python
traces_jit, params_jit = sim.generate_family(
    n_events=300,
    t0_jitter_range=(-sim.trace_duration * 0.1, sim.trace_duration * 0.1),
    n_QP_range=(5000, 5000), rng=rng
)
noisy_jit = traces_jit + noise_batch('pink', 300)
Jk = QPSimulator.estimate_psd(noise_cal_200, sim.frequency)[1]

# Time-shift OF filter bank using new API:
for j, t0 in enumerate(t0_grid):
    s_shift = sim.get_template_at_shift(t0)   # ← get_template_at_shift
    A_grid[j] = OF_amplitude(noisy, s_shift, Jk)
t_hat = t0_grid[np.argmax(A_grid, axis=1)]
# Ground truth: params_jit['t0_shift']
```

**Expected output / pass thresholds:**

| Metric | Time-shift OF | Fixed OF |
|---|---|---|
| Arrival-time RMSE | < 2 samples | N/A |
| Mean amplitude bias | < 5% | > 5% for large-jitter events |

**Deliverable:** (1) RMSE table; (2) scatter Â vs A_true; (3) timing-residual histogram.

---

## E9 — Convergence: EM iterations vs χ²

**Paper section:** §6 (`\label{subsec:convergence-theorem}`, Theorem `thm:convergence`).

**Purpose:** Verify monotone non-increase of χ² per EM step and characterize empirical
convergence speed. Test sensitivity to initialization.

**Data source:** SIM-batch (genuine 2D family — Setup B from E3)

**Input:**
```python
traces_conv, _ = sim.generate_family(
    n_events=500, tau_decay_range=(1e6, 5e6), rng=rng
)
noisy_conv = traces_conv + noise_batch('pink', 500)
# Run EMPCA 100 iterations for k=1,2,3; record χ²(r) each step
# Two inits: random Haar vs SVD-seeded
```

**Expected output / pass thresholds:**

| Metric | Pass |
|---|---|
| χ²(r) monotone | True for all r=1…100 |
| Convergence (k=1) | `|Δχ²| / χ²(0) < 1e-6` within 20 iters |
| Convergence (k=2,3) | within 50 iters |
| Init independence | `|χ²_rand(∞) − χ²_svd(∞)| / χ²(0) < 1e-5` |

**Deliverable:** Figure — χ²(r) vs r, k=1 and k=2 panels, two init curves each.

---

## E10 — Non-stationary noise robustness

**Paper section:** §6 (`\label{subsec:noise-assumptions}`).

**Purpose:** Show EMPCA degrades with non-stationary noise under global PSD; per-segment
PSD partially recovers performance.

**Data source:** SIM-batch

**Input:**
```python
traces_ns, _ = sim.generate_family(n_events=500, rng=rng)
tw = TemporalNoiseWrapper(base_ng, mode='piecewise', n_segments=4, scale_range=(0.7, 1.3))
noisy_ns = traces_ns + np.stack([tw.apply(np.zeros(N)) for _ in range(500)])
# Case A: single global PSD from all noise traces
# Case B: per-segment PSD (125 traces each)
```

**Expected output / pass thresholds:**

| Case | Test χ² | Amplitude RMSE |
|---|---|---|
| A (global PSD) | elevated | higher |
| B (per-segment) | closer to stationary | lower |
| Improvement B vs A | > 5% χ² reduction | quantified |

---

## E11 — Artifact robustness

**Paper section:** §6 (`\label{subsec:limitations}`), §8.3.

**Purpose:** Show glitch artifacts contaminate EMPCA subspace; χ²-threshold flagging restores
performance.

**Data source:** SIM-batch

**Input:**
```python
traces_art, params_art = sim.generate_family(n_events=500, rng=rng)
ai = ArtifactInjector(config=dict(glitch_rate=0.1, impulse_rate=0.05))
noisy_art = [trace + ng.generate_noise(N) + ai.apply(np.zeros(N)) for trace in traces_art]
# Pass 1: rank-2 EMPCA on all 500 (contaminated)
# Pass 2: flag χ²_noise > 5σ; re-run on clean subset
```

**Expected output / pass thresholds:**

| Pass | Amplitude RMSE | χ²_test |
|---|---|---|
| Without flagging | higher | elevated |
| After flagging | lower (> 10% improvement) | reduced |

---

## E12 — Real K-alpha data: full equivalence verification

**Paper section:** §7 (`\subsection{Verification of equivalence theorems}`),
§3 (`\label{tab:of-empca-verification}`), §5 (`\label{tab:empca_ae_primary}`).

**Purpose:** Validate E1 and E2 on real K-alpha calibration data. Add the amplitude histogram
(needed for CRB visual verification) currently missing from §7.

**Data source:** CAL-kalpha

**Input:**
```python
freqs, Jk_real = QPSimulator.estimate_psd(baseline_traces, sampling_frequency)
# Replicate E1 (OF vs rank-1 EMPCA) on real traces
# Replicate E2 (principal angles) on real traces
# NEW: amplitude histogram → fit Gaussian → extract σ_A_obs
# Compare σ_A_obs to 1/sqrt(N_Phi) from Jk_real
```

**Expected output / pass thresholds:**

| Metric | Pass |
|---|---|
| ρ_w cosine (E1 real data) | > 0.9999 (existing: 0.9999999655 ✓) |
| Principal-angle cosines (E2) | > 0.9999 |
| Amplitude histogram Shapiro-Wilk | p > 0.05 |
| `|σ_A_obs − 1/√N_Φ| / (1/√N_Φ)` | < 15% |

**Missing:** amplitude histogram figure + σ_A comparison row in `tab:of-empca-verification`.

**Deliverable:** (1) Amplitude histogram with Gaussian overlay; (2) updated table with σ_A row.

---

## Simulator status summary

| Addition | Status | Used by |
|---|---|---|
| `generate_family()` | ✅ implemented | E1, E2, E3, E6, E7, E8, E9, E10, E11 |
| `get_template_at_shift()` | ✅ implemented | E8 |
| `estimate_psd()` | ✅ implemented | E1–E12 (every experiment) |
| `generate(return_amplitude=True)` | ✅ implemented | E4, E5, E7 |

No changes needed to `NoiseGenerator`, `ArtifactInjector`, `TemporalNoiseWrapper`,
or `MultiChannelNoiseGenerator`.

---

## Result table template for §7

| Exp | Method | Noise / Data | Key metric | Value |
|---|---|---|---|---|
| E1 | OF vs rank-1 EMPCA | SIM pink | ρ_w cosine | ≥ 0.9999 |
| E1 | OF vs rank-1 EMPCA | SIM pink | amp correlation | ≥ 0.999 |
| E2 | EMPCA vs AE (k=1) | SIM pink | principal-angle cos | ≥ 0.9999 |
| E2 | EMPCA vs AE (k=2) | SIM pink | principal-angle cos | ≥ 0.9999 |
| E3 | χ²(k) plateau | SIM pink, fixed τ | Δχ²(2)/χ²(1) | < 1% |
| E3 | χ²(k) drop | SIM pink, varied τ | Δχ²(2)/χ²(1) | > 5% |
| E4 | CRB | SIM white | `|σ²_emp − 1/N_Φ|/(1/N_Φ)` | < 5% |
| E4 | CRB | SIM pink | same | < 5% |
| E5 | σ_E scaling | SIM pink sweep | slope log σ_E vs log pw | −0.5 ± 0.05 |
| E5 | σ_E formula | CAL K-alpha | `|σ_pred − σ_obs|/σ_obs` | < 15% |
| E6 | EMPCA vs PCA | SIM white | Δχ²/χ²_PCA | ≈ 0 (control) |
| E6 | EMPCA vs PCA | SIM pink | Δχ²/χ²_PCA | > 5% |
| E6 | EMPCA vs PCA | SIM brownian | Δχ²/χ²_PCA | > 15% |
| E7 | OF bias formula | SIM mismatched | `|E[Â_OF] − A·cos²θ|` | < 5% |
| E8 | time-shift OF | SIM jittered | t̂₀ RMSE | < 2 samples |
| E9 | EM convergence | SIM pink varied τ | iters to 1e-6 tol | ≤ 50 |
| E12 | Real K-alpha (E1) | CAL K-alpha | ρ_w cosine | 0.9999999655 ✓ |
| E12 | Real K-alpha (CRB) | CAL K-alpha | `|σ_obs − 1/√N_Φ|/...` | TBD |

---

## Priority order

1. **E1** — Theorem 1: direct numerical verification (§4)
2. **E2** — Bridge Theorem: direct numerical verification (§5)
3. **E6** — EMPCA vs PCA ablation: main practical claim (§7)
4. **E3** — χ²(k) curve: Proposition verification (§4)
5. **E4** — CRB: energy resolution bound (§3)
6. **E12** — Real data: amplitude histogram missing (§7)
7. **E7** — Template mismatch bias: limitation section (§3)
8. **E8** — Time-shift OF: extension verification (§3)
9. **E9** — Convergence plot: §6 theorem support
10. **E5** — σ_E scaling: simulation + real data (§3)
11. **E10/E11** — Robustness: supports §6 limitations discussion
