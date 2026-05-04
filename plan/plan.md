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
