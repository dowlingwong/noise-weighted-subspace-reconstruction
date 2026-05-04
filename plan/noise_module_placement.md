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
