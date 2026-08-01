# R1 Analysis plan
# Version 1.0 · 2026-08-01

protocol: R1-C
version: "1.0"

## Statistical estimands

Each estimand is defined before implementation with:
- population and task-family weighting
- treatment or predictor contrast
- outcome definition and time horizon
- handling of invalid, censored, and missing runs
- unit of analysis
- covariate and blocking adjustment
- uncertainty method

Primary estimands use equal weight across task families and task instances
unless a deployment distribution is explicitly defined. Prevalence-weighted
estimates are secondary.

## Hierarchical modeling

Generalized mixed models appropriate to outcome:
- Complementary log-log mixed model for discrete failure hazard
- Logistic mixed model for binary verified success and recovery
- Linear or generalized mixed model for cost and lead time
- Hierarchical multinomial or multilabel model for attribution

Random intercepts for task template and task instance. Model/provider or
fault random effects only when enough levels exist. With two model families,
treat model as fixed. Repeated seeds and policies nested within task/fault blocks.

## Uncertainty and resampling

Primary nonparametric intervals use hierarchical bootstrap:
1. Resample task templates
2. Resample task instances within template
3. Retain all repeated trajectories or resample them within instance as specified
4. Recompute complete metric including threshold selection from calibration split only

At least 2,000 bootstrap replicates for final intervals. Report bootstrap seed
and convergence diagnostics. Compare with model-based intervals as sensitivity.

## Multiplicity

H1-H4 each have one primary comparison. Secondary model/fault/workload contrasts
use Benjamini-Hochberg FDR control within declared families. Exploratory analyses
labeled exploratory and do not inherit confirmatory language. Do not select best
subgroup and report it as general effect.

## Missingness and invalid runs

Every non-complete trajectory classified as:
- policy_termination
- budget_exhaustion
- infrastructure_failure
- injector_failure
- observer_failure
- verifier_failure
- scientific_task_failure
- unknown

Intention-to-treat analysis retains all valid assignments. Per-protocol sensitivity
excludes injector or infrastructure failures but cannot replace primary analysis.
Report missingness by arm. Test whether it is policy-dependent.
