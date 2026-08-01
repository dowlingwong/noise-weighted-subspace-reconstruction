# R2 Analysis plan
# Version 1.0 · 2026-08-01

protocol: R2
version: "1.0"

## Controller analysis (R2-C1)

Primary contrast: adaptive policy vs strongest non-oracle baseline
at matched verification and human-effort budget.

Blocking: task_instance × model_configuration × sealed_fault_episode.
Within-block randomized policy order. Episode as random effect
when exact replay impossible.

Primary model: mixed-effects model with policy as fixed effect,
task_instance and fault_episode as random intercepts.

Supporting: success–cost frontier area, irreversible damage rate,
false-intervention rate on no-fault tasks, oracle regret.

Policy-induced calibration: account for selective observation under
adaptive verification. Off-policy evaluation with importance sampling
or doubly robust estimator where logging propensities available.

## Training analysis (R2-T)

Primary contrast: process-supervised arm (5) vs stronger of
recovery SFT (3) and outcome-only (4) at matched evaluation cost.

Co-primary: clean-task non-inferiority with margin -0.02.

Uncertainty: hierarchical over training seeds (within arm),
task families, and fault families. At least 2,000 bootstrap
replicates for primary intervals.

Checkpoint selection: fixed rule based on development outcomes only.
No test-set optimization.

## Factorial ablation analysis (R2-A)

2×2×2 design estimates main effects and two-factor interactions.
Primary interest: active_verification × process_supervision interaction.
Reuse exact checkpoints from primary arms where configuration matches.
No retraining under new labels.

## Multiplicity

H3 and H4 each have one primary comparison. Secondary contrasts
(baseline-to-baseline, per-workload, per-fault-family) use
Benjamini-Hochberg FDR within declared families. Exploratory
analyses labeled as such.

## Missingness and invalid runs

Same classification as R1. Training-view construction excludes
injector/observer/infrastructure failures. Evaluation retains
all valid assignments in intention-to-treat. Report missingness
by arm.
