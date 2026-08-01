# Reliable scientific agents: experiment and evidence plan

_Preregisterable design specification for Papers R1 and R2 · 2026-08-01_

## 🎯 Questions, claims, and evidence boundary

The experiments are organized around four hypotheses. Each hypothesis has one
primary contrast, supporting diagnostics, a minimum practically important effect,
and a fallback interpretation. Statistical significance alone is insufficient.

| ID | Primary question | Primary evidence | Supports |
| --- | --- | --- | --- |
| **H1** | Can execution-grounded history warn of terminal scientific failure before terminal checks? | Held-out warning lead time and integrated probability quality at a fixed false-alert budget | R1 predictive claim |
| **H2** | Does warning time contain information about whether a trajectory remains recoverable? | Mixed-effects recoverability curve under standardized recovery probes | R1 failure-dynamics law |
| **H3** | Does adaptive verification improve verified success at matched cost? | Paired policy comparison and success–cost frontier on held-out fault episodes | R2 controller claim |
| **H4** | Does verified process supervision improve recovery beyond recovery SFT and outcome-only learning? | Matched-data, matched-compute comparison on unseen fault compositions with clean-task non-inferiority | R2 learning claim |

The following are explicitly outside the primary claims:

- universal generalization to all scientific agents;
- causal attribution of naturally occurring failures without a controlled fault;
- safety guarantees under unrestricted deployment;
- superiority of a new neural architecture;
- claims based only on agent self-judgment or the training reward channel;
- treating every event in a long trace as an independent replicate.

## 🧮 Experimental units and replication

### Unit hierarchy

| Level | Definition | Statistical role |
| --- | --- | --- |
| Task template | Shared generator, repository, dataset, or workflow skeleton | Grouping unit for split isolation |
| Task instance | One independently parameterized scientific problem | Primary independent sampling unit |
| Fault episode | A sealed fault assignment within a task instance | Treatment unit nested in task |
| Trajectory | One agent run under one assigned condition and seed | Experimental unit for policy assignment |
| Event or step | Repeated observation within a trajectory | Longitudinal measurement, never an independent replicate |
| Checkpoint | Model state produced by one training seed | Training replicate nested in method arm |

The task instance is the principal unit for uncertainty and generalization. Repeated
agent seeds within a task estimate conditional variability, but they do not turn one
task into many independent scientific problems. Confidence intervals therefore use
task-grouped bootstrap or hierarchical models. Any model, task, or training-seed
average must preserve the nesting.

### Blocking and run order

Runs are block-randomized by task instance, model configuration, workload family,
and resource tier. Within a block, condition and policy order are generated from a
seeded schedule. Calendar date, provider version, GPU type, and adjudication batch
are recorded and either blocked or included as nuisance effects.

Never execute all controls before faults, all low severities before high severities,
or one policy in a single calendar batch. That would confound condition with drift,
model-service changes, or operator learning.

### Pilot-driven sample size

The trajectory counts below are starting designs, not post-hoc promises of power.
After the pilot, simulate the complete hierarchical analysis using observed:

- task-level and model-level variance;
- within-task correlation across seeds and conditions;
- fault prevalence and injection fidelity;
- censoring and missing-observation rates;
- baseline failure and recovery rates;
- warning lead-time distribution;
- adjudication ambiguity.

Choose the smallest confirmatory design attaining at least 80% power for the minimum
practical effect while maintaining the required held-out groups. If power is
inadequate, add task instances—not merely more trace steps or checkpoints. Freeze the
simulation code, assumptions, and resulting sample size before confirmatory runs.

## 🧪 R1 experiment family

### R1-P — Instrumentation and label pilot

**Purpose:** determine whether the trace contract observes the intended mechanisms,
whether injectors work without contaminating controls, and whether adjudicators can
apply the label book consistently.

Starting design:

| Factor | Pilot levels |
| --- | --- |
| Workload family | Repository repair; reproducible ML experiment |
| Task instances | Six per workload family |
| Model configuration | One open local model; one API or independently served model |
| Condition | No fault plus six fault mechanisms |
| Replicate seeds | Two per task–model–condition cell |
| Planned trajectories | `2 × 6 × 2 × 7 × 2 = 336` before invalid-run replacement |

Fault severity and injection time are balanced across task blocks using a
mixed-level D-optimal assignment rather than fully crossing every level. Use two
severity levels and three trajectory bands—early, middle, and late. Before running,
inspect the design matrix to confirm that severity and timing main effects and the
prespecified two-factor interactions are estimable at acceptable variance-inflation
levels. Add a small set of medium-severity cases to detect obvious nonlinearity.

Primary pilot measurements:

1. **Injection fidelity:** proportion of assigned faults whose intended immediate
   state difference occurred.
2. **Control contamination:** proportion of no-fault runs exhibiting the injected
   mechanism.
3. **Observation completeness:** fraction of required event and snapshot fields
   present at eligible steps.
4. **Causal label agreement:** agreement on controlled root-cause family after
   excluding injector implementation failures.
5. **Onset and recoverability agreement:** weighted agreement within a declared
   tolerance window.
6. **Replay fidelity:** proportion of deterministic runs reproducing the terminal
   state and trace-derived outcomes.
7. **Cost distribution:** tokens, wall time, tool calls, storage, and adjudication
   minutes per trajectory.

Pilot gates:

| Gate | Initial threshold | Response if missed |
| --- | ---: | --- |
| Injection fidelity | At least 90% per retained mechanism | Repair or drop injector; do not relabel failed injections as faults |
| No-fault contamination | At most 2% for the same mechanism | Fix task or observer; repeat affected controls |
| Required-field completeness | At least 98% | Fix instrumentation before scaling |
| Root-cause agreement | Krippendorff's alpha at least 0.70 | Narrow hierarchy or use set-valued/ambiguous labels |
| Onset agreement | Median absolute difference at most two eligible steps | Redefine onset rule or use interval-censored onset |
| Deterministic replay | At least 95% on replayable cases | Separate non-replayable tasks or repair snapshot boundary |

These thresholds are engineering gates, not claimed population estimates. A failed
gate changes the instrument during the pilot and triggers a fresh pilot sample.

### R1-C — Confirmatory failure-dynamics benchmark

**Purpose:** estimate failure hazard, propagation, warning, attribution, and
recoverability on frozen tasks and fault assignments.

Starting core design:

| Factor | Confirmatory levels |
| --- | --- |
| Workload families | L0 repository repair, L1 ML experiment, L2 signal detection, L3 nonstationary-noise experiment |
| Task instances | Twelve per family; expand to twenty if pilot ICC requires it |
| Primary model configurations | Two materially different model/provider families with a frozen scaffold |
| Conditions | No fault plus six retained fault mechanisms |
| Replicate seeds | Two per task–model–condition cell |
| Core minimum | `48 × 2 × 7 × 2 = 1,344` trajectories |
| Shadow transfer model | Third model on a stratified 20–25% subset |
| External-validity arm | L4 real-data tasks without causal fault labels; sized after pilot |

This design is partially crossed. Every primary task and model sees every retained
fault family and a no-fault control, but severity, persistence, and injection timing
are assigned by a blocked mixed-level D-optimal design whose estimability and
variance-inflation diagnostics are frozen with the manifest. The confirmatory
analysis estimates main effects and prespecified two-factor interactions; it does
not pretend that all high-order interactions are estimable.

#### Frozen split regimes

| Split | Construction | Question |
| --- | --- | --- |
| Development | Pilot-only task templates and injectors | Instrument and model development |
| Unseen task | Entire task templates and instances withheld | Does warning transfer within workload family? |
| Unseen injector | New implementation of a known causal mechanism | Is the model detecting a mechanism rather than injector artifacts? |
| Unseen fault family | One mechanism absent from model training | Can the system warn about unknown failures? Attribution is evaluated as `unknown`, not forced into a known class |
| Unseen composition | New pair or sequence of individually familiar faults | Does warning and multilabel attribution survive interaction? |
| External L4 | Naturally occurring real-data failures | Does the measured signal survive outside controlled injection? |

Task templates, repositories, datasets, and near-duplicate solutions are grouped
before splitting. Text or code similarity is used as a leakage screen, followed by
manual audit of borderline clusters.

### R1-E1 — Failure hazard and propagation

**Why:** end-task accuracy does not reveal when divergence begins or how subsystem
errors compound. This experiment establishes whether there is a reproducible
temporal law worth publishing.

Exposure variables:

- trajectory age and normalized remaining horizon;
- fault family, severity, persistence, and injection band;
- accumulated unresolved state divergence;
- evidence freshness and provenance breaks;
- model, workload, scaffold, and resource pressure.

Outcomes:

- first adjudicated failure-onset interval;
- terminal execution failure;
- terminal scientific invalidity;
- propagation edges between typed subsystems;
- irreversible-state threshold;
- censoring reason.

Analysis:

- Kaplan–Meier or cumulative-incidence summaries for description;
- discrete-time mixed-effects hazard model with complementary log–log link;
- task-instance and template random effects;
- model/workload/fault fixed effects and prespecified interactions;
- interval-censoring sensitivity analysis for uncertain onset;
- clustered bootstrap confidence intervals over task templates;
- competing-risk sensitivity for policy termination versus scientific failure.

Evidence required for the R1 failure-law claim:

1. The preregistered law has the predicted direction and a practically meaningful
   magnitude in the pooled confirmatory set.
2. Its grouped 95% confidence interval excludes the null.
3. The direction replicates in at least two workload families.
4. It survives removal of terminal-adjacent events and one alternative onset rule.
5. It is not explained solely by one model, task template, or injector.

If no law passes all five conditions, R1 becomes a benchmark and negative
measurement paper only if the artifact remains independently valuable.

### R1-E2 — Calibrated pre-terminal warning

**Why:** merging a predictive baseline into R1 prevents the benchmark from being
only descriptive and tests whether its traces contain actionable information.

Models:

1. terminal verifier only;
2. runtime-rule alarms;
3. agent self-confidence;
4. text-only trace critic;
5. typed-event temporal model without state differences;
6. typed-event temporal model with state differences and evidence provenance;
7. oracle fault metadata as an unattainable upper bound.

Primary endpoint:

> Median normalized warning lead time among terminally invalid trajectories at an
> operating point with no more than 5% false warnings on no-fault valid
> trajectories.

The operating threshold is selected on the calibration split, never on the test
set. A warning counts only if it precedes the first terminal-only signal and the
declared minimum reaction window.

Secondary endpoints:

- time-dependent AUPRC and AUROC;
- integrated Brier score and log loss;
- calibration intercept and slope with reliability plots;
- risk–coverage and abstention utility;
- lead-time distribution by fault and workload;
- alert burden per successful trajectory;
- performance after environment or provider shift.

Minimum evidence for H1:

- at least 10% of normalized trajectory horizon median lead-time gain, or at least
  two actionable steps when horizons are short, over the strongest non-oracle
  baseline at the fixed false-warning budget;
- at least 10% relative improvement in integrated Brier score over that baseline;
- grouped 95% intervals excluding no improvement for both primary quantities;
- improvement on unseen tasks and unseen injector implementations;
- no terminal leakage under the event-removal audit.

The two thresholds encode practical actionability and probability quality. If only
one passes, claim either early discrimination or calibrated probability improvement,
not both.

### R1-E3 — Root-cause attribution

**Why:** warning without a plausible cause may not support an efficient recovery
choice, but causal truth is only available for controlled faults.

Attribution is hierarchical:

```text
known single mechanism
known multi-mechanism composition
unknown mechanism
ambiguous among a declared set
insufficient evidence
```

Metrics include macro-F1 over known top-level families, top-k recall, hierarchical
distance, multilabel set F1 for compositions, unknown-family detection, and
calibration of the cause distribution. Report results both including and excluding
ambiguous cases.

Minimum supporting evidence:

- at least 0.10 absolute macro-F1 improvement over the strongest text-only or
  rule-based baseline on unseen injector implementations;
- positive improvement with a grouped 95% interval;
- no more than a predeclared degradation in unknown-family detection;
- confusion concentrated in scientifically plausible neighboring categories rather
  than one majority class.

Attribution is secondary to H1. Failure here narrows R1 to warning and dynamics; it
does not invalidate a strong benchmark.

### R1-E4 — Recoverability window

**Why:** H2 requires a controlled measure of whether earlier warning creates an
opportunity to recover, without introducing R2's learned policy.

At eligible snapshots, apply a standardized, family-appropriate canonical recovery
probe at early, middle, and late delays after adjudicated onset. The probe is fixed
before confirmatory data and is not chosen by a learned controller. Each faulty
trajectory is paired with a no-intervention continuation from the same snapshot
where replay semantics permit.

Outcomes:

- immediate fault removal;
- consistent downstream state;
- final scientific validity;
- recovery cost and additional damage;
- failure of the probe or verifier.

Analysis uses a mixed-effects logistic or survival model with recovery as the
outcome and normalized delay, fault family, severity, and workload as fixed effects;
task and fault episode are random effects. The primary quantity is the marginal
change in verified recovery probability per 10% delay in normalized horizon.

Minimum evidence for H2:

- a monotonic decline in recovery probability with delay whose grouped 95%
  interval excludes zero in the pooled analysis;
- the same direction in at least two workload families;
- predictive value of lead time after adjustment for severity, fault family, task
  difficulty, and model;
- sensitivity to at least one alternative onset interval and recovery definition.

This experiment supports an empirical recoverability law. It does not claim that
the canonical probe is an optimal recovery policy.

## 🔧 R2 experiment family

R2 begins only after R1's raw benchmark, split manifest, adjudicator versions, and
calibration protocol are frozen. R2 may train on R1 training/development traces but
cannot inspect R1's sealed test outcomes during method development.

### R2-P — Offline controller feasibility

**Purpose:** determine whether observable R1 belief states contain enough
information for verification decisions before live intervention collection.

Build a replay dataset of eligible decision points with action availability, cost,
subsequent evidence, and outcome. Use inverse-propensity or doubly robust estimates
only where logging support is adequate; unsupported counterfactual actions remain
unknown rather than imputed as truth.

Compare:

- never verify;
- always verify;
- fixed periodic or terminal verification;
- random verification at matched budget;
- threshold on R1 failure probability;
- myopic value of information;
- learned budget-aware controller;
- oracle fault-aware controller as an upper bound.

Feasibility gate:

- the value-of-information or learned controller must improve estimated verified
  success at matched cost over the strongest fixed non-oracle baseline;
- effective sample size under off-policy weighting must exceed a preregistered
  minimum;
- conclusions must agree in sign under direct replay subsets and a conservative
  unsupported-action sensitivity bound.

### R2-C1 — Randomized online verification policy study

**Why:** offline estimates are vulnerable to support failure and policy-induced
observation bias. A blocked online comparison is necessary for H3.

Experimental block:

> One task instance × model configuration × sealed fault episode with a forkable
> pre-decision snapshot.

Within each block, run the non-oracle policies from the same snapshot with randomized
order. If exact replay is impossible, use independently generated matched episodes
and include episode as a random effect.

Starting design:

| Element | Minimum plan |
| --- | --- |
| Independent task instances | At least 32 across four workload families |
| Fault episodes | At least 120 balanced across retained families and severities |
| Non-oracle policies | Six: never, always, fixed, random, threshold, adaptive |
| Replicate executions | Determined by pilot ICC; minimum two when stochastic |
| Primary comparison | Adaptive versus strongest non-oracle baseline at matched cost |

Primary endpoint:

> Difference in independently verified scientific success at a common verification
> and human-effort budget.

Supporting endpoints:

- area under the verified-success versus cost frontier;
- irreversible damage rate;
- false intervention on no-fault tasks;
- latency, compute, tool calls, and human escalation;
- calibration after policy-induced selective observation;
- regret relative to oracle within replayable simulations.

Minimum evidence for H3:

- at least five percentage points absolute verified-success improvement over the
  strongest non-oracle baseline at the matched budget;
- paired or hierarchical 95% interval excluding zero;
- no increase in irreversible damage;
- benefit on at least two workload families and no catastrophic reversal on any
  prespecified family;
- a non-dominated region of the success–cost frontier, not merely one hand-selected
  operating point.

### R2-C2 — Verified-intervention dataset construction

**Why:** R2 training needs process labels whose meaning survives outside the current
controller.

For each eligible transition, record:

- observable history and calibrated belief before action;
- all actions available under policy and resource constraints;
- selected action and selection propensity;
- action cost, state difference, and immediate postcondition;
- final scientific outcome under an independent adjudicator;
- replayed counterfactual outcomes when technically valid;
- ambiguity, verifier disagreement, and missing evidence;
- source split, deduplication cluster, and eligibility for each training view.

Sampling intentionally oversamples rare fault/action combinations for training but
retains prevalence weights for population evaluation. Training balance must never be
mistaken for deployment prevalence.

Dataset gate:

- each primary recovery action has adequate independent episode coverage;
- no single task template contributes more than a predeclared share of an action
  class;
- at least two adjudicator implementations or a human audit cover high-reward cases;
- duplicate and near-duplicate clusters are isolated by split;
- ambiguous and failed transitions remain available for audit.

### R2-T — Matched post-training experiment

**Why:** this is the causal comparison supporting H4. Adding more algorithms does
not substitute for matched data, compute, and evaluation.

Primary arms:

| Arm | Training information | Scientific purpose |
| --- | --- | --- |
| Base | None | Establish unadapted recovery and clean-task behavior |
| Successful-only SFT | Valid terminal trajectories only | Test ordinary successful imitation |
| Recovery SFT | Failed and verified-repaired transitions | Isolate the value of recovery examples |
| Outcome-only optimization | Independent terminal reward | Test whether terminal feedback is sufficient |
| Process-supervised optimization | Independent step and transition evidence | Test process-information value |
| Process plus calibrated abstention | Process evidence plus held-out calibration | Test selective safety and over-intervention |

Use one open model family and parameter-efficient adaptation for the primary study.
Run at least three independent training seeds per trainable arm. Match the number of
source trajectories, effective tokens, optimizer steps, and evaluation budget;
report any unavoidable mismatch.

The checkpoint selection rule is fixed before test evaluation and uses only
development outcomes. Testing every checkpoint and choosing the best test result is
prohibited.

#### Evaluation sets

| Set | Purpose |
| --- | --- |
| Seen task and seen fault | Verify basic learning and debug failure |
| Unseen task templates | Test task transfer |
| Unseen injector implementations | Test mechanism rather than artifact learning |
| Unseen fault compositions | Primary recovery generalization test |
| Unknown fault family | Test abstention and safe escalation |
| Clean tasks | Non-inferiority and unnecessary-intervention test |
| Verifier substitution | Reward-channel robustness |
| Environment/provider shift | Calibration and operational robustness |

Primary H4 endpoint:

> Verified recovery rate on unseen fault compositions, comparing process-supervised
> optimization with the stronger of recovery SFT and outcome-only optimization at
> matched evaluation cost.

Co-primary safety endpoint:

> Clean-task verified success under a non-inferiority margin of two percentage
> points relative to the same base model evaluation protocol.

Minimum evidence for H4:

- at least five percentage points absolute improvement in verified recovery over
  the strongest matched comparator;
- grouped 95% interval excluding zero;
- clean-task lower confidence bound above the `−0.02` non-inferiority margin;
- selective-risk or risk–coverage improvement after calibration;
- improvement survives verifier substitution and removal of near-duplicate traces;
- no single training seed, task family, or fault family explains the result.

If process supervision does not pass, report the negative comparison. R2 may still
be publishable around the controller and verified-intervention dataset if H3 and the
artifact gates are strong.

### R2-A — Mechanism and ablation study

Run only after the primary matrix. Use a compact factorial ablation over:

| Factor | Off | On |
| --- | --- | --- |
| Calibrated R1 belief | Raw score or no belief | Calibrated hazard and cause distribution |
| Active verification | Fixed/none | Budget-aware choice |
| Process supervision | Terminal outcome only | Independently verified transitions |

A full `2 × 2 × 2` design estimates main effects and interactions with eight arms.
Where an arm is identical to a primary arm, reuse the exact checkpoint and result;
do not retrain it under a new label. The critical interaction is active verification
× process supervision: does training help specifically because the policy learns
which evidence to obtain, rather than simply learning better generic task behavior?

## 📐 Statistical analysis plan

### Estimands

Define each estimand before implementation:

- population and task-family weighting;
- treatment or policy contrast;
- outcome definition and time horizon;
- handling of invalid, censored, and missing runs;
- unit of analysis;
- covariate and blocking adjustment;
- uncertainty method.

Primary estimands use equal weight across task families and task instances unless a
deployment distribution is explicitly defined. Report prevalence-weighted estimates
as secondary analyses.

### Hierarchical models

Use generalized mixed models appropriate to the outcome:

- complementary log–log mixed model for discrete failure hazard;
- logistic mixed model for binary verified success and recovery;
- linear or generalized mixed model for cost and lead time;
- hierarchical multinomial or multilabel model for attribution where appropriate.

Include random intercepts for task template and task instance. Add model/provider or
fault random effects only when enough levels exist; with two model families, treat
model as fixed. Repeated seeds and policies are nested within task/fault blocks.

### Uncertainty and resampling

Primary nonparametric intervals use hierarchical bootstrap:

1. resample task templates;
2. resample task instances within template;
3. retain all repeated trajectories or resample them within instance as specified;
4. recompute the complete metric, including threshold selection only from the
   calibration split.

Use at least 2,000 bootstrap replicates for final intervals. Report bootstrap seed
and convergence diagnostics. Compare with model-based intervals as a sensitivity
analysis.

### Multiplicity

H1–H4 each have one primary comparison. Secondary model/fault/workload contrasts
use Benjamini–Hochberg false-discovery-rate control within declared families.
Exploratory analyses are labeled exploratory and do not inherit confirmatory
language. Do not select the best subgroup and report it as a general effect.

### Missingness and invalid runs

Classify every non-complete trajectory as:

```text
policy termination
budget exhaustion
infrastructure failure
injector failure
observer failure
verifier failure
scientific task failure
unknown
```

The intention-to-treat analysis retains all valid assignments and treats failures
according to the preregistered endpoint rule. A per-protocol sensitivity analysis
may exclude injector or infrastructure failures, but it cannot replace the primary
analysis. Report missingness by arm and test whether it is policy-dependent.

## 🧾 Evidence-to-claim matrix

| Experiment | Required result | Figure/table | Claim enabled | Claim prohibited if absent |
| --- | --- | --- | --- | --- |
| R1-P | Reliable injectors, trace completeness, label agreement | Pilot quality table | Instrument is usable | Benchmark-scale causal language |
| R1-E1 | Replicated preregistered hazard/propagation law | Hazard and propagation figure | Failure has transferable temporal structure | General failure law |
| R1-E2 | Earlier and better-calibrated warning on held-out groups | Lead-time and calibration figure | Pre-terminal warning is actionable | Early warning beyond terminal checks |
| R1-E3 | Attribution improvement on unseen injectors | Hierarchical confusion figure | Cause belief adds information | Reliable diagnosis |
| R1-E4 | Delay predicts recovery after adjustment | Recoverability-window figure | Warning time has recovery value | Early detection improves recovery opportunity |
| R2-C1 | Adaptive policy dominates matched-cost fixed baselines | Success–cost frontier | Active verification adds value | Learned controller superiority |
| R2-C2 | Auditable, diverse intervention transitions | Dataset composition and audit table | Process dataset is reusable | General recovery training evidence |
| R2-T | Process arm improves unseen composition recovery and passes clean non-inferiority | Primary recovery and regression table | Verified process supervision adds value | Safer or more reliable post-training |
| R2-A | Prespecified main effects/interactions | Ablation plot | Mechanistic attribution of gains | Component-causality narrative |

## 📊 Planned figure and table inventory

### R1 primary displays

1. System and evidence-contract overview.
2. Benchmark workload, fault, model, and split composition.
3. Failure hazard and propagation by trajectory age.
4. Warning lead-time versus false-alert frontier.
5. Calibration and risk–coverage under held-out transfer.
6. Root-cause attribution matrix including unknown and ambiguous cases.
7. Recoverability probability versus warning/intervention delay.
8. Artifact quality, missingness, agreement, and cost table.

### R2 primary displays

1. Calibrated belief, verification, recovery, and adjudication loop.
2. Verified-success versus cost frontier for controller policies.
3. Intervention dataset composition and evidence-channel audit.
4. Six-arm verified-recovery comparison on unseen compositions.
5. Clean-task non-inferiority and false-intervention plot.
6. Risk–coverage and abstention calibration.
7. Verifier-substitution and reward-disagreement analysis.
8. Factorial ablation of belief, verification, and process supervision.

## ✅ Freeze and audit checklist

Before each confirmatory run, verify:

- [ ] Hypothesis and primary contrast are registered.
- [ ] Task, model, scaffold, fault, severity, and timing manifests are hashed.
- [ ] Experimental unit and block variables are present in the run manifest.
- [ ] Randomization schedule and seed are archived but treatment concealment is
      maintained where required.
- [ ] Training, calibration, and test groups pass leakage and near-duplicate audits.
- [ ] Primary endpoint code passes golden tests.
- [ ] Missingness, censoring, and invalid-run rules are executable.
- [ ] Sample-size simulation and assumptions are frozen.
- [ ] Adjudicators and reward channels are versioned and separated.
- [ ] Compute, storage, API, and human-adjudication budgets are approved.
- [ ] Stopping rules and fallback claims are documented.

After collection, verify:

- [ ] Raw traces are locked before primary analysis unblinding.
- [ ] All assigned runs appear in a disposition table.
- [ ] No threshold or checkpoint was selected on the test set.
- [ ] Grouped uncertainty respects task/template nesting.
- [ ] Primary, secondary, sensitivity, and exploratory outputs are labeled.
- [ ] Negative outcomes and failed validations remain in the artifact.
- [ ] Each manuscript claim maps to the evidence-to-claim matrix.
