# Paper 3 collaboration package

This directory is now a serious proposal package, not a manuscript or a claim
that the confirmatory study is complete.

## Send/read in this order

1. `PROPOSAL.md` — the primary 4–6 page document: problem, gap, three claims,
   experiment table, resources, risks, stopping rules, and collaborator/advisor
   asks.
2. `NUBENCH_FEASIBILITY_AUDIT.md` — checkpoint/task/geometry availability,
   exact archive sizes, embedding hook, compute, checksums, and V0/P0 status.
3. `results/nubench_hexagon_ice_le_dynedge/RESULT.md` — the close-but-not-strict
   published-metric reproduction, checkpoint compatibility blocker, and
   labeled exploratory module-dropout curve.
4. `CLOSEST_WORK_COMPARISON.md` and `references.bib` — novelty boundary and
   working bibliography.
5. `COLLABORATOR_REVIEW_BRIEF.md` — one-page meeting pre-read.
6. `PAPER3_PLAN.md` — retained as the detailed technical appendix, not the
   primary proposal.

## Current readiness

- **Ready to send to the postdoc for technical/scoping review.** There is now a
  concrete public cell, resource audit, reproducible code, preliminary curve,
  and a specific checkpoint-compatibility question for a NuBench expert.
- **Not ready to present as an execution-ready or result-bearing paper.** The
  full released prediction rescoring is close to Table 6 but misses the strict
  0.01 tolerance; CPU checkpoint re-inference also differs event by event from
  released predictions. V0 remains open.
- **The pilot is exploratory.** It demonstrates sensitivity and estimates
  compute, but its selected high-multiplicity sample is not representative and
  cannot support the three scientific claims before V0 and registration.

## Evidence gates

- **V0-A:** explain or agree a tolerance for the ≤0.055° / ≤0.051 percentage
  point difference between released-prediction rescoring and Table 6.
- **V0-B:** recover the author environment/recipe or document acceptable
  backend variation for checkpoint re-inference.
- **E1:** domain-review perturbations, choose smallest relevant effects and
  equivalence margins, power the event/replicate count, and freeze the analysis.
- **Track B:** confirm input modality, raw-input and embedding access,
  real/simulated status, outcomes, covariates, compute, permissions, and the
  minimum collaborator-domain result.

Do not describe NuBench geometries as a causal detector-layout experiment, pool
waveform and hit-level severities, demand nuisance-undecodable embeddings, call
miscalibration a software bug, or activate anomaly detection without a
domain-defined anomaly class and decision.
