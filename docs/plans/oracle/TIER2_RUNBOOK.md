# Tier 2 runbook — ORACLE-Paired (cluster)

_Cannot run here: needs Prometheus, the NuBench DynEdge checkpoint and the GPU nodes.
Everything below is dispatchable once the Tier-1 predictions (PREREGISTRATION.md §5–§6)
are frozen._

## Inputs
- `ORACLE/src/oracle_paired` (events, geometry, response, interventions, matching, strata);
  `ORACLE/scripts/nubench/` (checkpoint loading, hooks, rescoring); `ORACLE/docs/IMPLEMENTATION_PLAN.md` WP9.
- Monitors and analysis: `experiments/oracle/oracle_cov/{monitors,baselines,consequence,abstain,report}.py`
  are subject-agnostic; a Tier-2 subject needs only `represent(X, stage)`, `outputs(X)`
  (angular error target) and `jac_recon`/`jac_output` via autograd on the hooked stage.

## Steps
1. **Parity gate (WP9-0).** Re-infer the released Hexagon Ice LE DynEdge on CPU and GPU;
   CPU = GPU ≠ release ⇒ preprocessing/version skew, CPU ≠ GPU ⇒ k-NN backend.  A stable
   offset across severity strata is a predeclared tolerance; severity-dependent drift keeps the gate closed.
2. **Production (WP9-1).** One seeded LeptonInjector run; Prometheus photon propagation in
   ORCA and ARCA (both water); `oracle_paired.response` for the NuBench detector response;
   cells: `clean`, N1–N5 (module loss, hit thinning, jitter, gain drift, noise rate),
   S1–S5 (low-density strata in energy/topology/vertex/multiplicity, estimated from training
   only), U1–U4 (τ double-bang, through-going μ, pile-up, correlated noise).  Content-matched
   clean twins for every N cell (`oracle_paired.matching`).  `injection_id` is the pairing key.
3. **Hooks (WP1 on DynEdge).** Four convolution blocks + post-processing + readout; store
   per-window moments or sketches (`hooks.benchmark` for C3).
4. **Subject adapter.** Implement `DynEdgeSubject` with the interface above; `jac_output` by
   autograd of the direction output; the pullback-whitened monitor is not available
   (no Σ) — the Jacobian-projected monitor carries the deployability claim.
5. **Run once.** `run_tier1`-style loop over cells with 20 confirmatory seeds; score the
   frozen prediction table (§6) as one summary; C1/C2/C4 as in Tier 1 with K = angular error.
6. **Report.** Same `analyze_tier1` layout; the bridge score is the headline.

## Compute
Production ~ cluster-days (see `ORACLE/docs/archive/DATASET_PRODUCTION_PLAN.md`); inference and
monitoring: hours on one GPU.
