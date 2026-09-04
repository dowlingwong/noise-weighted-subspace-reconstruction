# Tier 2 runbook — ORACLE-Paired (cluster)

_Cannot run here: needs Prometheus, the NuBench DynEdge checkpoint and the GPU nodes.
Everything below is dispatchable once the Tier-1 predictions (PREREGISTRATION.md §5–§6)
are frozen._

_Extended 2026-09-03 with the simulation mechanics (§A–§F below) after reading the
Prometheus source. **One finding revises §Steps.2: the ORCA/ARCA pair is not viable
as written** — see §B._

## Inputs
- `ORACLE/src/oracle_paired` (events, geometry, response, interventions, matching, strata);
  `ORACLE/scripts/nubench/` (checkpoint loading, hooks, rescoring); `ORACLE/docs/IMPLEMENTATION_PLAN.md` WP9.
- Monitors and analysis: `experiments/oracle/oracle_cov/{monitors,baselines,consequence,abstain,report}.py`
  are subject-agnostic; a Tier-2 subject needs only `represent(X, stage)`, `outputs(X)`
  (angular error target) and `jac_recon`/`jac_output` via autograd on the hooked stage.
- **New:** `ORACLE/src/prometheus_simulation/` — injection sharing, re-centring, medium
  override, pair feasibility, the pairing null, readout, baseline reconstruction
  and a headless analysis report (`analyze.py`). Configs carry `[nubench]`/`[ours]`/`[ASK]`
  provenance on every value.

## Steps
1. **Parity gate (WP9-0).** Re-infer the released Hexagon Ice LE DynEdge on CPU and GPU;
   CPU = GPU ≠ release ⇒ preprocessing/version skew, CPU ≠ GPU ⇒ k-NN backend.  A stable
   offset across severity strata is a predeclared tolerance; severity-dependent drift keeps the gate closed.
2. **Production (WP9-1).** One seeded LeptonInjector run; Prometheus photon propagation in
   ~~ORCA and ARCA~~ **ARCA and IceCube** (both water — see §B) ;
   `oracle_paired.response` for the NuBench detector response;
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
monitoring: hours on one GPU. Simulation itself: 1–10 CPU-s per event per geometry, so
100k × 2 geometries ≈ 55–550 CPU-h, embarrassingly parallel, plus the null at reduced N.
Ice/PPC is GPU-oriented; the ice arm (pair P3) is the expensive one on CPU.

---

# §A. Gates before any production run

| Gate | Statement | Status |
|---|---|---|
| **V0** | CPU re-inference of the released DynEdge matches the released predictions to 0.01 | **OPEN** (`DATASETS.md`; = Steps.1) |
| **G1** | Our §3.2 pulse-count distribution matches theirs per energy decade (stated KS distance) | not started |
| **G2** | Our per-pulse charge and time-residual distributions match theirs | not started |
| **G3** | The **released** DynEdge, applied to **our** events, reproduces published Table-6 metrics within tolerance | blocked by V0 |

G3 is the load-bearing one. NuBench's detector response (§3.2 of arXiv:2511.13111) was
**never released as code**; ours is a re-implementation. Any mismatch is an acquisition
shift indistinguishable from the N-contract violations we claim to detect — the monitor
would fire on our own emulator and we could not tell. If G3 fails, Tier 2 must train its
own models and the cost estimate rises by roughly an order of magnitude.
**No C1–C4 number from a released checkpoint on our events may be reported until G3 passes.**

# §B. Choosing the pair — and why ORCA/ARCA fails

Measured from the geofiles shipped in `prometheus/resources/geofiles/`
(`prometheus_simulation/geometry.py`, run it to reproduce):

| geo | modules | offset z (m) | r_horiz (m) | half-height (m) | header medium |
|---|---|---|---|---|---|
| orca (Flower S) | 3300 | +95.4 | 100.2 | 95.4 | mediterranean |
| arca (Flower L) | 2070 | −3194.0 | 500.9 | 306.0 | mediterranean |
| trident (Flower XL) | 24220 | −3090.0 | 1949.6 | 285.0 | water |
| pone_triangle | 60 | −0.0 | 57.7 | 500.0 | water |
| gvd (Cluster) | 288 | −970.0 | 60.0 | 270.0 | water |
| icecube (Hexagon) | 5160 | −1972.0 | 596.3 | 518.7 | ice |

Pairing forces ONE injection cylinder, sized on the larger detector. The smaller one then
sees only part of it:

| pair | scale ratio | instrumented-volume fraction |
|---|---|---|
| **orca ↔ arca** (as in Steps.2) | **5.00** | **orca 0.005**, arca 0.385 |
| arca ↔ icecube | 1.19 | arca 0.196, icecube 0.470 |
| arca ↔ trident | 3.89 | arca 0.041, trident 0.577 |

ORCA would see **0.5%** of the injected volume: 99.5% of the CPU spent on events it cannot
detect, and the surviving sample is a heavily selected sliver, not a paired sample.
ORCA is also the only NuBench dataset capped at 10³ GeV rather than 10⁵, so at a fixed
spectrum a Flower S comparison is already a support shift — an S-contract change — not a
geometry change. **Use ARCA ↔ IceCube.** Full ranking in
`experiments/oracle/paired_geometry/config/geometries.yaml:pairs`.

**P3 (Hexagon ↔ Hexagon Ice LE) is a required negative control, not an option.** NuBench ran
all six geometries in **water** and added **one ice** dataset on the Hexagon geometry, so
without the medium-only control any geometry effect can be read as a medium effect. Note the
shipped `.geo` headers disagree with what NuBench simulated (orca/arca say "mediterranean",
icecube says "ice"); `run_geometry.override_medium` rewrites the `Medium:` line, which is
what selects PPC (ice) vs olympus (water).

# §C. The four mechanics that break naive injection re-use

Read from the Prometheus source, not from its docs:

1. **No shared coordinate frame.** `detector.py:162` defines the detector offset as the mean
   module position. `lepton_injector_utils.apply_detector_offset` adds it to every vertex
   **in place in the injection file**, and **only on the `inject=True` path**;
   `LI_injection.injection_from_LI_output` takes `**_` and **ignores** `detector_offset` when
   loading. So a stored injection is in the reference detector's absolute frame and re-using
   it verbatim puts every event 1.2 km from IceCube. → `geo.recentre_delta` +
   `run_geometry.recentre_injection` (mutates a copy; the reference file is the pairing key
   and is immutable).
2. **The cylinder belongs to one detector.** → `geo.shared_cylinder`, sized on the larger,
   containment reported for both.
3. **There is no zero-point.** Photon propagation is re-run per geometry and is seedable only
   to Poisson level, so the same event in the *same* detector gives different pulses. Without
   that floor a cross-geometry displacement has no scale. → §D.
4. **Depth changes with the geometry.** The Earth model is built at `-detector_offset[2]`
   (`lepton_injector_utils.py:107`), so re-centring changes the overburden and LeptonInjector
   weights do not transfer across the pair. → Tier 2 reports **unweighted, per-event,
   truth-referenced** metrics only. Nothing flux-weighted crosses a geometry boundary.

# §D. Run order — the null goes first

```bash
export ORACLE_SIM_ROOT=/ceph/dwong/oracle_sim      # never inside the repo
bash experiments/oracle/paired_geometry/bootstrap_prometheus.sh   # clones + pins + smoke-tests
source $ORACLE_SIM_ROOT/.venv/bin/activate         # h5py/numpy/pyyaml

G=$ORACLE_SIM_ROOT/prometheus/resources/geofiles
O=$ORACLE_SIM_ROOT/runs/P1_scale_matched

# 1. ONE injection, cylinder sized for the larger detector
python -m experiments.oracle.paired_geometry.make_injection \
    --pair P1_scale_matched --geodir $G --out $O/injection --execute

# 2. replay through each geometry (recentres, overrides medium)
for ds in flower_l hexagon; do
  python -m experiments.oracle.paired_geometry.run_geometry \
      --pair P1_scale_matched --dataset $ds \
      --injection $O/injection/*.h5 --lic $O/injection/*.lic \
      --geodir $G --out $O/$ds --photon-seed 1 --execute
done

# 3. THE NULL — same geometry, same events, different photon seed
python -m experiments.oracle.paired_geometry.run_geometry \
    --pair P1_scale_matched --dataset hexagon \
    --injection $O/injection/*.h5 --lic $O/injection/*.lic \
    --geodir $G --out $O/hexagon_seed2 --photon-seed 2 --execute
```

**Run step 3 first, at ~1k events.** If photon stochasticity alone moves the representation
as much as the geometry change does, the pair is unusable and that is learned for ~1% of the
cost. `check_pairing.effect_over_null` returns the verdict; every cross-geometry displacement
is reported as a **multiple of the null's p95**, never as a raw distance.
`check_pairing.check_identity` separately asserts that truth matches event-for-event up to
the re-centring delta — if it fails, nothing downstream is interpretable.

# §E. Deviations to declare in the paper

1. Spectral index, zenith/azimuth range, injection volume, Earth model, per-OM noise rate,
   per-dataset TTS and the ice angular-acceptance curve are **not published**. Our values are
   choices, marked `[ASK]` in `paired_geometry/config/`.
2. The §3.2 response code was never released; ours is a re-implementation calibrated only to
   the distributions in G1–G2.
3. Weights do not transfer across the pair (§C.4) — unweighted metrics only.
4. Pairing is exact at truth level, approximate at light-yield level (§C.3), quantified by the
   null.
5. Prometheus is LGPL-2.1. Used as a library; our changes stay in this repo, the clone is
   never patched.

# §F. What we actually need from the NuBench authors

_Revised 2026-09-03 after checking the licences directly._

## The licences, verified

| Artefact | Licence | Consequence |
|---|---|---|
| Prometheus (`Harvard-Neutrino/prometheus`) | **LGPL-2.1** (`LICENSE.md`) | simulate freely; keep our changes out of the clone |
| Geometry files (`resources/geofiles/`) | ships with Prometheus | **the six NuBench geometries come from here, not from NuBench** |
| GraphNeT (`graphnet-team/graphnet`) | **Apache-2.0** | DynEdge/ParticleNet **architecture and training code are free to use** |
| NuBench repo (`graphnet-team/NuBench`) | **no LICENSE file** (404 on `main` and `master`) | datasets, predictions and **trained weights are unlicensed** |
| NuBench README | asks that NuBench, Prometheus and GraphNeT be cited | that is the whole stated obligation |

The split that matters: **the architecture is Apache-2.0, only the weights are
unlicensed.** So training our own DynEdge is unencumbered; using theirs is not.

## Three postures, three dependencies

| Posture | What we use of theirs | What we need from them |
|---|---|---|
| **(a) Self-simulated** — our events, our response, **our trained models** | nothing but the citation | **nothing** |
| **(b) Their models** — our events, their checkpoints | unlicensed weights + predictions | a licence, **and** every §3.2 parameter (gate G3) |
| **(c) Their events** — their injection files replayed through other geometries | unlicensed injection files | a licence + the files, **if they kept them** |

**Posture (a) needs nothing from them.** Going fully self-simulated does not
merely dodge the pairing problem — it dissolves the licence problem, because the
licence problem was only ever about their artefacts. Everything posture (a)
touches is LGPL-2.1 or Apache-2.0 or a published fact.

## Why (a) is the strong version, not the fallback

Training our own subject means **we know Σ̂**, the covariance the model was
trained under. Their released checkpoint does not come with one, which is why
Steps.4 above says the pullback-whitened monitor "is not available (no Σ)" and
falls back to the Jacobian-projected monitor.

If we train the subject, Σ̂ is ours to set — including deliberately wrong, for
the κ(Σ̂⁻¹Σ) sweep. That makes Tier 2 carry the covariance-geometry result on a
real neutrino telescope instead of only on the controlled simulator. It moves
Tier 2 from "domain transfer" to the ⭐ novelty item of the August audit.

Cost: GPU-days to train DynEdge on ~10⁵–10⁶ events, versus free inference on a
released checkpoint. That is the trade, and it buys the headline claim.

It is also, by some distance, **the lowest-gatekeeping testbed on the list.**
SPINE is collaboration data, MicroBooNE open data carries terms, Panda is a
collaboration question. NuBench-via-Prometheus asks nobody for permission.

## Still worth one email — for one question

Even on posture (a), ask this, and ask it first:

> **Did you keep the LeptonInjector injection files, or the seeds that generated them?**

If yes, posture (c) opens and it is strictly better than (a): the same *paired*
events, and directly comparable to their published numbers. One question, large
upside, no cost if the answer is no.

Bundle the rest as a short numbered list, ordered by what it would change:

2. **Per-OM noise rate.** Not published. Sets the pulse-multiplicity floor,
   hence the <4-pulse cut efficiency, hence event yield and the low-energy
   sample composition. The most consequential single unknown.
3. **Per-dataset TTS.** The paper says the merging window is "set to the TTS per
   dataset" without giving the values; it sets pulse count and charge.
4. **The ice angular-acceptance curve.** "Reduced for down-going photons" is
   qualitative; no function is given. Only affects the P3 medium control.
5. **Injection configuration:** spectral index, zenith/azimuth range, ranged vs
   volume mode, cylinder dimensions, Earth model, cross-section splines.
6. **A licence statement** on datasets, predictions and checkpoints — CC-BY-4.0
   for data, MIT/Apache-2.0 for models is the usual pairing. Needed only if we
   ever move to posture (b) or (c), but cheap to ask alongside.
7. **The V0 discrepancy:** what hardware, precision and GraphNeT version
   produced the released predictions? Our CPU re-inference differs event by
   event.

## What items 2–5 actually buy

Nothing for internal validity. Our claims are **within-event, paired A-vs-B**
comparisons; they never reference a NuBench number, so they stand whatever the
authors say. Items 2–5 buy the right to one sentence:

> "Our re-simulation reproduces NuBench's Table-6 metrics on the Hexagon Ice LE
> configuration to within X."

That sentence is worth having — it pre-empts the obvious referee question — but
it is a credibility bonus, not a dependency. Without it the honest label is
**"NuBench-geometry"**, not "NuBench-configured", and every deviation goes in
§E. Decide the label now, because it changes how §3 of the paper is written.

**Offer:** we are building the event-paired cross-geometry extension their design
cannot express, and will share the configuration and the paired dataset back.
Co-authorship or acknowledgement to their preference; ask whether they would
rather it land as a NuBench PR.

**Tone:** short. One paragraph of context, then the numbered list. They are a
benchmark team; unpublished-parameter questions are expected traffic.
