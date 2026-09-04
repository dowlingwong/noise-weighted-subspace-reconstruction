# ORACLE-Paired — same event, different geometry

> **Superseded for simulation mechanics.** `src/prometheus_simulation/` is now the
> canonical package (geometry, injection, replay, readout, physics record, agent
> prompt, notebook). This directory keeps only the Tier-2 experiment design: the
> pair ranking, the NuBench response gates G1-G3, and the cell definitions.

Tier 2 of the Paper 3 plan (`docs/EXPERIMENT_PLAN_PAPER3.md` §6). Produces the
event-paired cross-geometry dataset that **NuBench does not have**, by
re-running one Prometheus injection through two detector geometries.

Runbook: `docs/plans/oracle/TIER2_RUNBOOK.md`.

## Why re-simulate at all

NuBench's seven released datasets are independently injected. Comparing
geometry A to geometry B therefore compares two *populations*, so any change in
reconstruction error mixes the geometry effect with the event-sample
difference. ORACLE's consequence variable K is a per-event quantity at known
truth; it needs the same event in both detectors. Prometheus supports exactly
this: `injection.lepton_injector.inject = False` plus `paths.injection_file`
replays a stored injection.

## The four things that break it (all handled here)

1. **The geometries do not share a coordinate frame.** Measured from the
   shipped geofiles, detector centres run from z = +95 m (orca) to
   z = −3194 m (arca), with horizontal radii from 58 m to 1950 m. Prometheus
   applies the detector offset to the injection file *in place*, and only on
   the `inject=True` path — `injection_from_LI_output` ignores `detector_offset`
   on load. So naive re-use puts every event kilometres from detector B.
   → `geo.recentre_delta` + `run_geometry.recentre_injection`.

2. **The injection cylinder belongs to one detector.** Sized for A, it is far
   too small for a larger B (or wastefully large for a smaller one).
   → `geo.shared_cylinder` sizes it on the larger member and reports the
   instrumented-volume fraction for both. A pair with a horizontal scale ratio
   above ~2 is degenerate; say so rather than hiding it.

3. **There is no zero-point.** Photon propagation is re-run per geometry and is
   seedable only to Poisson level, so "the same event" differs even in the same
   detector. Without measuring that floor, a cross-geometry displacement has no
   scale. → `check_pairing.photon_seed_null` and `effect_over_null`. **Run the
   null before the pair.**

4. **Depth changes with the geometry.** The Earth model is built at
   `-detector_offset[2]`, so re-centring changes the overburden and invalidates
   flux weights across the pair. → Tier 2 reports unweighted, per-event,
   truth-referenced metrics only. Stated in `config/injection.yaml`.

## NuBench configuration — what we can and cannot match

Matched from the paper: ν_μ CC+NC, 10 GeV–100 TeV, OM r = 30 cm, QE 20%,
≥5 µs trigger window, uniform noise photons in-window, TTS photon merging,
1 ns time and 0.25 pe charge smearing, <4 or >10⁶ pulse cut, no charge
saturation. Also: NuBench ran **all six geometries in water**, plus one ice
dataset on the Hexagon geometry — the shipped `.geo` headers disagree and are
overridden (`run_geometry.override_medium`).

Not published, so not matchable (marked `[ASK]` in the configs): spectral
index, zenith/azimuth range, injection volume, Earth model, per-OM noise rate,
per-dataset TTS, and the ice angular-acceptance curve. **The detector-response
code of §3.2 was never released at all.**

That last point is the real risk. Our §3.2 is a re-implementation, so any
mismatch is an acquisition shift indistinguishable from the N-contract
violation we are trying to detect. `config/response.yaml` defines gates
G1–G3 that must pass before any ORACLE claim uses this module; G3 (released
DynEdge reproduces its published metrics on our events) is currently blocked by
the open V0 issue in `DATASETS.md`.

## Layout

```
bootstrap_prometheus.sh   clone + pin + install OUTSIDE the repo (LGPL-2.1)
geo.py                    geofile parsing, offsets, recentring, pair feasibility
make_injection.py         step 1 — one injection, sized for the pair
run_geometry.py           step 2 — replay it per geometry (recentre + medium)
check_pairing.py          step 3 — identity check, and the photon-seed null
config/geometries.yaml    7 NuBench datasets ↔ geofiles, measured extents, pairs
config/injection.yaml     LI settings, [nubench]/[ours]/[ASK] provenance
config/response.yaml      §3.2 re-implementation parameters + gates G1–G3
```

Simulation output never enters this repo. Set `ORACLE_SIM_ROOT`.

## Status

Scaffold only — no simulation has been run. `geo.py` and `check_pairing.py` are
complete and verified against the real geofiles; `make_injection.py` and
`run_geometry.py` build their manifests without Prometheus and call it under
`--execute`; `response.py` (the §3.2 re-implementation) is **not yet written** —
its parameters and gates are specified in `config/response.yaml`.
