# VOID — superseded 2026-08-08

This document planned rung 4 of the architecture ladder using DELight simulation results.

**It is void. Do not follow it. No DELight data, simulation, physics, or trained checkpoint appears anywhere in this paper.**

Rung 4 is now instantiated on **TIDMAD** (public, CC-BY-4.0). See:

- `PLAN_03_RUNG4_TIDMAD_2026-08-08.md` — the replacement
- `PLAN_00_MASTER_RESTRUCTURE_2026-08-08.md` — governing document

Reason for the change: collaboration-level resistance to external use of DELight data, simulation, and physics. Rather than seek clearance, rung 4 was moved to a public benchmark on which Paper 1 already has archived results and a reproduced calibration anchor. The scientific claim is unchanged and the evidence is stronger, because TIDMAD carries published baselines a reviewer can check.

Two constraints carried over from this document and still binding:

1. **Repository hygiene.** `src/scripts/TraceSimulator.py` and the `artifacts/` checkpoint tree are DELight property and must not appear in any published or anonymized repository. See `PLAN_03_RUNG4_TIDMAD` §2A.5.
2. **Attribution.** The architecture derives from prior work. The three conditions governing acknowledgement are in `PLAN_03_RUNG4_TIDMAD` §5.
