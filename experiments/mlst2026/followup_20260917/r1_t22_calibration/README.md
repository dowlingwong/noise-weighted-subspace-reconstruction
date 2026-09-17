# R1 — T22 small-file interval calibration

Status: configured, locally tested on development fixtures; **ready to transfer**
pending scientific review of the frozen candidate family.

## Question

Can a prespecified interval achieve the archived coverage band at few independent
files while keeping a coherent type-I rule?

## Design (new protocol)

- `PROTOCOL_R1_T22_CALIBRATION_v1.json`, package `r1_t22_calibration`, supersedes
  T22-S v2. Designed after the September results were observed.
- Frozen candidate family (`candidates.py`): `two_stage` (named baseline),
  `t_equal_file`, `two_stage_t`, `welch_fe_t`, `re_t`. Each candidate reports a
  point, an interval and a p-value from the **same** pivot (`coherent_pvalue`),
  so a repaired interval cannot be scored with an incompatible significance rule.
- Retains the 65 non-drift scenarios, four drift stress scenarios, n_files =
  4/8/16, declared generators, effect scales, the [0.925, 0.975] Wilson band, the
  0.07 type-I threshold and the three-claim Holm family.
- Selection: an all-scenario adequacy rule on fresh pilot seeds; at most one
  primary replacement; `"none"` is a valid outcome. Frozen in
  `FROZEN_CANDIDATE.json`; confirmation cells refuse without it.
- Seed namespaces: development [920000, 920999], pilot [21000, 21299],
  confirmation [22000, 22499] (disjoint from every archived package).

## Commands

```sh
P=experiments/mlst2026/followup_20260917/r1_t22_calibration/PROTOCOL_R1_T22_CALIBRATION_v1.json
# local development fixture
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration run  --protocol $P --out $OUT --fixture --families sim_pilot control
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration reduce --protocol $P --out $OUT --allow-partial
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration run  --protocol $P --out $OUT --fixture --families sim_confirm
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration reduce --protocol $P --out $OUT
# production (remote) -- pilot first, then reduce to freeze, then confirmation
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration run  --protocol $P --out $RUN --families sim_pilot control
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration reduce --protocol $P --out $RUN
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration run  --protocol $P --out $RUN --families sim_confirm
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration reduce --protocol $P --out $RUN
```

## Tests

`tests/test_followup_20260917_r1.py` (11 tests): validate/enumerate compute
nothing, new-version provenance, candidate math (ordering, coherent pivot,
coverage), namespace disjointness vs the archived t22s, fixture refusal without a
freeze, fixture end-to-end freeze, tampered-record refusal, and frozen selection.

## Limits

No production run has been executed. Fixture output is development-only and never
citable. The frozen candidate in the fixture is not a scientific result.
