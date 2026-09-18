> **SUPERSEDED — do not use.** The binding section searched confirmation cells for the freeze *file byte hash*. That is the mechanism for T19 and T22 only; T20 and T21 bind differently, so their "not_bound" result is a method error, not a finding. See FREEZE_ARTIFACT_INDEX.md.

# Freeze artifact index

Each freeze was written by a partial reduction before confirmation cells were drawn. Hashes are of the files in place.

| Selection | Artifact | sha256 | mtime | protocol_sha256 | development_only | source |
|---|---|---|---|---|---|---|
| T19-S | `t19s-continuation/PILOT_SELECTION.json` | `298f80036d1754795a7a55c85627b271d341760c2f32f4e1b993eef129e1b72e` | 2026-09-13T23:16:13 | `9436266b1edadc45…` | False | pilot |
| T20-S | `t20s/FROZEN_SELECTION.json` | `e6bf0a22b68f451277ddd597c84ccd509e4140882a8d009e70f88c4306c2c1f0` | 2026-09-13T16:41:13 | `f3fda7cd4cc02a2e…` | False | — |
| T21-S | `t21s/FROZEN_PREDICTOR.json` | `50ff515c3a1c363d9faa1597ed3a7b83c69dc4ea27ec3451eda5a11af567ff3d` | 2026-09-13T15:34:59 | `b84856fe769ddea6…` | False | — |
| T22-S | `t22s/FROZEN_CALIBRATION.json` | `946656fc7f191a6802909c1fb90f58e77811f21d12cefe1d289678c527f5dd79` | 2026-09-13T21:11:17 | `30acaa43509832eb…` | False | pilot cells: 32 |

**CRESST archive / calibration:** no freeze file at the root; see the scientific audit for why.

## Confirmation-cell binding

- **T19-S**: 100 confirmation cells; {'bound': 100} to `298f80036d175479…`
- **T20-S**: 576 confirmation cells; {'not_bound': 576} to `e6bf0a22b68f4512…`
- **T21-S**: 2080 confirmation cells; {'not_bound': 2080} to `50ff515c3a1c363d…`
- **T22-S**: 690 confirmation cells; {'bound': 690} to `946656fc7f191a68…`
