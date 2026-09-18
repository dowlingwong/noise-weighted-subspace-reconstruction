# Freeze artifact index (package-specific binding checks)

Hashes are of the files in place on /ceph. Copies are in this directory as `<root>__<name>.json`.

| Selection | Artifact | sha256 | written (UTC) | first confirmation attempt, any state (UTC) | raw comparison |
|---|---|---|---|---|---|
| T19-S | `t19s-continuation/PILOT_SELECTION.json` | `298f80036d1754795a7a55c85627b271d341760c2f32f4e1b993eef129e1b72e` | 2026-09-13T21:16:13+00:00 | 2026-09-13T13:20:24+00:00 | **False** |
| T20-S | `t20s/FROZEN_SELECTION.json` | `e6bf0a22b68f451277ddd597c84ccd509e4140882a8d009e70f88c4306c2c1f0` | 2026-09-13T14:41:13+00:00 | 2026-09-13T14:41:14+00:00 | **True** |
| T21-S | `t21s/FROZEN_PREDICTOR.json` | `50ff515c3a1c363d9faa1597ed3a7b83c69dc4ea27ec3451eda5a11af567ff3d` | 2026-09-13T13:34:59+00:00 | 2026-09-13T13:34:59+00:00 | **False** |
| T22-S | `t22s/FROZEN_CALIBRATION.json` | `946656fc7f191a6802909c1fb90f58e77811f21d12cefe1d289678c527f5dd79` | 2026-09-13T19:11:17+00:00 | 2026-09-13T19:11:19+00:00 | **True** |

### Freeze ordering — resolved

The raw comparison above uses the first confirmation attempt **of any state**, compared at mixed precision.
It shows `False` for T19 and T21. Neither is a provenance problem.

| Selection | Freeze written (UTC) | First **completed** confirmation attempt | Completed confirmation cells before the freeze | Explanation |
|---|---|---|---|---|
| T19-S | 21:16:13.713 | 21:16:14 | **0** | The one earlier attempt (13:20:24, cell `5493e66358a9c0cf`, pid 2926850) is the **refused** attempt from the defective launcher. It ended in a traceback, remains charged and produced no cell. The cell later completed on a new attempt at 21:16:23. |
| T20-S | 14:41:13.057 | 14:41:14 | **0** | — |
| T21-S | 13:34:59.242 | 13:34:59 | **0** | Same second only because ledger timestamps are whole seconds. Independent clock: the launcher copied `01-REDUCED.json` at 13:34:59.384, and confirmation (phase 02) could not start until that phase returned. |
| T22-S | 19:11:17.863 | 19:11:19 | **0** | — |

**In all four selections, every completed confirmation cell was drawn after its freeze existed.**

## Binding, by each package's own mechanism

- **T19-S** (per-cell `result.selection_sha256`): 100/100 confirmation cells record the freeze digest.
- **T22-S** (per-cell `result.summary.frozen_sha256`): 690/690 confirmation cells record the freeze digest.
- **T20-S** (freeze-time `pilot_digest`; no per-cell hash by design): recorded `210bc99cbe64ce8c…`, recomputed now `210bc99cbe64ce8c…`, match **True**; `confirmation_cells_present_at_freeze` = 0; pilot cells 384.
- **T21-S** (design-only predictor; no per-cell hash by design): `confirmation_permitted` = True; `coefficients_fitted_on_any_data` = False; freeze lists 2080 confirmation cells in 104 configurations; completed confirmation cells 2080; sets equal **True**.

**CRESST archive / calibration:** no freeze artifact at the root. The authorized selections stop at partial reductions; see the scientific audit.
