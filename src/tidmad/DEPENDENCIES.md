# Dependencies of `src/tidmad`

*Updated 2026-08-08 after the migration into this repository. The earlier version
of this note described a cross-repository arrangement that no longer exists.*

## What changed

`tidmad` previously lived in a separate `transformer` repository and reached
outward for two things: the backbone (`reconstruction_model.*`, a sibling
package) and Paper 1's noise machinery (`noise_geometry`, reached by a
`sys.path` insertion that walked up two directories). Both couplings are gone.

- **The backbone is vendored** at `src/tidmad/backbone/` — `model.py`, `muon.py`,
  `schedulers.py`, `checkpoints.py`, `registry.py`. No import of
  `reconstruction_model` remains anywhere in the package.
- **`noise_geometry` is now a same-repository import**, resolved normally via
  `src/` on the path. No `sys.path` manipulation.

## Internal dependency: `noise_geometry`

`tidmad` reuses Paper 1's TIDMAD machinery by import, deliberately, so that the
`J(f)` used at rung 4 is the same object the paper cites (PLAN_04 §3.2, §3.5). A
reimplementation that disagreed by a normalisation factor would silently break
comparability with the archived Phase-4 result.

| `tidmad` module | Imports from `noise_geometry` |
|---|---|
| `train.py` | `noise_geometry.tidmad.loader` — `TidmadFile`, `load_contract` |
| `psd.py` | Paper 1 Phase-4 median-periodogram estimator |
| `score.py` | `noise_geometry.tidmad.score.run_upstream_benchmark` |
| `infer.py` | Paper 1's denoised-h5 writer |

`characterise_injections.py` deliberately depends on **nothing** beyond `numpy`
and `h5py`: it must be runnable before any of the above is trusted, since it
decides whether the modelled band is valid at all.

## External

`numpy`, `scipy`, `h5py`, `PyYAML` from the base install; `torch` from the
`tidmad` extra; `wandb` (optional, only with `--wandb`) from the `paper2` extra.

## Running

`python -m tidmad.*` needs `src/` on the path:

```bash
PYTHONPATH=src python -m tidmad.train ...
```

`scripts/train_tidmad_local.sh` sets this itself.

## For the D5 anonymized artifact

The published subtree must carry `src/tidmad/` **and** the `noise_geometry`
modules listed above, or vendor them. The backbone no longer needs declaring —
it is inside the package. Nothing in this package imports detector-derived code:
`src/transformer/` was removed and `src/tidmad` never depended on it.
