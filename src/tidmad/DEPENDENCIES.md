# Dependencies of `tidmad/` (round-2 correction T6)

`tidmad/` is not self-contained by design: it reuses Paper 1's noise/PSD/h5
machinery by import so the `J(f)` used here is the same object the paper cites
(PLAN_04 §3.2, §3.5). For the working setup this is fine — the parent repo's
`src/` is placed on `sys.path` by `tidmad/__init__.py`. For the **D5 anonymized
artifact** it must be declared or vendored.

## Import surface (the full set)

| `tidmad` module | Imports from `noise_geometry` |
|---|---|
| `psd.py` | `noise_geometry.tidmad.psd.welch_psd_from_windows` |
| `data.py` | (none — self-contained STFT) |
| `train.py` | `noise_geometry.tidmad.loader` (`TidmadFile`, `load_contract`) |
| `infer.py` | `noise_geometry.tidmad.loader`, `noise_geometry.tidmad.score` (`DenoisedFileSpec`, `write_denoised_file`) |
| `score.py` | `noise_geometry.tidmad.score` (`run_upstream_benchmark`) |
| tests | `noise_geometry.tidmad.splits` (chronological split, leakage audit) |

## Decision (recorded 2026-08-08)

**Vendor** the four `noise_geometry.tidmad` modules (`psd.py`, `loader.py`,
`score.py`, `splits.py`) into the D5 artifact's subtree. They are small (~700
lines total), have no detector coupling, and are the same estimator the paper
cites; vendoring keeps the published subtree self-contained and the J(f) object
identical. Do not vendor the whole `noise_geometry` package — only the four
modules and their direct imports (`noise_geometry.noise.psd`,
`noise_geometry.noise` for `inverse_psd_weights`).

If vendoring is rejected, the alternative is to declare `noise_geometry` as a
dependency in the artifact's `pyproject.toml` and document the pinned commit —
but that couples the public artifact to a non-public repo, so vendoring is
preferred.
