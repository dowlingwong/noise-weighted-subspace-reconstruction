"""P1-E1 — Figure-1 data product: learned versus planted factors.

For each seed at the exactly separable condition (alpha = 0) refit NFPA and
the isotropic factored model from the frozen configuration, save the learned
and planted channel/temporal projectors, one example event with both
reconstructions, and the decoded amplitude / rise / decay per method.  The
channel projector error of every seed is regressed against the archived E8
values (``results/rps2026/E8_semantics/separable_alpha_0/seed_XX.csv``).

Usage::

    python -m experiments.paper1.fig1_factors --seeds 0 1 2 3 4 5 6 7 8 9 [--smoke]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_semantics as sem
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.generator import make_dataset
from experiments.synthetic.nfpa_confirmatory.methods import reconstruct
from experiments.synthetic.nfpa_confirmatory import r9_common as rc

from . import common

EXPERIMENT_ID = "P1-E1"
METHODS = ("NFPA", "Iso-MPCA")
ARCHIVE = common.REPO_ROOT / "results" / "rps2026" / "E8_semantics" / "separable_alpha_0"


def archived_channel_error(seed: int, method: str) -> float | None:
    path = ARCHIVE / f"seed_{seed:02d}.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["method"] == method and row["channel_projector_error"] not in ("", "None"):
                return float(row["channel_projector_error"])
    return None


def run(seeds: list[int], *, smoke: bool) -> Path:
    cfg = replace(common.smoke_config() if smoke else PRIMARY, nonseparability_alpha=0.0)
    out = common.out_dir_for(EXPERIMENT_ID, smoke=smoke)
    projectors: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    example_saved = False
    for seed in seeds:
        ds = make_dataset(seed, "colored", cfg)
        fits = common.fit_methods(ds, METHODS)
        for name, fit in fits.items():
            V_t, T_t = sem.planted_factors_in_method_coords(fit, ds)
            key = f"seed{seed:02d}_{name.replace('-', '_')}"
            projectors[f"{key}_P_channel_learned"] = fit.U_c @ fit.U_c.T
            projectors[f"{key}_P_channel_planted"] = V_t @ V_t.T
            projectors[f"{key}_P_temporal_learned"] = fit.U_t @ fit.U_t.T
            projectors[f"{key}_P_temporal_planted"] = T_t @ T_t.T
            errs = sem.planted_errors(fit, ds)
            semantics = sem.evaluate_semantics(fit, ds)
            archived = None if smoke else archived_channel_error(seed, name)
            rows.append({
                "config_hash": config_hash(cfg), "seed": seed, "method": name,
                "channel_projector_error": errs["channel_projector_error"],
                "temporal_projector_error": errs["temporal_projector_error"],
                "archived_channel_projector_error": archived,
                "regression_abs_diff": None if archived is None else abs(archived - errs["channel_projector_error"]),
                "amplitude_rmse": semantics.get("amplitude_rmse"),
                "amplitude_correlation": semantics.get("amplitude_correlation"),
                "rise_time_rmse": semantics.get("rise_time_rmse"),
                "decay_time_rmse": semantics.get("decay_time_rmse"),
                "runtime_seconds": fit.diag.get("runtime_seconds"),
            })
        if not example_saved:
            i = 0
            X = ds.X_test[i:i + 1]
            np.savez(out / "event_example.npz",
                     noisy=X[0], clean=ds.clean_test[i], seed=seed,
                     **{f"recon_{n.replace('-', '_')}": reconstruct(f, ds, X)[0] for n, f in fits.items()},
                     amplitude=ds.latents_test["amplitude"][i], rise_time=ds.latents_test["rise_time"][i],
                     decay_time=ds.latents_test["decay_time"][i])
            example_saved = True
        print(f"[P1-E1] seed {seed} done", flush=True)
    np.savez(out / "projectors.npz", **projectors)
    cols = list(rows[0].keys())
    rc.write_rows_csv(out / "decoding.csv", rows, cols)
    common.write_manifest(out, experiment_id=EXPERIMENT_ID, cfgs=[cfg], seeds=seeds,
                          extra={"methods": list(METHODS), "regression_reference": str(ARCHIVE.relative_to(common.REPO_ROOT)),
                                 "max_regression_abs_diff": max([r["regression_abs_diff"] or 0.0 for r in rows])})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    out = run(a.seeds, smoke=a.smoke)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
