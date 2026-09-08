"""E-0 — environment and regression baseline (non-destructive).

``fig1_factors.py --smoke`` only checks wiring: in smoke mode the archived
lookup is skipped, so its ``max_regression_abs_diff`` is vacuously ``0.0``.
This runner performs the *actual* regression gate of task card E-0 without
touching ``results/paper1/P1-E1`` (ground rule 5): it refits the frozen
``PRIMARY`` configuration at ``alpha = 0`` for the requested seeds and compares
the planted channel/temporal projector errors against the archived R/PS values
in ``results/rps2026/E8_semantics/separable_alpha_0/seed_XX.csv``.

Usage::

    python -m experiments.paper1.e0_regression --seeds 0
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory import r9_semantics as sem
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.generator import make_dataset

from . import common

EXPERIMENT_ID = "P1-E0"
METHODS = ("NFPA", "Iso-MPCA")
ARCHIVE = common.REPO_ROOT / "results" / "rps2026" / "E8_semantics" / "separable_alpha_0"
COLUMNS = [
    "config_hash", "seed", "method",
    "channel_projector_error", "archived_channel_projector_error", "abs_diff_channel",
    "temporal_projector_error", "archived_temporal_projector_error", "abs_diff_temporal",
    "runtime_seconds",
]


def _archived_row(seed: int, method: str) -> dict | None:
    path = ARCHIVE / f"seed_{seed:02d}.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["method"] == method:
                return row
    return None


def _f(row: dict | None, key: str) -> float | None:
    if row is None:
        return None
    v = row.get(key, "")
    if v in ("", "None"):
        return None
    return float(v)


def run(seeds: list[int]) -> Path:
    cfg = replace(PRIMARY, nonseparability_alpha=0.0)
    out = common.out_dir_for(EXPERIMENT_ID)
    rows: list[dict] = []
    t_start = time.time()
    for seed in seeds:
        ds = make_dataset(seed, "colored", cfg)
        fits = common.fit_methods(ds, METHODS)
        for name, fit in fits.items():
            errs = sem.planted_errors(fit, ds)
            arch = _archived_row(seed, name)
            a_c, a_t = _f(arch, "channel_projector_error"), _f(arch, "temporal_projector_error")
            c, t = errs["channel_projector_error"], errs["temporal_projector_error"]
            rows.append({
                "config_hash": config_hash(cfg), "seed": seed, "method": name,
                "channel_projector_error": c,
                "archived_channel_projector_error": a_c,
                "abs_diff_channel": None if (a_c is None or c is None) else abs(a_c - c),
                "temporal_projector_error": errs["temporal_projector_error"],
                "archived_temporal_projector_error": a_t,
                "abs_diff_temporal": None if (a_t is None or t is None) else abs(a_t - t),
                "runtime_seconds": fit.diag.get("runtime_seconds"),
            })
        print(f"[P1-E0] seed {seed} done", flush=True)
    rc.write_rows_csv(out / "regression_check.csv", rows, COLUMNS)
    diffs = [r[k] for r in rows for k in ("abs_diff_channel", "abs_diff_temporal") if r[k] is not None]
    n_compared = len(diffs)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "seeds": seeds,
        "regression_reference": str(ARCHIVE.relative_to(common.REPO_ROOT)),
        "n_comparisons": n_compared,
        "max_regression_abs_diff": float(max(diffs)) if diffs else None,
        "gate_tolerance": 1e-6,
        "gate_passed": bool(diffs) and float(max(diffs)) <= 1e-6,
    }
    rc.write_json(out / "summary.json", common.as_jsonable(summary))
    common.write_manifest(
        out, experiment_id=EXPERIMENT_ID, cfgs=[cfg], seeds=seeds,
        extra={"methods": list(METHODS),
               "regression_reference": summary["regression_reference"],
               "max_regression_abs_diff": summary["max_regression_abs_diff"],
               "gate_passed": summary["gate_passed"],
               "walltime_s": time.time() - t_start,
               "host": platform.node(),
               "platform": platform.platform(),
               "numpy": np.__version__},
    )
    print(f"[P1-E0] max_regression_abs_diff = {summary['max_regression_abs_diff']!r} "
          f"over {n_compared} comparisons; gate_passed={summary['gate_passed']}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args(argv)
    print(f"wrote {run(a.seeds)}")


if __name__ == "__main__":
    main()
