"""R3 measured kernel benchmark -> BENCHMARK.json.

Times the R3 per-cell kernel (the archived predictor's exact-K eigen solve, the
geometry construction, and one rank-k PCA fit) at every PRODUCTION family
dimension and writes the measured ``BENCHMARK.json`` that ``common.cli.estimate``
reads beside the protocol.

This is a bounded benchmark, not a production run: it writes no cell files, uses
an explicit benchmark seed outside every reserved namespace, and records the
measured per-cell seconds only.  The estimate's *cell compute* is then
``sum_family (measured seconds per cell x production cells)``.

Usage::

    python -m experiments.mlst2026.followup_20260917.r3_t21_finite.benchmark \
        --protocol PROTOCOL_R3_T21_FINITE_v1.json --out BENCHMARK.json --repeats 5
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from ...successor_20260913.common.protocol import load_protocol
from ...successor_20260913.common.results import write_json_atomic

from . import runner as RUN

BENCHMARK_SEED = 1          # outside every reserved namespace (not a scientific seed)
HERE = Path(__file__).resolve().parent


def _time_cell(P: dict, cell: dict, repeats: int) -> dict:
    times, peaks = [], []
    for r in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        if cell["role"] == "algebraic":
            RUN.run_algebraic_cell(P, cell, False)
        else:
            RUN.run_data_cell(P, cell)
        dt = time.perf_counter() - t0
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if r > 0:                                   # drop the first (warm-up) repeat
            times.append(dt)
            peaks.append(peak)
    if not times:
        times, peaks = [dt], [peak]
    return {"median_s": float(statistics.median(times)), "min_s": float(min(times)), "max_s": float(max(times)),
            "repeats": len(times), "memory": {"tracemalloc_peak_bytes": int(max(peaks)),
            "rss_high_water_bytes": None, "memory_measured_in": "this process"}}


def build_benchmark(P: dict, repeats: int = 5) -> dict:
    matrix = RUN.build_matrix(P, False)
    prod = matrix.subset(roles=[r for r in ("pilot", "confirmation", "algebraic")])
    families: dict[str, dict] = {}
    kernels: dict[str, dict] = {}
    t_start = time.perf_counter()
    for fam in sorted({c["family"] for c in prod.cells}):
        members = [c for c in prod.cells if c["family"] == fam]
        cell = dict(members[0])
        if cell["role"] != "algebraic":
            cell["seed"] = BENCHMARK_SEED
        stats = _time_cell(P, cell, repeats)
        s = stats["median_s"]
        families[fam] = {"basis": "measured", "calls_per_cell": 1, "kernel": f"r3_cell_{fam}",
                         "seconds_per_cell": {"median": s, "low": stats["min_s"], "high": stats["max_s"]},
                         "size": int((cell["params"] or {}).get("d", 0)) or None, "size_symbol": "d",
                         "extrapolation_factor_in_dimension": 1.0,
                         "projection_label": "measured at the production size",
                         "how": ("the production dimension was run directly through the R3 cell kernel %d times "
                                 "(first repeat dropped as warm-up); the range is the measured min/max over repeats"
                                 % (stats["repeats"] + 1))}
        kernels[f"r3_cell_{fam}"] = {"kernel": f"r3_cell_{fam}", "call_path": "r3_t21_finite.runner.run_data_cell / run_algebraic_cell",
                                     "size_symbol": "d", "notes": "the R3 per-cell kernel", "points": [
                                         {"size": families[fam]["size"], "median_s": s, "min_s": stats["min_s"],
                                          "max_s": stats["max_s"], "repeats": stats["repeats"],
                                          "memory": stats["memory"], "memory_measured_in": "this process"}]}
    wall = time.perf_counter() - t_start
    counts = prod.counts()
    point = sum(families[f]["seconds_per_cell"]["median"] * counts["by_family"][f]["cells"] for f in families)
    return {"schema": "mlst-local-package-benchmark/1", "package": RUN.PACKAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark_wall_seconds": wall,
            "hardware": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version.split()[0],
                         "numpy": np.__version__, "cpu_count": os.cpu_count()},
            "thread_settings": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "families": families, "kernels": kernels,
            "reduce": {"measured": False, "why": "no fixture run root was passed to the benchmark; the reduce projection is unavailable"},
            "harness": {"measured": False, "why": "the harness is measured by the CLI estimate pass over a fixture run root, not here"},
            "storage": {"bytes_per_cell_by_family": {}}, "work_accounting": {},
            "setup_and_transfer": {"transfer": {"archive_bytes": None}, "note": "local development only; no server transfer measured"},
            "notes": ["MEASURED benchmark of the production-sized R3 cell kernels; no production run, no cell file, no reserved seed",
                      "cell_compute projection over the production matrix = %.3f s from these medians" % point,
                      "these timings are single-process on one laptop; server speed, core count and contention are unknown"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R3 measured kernel benchmark")
    ap.add_argument("--protocol", type=Path, default=HERE / "PROTOCOL_R3_T21_FINITE_v1.json")
    ap.add_argument("--out", type=Path, default=HERE / "BENCHMARK.json")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)
    P = load_protocol(args.protocol)
    bench = build_benchmark(P, repeats=int(args.repeats))
    write_json_atomic(args.out, bench)
    print(json.dumps({"out": str(args.out), "benchmark_wall_seconds": bench["benchmark_wall_seconds"],
                      "families": len(bench["families"]),
                      "slowest": max((v["seconds_per_cell"]["median"], k) for k, v in bench["families"].items())[1]}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
