#!/usr/bin/env python3
"""Measure the dominant kernel of every local package and write ``BENCHMARK.json``.

    python3 experiments/mlst2026/successor_20260913/run_benchmarks.py \
        --ms-evidence /path/to/MS/evidence/data [--packages t19s,t20s,...]

For each package this driver

* identifies the estimator call the production cells actually spend their time
  in (by reading the runner; the choice is recorded in ``kernels[*].call_path``),
* times **the package's own function** at four or more problem sizes that
  bracket or reach the production dimension,
* fits ``t = a n^b`` and prints the fitted exponent beside the algorithm's,
* measures peak memory (``tracemalloc`` in-process and ``ru_maxrss`` in a fresh
  subprocess, which is the only way an RSS high-water mark means one size),
* measures the ``reduce`` step and the on-disk size of a cell file, and
* writes ``<pkg>/BENCHMARK.json``, which ``common.cli._estimate`` reads.

Nothing here re-implements a package: every timed callable is imported from the
package under test.  Where a production cell cannot be executed here at all (the
CRESST robustness cells need raw pulse traces that are not in this container),
the *dimensions* are reproduced synthetically with the package's own fixture
writer and the substitution is recorded in ``notes``.

``--probe PKG KERNEL SIZE`` runs one call in this process and prints a single
JSON line; it is how the memory subprocess probe is spawned.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# BLAS threading must be pinned before numpy binds it.
from experiments.mlst2026.successor_20260913.common import benchmark as B  # noqa: E402

B.pin_single_thread()

import numpy as np  # noqa: E402,F401

from experiments.mlst2026.successor_20260913.common import fitcount  # noqa: E402
from experiments.mlst2026.successor_20260913.common.protocol import load_protocol  # noqa: E402

PKGS = ("t19s", "t20s", "t21s", "t22s", "cresst_s")
PROTOCOLS = {"t19s": "PROTOCOL_T19S_v2.json", "t20s": "PROTOCOL_T20S_v2.json", "t21s": "PROTOCOL_T21S_v2.json",
             "t22s": "PROTOCOL_T22S_v2.json", "cresst_s": "PROTOCOL_CRESST_S_v1.json"}


def protocol(pkg: str) -> dict:
    return load_protocol(HERE / pkg / PROTOCOLS[pkg])


def _log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def _counting(fn):
    """Wrap a callable so the package's fitcount instrumentation has a region."""
    def call():
        with fitcount.counting():
            return fn()
    return call


#: Set by ``main`` so that a spawned probe inherits the same inputs.
_ARGS = None


def _mem_probe(pkg: str, kernel: str):
    """A subprocess memory probe for one size of one kernel."""
    def probe(size):
        argv = [sys.executable, str(Path(__file__).resolve()), "--probe", pkg, kernel, repr(size)]
        if _ARGS is not None and getattr(_ARGS, "ms_evidence", None):
            argv += ["--ms-evidence", str(_ARGS.ms_evidence)]
        return B.probe_peak_memory_subprocess(argv, cwd=str(ROOT))
    return probe


# =============================================================================
# t19s -- G1 covariance-weighted PCA, paired cells
# =============================================================================
def _t19s_paired_call(d: int):
    from experiments.mlst2026.successor_20260913.t19s import synthetic as S
    P = protocol("t19s")
    pp = P["design"]["production"]["paired"]
    params = {"kind": "paired", "d": int(d), "k": 2, "n_train": 2 * int(d), "eps_grid": pp["eps_grid"],
              "m_over_d_grid": pp["m_over_d_grid"], "n_train_ratio_grid": pp["n_train_ratio_grid"],
              "rng_offsets": pp["rng_offsets"]}
    return _counting(lambda: S.run_paired_cell(params, 1000, selection_sha256=None))


def _t19s_det_call(d: int, family: str = "generic"):
    from experiments.mlst2026.successor_20260913.t19s import runner as R
    P = protocol("t19s")
    det = P["design"]["production"]["deterministic"]
    params = {"kind": "deterministic", "model": "covariance", "family": family, "d": int(d), "k": 2,
              "operator": {"kind": "g1_population", "d": int(d)}, "eps_grid": det["eps_grid"],
              "direction_seed": 20260913}
    return _counting(lambda: R.run_deterministic_cell(params))


def bench_t19s(args) -> dict:
    from experiments.mlst2026.successor_20260913.t19s import runner as R, synthetic as S
    P = protocol("t19s")
    D = P["design"]["production"]
    d_prod = int(D["d"])
    pp = D["paired"]
    n_fits, n_solves = S.plan_counts({"eps_grid": pp["eps_grid"], "m_over_d_grid": pp["m_over_d_grid"],
                                      "n_train_ratio_grid": pp["n_train_ratio_grid"]})
    _log("t19s paired cell, d = 32 64 128 256 (production d = 256, reached)")
    paired = B.scaling_benchmark(
        "paired_cell", _t19s_paired_call, [32, 64, 128, 256], theory_key="eigendecomposition",
        size_symbol="d", production_size=d_prod, fixture_size=int(P["design"]["fixture"]["d"]),
        repeats=3, warmup=1, memory_sizes=[d_prod], memory_probe=_mem_probe("t19s", "paired"),
        call_path="t19s.synthetic.run_paired_cell (32 cw_pca fits + 29 eigen solves per cell; the m/d = 2048 "
                  "noise-only calibration stream of 2048*d rows dominates)",
        notes="N is held at the production aspect ratio N = 2d so the d = 256 point IS one production paired cell")
    _log("t19s deterministic cell, d = 32 64 128 256")
    det = B.scaling_benchmark(
        "deterministic_cell", _t19s_det_call, [32, 64, 128, 256], theory_key="eigendecomposition",
        size_symbol="d", production_size=d_prod, fixture_size=int(P["design"]["fixture"]["d"]),
        repeats=3, warmup=1, memory_sizes=[d_prod],
        call_path="t19s.runner.run_deterministic_cell on the G1 population operator (8 eps points, one eigen "
                  "solve each)",
        notes="every d = 256 deterministic/control family runs this same code path with a different direction")
    two_d = B.timed(_t19s_det_two_d(), size=2, repeats=3, warmup=1)

    # measured cell-file bytes at the production size
    with fitcount.counting():
        res_paired, _, _ = S.run_paired_cell(
            {"kind": "paired", "d": d_prod, "k": 2, "n_train": int(D["n_train"]), "eps_grid": pp["eps_grid"],
             "m_over_d_grid": pp["m_over_d_grid"], "n_train_ratio_grid": pp["n_train_ratio_grid"],
             "rng_offsets": pp["rng_offsets"]}, 1000, selection_sha256=None)
    det_res, _, _ = R.run_deterministic_cell(
        {"kind": "deterministic", "model": "covariance", "family": "generic", "d": d_prod, "k": 2,
         "operator": {"kind": "g1_population", "d": d_prod}, "eps_grid": D["deterministic"]["eps_grid"],
         "direction_seed": 20260913})

    per = {"paired": paired["projection"], "det": det["projection"]}
    fams = {}
    for fam in ("det_generic", "det_pure_coupling", "ctrl_block_diagonal", "ctrl_zero",
                "ctrl_additive_generic", "ctrl_additive_pure_coupling"):
        fams[fam] = _family_from(det, "deterministic_cell", 1, d_prod, "d")
    fams["det_two_d"] = {"kernel": "deterministic_cell_2d", "calls_per_cell": 1, "basis": "measured",
                         "size": 2, "size_symbol": "d",
                         "seconds_per_cell": {"median": two_d.median_s, "low": two_d.min_s, "high": two_d.max_s},
                         "how": "the 2-D closed-form cell is its own production size; measured directly"}
    fams["paired"] = _family_from(paired, "paired_cell", 1, d_prod, "d")
    return {
        "kernels": {"paired_cell": paired, "deterministic_cell": det,
                    "deterministic_cell_2d": {"kernel": "deterministic_cell_2d", "points": [two_d.as_dict()],
                                              "projection": {"basis": "measured", "seconds_per_call_median": two_d.median_s,
                                                             "seconds_per_call_low": two_d.min_s,
                                                             "seconds_per_call_high": two_d.max_s,
                                                             "label": "measured at the production size"}}},
        "families": fams,
        "work_accounting": {
            "estimator_calls_per_paired_cell": n_fits,
            "restarts": "none: the CW-PCA estimator is closed form (SVD), it has no random restarts",
            "solves_per_paired_cell": n_solves,
            "simulation_replicates_per_paired_cell": {
                "training_rows_drawn": max(int(round(int(D["n_train"]) * r)) for r in pp["n_train_ratio_grid"]),
                "calibration_rows_streamed": max(pp["m_over_d_grid"]) * d_prod,
                "note": "the calibration pool is nested: the largest m is the whole stream"},
            "bootstrap_replicates_in_reduce": int(P["inference"].get("bootstrap", {}).get("replicates", 0) or 0),
        },
        "storage": {"paired_cell_result_bytes_measured": B.json_bytes(res_paired),
                    "deterministic_cell_result_bytes_measured": B.json_bytes(det_res),
                    "basis": "measured: the production-size result dict serialised with the runner's encoder"},
        "notes": [
            "The production paired cell (d = 256, N = 512) is inside the benchmarked range, so the paired "
            "projection is a MEASUREMENT, not an extrapolation.",
            "Within one paired cell the cost is dominated by the m/d = 2048 calibration stream "
            "(2048 d rows accumulated as Z^T Z, i.e. O(2048 d^3)), not by the 32 CW-PCA fits.",
        ],
        "_per": per,
    }


def _t19s_det_two_d():
    from experiments.mlst2026.successor_20260913.t19s import runner as R
    P = protocol("t19s")
    det = P["design"]["production"]["deterministic"]
    params = {"kind": "deterministic", "model": "covariance", "family": "two_d_example", "d": 2, "k": 1,
              "operator": {"kind": "two_d_example"}, "eps_grid": det["two_d_example_eps_grid"],
              "reference": det["two_d_reference"]}
    return _counting(lambda: R.run_deterministic_cell(params))


# =============================================================================
# t20s -- separable ALS against closed-form PCA
# =============================================================================
def _t20s_fit_call(N: int):
    from experiments.mlst2026.successor_20260913.t20s import runner as R
    P = protocol("t20s")
    D = R.design(P, False)
    base = {k: D[k] for k in ("C", "T", "k_c", "k_t", "spectrum", "geometry_seed", "R_pop",
                              "pop_als_n_iter", "pop_als_tol")}
    p = {**base, "kind": "fit", "phase": "pilot", "theta": 0.25, "theta_index": 2, "N": int(N), "N_index": 1,
         "band": "core", "R": int(D["R"]), "als_n_iter": int(D["als_n_iter"]), "als_tol": float(D["als_tol"]),
         "control": None, "fixture": False}
    cell = {"role": "pilot", "seed": 3000, "family": "pilot", "cell_id": "bench", "n_fits": 1 + int(D["R"]),
            "n_solves": 0, "params": p}
    ctx = {"out": Path(tempfile.gettempdir()), "fixture_overrides": None}
    return _counting(lambda: R.run_fit_cell(P, cell, D, ctx))


def _t20s_algebraic_call(theta_index: int = 2):
    from experiments.mlst2026.successor_20260913.t20s import runner as R
    P = protocol("t20s")
    D = R.design(P, False)
    base = {k: D[k] for k in ("C", "T", "k_c", "k_t", "spectrum", "geometry_seed", "R_pop",
                              "pop_als_n_iter", "pop_als_tol")}
    cell = {"role": "algebraic", "seed": None, "family": "algebraic", "cell_id": "alg",
            "n_fits": 0, "n_solves": 1 + (D["R_pop"] + 1) + 1,
            "params": {**base, "kind": "algebraic", "theta": D["thetas"][theta_index],
                       "theta_index": theta_index, "thetas_all": D["thetas"], "control": None, "fixture": False}}
    return lambda: R.run_algebraic_cell(P, cell, D)


def bench_t20s(args) -> dict:
    from experiments.mlst2026.successor_20260913.t20s import runner as R, generator as GEN
    P = protocol("t20s")
    D = R.design(P, False)
    N_grid = list(D["N_grid"])
    _log(f"t20s fit cell at d = {D['d']}, N = {N_grid} (the whole production N grid)")
    fit = B.scaling_benchmark(
        "fit_cell", _t20s_fit_call, N_grid, theory_key="als_sweep", size_symbol="N",
        production_size=max(N_grid), fixture_size=int(P["fixture"]["N_core"][0]) if P["fixture"].get("N_core") else None,
        repeats=3, warmup=1, memory_sizes=[max(N_grid)], memory_probe=_mem_probe("t20s", "fit"),
        call_path="t20s.runner.run_fit_cell -> np.linalg.eigh(S) (1 PCA fit) + "
                  "experiments.synthetic.nfpa_confirmatory.methods.als_fit (R = 4 restarts x 150 sweeps)",
        notes="d = C.T = 256 is fixed by the design; N is the only production size that varies, and every "
              "production N was run")
    _log("t20s algebraic cell (population reference, cold cache)")
    alg = B.timed(_t20s_algebraic_call(), size=1, repeats=3, warmup=1)
    # the per-process cold population reference a fit cell pays if its worker has not
    # already computed that theta
    GEN._population_reference_cached.cache_clear()
    t0 = time.perf_counter()
    R._reference(D, 0.25)
    cold_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    R._reference(D, 0.25)
    warm_ref = time.perf_counter() - t0

    with fitcount.counting():
        res, _ = R.run_fit_cell(P, {"role": "pilot", "seed": 3000, "family": "pilot", "cell_id": "b",
                                    "n_fits": 5, "n_solves": 0,
                                    "params": {**{k: D[k] for k in ("C", "T", "k_c", "k_t", "spectrum",
                                                                    "geometry_seed", "R_pop", "pop_als_n_iter",
                                                                    "pop_als_tol")},
                                               "kind": "fit", "phase": "pilot", "theta": 0.25, "theta_index": 2,
                                               "N": 1600, "N_index": 4, "band": "core", "R": 4,
                                               "als_n_iter": 150, "als_tol": 1e-10, "control": None,
                                               "fixture": False}},
                                D, {"out": Path(tempfile.gettempdir()), "fixture_overrides": None})
    alg_res = _t20s_algebraic_call()()

    # every (seed, theta) contributes one cell at each of the six N, so the per-cell
    # mean over a family is the unweighted mean of the six measured N points
    pts = {r["size"]: r for r in fit["points"]}
    med = sum(pts[n]["median_s"] for n in N_grid) / len(N_grid)
    lo = sum(pts[n]["min_s"] for n in N_grid) / len(N_grid)
    hi = sum(pts[n]["max_s"] for n in N_grid) / len(N_grid)
    fam_fit = {"kernel": "fit_cell", "calls_per_cell": 1, "basis": "measured", "size": "N in %s" % (N_grid,),
               "size_symbol": "N", "seconds_per_cell": {"median": med, "low": lo, "high": hi},
               "fitted_exponent": fit["fit"]["b"], "theoretical_exponent": fit["theory"]["exponent"],
               "extrapolation_factor_in_dimension": 1.0,
               "how": ("mean over the six production N of the measured per-cell time "
                       + ", ".join("N=%d:%.3fs" % (n, pts[n]["median_s"]) for n in N_grid)
                       + "; each (seed, theta) contributes exactly one cell at each N")}
    fams = {"pilot": dict(fam_fit), "confirmation": dict(fam_fit),
            "algebraic": {"kernel": "algebraic_cell", "calls_per_cell": 1, "basis": "measured", "size": 4,
                          "size_symbol": "thetas", "extrapolation_factor_in_dimension": 1.0,
                          "seconds_per_cell": {"median": alg.median_s, "low": alg.min_s, "high": alg.max_s},
                          "how": "measured at the production design with the population-reference cache cleared, "
                                 "as the cell itself does"}}
    return {
        "kernels": {"fit_cell": fit,
                    "algebraic_cell": {"kernel": "algebraic_cell", "points": [alg.as_dict()],
                                       "projection": {"basis": "measured", "seconds_per_call_median": alg.median_s,
                                                      "seconds_per_call_low": alg.min_s,
                                                      "seconds_per_call_high": alg.max_s,
                                                      "label": "measured at the production size"}}},
        "families": fams,
        "work_accounting": {
            "estimator_calls_per_fit_cell": 1 + int(D["R"]),
            "restarts": {"als_restarts_per_fit_cell": int(D["R"]),
                         "als_sweeps_per_restart_max": int(D["als_n_iter"]),
                         "note": "1 closed-form PCA fit + R = 4 ALS restarts; the restart count is NOT folded "
                                 "into the fit number, it is the fit number minus one"},
            "solves_per_algebraic_cell": 1 + int(D["R_pop"]) + 1 + 1,
            "population_als_inits_per_algebraic_cell": int(D["R_pop"]) + 1,
            "simulation_replicates": {"draws_per_fit_cell": 1,
                                      "rows_drawn_per_fit_cell": "N (400..3200) rows of d = 256"},
            "per_worker_cold_population_reference_seconds_measured": cold_ref,
            "per_worker_warm_population_reference_seconds_measured": warm_ref,
        },
        "storage": {"fit_cell_result_bytes_measured": B.json_bytes(res),
                    "algebraic_cell_result_bytes_measured": B.json_bytes(alg_res),
                    "basis": "measured: the production-size result dict serialised with the runner's encoder"},
        "notes": [
            "run_fit_cell caches the population reference per (process, theta) with an LRU cache. The measured "
            "per-cell times above are WARM-cache times. A worker that runs one cell per process pays the cold "
            "reference (%.2f s measured) on every cell; a long-lived worker pays it four times in total (one "
            "per theta)." % cold_ref,
            "The dominant cost inside a fit cell is als_fit (4 restarts x up to 150 sweeps), not the closed-form "
            "PCA: the PCA is a single eigh of a 256x256 matrix.",
        ],
    }


# =============================================================================
# t21s -- rank-k PCA and the DGJ predictor
# =============================================================================
def _t21s_data_call(d: int, arm: str = "hi", geometry: str = "correlated"):
    from experiments.mlst2026.successor_20260913.t21s import runner as R
    P = protocol("t21s")
    n = max(2, int(round(d / 2))) if arm == "hi" else int(4 * d)
    cell = {"role": "pilot", "seed": 5000, "cell_id": "bench", "n_fits": 1, "n_solves": 1,
            "params": {"geometry": geometry, "gamma_arm": arm, "d": int(d), "n_train": n, "k": 4,
                       "kind": "grid", "varied_direction": 1, "ratio": 1.0, "phase": "pilot"}}
    return _counting(lambda: R.run_data_cell(P, cell))


def _t21s_alg_call(which: str = "geometry_invariants"):
    from experiments.mlst2026.successor_20260913.t21s import runner as R
    P = protocol("t21s")
    d = int(P["design"]["d"])
    params = ({"check": "geometry_invariants", "geometry": "correlated", "d": d, "k": 4}
              if which == "geometry_invariants" else {"check": "predictor_sanity", "d": d, "k": 4})
    cell = {"role": "algebraic", "seed": None, "cell_id": "alg", "n_fits": 0, "n_solves": 1, "params": params}
    return lambda: R.run_algebraic_cell(P, cell)


def bench_t21s(args) -> dict:
    from experiments.mlst2026.successor_20260913.t21s import runner as R
    P = protocol("t21s")
    d_prod = int(P["design"]["d"])
    d_fx = int(P["fixture"]["d"])
    sizes = [d_fx, 64, 128, d_prod]
    out_kernels = {}
    fams = {}
    for arm, n_prod in sorted(P["design"]["n_train"].items()):
        _log(f"t21s data cell, gamma arm {arm} (N = {n_prod} at d = 256), d = {sizes}")
        k = B.scaling_benchmark(
            f"data_cell_{arm}", (lambda a: (lambda d: _t21s_data_call(d, a)))(arm), sizes,
            theory_key="svd", size_symbol="d", production_size=d_prod, fixture_size=d_fx,
            repeats=5, warmup=2, memory_sizes=[d_prod], memory_probe=_mem_probe("t21s", f"data_{arm}"),
            call_path="t21s.runner.run_data_cell -> compute_predictions (eigh of the d x d population K, one "
                      "solve) then fit_pca (thin SVD of the centred N x d draw, one fit)",
            notes=f"N is held at the production aspect ratio for the {arm} arm (gamma = d/N fixed), so the "
                  f"d = 256 point IS a production {arm}-arm cell")
        out_kernels[f"data_cell_{arm}"] = k
        for phase in ("pilot", "confirmation"):
            for geom in P["design"]["geometry"]["arms"]:
                fams[f"{phase}-{geom}-{arm}"] = _family_from(k, f"data_cell_{arm}", 1, d_prod, "d")
    _log("t21s algebraic cells")
    alg_g = B.timed(_t21s_alg_call("geometry_invariants"), size=d_prod, repeats=5, warmup=2)
    alg_p = B.timed(_t21s_alg_call("predictor_sanity"), size=d_prod, repeats=5, warmup=2)
    alg_med = (2 * alg_g.median_s + alg_p.median_s) / 3.0
    fams["algebraic"] = {"kernel": "algebraic_cell", "calls_per_cell": 1, "basis": "measured", "size": d_prod,
                         "size_symbol": "d", "extrapolation_factor_in_dimension": 1.0,
                         "seconds_per_cell": {"median": alg_med,
                                              "low": min(alg_g.min_s, alg_p.min_s),
                                              "high": max(alg_g.max_s, alg_p.max_s)},
                         "how": "the three algebraic cells are two geometry-invariant checks (%.4f s each) and "
                                "one predictor sanity check (%.4f s); the family mean is their average"
                                % (alg_g.median_s, alg_p.median_s)}
    out_kernels["algebraic_cell"] = {
        "kernel": "algebraic_cell",
        "points": [dict(alg_g.as_dict(), what="geometry_invariants"), dict(alg_p.as_dict(), what="predictor_sanity")],
        "projection": {"basis": "measured", "seconds_per_call_median": alg_med,
                       "seconds_per_call_low": min(alg_g.min_s, alg_p.min_s),
                       "seconds_per_call_high": max(alg_g.max_s, alg_p.max_s),
                       "label": "measured at the production size"}}

    with fitcount.counting():
        res = R.run_data_cell(P, {"role": "pilot", "seed": 5000, "cell_id": "b", "n_fits": 1, "n_solves": 1,
                                  "params": {"geometry": "correlated", "gamma_arm": "lo", "d": d_prod,
                                             "n_train": 1024, "k": 4, "kind": "grid", "varied_direction": 1,
                                             "ratio": 1.0, "phase": "pilot"}})
    return {
        "kernels": out_kernels, "families": fams,
        "work_accounting": {
            "estimator_calls_per_data_cell": 1,
            "restarts": "none: the learner is closed-form rank-k PCA (thin SVD), it has no restarts",
            "solves_per_data_cell": 1,
            "simulation_replicates": {"draws_per_data_cell": 1, "rows_drawn": "N = 128 (hi) or 1024 (lo)"},
            "bootstrap_replicates_in_reduce": int(getattr(R, "BOOTSTRAP_REPLICATES", 0)),
        },
        "storage": {"data_cell_result_bytes_measured": B.json_bytes(res),
                    "basis": "measured: the production-size result dict serialised with the runner's encoder"},
        "notes": [
            "The protocol derives its flat scale from the PCA flop count N d min(N, d). That is NOT where the "
            "time goes at the production size: compute_predictions (the d x d eigendecomposition of K plus the "
            "QR of the direction frame) is comparable to or larger than the SVD, and at the fixture size both "
            "are dwarfed by Python overhead.",
            "Both production sizes (d = 256 with N = 128 and with N = 1024) were run directly, so the t21s "
            "projection is a MEASUREMENT.",
        ],
    }


# =============================================================================
# t22s -- Monte Carlo over the archived T22 inference pipeline
# =============================================================================
def _t22s_scenario(name: str):
    from experiments.mlst2026.successor_20260913.t22s import simulator as sim
    for s in list(sim.confirmation_scenarios()) + list(sim.pilot_scenarios()):
        if s.name == name:
            return s
    raise KeyError(name)


def _t22s_replicate_call(file_reps: int, *, scenario: str, ratio=(1, 5, 0.2)):
    from experiments.mlst2026.successor_20260913.t22s import pipeline as pl
    sc = _t22s_scenario(scenario)
    f = int(file_reps)
    hier = int(round(f * ratio[1]))
    curve = max(1, int(round(f * ratio[2])))
    return lambda: pl.run_replicate(1234, sc, file_replicates=f, hier_replicates=hier, curve_replicates=curve)


def bench_t22s(args) -> dict:
    from experiments.mlst2026.successor_20260913.t22s import pipeline as pl, runner as RN, simulator as sim
    P = protocol("t22s")
    M = P["matrix"]["production"]
    arch = M["replicate_settings"]["archived"]
    red = M["replicate_settings"]["reduced"]
    fx = P["matrix"]["fixture"]["replicate_settings"]["archived"]
    mc = int(M["mc_per_cell"])
    base = "null-complete-n4-phi0.6-tau0.01"
    sizes = [fx["file"], 250, 1000, arch["file"]]
    _log(f"t22s pipeline replicate, file-bootstrap replicates = {sizes} (production archived = {arch['file']})")
    rep = B.scaling_benchmark(
        "pipeline_replicate", (lambda f: _t22s_replicate_call(f, scenario=base)), sizes,
        theory_key="bootstrap", size_symbol="R_file", production_size=arch["file"], fixture_size=fx["file"],
        repeats=3, warmup=1, memory_sizes=[arch["file"]], memory_probe=_mem_probe("t22s", "replicate"),
        call_path="t22s.pipeline.run_replicate: te.joint_bootstrap per evaluation file (R_file), "
                  "t22.combine_stage two-stage hierarchical bootstrap (R_hier = 5 R_file) and "
                  "te.block_variance_curve on the development files (R_curve = R_file/5)",
        notes="the three replicate counts are moved together at the archived ratio 1 : 5 : 0.2, so the "
              "size axis is the whole replicate budget of one Monte-Carlo replicate; the production "
              "(archived) setting was run directly")
    # cost by number of files, at the production replicate setting
    by_files = {}
    for nm in ("null-complete-n4-phi0.6-tau0.01", "null-complete-n8-phi0.6-tau0.01",
               "null-complete-n16-phi0.6-tau0.01"):
        sc = _t22s_scenario(nm)
        _log(f"t22s replicate at the archived setting, n_files = {sc.n_files}")
        t = B.timed(lambda s=sc: pl.run_replicate(1234, s, file_replicates=arch["file"],
                                                  hier_replicates=arch["hier"], curve_replicates=arch["curve"]),
                    size=sc.n_files, repeats=3, warmup=1)
        by_files[sc.n_files] = t.as_dict()
    files_fit = B.fit_power_law(sorted(by_files), [by_files[k]["median_s"] for k in sorted(by_files)])
    # shape variations at fixed n_files, to bound the scenario-to-scenario spread
    shape = {}
    for nm in ("null-partial-n4-phi0.6-tau0.01-drift0.15", "null-partial-n4-phi0.6-tau0.01-t3",
               "alt-large-n4-phi0.85-tau0.01"):
        sc = _t22s_scenario(nm)
        t = B.timed(lambda s=sc: pl.run_replicate(1234, s, file_replicates=arch["file"],
                                                  hier_replicates=arch["hier"], curve_replicates=arch["curve"]),
                    size=sc.n_files, repeats=2, warmup=1)
        shape[nm] = t.as_dict()
    # the reduced setting, which the pilot may freeze instead of the archived one
    sc4 = _t22s_scenario(base)
    t_red = B.timed(lambda: pl.run_replicate(1234, sc4, file_replicates=red["file"], hier_replicates=red["hier"],
                                             curve_replicates=red["curve"]), size=red["file"], repeats=3, warmup=1)

    def per_replicate(n_files: int, key: str) -> dict:
        row = by_files[n_files]
        return {"median": row["median_s"], "low": row["min_s"], "high": row["max_s"]}

    # census of the production catalogue by n_files
    conf = list(sim.confirmation_scenarios())
    pil = list(sim.pilot_scenarios())
    conf_census = {}
    for s in conf:
        conf_census[s.n_files] = conf_census.get(s.n_files, 0) + 1
    pil_census = {}
    for s in pil:
        pil_census[s.n_files] = pil_census.get(s.n_files, 0) + 1
    n_conf_chunks = M["mc_replicates_per_scenario"] // mc
    n_pil_chunks = M["pilot_mc_per_scenario"] // mc

    def weighted(census, ratio_reduced=0.0):
        tot = sum(census.values())
        med = sum(by_files[n]["median_s"] * c for n, c in census.items()) / tot
        lo = sum(by_files[n]["min_s"] * c for n, c in census.items()) / tot
        hi = sum(by_files[n]["max_s"] * c for n, c in census.items()) / tot
        return med, lo, hi

    c_med, c_lo, c_hi = weighted(conf_census)
    p_med, p_lo, p_hi = weighted(pil_census)
    # pilot cells: half at the archived setting, half at the reduced one
    ratio_red = t_red.median_s / by_files[4]["median_s"]
    p_med_mix = p_med * 0.5 * (1.0 + ratio_red)
    p_lo_mix = p_lo * 0.5 * (1.0 + ratio_red)
    p_hi_mix = p_hi * 0.5 * (1.0 + ratio_red)

    _log("t22s production control cells")
    controls = {}
    ctrl_cells = RN._control_cells(P, "production")
    ctx = {"out": Path(tempfile.gettempdir()), "fixture_overrides": None}
    for cell in ctrl_cells:
        nm = cell["params"]["control"]
        t = B.timed(lambda c=cell: RN._run_control_cell(P, c, False, ctx), size=1, repeats=2, warmup=0)
        controls[nm] = t.as_dict()
    ctrl_med = sum(v["median_s"] for v in controls.values()) / len(controls)
    ctrl_lo = sum(v["min_s"] for v in controls.values()) / len(controls)
    ctrl_hi = sum(v["max_s"] for v in controls.values()) / len(controls)

    one_rec = pl.run_replicate(999, sc4, file_replicates=arch["file"], hier_replicates=arch["hier"],
                               curve_replicates=arch["curve"])
    fams = {
        "sim_confirm": {"kernel": "pipeline_replicate", "calls_per_cell": mc, "basis": "measured",
                        "size": arch["file"], "size_symbol": "R_file",
                        "extrapolation_factor_in_dimension": 1.0,
                        "fitted_exponent": rep["fit"]["b"], "theoretical_exponent": rep["theory"]["exponent"],
                        "seconds_per_cell": {"median": mc * c_med, "low": mc * c_lo, "high": mc * c_hi},
                        "how": ("%d Monte-Carlo replicates per cell x the n_files-weighted measured replicate "
                                "time (%s) at the ARCHIVED replicate setting; census of the 69 confirmation "
                                "scenarios by n_files = %s"
                                % (mc, ", ".join("n=%d:%.3fs" % (n, by_files[n]["median_s"]) for n in sorted(by_files)),
                                   conf_census))},
        "sim_pilot": {"kernel": "pipeline_replicate", "calls_per_cell": mc, "basis": "measured",
                      "size": arch["file"], "size_symbol": "R_file",
                      "extrapolation_factor_in_dimension": 1.0,
                      "fitted_exponent": rep["fit"]["b"], "theoretical_exponent": rep["theory"]["exponent"],
                      "seconds_per_cell": {"median": mc * p_med_mix, "low": mc * p_lo_mix, "high": mc * p_hi_mix},
                      "how": ("%d replicates per cell; half the pilot cells run the archived setting and half "
                              "the reduced one (measured ratio reduced/archived = %.3f), n_files census %s"
                              % (mc, ratio_red, pil_census))},
        "control": {"kernel": "control_cell", "calls_per_cell": 1, "basis": "measured", "size": 1,
                    "size_symbol": "cell", "extrapolation_factor_in_dimension": 1.0,
                    "seconds_per_cell": {"median": ctrl_med, "low": ctrl_lo, "high": ctrl_hi},
                    "how": "each of the five production control cells run once at its production configuration: "
                           + ", ".join("%s %.3fs" % (k, v["median_s"]) for k, v in sorted(controls.items()))},
    }
    return {
        "kernels": {"pipeline_replicate": rep,
                    "replicate_by_n_files": {"kernel": "replicate_by_n_files", "size_symbol": "n_files",
                                             "points": [dict(by_files[k], size=k) for k in sorted(by_files)],
                                             "fit": files_fit.as_dict(),
                                             "theory": {"key": "bootstrap", "exponent": 1.0,
                                                        "note": "the joint bootstrap is run once per evaluation "
                                                                "file, so the replicate is linear in n_files "
                                                                "above a fixed simulation cost"},
                                             "exponent_check": B.exponent_verdict(files_fit.b, 1.0),
                                             "projection": {"basis": "measured",
                                                            "label": "every production n_files (4, 8, 16) was run"}},
                    "control_cell": {"kernel": "control_cell", "points": [dict(v, what=k) for k, v in sorted(controls.items())],
                                     "projection": {"basis": "measured", "seconds_per_call_median": ctrl_med,
                                                    "seconds_per_call_low": ctrl_lo, "seconds_per_call_high": ctrl_hi,
                                                    "label": "measured at the production configuration"}}},
        "families": fams,
        "work_accounting": {
            "estimator_calls": 0,
            "estimator_calls_note": "the production matrix plans 0 estimator fits: the simulator generates the "
                                    "learned estimates, so no estimator is invoked on drawn data. All the work "
                                    "is simulation and bootstrap replicates.",
            "restarts": "not applicable (nothing is fitted)",
            "simulation_replicates_per_confirmation_cell": {
                "monte_carlo": mc,
                "joint_bootstrap_draws": "mc x n_files x R_file = %d x n_files x %d" % (mc, arch["file"]),
                "two_stage_bootstrap_draws": "mc x 4 contrasts x R_hier = %d x 4 x %d" % (mc, arch["hier"]),
                "block_curve_draws": "mc x n_dev x (4 x %d + %d) x R_curve"
                                     % (len(pl.BLOCKS_EVAL), len(pl.BLOCKS_CAL))},
            "replicate_settings_production": {"archived": arch, "reduced": red},
            "frozen_setting_unknown": "the confirmation setting is frozen by the pilot; the archived setting is "
                                      "the more expensive of the two and is what is priced here, so the "
                                      "confirmation total is an UPPER bound (measured ratio reduced/archived "
                                      "= %.3f at n_files = 4)" % ratio_red,
        },
        "storage": {"simulation_cell_record_bytes_measured": B.json_bytes(one_rec),
                    "records_per_cell": mc,
                    "cell_result_bytes_projected": B.json_bytes(one_rec) * mc,
                    "basis": "measured for one Monte-Carlo record at n_files = 4 and the archived setting; "
                             "a cell holds mc = %d such records, so the cell file is projected linearly in mc "
                             "(the records list is exactly mc entries long)" % mc},
        "scenario_shape_spread": shape,
        "notes": [
            "The production replicate setting (file 2000 / hier 10000 / curve 400) was run directly, so the "
            "per-replicate cost is a MEASUREMENT; only the multiplication by mc = %d and by the cell count is "
            "arithmetic." % mc,
            "The dominant cost is te.joint_bootstrap, which runs once per evaluation file: the replicate time "
            "is close to linear in n_files, and the confirmation catalogue is 1/3 at n_files = 16.",
        ],
    }


# =============================================================================
# cresst_s -- archive recomputation, cluster reporting, EMPCA robustness, calibration
# =============================================================================
def _cresst_pulse_root():
    """A synthetic pulse root at the PRODUCTION dimensions, written by the package's
    own fixture writer.  The real CRESST traces are not in this container; the EMPCA
    cost depends on the dimensions and the fixed iteration count, not on the values."""
    from experiments.mlst2026.successor_20260913.cresst_s import robustness_design as RD
    P = protocol("cresst_s")
    dims = P["robustness"]["production_dimensions"]
    root = Path(tempfile.mkdtemp(prefix="cresst_prod_dims_"))
    RD.write_fixture_data_root(root, seed=20260913, n_events=int(dims["n_events"]),
                               n_samples=int(dims["n_samples"]), n_noise=int(dims["n_noise"]))
    return root, dims


_CRESST_CH = {}


def _cresst_channel():
    if "ch" not in _CRESST_CH:
        from experiments.mlst2026.successor_20260913.cresst_s import robustness_design as RD
        root, dims = _cresst_pulse_root()
        _CRESST_CH["ch"] = RD.load_pulse_channel(root, *RD.ARCHIVE_CHANNELS[0])
        _CRESST_CH["dims"] = dims
    return _CRESST_CH["ch"], _CRESST_CH["dims"]


def _cresst_robust_call(rank: int, time_cut: str = "full"):
    from experiments.mlst2026.successor_20260913.cresst_s import robustness_design as RD
    P = protocol("cresst_s")
    ch, _ = _cresst_channel()
    niter = int(P["robustness"]["empca_iterations"])
    sh = float(P["robustness"]["primary_setting"]["shrinkage"])
    return _counting(lambda: RD.fit_and_score(ch["pulses"], ch["times"], ch["noise"], rank=int(rank),
                                              shrinkage=sh, time_cut=time_cut, seed=1, niter=niter))


def bench_cresst_s(args) -> dict:
    from experiments.mlst2026.successor_20260913.cresst_s import (archive_recompute as AR, calibration as CB,
                                                                  robustness_design as RD, runner as RU)
    P = protocol("cresst_s")
    if not args.ms_evidence:
        raise SystemExit("cresst_s benchmarks need --ms-evidence <MS>/evidence/data")
    RU.set_roots(archive_root=args.ms_evidence)
    S = P["matrix"]["production"]
    reps_prod = int(S["archive_replicates"])
    reps_fx = int(P["matrix"]["fixture"]["archive_replicates"])
    A = RU._archive()

    _log("cresst_s archive interval, replicates = %s" % [reps_fx, 2500, 5000, reps_prod])
    interval = B.scaling_benchmark(
        "archive_interval", (lambda r: (lambda: AR.recompute_interval("headline", A["clusters"],
                                                                      replicates=int(r), seed=AR.ARCHIVE_SEED))),
        [reps_fx, 2500, 5000, reps_prod], theory_key="bootstrap", size_symbol="R", production_size=reps_prod,
        fixture_size=reps_fx, repeats=3, warmup=1, memory_sizes=[reps_prod],
        memory_probe=_mem_probe("cresst_s", "interval"),
        call_path="cresst_s.archive_recompute.recompute_interval: a cluster bootstrap over the "
                  "%d archived evaluation events in 7 channels" % RU.N_ARCHIVE_EVENTS,
        notes="the production replicate count was run directly")
    _log("cresst_s archive interval, all %d constructions at production replicates" % len(AR.CONSTRUCTIONS))
    per_constr = {}
    for name in sorted(AR.CONSTRUCTIONS):
        t = B.timed(lambda n=name: AR.recompute_interval(n, A["clusters"], replicates=reps_prod,
                                                         seed=AR.ARCHIVE_SEED), size=reps_prod, repeats=1, warmup=0)
        per_constr[name] = t.as_dict()
    iv_med = sum(v["median_s"] for v in per_constr.values()) / len(per_constr)
    iv_lo = min(v["min_s"] for v in per_constr.values())
    iv_hi = max(v["max_s"] for v in per_constr.values())

    _log("cresst_s cluster report, replicate scaling and all five reports")
    cr_scale = B.scaling_benchmark(
        "cluster_report", (lambda r: (lambda: RU._cell_cluster_report(
            P, {"cell_id": "b", "role": "control", "seed": None,
                "params": {"analysis": "cluster_report", "report": "per_channel_effects", "replicates": int(r),
                           "archive_seed": AR.ARCHIVE_SEED, "scale": "production"}}))),
        [reps_fx, 2500, 5000, reps_prod], theory_key="bootstrap", size_symbol="R", production_size=reps_prod,
        fixture_size=reps_fx, repeats=2, warmup=1, memory_sizes=[reps_prod],
        call_path="cresst_s.runner._cell_cluster_report -> cluster_report.* (cluster bootstrap per report)",
        notes="the production replicate count was run directly")
    per_report = {}
    for rep in RU.CLUSTER_REPORTS:
        t = B.timed(lambda r=rep: RU._cell_cluster_report(
            P, {"cell_id": "b", "role": "control", "seed": None,
                "params": {"analysis": "cluster_report", "report": r, "replicates": reps_prod,
                           "archive_seed": AR.ARCHIVE_SEED, "scale": "production"}}),
            size=reps_prod, repeats=1, warmup=0)
        per_report[rep] = t.as_dict()
    cr_med = sum(v["median_s"] for v in per_report.values()) / len(per_report)
    cr_lo = min(v["min_s"] for v in per_report.values())
    cr_hi = max(v["max_s"] for v in per_report.values())

    _log("cresst_s robustness (EMPCA) at the production dimensions, rank 2..12, both time-cut classes")
    ch, dims = _cresst_channel()
    robust = B.scaling_benchmark(
        "empca_robustness", _cresst_robust_call, [2, 4, 6, 8, 12], theory_key="em_sweep", size_symbol="rank k",
        production_size=6, fixture_size=4, repeats=3, warmup=1, memory_sizes=[12],
        memory_probe=_mem_probe("cresst_s", "robust"),
        call_path="cresst_s.robustness_design.fit_and_score -> src.canonical.empca.classic_pca (1 fit) + "
                  "empca (EMPCA_RESTARTS = 2 fits, niter = 25)",
        notes="the real CRESST pulse traces are NOT in this container; the synthetic pulse root written by the "
              "package's own write_fixture_data_root is used AT THE PRODUCTION DIMENSIONS "
              "(n_events %d, n_samples %d, n_noise %d). The EMPCA cost is set by those dimensions and by the "
              "fixed niter, not by the values, but the substitution is a substitution and is labelled here."
              % (dims["n_events"], dims["n_samples"], dims["n_noise"]))
    robust_by = {}
    for cut in ("full", "first_half"):
        for rank in (2, 4, 6, 8, 12):
            t = B.timed(_cresst_robust_call(rank, cut), size=rank, repeats=3, warmup=1)
            robust_by[f"{cut}|{rank}"] = t.as_dict()

    _log("cresst_s science and calibration controls at their production configurations")
    sci = {}
    for name in RU.SCIENCE_CONTROLS:
        cfg = dict(S["controls"][name])
        cell = {"cell_id": "b", "role": "control", "seed": None,
                "params": {"analysis": "science_control", "control": name, "scale": "production", **cfg}}
        t = B.timed(_counting(lambda c=cell: RU._cell_science_control(P, c)), size=1, repeats=2, warmup=0)
        sci[name] = t.as_dict()
    sci_med = sum(v["median_s"] for v in sci.values()) / len(sci)
    cc_cfg = dict(S["controls"]["calibration_null_dependence"])
    cc_cell = {"cell_id": "b", "role": "control", "seed": None,
               "params": {"analysis": "calibration_control", "control": "null_dependence",
                          "scale": "production", **cc_cfg}}
    calib_ctrl = B.timed(lambda: RU._cell_calibration_control(P, cc_cell), size=1, repeats=2, warmup=0)

    _log("cresst_s calibration designs (all 24, at their production replicate budgets)")
    calib = {"pilot": {}, "confirmation": {}}
    for arm in ("pilot", "confirmation"):
        for i, d in enumerate(RU._calibration_designs(P, arm, "production")):
            t = B.timed(lambda dd=d: CB.coverage(dd, seed=1), size=d.mc_replicates * d.bootstrap_replicates * d.n_clusters,
                        repeats=2, warmup=0)
            calib[arm][f"runs{d.n_runs}|cpr{d.channels_per_run}|sd{d.sd_run}"] = dict(t.as_dict(), n_clusters=d.n_clusters)
    # a proper replicate-scaling series at the reference design
    ref = RU._calibration_designs(P, "confirmation", "production")[0]

    def _calib_at(mc):
        import dataclasses
        d = dataclasses.replace(ref, mc_replicates=int(mc))
        return lambda: CB.coverage(d, seed=1)
    calib_scale = B.scaling_benchmark(
        "calibration_coverage", _calib_at, [50, 100, 200, ref.mc_replicates], theory_key="bootstrap",
        size_symbol="mc", production_size=ref.mc_replicates, fixture_size=120, repeats=3, warmup=1,
        memory_sizes=[ref.mc_replicates],
        call_path="cresst_s.calibration.coverage: mc Monte-Carlo datasets x bootstrap_replicates cluster "
                  "bootstrap draws each",
        notes="at the reference design (n_runs %d, channels_per_run %d); the production mc was run directly"
              % (ref.n_runs, ref.channels_per_run))

    ap_cell = {"cell_id": "b", "role": "algebraic", "seed": None,
               "params": {"analysis": "archive_point_estimates", "scale": "production"}}
    ap = B.timed(lambda: RU._cell_archive_point(P, ap_cell), size=1, repeats=3, warmup=1)

    def _mean(d, key="median_s"):
        return sum(v[key] for v in d.values()) / len(d)

    # per-cell seconds for the robustness families, from the measured (cut, rank) table
    def robust_family(arm):
        rows = RD.grid_rows(P, arm)
        vals = []
        for r in rows:
            cut = "full" if r["time_cut"] == "full" else "first_half"
            vals.append(robust_by[f"{cut}|{int(r['rank'])}"])
        return ({"median": sum(v["median_s"] for v in vals) / len(vals),
                 "low": sum(v["min_s"] for v in vals) / len(vals),
                 "high": sum(v["max_s"] for v in vals) / len(vals)}, len(rows))

    rp, n_rp = robust_family("pilot")
    rc, n_rc = robust_family("confirmation")

    def calib_family(arm):
        v = calib[arm]
        return {"median": _mean(v), "low": _mean(v, "min_s"), "high": _mean(v, "max_s")}

    fams = {
        "archive_point": {"kernel": "archive_point", "calls_per_cell": 1, "basis": "measured", "size": 1,
                          "size_symbol": "cell", "extrapolation_factor_in_dimension": 1.0,
                          "seconds_per_cell": {"median": ap.median_s, "low": ap.min_s, "high": ap.max_s},
                          "how": "measured on the real archived inputs"},
        "archive_interval": {"kernel": "archive_interval", "calls_per_cell": 1, "basis": "measured",
                             "size": reps_prod, "size_symbol": "R",
                             "extrapolation_factor_in_dimension": 1.0,
                             "fitted_exponent": interval["fit"]["b"], "theoretical_exponent": 1.0,
                             "seconds_per_cell": {"median": iv_med, "low": iv_lo, "high": iv_hi},
                             "how": "mean over the ten archived constructions, each run once at the production "
                                    "10000 replicates: " + ", ".join("%s %.2fs" % (k, v["median_s"])
                                                                     for k, v in sorted(per_constr.items()))},
        "cluster_report": {"kernel": "cluster_report", "calls_per_cell": 1, "basis": "measured", "size": reps_prod,
                           "size_symbol": "R", "extrapolation_factor_in_dimension": 1.0,
                           "fitted_exponent": cr_scale["fit"]["b"], "theoretical_exponent": 1.0,
                           "seconds_per_cell": {"median": cr_med, "low": cr_lo, "high": cr_hi},
                           "how": "mean over the five reports at the production 10000 replicates: "
                                  + ", ".join("%s %.2fs" % (k, v["median_s"]) for k, v in sorted(per_report.items()))},
        "science_control": {"kernel": "science_control", "calls_per_cell": 1, "basis": "measured", "size": 1,
                            "size_symbol": "cell", "extrapolation_factor_in_dimension": 1.0,
                            "seconds_per_cell": {"median": sci_med, "low": min(v["min_s"] for v in sci.values()),
                                                 "high": max(v["max_s"] for v in sci.values())},
                            "how": "mean over the four production science controls: "
                                   + ", ".join("%s %.3fs" % (k, v["median_s"]) for k, v in sorted(sci.items()))},
        "calib_control": {"kernel": "calibration_control", "calls_per_cell": 1, "basis": "measured", "size": 1,
                          "size_symbol": "cell", "extrapolation_factor_in_dimension": 1.0,
                          "seconds_per_cell": {"median": calib_ctrl.median_s, "low": calib_ctrl.min_s,
                                               "high": calib_ctrl.max_s},
                          "how": "measured at the production configuration (mc %d x bootstrap %d, two arms)"
                                 % (cc_cfg["mc_replicates"], cc_cfg["bootstrap_replicates"])},
        "robustness_pilot": {"kernel": "empca_robustness", "calls_per_cell": 1, "basis": "measured-on-substitute",
                             "size": "rank 2..12", "size_symbol": "rank k",
                             "extrapolation_factor_in_dimension": 1.0,
                             "fitted_exponent": robust["fit"]["b"], "theoretical_exponent": 1.0,
                             "seconds_per_cell": rp,
                             "how": "mean over the %d pilot grid rows of the measured (time cut, rank) table at "
                                    "the production dimensions on SYNTHETIC pulses (no CRESST raw data here)" % n_rp},
        "robustness_confirm": {"kernel": "empca_robustness", "calls_per_cell": 1, "basis": "measured-on-substitute",
                               "size": "rank 2..12", "size_symbol": "rank k",
                               "extrapolation_factor_in_dimension": 1.0,
                               "fitted_exponent": robust["fit"]["b"], "theoretical_exponent": 1.0,
                               "seconds_per_cell": rc,
                               "how": "mean over the %d confirmation grid rows of the measured (time cut, rank) "
                                      "table at the production dimensions on SYNTHETIC pulses" % n_rc},
        "calib_pilot": {"kernel": "calibration_coverage", "calls_per_cell": 1, "basis": "measured",
                        "size": "mc 200 x boot 300", "size_symbol": "mc.boot.clusters",
                        "extrapolation_factor_in_dimension": 1.0,
                        "fitted_exponent": calib_scale["fit"]["b"], "theoretical_exponent": 1.0,
                        "seconds_per_cell": calib_family("pilot"),
                        "how": "every one of the 12 production pilot designs was run at its own replicate budget; "
                               "the family value is their mean"},
        "calib_confirm": {"kernel": "calibration_coverage", "calls_per_cell": 1, "basis": "measured",
                          "size": "mc 400 x boot 400", "size_symbol": "mc.boot.clusters",
                          "extrapolation_factor_in_dimension": 1.0,
                          "fitted_exponent": calib_scale["fit"]["b"], "theoretical_exponent": 1.0,
                          "seconds_per_cell": calib_family("confirmation"),
                          "how": "every one of the 12 production confirmation designs was run at its own "
                                 "replicate budget; the family value is their mean"},
    }
    res_iv = RU._cell_archive_interval(P, {"cell_id": "b", "role": "control", "seed": None,
                                           "params": {"analysis": "archive_interval", "construction": "headline",
                                                      "replicates": reps_prod, "archive_seed": AR.ARCHIVE_SEED,
                                                      "scale": "production"}})
    return {
        "kernels": {"archive_interval": interval, "cluster_report": cr_scale, "empca_robustness": robust,
                    "calibration_coverage": calib_scale,
                    "archive_point": {"kernel": "archive_point", "points": [ap.as_dict()],
                                      "projection": {"basis": "measured", "seconds_per_call_median": ap.median_s,
                                                     "seconds_per_call_low": ap.min_s, "seconds_per_call_high": ap.max_s,
                                                     "label": "measured at the production size"}}},
        "families": fams,
        "measured_tables": {"interval_by_construction": per_constr, "cluster_report_by_report": per_report,
                            "robustness_by_cut_and_rank": robust_by, "science_controls": sci,
                            "calibration_by_design": calib},
        "work_accounting": {
            "estimator_calls_production": int(P["budget"]["fits_planned_production"]),
            "restarts": {"empca_restarts_per_robustness_cell": int(RD.EMPCA_RESTARTS),
                         "fits_per_robustness_cell": int(RD.FITS_PER_CELL),
                         "note": "1 classic PCA + 2 EMPCA restarts = 3 estimator invocations per robustness "
                                 "cell; the restart count is reported separately, not folded in"},
            "empca_iterations": int(P["robustness"]["empca_iterations"]),
            "solves": "0 by construction (protocol solve_definition)",
            "simulation_replicates": {
                "archive_interval_per_cell": reps_prod,
                "cluster_report_per_cell": "1x to 14x %d depending on the report" % reps_prod,
                "calibration_pilot_per_cell": 200 * 300, "calibration_confirmation_per_cell": 400 * 400,
                "calibration_control": 2 * cc_cfg["mc_replicates"] * cc_cfg["bootstrap_replicates"]},
        },
        "storage": {"archive_interval_result_bytes_measured": B.json_bytes(res_iv),
                    "basis": "measured: a production-configuration result dict serialised with the runner's encoder"},
        "notes": [
            "Every cresst_s family except the robustness grid was benchmarked at its exact production "
            "configuration on the real archived inputs, so those projections are MEASUREMENTS.",
            "The robustness cells cannot be executed here at all: they need raw CRESST pulse traces that are "
            "not in this container. Their cost was measured on a synthetic pulse root written at the "
            "production dimensions by the package's own writer. That is a substitute for the data, not for "
            "the dimensions; a real channel with a different event count would scale linearly in n_train.",
        ],
    }


# =============================================================================
# generic helpers
# =============================================================================
def _family_from(kernel: dict, name: str, calls: int, size, symbol: str) -> dict:
    p = kernel["projection"]
    return {"kernel": name, "calls_per_cell": int(calls), "basis": p["basis"], "size": size,
            "size_symbol": symbol,
            "fitted_exponent": kernel["fit"]["b"], "theoretical_exponent": kernel["theory"]["exponent"],
            "extrapolation_factor_in_dimension": p.get("extrapolation_factor_in_size", 1.0),
            "seconds_per_cell": {"median": calls * p["seconds_per_call_median"],
                                 "low": calls * p["seconds_per_call_low"],
                                 "high": calls * p["seconds_per_call_high"]},
            "how": p["how"] if "how" in p else p.get("label", ""), "projection_label": p.get("label", "")}



# =============================================================================
# output storage: the bytes one PRODUCTION cell file costs, per family
# =============================================================================
def _bytes(obj, how, basis="measured"):
    return {"bytes": B.json_bytes(obj), "basis": basis, "how": how}


def storage_t19s(args, _bench):
    from experiments.mlst2026.successor_20260913.t19s import runner as R, synthetic as S
    P = protocol("t19s")
    D = P["design"]["production"]
    pp = D["paired"]
    with fitcount.counting():
        paired, _, _ = S.run_paired_cell(
            {"kind": "paired", "d": int(D["d"]), "k": 2, "n_train": int(D["n_train"]), "eps_grid": pp["eps_grid"],
             "m_over_d_grid": pp["m_over_d_grid"], "n_train_ratio_grid": pp["n_train_ratio_grid"],
             "rng_offsets": pp["rng_offsets"]}, 1000, selection_sha256=None)
    det, _, _ = R.run_deterministic_cell(
        {"kind": "deterministic", "model": "covariance", "family": "generic", "d": int(D["d"]), "k": 2,
         "operator": {"kind": "g1_population", "d": int(D["d"])}, "eps_grid": D["deterministic"]["eps_grid"],
         "direction_seed": 20260913})
    two, _, _ = R.run_deterministic_cell(
        {"kind": "deterministic", "model": "covariance", "family": "two_d_example", "d": 2, "k": 1,
         "operator": {"kind": "two_d_example"}, "eps_grid": D["deterministic"]["two_d_example_eps_grid"],
         "reference": D["deterministic"]["two_d_reference"]})
    out = {"paired": _bytes(paired, "one production paired cell result (d = 256, N = 512)"),
           "det_two_d": _bytes(two, "the 2-D closed-form cell result")}
    for fam in ("det_generic", "det_pure_coupling", "ctrl_block_diagonal", "ctrl_zero",
                "ctrl_additive_generic", "ctrl_additive_pure_coupling"):
        out[fam] = _bytes(det, "one production deterministic cell result (d = 256, 8 eps points)")
    return out


def storage_t20s(args, _bench):
    from experiments.mlst2026.successor_20260913.t20s import runner as R
    P = protocol("t20s")
    D = R.design(P, False)
    base = {k: D[k] for k in ("C", "T", "k_c", "k_t", "spectrum", "geometry_seed", "R_pop",
                              "pop_als_n_iter", "pop_als_tol")}
    cell = {"role": "pilot", "seed": 3000, "family": "pilot", "cell_id": "b", "n_fits": 5, "n_solves": 0,
            "params": {**base, "kind": "fit", "phase": "pilot", "theta": 0.25, "theta_index": 2, "N": 1600,
                       "N_index": 4, "band": "core", "R": 4, "als_n_iter": 150, "als_tol": 1e-10,
                       "control": None, "fixture": False}}
    with fitcount.counting():
        fit, _ = R.run_fit_cell(P, cell, D, {"out": Path(tempfile.gettempdir()), "fixture_overrides": None})
    alg = _t20s_algebraic_call()()
    return {"pilot": _bytes(fit, "one production fit cell result (N = 1600)"),
            "confirmation": _bytes(fit, "one production fit cell result (N = 1600)"),
            "algebraic": _bytes(alg, "one production algebraic cell result")}


def storage_t21s(args, _bench):
    from experiments.mlst2026.successor_20260913.t21s import runner as R
    P = protocol("t21s")
    d = int(P["design"]["d"])
    out = {}
    for arm, n in sorted(P["design"]["n_train"].items()):
        with fitcount.counting():
            res = R.run_data_cell(P, {"role": "pilot", "seed": 5000, "cell_id": "b", "n_fits": 1, "n_solves": 1,
                                      "params": {"geometry": "correlated", "gamma_arm": arm, "d": d,
                                                 "n_train": int(n), "k": 4, "kind": "grid",
                                                 "varied_direction": 1, "ratio": 1.0, "phase": "pilot"}})
        for phase in ("pilot", "confirmation"):
            for geom in P["design"]["geometry"]["arms"]:
                out[f"{phase}-{geom}-{arm}"] = _bytes(res, f"one production data cell result (d = {d}, N = {n})")
    out["algebraic"] = _bytes(_t21s_alg_call("geometry_invariants")(), "one geometry-invariant algebraic cell")
    return out


def storage_t22s(args, _bench):
    from experiments.mlst2026.successor_20260913.t22s import pipeline as pl, runner as RN
    P = protocol("t22s")
    M = P["matrix"]["production"]
    mc = int(M["mc_per_cell"])
    red = P["matrix"]["fixture"]["replicate_settings"]["reduced"]
    recs = {}
    for nm in ("null-complete-n4-phi0.6-tau0.01", "null-complete-n8-phi0.6-tau0.01",
               "null-complete-n16-phi0.6-tau0.01"):
        sc = _t22s_scenario(nm)
        recs[sc.n_files] = pl.run_replicate(7, sc, file_replicates=red["file"], hier_replicates=red["hier"],
                                            curve_replicates=red["curve"])
    from experiments.mlst2026.successor_20260913.t22s import simulator as sim
    conf, pil = list(sim.confirmation_scenarios()), list(sim.pilot_scenarios())

    def weighted(scs):
        return sum(B.json_bytes(recs[s.n_files]) for s in scs) / len(scs)
    ctx = {"out": Path(tempfile.gettempdir()), "fixture_overrides": None}
    ctrl = RN._run_control_cell(P, RN._control_cells(P, "production")[0], False, ctx)
    how = ("mc = %d Monte-Carlo records per cell; one record measured per n_files (4/8/16) and averaged over "
           "the scenario catalogue. The record was produced at the reduced replicate setting: a record's SIZE "
           "does not depend on the replicate counts (it holds summaries, not draws), only on n_files."
           % mc)
    return {"sim_confirm": {"bytes": int(mc * weighted(conf)), "basis": "projected",
                            "how": how + " Projected linearly in mc (the records list is exactly mc long)."},
            "sim_pilot": {"bytes": int(mc * weighted(pil)), "basis": "projected", "how": how},
            "control": _bytes(ctrl, "one production control cell result")}


def storage_cresst_s(args, _bench):
    from experiments.mlst2026.successor_20260913.cresst_s import (archive_recompute as AR, calibration as CB,
                                                                  robustness_design as RD, runner as RU)
    import dataclasses
    P = protocol("cresst_s")
    RU.set_roots(archive_root=args.ms_evidence)
    S = P["matrix"]["production"]
    reps = int(S["archive_replicates"])
    cheap = int(P["matrix"]["fixture"]["archive_replicates"])
    note = ("run at the fixture replicate count: the result is a summary whose size does not depend on the "
            "number of bootstrap draws")
    out = {}
    out["archive_point"] = _bytes(RU._cell_archive_point(P, {"cell_id": "b", "role": "algebraic", "seed": None,
                                                             "params": {"analysis": "archive_point_estimates",
                                                                        "scale": "production"}}),
                                  "one production archive-point cell result")
    out["archive_interval"] = _bytes(
        RU._cell_archive_interval(P, {"cell_id": "b", "role": "control", "seed": None,
                                      "params": {"analysis": "archive_interval", "construction": "headline",
                                                 "replicates": reps, "archive_seed": AR.ARCHIVE_SEED,
                                                 "scale": "production"}}),
        "one production archive-interval cell result (headline construction, %d replicates)" % reps)
    sizes = []
    for rep in RU.CLUSTER_REPORTS:
        sizes.append(B.json_bytes(RU._cell_cluster_report(
            P, {"cell_id": "b", "role": "control", "seed": None,
                "params": {"analysis": "cluster_report", "report": rep, "replicates": cheap,
                           "archive_seed": AR.ARCHIVE_SEED, "scale": "production"}})))
    out["cluster_report"] = {"bytes": int(sum(sizes) / len(sizes)), "basis": "measured",
                             "how": "mean over the five cluster reports, " + note}
    sci = []
    for name in RU.SCIENCE_CONTROLS:
        cfg = dict(S["controls"][name])
        with fitcount.counting():
            sci.append(B.json_bytes(RU._cell_science_control(
                P, {"cell_id": "b", "role": "control", "seed": None,
                    "params": {"analysis": "science_control", "control": name, "scale": "production", **cfg}})))
    out["science_control"] = {"bytes": int(sum(sci) / len(sci)), "basis": "measured",
                              "how": "mean over the four production science controls"}
    cc = dict(S["controls"]["calibration_null_dependence"])
    cc_small = {**cc, "mc_replicates": 40, "bootstrap_replicates": 60}
    out["calib_control"] = _bytes(RU._cell_calibration_control(
        P, {"cell_id": "b", "role": "control", "seed": None,
            "params": {"analysis": "calibration_control", "control": "null_dependence", "scale": "production",
                       **cc_small}}), "the calibration control cell result, " + note)
    ch, dims = _cresst_channel()
    with fitcount.counting():
        rb = RD.fit_and_score(ch["pulses"], ch["times"], ch["noise"], rank=6,
                              shrinkage=float(P["robustness"]["primary_setting"]["shrinkage"]),
                              time_cut="full", seed=1, niter=int(P["robustness"]["empca_iterations"]))
    for fam in ("robustness_pilot", "robustness_confirm"):
        out[fam] = _bytes(rb, "one robustness cell result at the production dimensions (synthetic pulses)")
    for arm, fam in (("pilot", "calib_pilot"), ("confirmation", "calib_confirm")):
        d0 = RU._calibration_designs(P, arm, "production")[0]
        out[fam] = _bytes(CB.coverage(dataclasses.replace(d0, mc_replicates=25), seed=1),
                          "one calibration cell result, " + note)
    return out


STORAGE_PROBES = {"t19s": storage_t19s, "t20s": storage_t20s, "t21s": storage_t21s,
                  "t22s": storage_t22s, "cresst_s": storage_cresst_s}


REDUCE_ARGS = {"cresst_s": ["--archive-root", "{ms}"]}


def measure_reduce_and_storage(pkg: str, run_root: Path, args) -> dict:
    """Time the package's own ``reduce`` on a copy of the fixture run root, and
    measure what a cell file and a ledger actually cost on disk."""
    if not (run_root / "run_matrix_summary.json").exists():
        return {"measured": False, "why": f"no fixture run root at {run_root}"}
    tmp = Path(tempfile.mkdtemp(prefix=f"reduce_{pkg}_"))
    dst = tmp / "root"
    shutil.copytree(run_root, dst)
    for stale in ("REDUCED.json", "RESOURCE_ESTIMATE.json"):
        (dst / stale).unlink(missing_ok=True)
    cells = sorted((dst / "cells").glob("*.json"))
    extra = [a.format(ms=args.ms_evidence or "") for a in REDUCE_ARGS.get(pkg, [])]
    cmd = [sys.executable, "-m", f"experiments.mlst2026.successor_20260913.{pkg}", "reduce",
           "--protocol", str(HERE / pkg / PROTOCOLS[pkg]), "--out", str(dst)] + extra
    env = dict(os.environ)
    env.update(B.THREAD_ENV)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    wall = time.perf_counter() - t0
    ok = proc.returncode == 0
    # the same reduce, second time, to separate interpreter start-up from the work
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    wall2 = time.perf_counter() - t0
    startup = _interpreter_startup(pkg, env)
    sizes = [p.stat().st_size for p in cells]
    ledger = (dst / "EXECUTION_LEDGER.jsonl")
    reduced = (dst / "REDUCED.json")
    out = {
        "measured": bool(ok),
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-400:] if not ok else "",
        "cells_read": len(cells),
        "wall_seconds_total_including_interpreter": min(wall, wall2),
        "interpreter_and_import_seconds_measured": startup,
        "wall_seconds_reduce_work": max(0.0, min(wall, wall2) - startup),
        "seconds_per_cell": (max(0.0, min(wall, wall2) - startup) / len(cells)) if cells else None,
        "input_bytes_read": sum(sizes) + (ledger.stat().st_size if ledger.exists() else 0),
        "throughput_bytes_per_second": ((sum(sizes) / max(1e-9, min(wall, wall2) - startup))
                                        if cells and min(wall, wall2) > startup else None),
        "reduced_output_bytes": reduced.stat().st_size if reduced.exists() else None,
        "ledger_bytes_fixture": ledger.stat().st_size if ledger.exists() else None,
        "ledger_bytes_per_cell": ((ledger.stat().st_size / len(cells)) if ledger.exists() and cells else None),
        "fixture_cell_bytes_median": (sorted(sizes)[len(sizes) // 2] if sizes else None),
        "fixture_cell_bytes_total": sum(sizes),
        "bound": None,
    }
    if out["throughput_bytes_per_second"]:
        # 200 MB/s is a conservative floor for a warm page cache on any modern host;
        # anything far below it is compute, not I/O.
        out["bound"] = ("io" if out["throughput_bytes_per_second"] > 5e7 else "compute")
        out["bound_note"] = ("reduce reads every cell file once; measured throughput %.1f MB/s against a warm "
                             "page cache, so the step is %s-bound here"
                             % (out["throughput_bytes_per_second"] / 1e6, out["bound"]))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def _interpreter_startup(pkg: str, env: dict) -> float:
    cmd = [sys.executable, "-c", f"import experiments.mlst2026.successor_20260913.{pkg}.runner"]
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        subprocess.run(cmd, capture_output=True, cwd=str(ROOT), env=env)
        best = min(best, time.perf_counter() - t0)
    return best


def measure_setup(args) -> dict:
    """Transfer size and the receiving side's first two steps, measured."""
    out = {"transfer": {"archive_bytes": None, "note": "the wall time of the transfer depends on the link, which is "
                                                       "UNKNOWN; only the size is a fact. Measured on a build of "
                                                       "this handoff; successive builds differ by a few kB "
                                                       "(timestamps and the report file)."}}
    tars = sorted(Path("/tmp").glob("hf*/*.tar.gz")) + sorted(Path("/tmp").glob("*.tar.gz"))
    if args.handoff_tar:
        tars = [Path(args.handoff_tar)] + tars
    for t in tars:
        if t.is_file():
            out["transfer"]["archive_bytes"] = t.stat().st_size
            out["transfer"]["archive_path_measured"] = str(t)
            break
    env = dict(os.environ)
    env.update(B.THREAD_ENV)
    if args.handoff_dir and (Path(args.handoff_dir) / "verify_handoff.py").exists():
        hd = Path(args.handoff_dir)
        for label, extra in (("verify_handoff", []), ("verify_handoff_tests", ["--tests"])):
            times, rcs = [], []
            for _ in range(2):
                # each run gets a pristine copy: verify_handoff.py checks that no file
                # is outside MANIFEST.json, and a previous run leaves a .pytest_cache
                # behind, so re-running in place would time a different (failing) check
                tmp = Path(tempfile.mkdtemp(prefix="verify_"))
                work = tmp / hd.name
                shutil.copytree(hd, work, ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"))
                t0 = time.perf_counter()
                pr = subprocess.run([sys.executable, "verify_handoff.py"] + extra, cwd=str(work),
                                    capture_output=True, text=True, env=env)
                times.append(time.perf_counter() - t0)
                rcs.append(pr.returncode)
                shutil.rmtree(tmp, ignore_errors=True)
            out[label] = {"seconds_min": min(times), "seconds_max": max(times), "returncode": max(rcs),
                          "measured": True, "runs": len(times),
                          "note": "each run in a pristine copy of the extracted handoff; the copy itself "
                                  "is not timed"}
    # requirements: are they already satisfied, and what does checking cost?
    reqs = sorted(HERE.glob("*/requirements.txt"))
    lines = sorted({l.strip() for r in reqs for l in r.read_text().splitlines()
                    if l.strip() and not l.strip().startswith("#")})
    t0 = time.perf_counter()
    chk = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps", "--quiet", *lines],
                         capture_output=True, text=True, env=env)
    dry = time.perf_counter() - t0
    t0 = time.perf_counter()
    already = subprocess.run([sys.executable, "-c",
                              "import numpy, scipy, h5py, yaml; print(numpy.__version__)"],
                             capture_output=True, text=True, env=env)
    import_s = time.perf_counter() - t0
    out["requirements"] = {
        "pinned_lines": lines,
        "already_satisfied_here": already.returncode == 0,
        "import_check_seconds_measured": import_s,
        "pip_dry_run_seconds_measured": dry,
        "pip_dry_run_returncode": chk.returncode,
        "pip_dry_run_note": ("a real `pip install` of these pins was NOT timed: this container already "
                             "satisfies them and has no guaranteed network egress, so the honest figure is "
                             "the check time above. On a remote host without them, budget the download of "
                             "numpy+scipy wheels (~90 MB) plus install, which is link-dependent and UNKNOWN."),
    }
    return out


# =============================================================================
# probe mode (subprocess memory measurement)
# =============================================================================
PROBES = {
    ("t19s", "paired"): lambda s: _t19s_paired_call(int(s)),
    ("t19s", "det"): lambda s: _t19s_det_call(int(s)),
    ("t20s", "fit"): lambda s: _t20s_fit_call(int(s)),
    ("t21s", "data_hi"): lambda s: _t21s_data_call(int(s), "hi"),
    ("t21s", "data_lo"): lambda s: _t21s_data_call(int(s), "lo"),
    ("t22s", "replicate"): lambda s: _t22s_replicate_call(int(s), scenario="null-complete-n4-phi0.6-tau0.01"),
    ("cresst_s", "robust"): lambda s: _cresst_robust_call(int(s)),
}


def run_probe(pkg: str, kernel: str, size: str, args) -> int:
    if (pkg, kernel) == ("cresst_s", "interval"):
        from experiments.mlst2026.successor_20260913.cresst_s import archive_recompute as AR, runner as RU
        RU.set_roots(archive_root=args.ms_evidence)
        A = RU._archive()
        fn = lambda: AR.recompute_interval("headline", A["clusters"], replicates=int(float(size)),
                                           seed=AR.ARCHIVE_SEED)
    else:
        fn = PROBES[(pkg, kernel)](float(size))
    m = B.peak_memory(fn)
    print(json.dumps({"package": pkg, "kernel": kernel, "size": float(size), **m}))
    return 0


BENCH = {"t19s": bench_t19s, "t20s": bench_t20s, "t21s": bench_t21s, "t22s": bench_t22s,
         "cresst_s": bench_cresst_s}
FIXTURE_ROOTS = {p: Path("/tmp/est") / p for p in PKGS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packages", default=",".join(PKGS))
    ap.add_argument("--ms-evidence", default=None)
    ap.add_argument("--fixture-roots", default="/tmp/est", help="directory holding <pkg>/ fixture run roots")
    ap.add_argument("--handoff-dir", default=None, help="an extracted handoff, to time verify_handoff.py")
    ap.add_argument("--handoff-tar", default=None)
    ap.add_argument("--probe", nargs=3, metavar=("PKG", "KERNEL", "SIZE"), default=None)
    ap.add_argument("--storage-only", action="store_true",
                    help="re-measure only the per-family production cell-file bytes and patch them in")
    ap.add_argument("--setup-only", action="store_true",
                    help="re-measure transfer/setup and patch it into the existing BENCHMARK.json files")
    args = ap.parse_args(argv)
    global _ARGS
    _ARGS = args
    if args.probe:
        return run_probe(args.probe[0], args.probe[1], args.probe[2], args)
    setup = measure_setup(args)
    if args.setup_only or args.storage_only:
        for pkg in [p.strip() for p in args.packages.split(",") if p.strip()]:
            p = HERE / pkg / "BENCHMARK.json"
            doc = json.loads(p.read_text())
            if args.setup_only:
                doc["setup_and_transfer"] = setup
            if args.storage_only:
                doc["storage"]["bytes_per_cell_by_family"] = STORAGE_PROBES[pkg](args, doc)
            p.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
            _log(f"{pkg}: patched {p}")
        return 0
    for pkg in [p.strip() for p in args.packages.split(",") if p.strip()]:
        t0 = time.perf_counter()
        _log(f"=== {pkg}")
        data = BENCH[pkg](args)
        data.pop("_per", None)
        root = Path(args.fixture_roots) / pkg
        data["reduce"] = measure_reduce_and_storage(pkg, root, args)
        _log(f"{pkg}: measuring production cell-file bytes by family")
        data["storage"]["bytes_per_cell_by_family"] = STORAGE_PROBES[pkg](args, data)
        doc = {"schema": "mlst-local-package-benchmark/1", "package": pkg,
               "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "hardware": B.HARDWARE_BLOCK, "thread_settings": B.thread_settings_observed(),
               "benchmark_wall_seconds": None, "setup_and_transfer": setup, **data}
        doc["benchmark_wall_seconds"] = time.perf_counter() - t0
        p = HERE / pkg / "BENCHMARK.json"
        p.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
        _log(f"{pkg}: wrote {p} ({time.perf_counter() - t0:.1f} s of benchmarking)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
