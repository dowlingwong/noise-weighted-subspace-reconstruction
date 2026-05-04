"""generate_data.py — Multi-seed experiment runner for the paper.

Run this script from the repository root to populate ``results/data_cache.h5``
with all synthetic experiment results (E1-E9).  Each experiment is saved under
its own h5 group so the script can be safely interrupted and re-run; existing
groups are skipped unless ``--force`` is passed.

Usage (from repo root):
    python implementation/generate_data.py
    python implementation/generate_data.py --force          # re-run everything
    python implementation/generate_data.py --exps e1,e5,e6  # run subset
    python implementation/generate_data.py --n_events 500   # larger samples

The resulting h5 file is loaded by experiments_master.ipynb for plotting.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Repo-root discovery (same logic as notebook_support, but standalone)
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (
            (candidate / "QP_simulator").exists()
            and (candidate / "implementation").exists()
        ):
            return candidate
    raise RuntimeError(f"Cannot find repo root from {start}")


REPO_ROOT = _find_repo_root()
for _p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "QP_simulator", REPO_ROOT / "implementation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from notebook_support import (  # noqa: E402
    CanonicalConfig,
    run_block04_theorem_suite,
    run_block05_bridge_suite,
    run_block06_convergence_suite,
    run_block07_ablation_suite,
    run_block09_robustness_suite,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [314159, 271828, 161803, 141421, 100003, 200003, 300007, 400009]

DEFAULT_H5 = REPO_ROOT / "results" / "data_cache.h5"

# ---------------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------------

def _grp_exists(h5: h5py.File, path: str) -> bool:
    return path in h5


def _save_scalars(grp: h5py.Group, **kwargs) -> None:
    for k, v in kwargs.items():
        grp.attrs[k] = v


def _save_arr(grp: h5py.Group, name: str, arr) -> None:
    arr = np.asarray(arr)
    if arr.dtype.kind in ("U", "S", "O"):
        arr = np.array([s.encode() for s in arr])
    grp.create_dataset(name, data=arr)


def _save_df(grp: h5py.Group, df) -> None:
    """Save a pandas DataFrame column-by-column into an h5 group."""
    import pandas as pd
    for col in df.columns:
        vals = df[col].values
        if vals.dtype == object or (hasattr(vals.dtype, "kind") and vals.dtype.kind in ("U", "S")):
            vals = np.array([str(v).encode() for v in vals])
        elif isinstance(vals[0], (list, np.ndarray)):
            # columns holding lists (e.g. principal_angle_cosines)
            try:
                vals = np.array([np.asarray(v, dtype=float) for v in vals])
            except Exception:
                vals = np.array([str(v).encode() for v in vals])
        grp.create_dataset(col, data=vals)


# ---------------------------------------------------------------------------
# Per-experiment savers
# ---------------------------------------------------------------------------

def _make_cfg(seed: int, n_events_override: int | None = None) -> CanonicalConfig:
    cfg = CanonicalConfig(seed=seed)
    if n_events_override is not None:
        cfg = replace(cfg, sim_events_medium=n_events_override, sim_events_large=n_events_override)
    return cfg.validate()


def run_e1_e4_e5(h5: h5py.File, seed: int, n_events: int, force: bool) -> None:
    """Block 04 — E1 (rank-1 theorem), E4 (CRB), E5 (resolution scaling)."""
    grp_key = f"e1_e4_e5/seed_{seed}"
    if _grp_exists(h5, grp_key) and not force:
        print(f"  [skip] {grp_key}")
        return
    if grp_key in h5:
        del h5[grp_key]
    t0 = time.time()
    cfg = _make_cfg(seed, n_events)
    out = run_block04_theorem_suite(cfg)

    grp = h5.require_group(grp_key)

    # E1
    e1 = grp.require_group("e1")
    _save_df(e1, out["rank1_summary_df"])

    # E4
    e4 = grp.require_group("e4")
    _save_df(e4, out["crb_df"])

    # E5
    e5 = grp.require_group("e5")
    _save_df(e5, out["resolution_df"])
    e5.attrs["loglog_slope"] = out["resolution_summary"]["loglog_slope_sigma_emp_vs_noise_power"]

    h5.flush()
    print(f"  [done] {grp_key}  ({time.time()-t0:.1f}s)")


def run_e2(h5: h5py.File, seed: int, n_events: int, force: bool) -> None:
    """Block 05 — E2 (Bridge Theorem: EMPCA ≡ noise-aware linear AE)."""
    grp_key = f"e2/seed_{seed}"
    if _grp_exists(h5, grp_key) and not force:
        print(f"  [skip] {grp_key}")
        return
    if grp_key in h5:
        del h5[grp_key]
    t0 = time.time()
    cfg = _make_cfg(seed, n_events)
    out = run_block05_bridge_suite(cfg)

    grp = h5.require_group(grp_key)
    bridge = out["bridge_df"].copy()
    # Extract min cosine per k (list column → scalar)
    bridge["min_principal_cosine"] = bridge["principal_angle_cosines"].map(
        lambda x: float(min(x)) if hasattr(x, "__iter__") else float(x)
    )
    _save_df(grp, bridge[["k", "min_principal_cosine"]])

    # save residual comparison columns if present
    for col in ("mean_residual_diff", "weighted_residual_mean", "isotropic_residual_mean"):
        if col in bridge.columns:
            _save_arr(grp, col, bridge[col].values.astype(float))

    h5.flush()
    print(f"  [done] {grp_key}  ({time.time()-t0:.1f}s)")


def run_e3_e9(h5: h5py.File, seed: int, n_events: int, force: bool) -> None:
    """Block 06 — E3 (rank saturation / χ² monotone) and E9 (convergence)."""
    grp_key = f"e3_e9/seed_{seed}"
    if _grp_exists(h5, grp_key) and not force:
        print(f"  [skip] {grp_key}")
        return
    if grp_key in h5:
        del h5[grp_key]
    t0 = time.time()
    cfg = _make_cfg(seed, n_events)
    out = run_block06_convergence_suite(cfg)

    grp = h5.require_group(grp_key)

    # E3: monotonicity / rank saturation
    e3 = grp.require_group("e3")
    _save_df(e3, out["monotonicity_df"])

    # E9: convergence traces + summary
    e9 = grp.require_group("e9")
    _save_df(e9, out["convergence_df"])
    _save_df(e9.require_group("summary"), out["convergence_summary_df"])

    # rank summary (derived from setup_B_multiD)
    _save_df(grp.require_group("rank_summary"), out["rank_summary_df"][["k", "chi2_proxy_mean", "chi2_proxy_std"]])

    h5.flush()
    print(f"  [done] {grp_key}  ({time.time()-t0:.1f}s)")


def run_e6_e7(h5: h5py.File, seed: int, n_events: int, force: bool) -> None:
    """Block 07 — E6 (noise-aware ablation) and E7 (template mismatch)."""
    grp_key = f"e6_e7/seed_{seed}"
    if _grp_exists(h5, grp_key) and not force:
        print(f"  [skip] {grp_key}")
        return
    if grp_key in h5:
        del h5[grp_key]
    t0 = time.time()
    cfg = _make_cfg(seed, n_events)
    out = run_block07_ablation_suite(cfg)

    grp = h5.require_group(grp_key)

    # E6
    e6 = grp.require_group("e6")
    _save_df(e6, out["ablation_df"])

    # E7 — mismatch summary table
    e7 = grp.require_group("e7")
    _save_df(e7, out["mismatch_df"])

    # E7 — mismatch curve (deterministic / seed-independent, but save per seed for bookkeeping)
    e7_curve = e7.require_group("curve")
    _save_df(e7_curve, out["mismatch_curve_df"])

    # E8 — time-shift OF
    e8 = grp.require_group("e8")
    _save_df(e8, out["shift_df"])

    h5.flush()
    print(f"  [done] {grp_key}  ({time.time()-t0:.1f}s)")


def run_e10_e11(h5: h5py.File, seed: int, n_events: int, force: bool) -> None:
    """Block 09 — E10 (non-stationary noise) and E11 (artifact robustness)."""
    grp_key = f"e10_e11/seed_{seed}"
    if _grp_exists(h5, grp_key) and not force:
        print(f"  [skip] {grp_key}")
        return
    if grp_key in h5:
        del h5[grp_key]
    t0 = time.time()
    cfg = _make_cfg(seed, n_events)
    try:
        out = run_block09_robustness_suite(cfg)
    except Exception as exc:
        print(f"  [warn] {grp_key} failed: {exc}")
        return

    grp = h5.require_group(grp_key)
    for key in out:
        import pandas as pd
        if isinstance(out[key], pd.DataFrame):
            _save_df(grp.require_group(key), out[key])
        elif isinstance(out[key], np.ndarray):
            _save_arr(grp, key, out[key])
        elif isinstance(out[key], (int, float)):
            grp.attrs[key] = out[key]

    h5.flush()
    print(f"  [done] {grp_key}  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

EXP_MAP = {
    "e1": run_e1_e4_e5,
    "e4": run_e1_e4_e5,
    "e5": run_e1_e4_e5,
    "e2": run_e2,
    "e3": run_e3_e9,
    "e9": run_e3_e9,
    "e6": run_e6_e7,
    "e7": run_e6_e7,
    "e8": run_e6_e7,
    "e10": run_e10_e11,
    "e11": run_e10_e11,
}

# De-duplicate (multiple experiment aliases map to same runner)
_RUNNERS = {
    "e1_e4_e5": run_e1_e4_e5,
    "e2":       run_e2,
    "e3_e9":    run_e3_e9,
    "e6_e7":    run_e6_e7,
    "e10_e11":  run_e10_e11,
}

_RUNNER_EXP_TAGS = {
    "e1_e4_e5": {"e1", "e4", "e5"},
    "e2":       {"e2"},
    "e3_e9":    {"e3", "e9"},
    "e6_e7":    {"e6", "e7", "e8"},
    "e10_e11":  {"e10", "e11"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-run even if h5 group already exists")
    parser.add_argument("--exps", default="all", help="Comma-separated experiment tags, e.g. e1,e5,e6 (default: all)")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS), help="Comma-separated seeds")
    parser.add_argument("--n_events", type=int, default=None, help="Override sim_events_medium (default: 300)")
    parser.add_argument("--output", default=str(DEFAULT_H5), help="Path to output h5 file")
    parser.add_argument("--robustness", action="store_true", help="Also run E10/E11 robustness suite (slower)")
    args = parser.parse_args()

    requested_exps = set(args.exps.lower().split(",")) if args.exps != "all" else set(EXP_MAP.keys())
    seeds = [int(s) for s in args.seeds.split(",")]
    n_events = args.n_events if args.n_events is not None else 1000

    h5_path = Path(args.output)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output: {h5_path}")
    print(f"Seeds:  {seeds}")
    print(f"Events per seed: {n_events}")
    print(f"Requested experiments: {sorted(requested_exps)}")
    print()

    with h5py.File(h5_path, "a") as h5:
        for runner_key, runner_fn in _RUNNERS.items():
            relevant = _RUNNER_EXP_TAGS[runner_key] & requested_exps
            if not relevant:
                continue
            if runner_key == "e10_e11" and not args.robustness:
                print(f"Skipping {runner_key} (pass --robustness to enable)")
                continue
            print(f"\n--- {runner_key.upper()} ---")
            for seed in seeds:
                runner_fn(h5, seed=seed, n_events=n_events, force=args.force)

    print("\nDone. Results written to", h5_path)


if __name__ == "__main__":
    main()
