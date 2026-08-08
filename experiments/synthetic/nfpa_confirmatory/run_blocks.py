"""Run Blocks A, B, and H for the NFPA confirmatory study.

For each seed and noise regime: build the dataset, run the covariance checks,
fit every method at matched total rank on the training split only, evaluate on
the held-out test split, fit the semantic decoders (Block H), and write one raw
row per (seed, method) to disk before any aggregation.

Usage:
    python -m experiments.synthetic.nfpa_confirmatory.run_blocks --block A --seeds 0 1
    python -m experiments.synthetic.nfpa_confirmatory.run_blocks --block B
    python -m experiments.synthetic.nfpa_confirmatory.run_blocks --block A --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from .config import PRIMARY, PRIMARY_HASH, RunPaths, seed_list
from .covariance import assert_checks_pass, full_covariance, run_numerical_checks
from .generator import make_dataset
from .methods import FIT_FACTORY, fit_cw_pca
from .output import build_row, write_raw_rows


def run_seed(seed: int, regime: str, methods: list[str], paths: RunPaths) -> list[dict]:
    ds = make_dataset(seed, regime, PRIMARY)
    report = run_numerical_checks(PRIMARY, ds.covariances["Sigma"])
    assert_checks_pass(report)
    ds.covariance_report = report

    cw_pca = fit_cw_pca(ds)
    rows = []
    for name in methods:
        t0 = time.time()
        fit = FIT_FACTORY[name](ds)
        runtime = time.time() - t0
        rows.append(build_row(ds, fit, cw_pca, runtime=runtime))
    return rows


def run_block(block: str, regime: str, seeds: list[int], methods: list[str], paths: RunPaths) -> Path:
    all_rows = []
    for seed in seeds:
        rows = run_seed(seed, regime, methods, paths)
        all_rows.extend(rows)
        print(f"[{block}] seed {seed}: fitted {len(rows)} methods ({regime})", flush=True)
    out = Path(paths.raw) / f"block_{block}_{regime}_raw.csv"
    write_raw_rows(all_rows, out)
    print(f"wrote {out} ({len(all_rows)} rows)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=["A", "B"], default="A")
    parser.add_argument("--regime", default=None, help="override regime (default: colored for A, white for B)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="seed list override")
    parser.add_argument("--methods", nargs="*", default=["CW-PCA", "NFPA", "Iso-MPCA", "IsoPCA", "OF"])
    parser.add_argument("--smoke", action="store_true", help="2-seed smoke run")
    args = parser.parse_args()

    paths = RunPaths()
    regime = args.regime or ("colored" if args.block == "A" else "white")
    seeds = args.seeds or ([0, 1] if args.smoke else seed_list(30 if args.block == "A" else 20))
    if args.smoke:
        seeds = [0, 1]
    print(f"config_hash={PRIMARY_HASH}")
    run_block(args.block, regime, seeds, args.methods, paths)


if __name__ == "__main__":
    main()
