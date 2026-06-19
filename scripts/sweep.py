"""Multi-seed sweep with bootstrap confidence intervals (Paper 1 item 3).

Runs a config-driven synthetic experiment across many seeds, writes the
one-row-per-seed CSV plus a summary JSON with 68%/95% bootstrap CIs, and prints
a short table. Optional ``--pairs`` reports paired differences (e.g. PCA vs
weighted PCA) with the paired 95% interval and whether it excludes zero.

Examples
--------
    uv run python scripts/sweep.py --config configs/synthetic/s5_metric_reversal.yaml \
        --seeds 0-19 --set test_frac=0.5 \
        --pairs pca_weighted_residual_to_observed:weighted_pca_weighted_residual_to_observed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.experiments import run_synthetic_experiment  # noqa: E402
from noise_geometry.validation import (  # noqa: E402
    paired_difference,
    run_seed_sweep,
    summarize,
    write_json,
    write_rows_csv,
)


def _parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed sweep with bootstrap CIs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", default="0-19", help="e.g. '0-19' or '1,2,3' or '0-9,20'")
    parser.add_argument("--set", action="append", default=[], help="config override key=value")
    parser.add_argument("--pairs", default="", help="comma list of 'key_a:key_b' paired diffs")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "sweeps")
    args = parser.parse_args()

    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for item in args.set:
        key, _, value = item.partition("=")
        base_config[key.strip()] = _coerce(value.strip())

    seeds = _parse_seeds(args.seeds)
    rows = run_seed_sweep(run_synthetic_experiment, base_config, seeds)
    summary = summarize(rows, n_boot=args.n_boot, seed=args.boot_seed)

    pairs = []
    for spec in filter(None, (s.strip() for s in args.pairs.split(","))):
        a, _, b = spec.partition(":")
        pairs.append(paired_difference(rows, a, b, n_boot=args.n_boot, seed=args.boot_seed))

    exp_id = str(base_config.get("experiment_id", "experiment"))
    stem = f"{exp_id}_sweep_{len(seeds)}seeds"
    csv_path = write_rows_csv(rows, args.out_dir / f"{stem}.csv")
    summary_path = write_json(
        {"experiment_id": exp_id, "seeds": seeds, "config": base_config, "summary": summary, "pairs": pairs},
        args.out_dir / f"{stem}.json",
    )

    print(f"experiment={exp_id}  seeds={len(seeds)}")
    print(f"{'metric':52s} {'mean':>12s} {'95% CI of mean':>28s}")
    for key, s in summary.items():
        ci = f"[{s['ci95'][0]:.4g}, {s['ci95'][1]:.4g}]"
        print(f"{key:52s} {s['mean']:12.4g} {ci:>28s}")
    for p in pairs:
        verdict = "EXCLUDES 0" if p["ci95_excludes_zero"] else "includes 0"
        print(
            f"\npaired: {p['comparison']}\n  mean_diff={p['mean_difference']:.4g}  "
            f"95% CI=[{p['ci95'][0]:.4g}, {p['ci95'][1]:.4g}]  ({verdict})  "
            f"sign-consistent in {p['fraction_positive']*100:.0f}% of seeds"
        )
    print(f"\nrows -> {csv_path}\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
