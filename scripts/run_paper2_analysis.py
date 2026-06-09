#!/usr/bin/env python3
"""Analyze real Paper 2 run artifacts and generate comparison figures.

Examples
--------
Analyze local run folders using existing metrics only:

    PYTHONPATH=. python scripts/run_paper2_analysis.py \
      --results-dir paper2/results \
      --evaluate-mode metrics-only

Re-evaluate checkpoints and build actual AE/transformer plots:

    PYTHONPATH=. python scripts/run_paper2_analysis.py \
      --results-dir /ceph/dwong/noise-weighted-subspace-reconstruction/paper2/results \
      --evaluate-mode checkpoints \
      --template-path data/k_alpha/template_K_alpha_tight.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper2.analysis.reporting import analyze_results_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="paper2/results",
        help="Paper 2 results root that contains one folder per experiment.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Analysis output folder. Defaults to <results-dir>/_analysis/latest.",
    )
    parser.add_argument(
        "--evaluate-mode",
        choices=["auto", "metrics-only", "checkpoints"],
        default="auto",
        help=(
            "auto: use existing analysis metrics when present, otherwise evaluate checkpoints; "
            "metrics-only: never touch checkpoints; "
            "checkpoints: force checkpoint-backed evaluation for every eligible run."
        ),
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Recompute predictions_test.h5 and analysis_metrics.json even if they already exist.",
    )
    parser.add_argument(
        "--template-path",
        default="data/k_alpha/template_K_alpha_tight.npy",
        help="Template used to derive amplitude/time metrics from reconstructed traces.",
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        default=2.5e5,
        help="Sampling frequency passed into the OptimumFilter helper.",
    )
    parser.add_argument(
        "--of-mode",
        choices=["shifted", "fixed", "none"],
        default="shifted",
        help="How to estimate amplitude/time from reconstructed traces.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated experiment names to analyze.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "_analysis" / "latest"
    only = None
    if args.only:
        only = [item.strip() for item in args.only.split(",") if item.strip()]

    runs, paths = analyze_results_tree(
        results_dir=results_dir,
        output_dir=output_dir,
        evaluate_mode=args.evaluate_mode,
        force_eval=args.force_eval,
        template_path=args.template_path,
        sampling_frequency=args.sampling_frequency,
        of_mode=args.of_mode,
        only=only,
    )

    print(f"[paper2-analysis] runs={len(runs)} results_dir={results_dir}")
    print(f"[paper2-analysis] summary={paths.summary_csv}")
    print(f"[paper2-analysis] best={paths.best_csv}")
    print(f"[paper2-analysis] manifest={paths.manifest_json}")
    print(f"[paper2-analysis] figures_dir={paths.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
