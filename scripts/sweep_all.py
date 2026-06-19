"""Roll the seed-sweep / CI harness across all synthetic gates (S1-S9).

Runs each gate across N seeds, archives the one-row-per-seed CSV and a summary
JSON (mean + 68%/95% bootstrap CIs, plus the headline paired comparisons) under
``results/sweeps/``, and prints a compact per-gate headline table.

    uv run python scripts/sweep_all.py --seeds 0-9
    uv run python scripts/sweep_all.py --seeds 0-19 --gates S5,S9
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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

# gate -> (base config, headline metric, [(a, b) paired comparisons])
GATES: dict[str, tuple[dict, str, list[tuple[str, str]]]] = {
    "S1": (
        {"experiment_id": "S1", "n_traces": 2048},
        "sigma_over_crb",
        [],
    ),
    "S2": (
        {"experiment_id": "S2", "n_traces": 1024, "amplitude_model": "real"},
        "weighted_angle_deg",
        [],
    ),
    "S3": (
        {"experiment_id": "S3", "optimizers": ["lbfgs"]},
        "relative_gap",
        [],
    ),
    "S4": (
        {"experiment_id": "S4", "test_frac": 0.5},
        "max_principal_angle_deg",
        [("pca_weighted_residual", "empca_weighted_residual")],
    ),
    "S5": (
        {"experiment_id": "S5", "test_frac": 0.5},
        "subspace_angle_deg_max",
        [("pca_weighted_residual_to_observed", "weighted_pca_weighted_residual_to_observed")],
    ),
    "S6": (
        {"experiment_id": "S6", "test_frac": 0.5},
        "best_rank_by_clean_mse",
        [("weighted_residual_by_rank.0", "weighted_residual_by_rank.5")],
    ),
    "S7": (
        {"experiment_id": "S7"},
        "sigma_over_oracle.4",
        [],
    ),
    "S8": (
        {"experiment_id": "S8"},
        "mean_chi2_per_dof",
        [],
    ),
    "S9": (
        {"experiment_id": "S9"},
        "diagonal_over_full_sigma",
        [("diagonal_true_weighted_residual", "full_true_weighted_residual")],
    ),
}


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep all synthetic gates with bootstrap CIs.")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--gates", default=",".join(GATES), help="comma list, e.g. 'S5,S9'")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "sweeps")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    gates = [g.strip().upper() for g in args.gates.split(",") if g.strip()]

    print(f"seeds={len(seeds)}\n{'gate':5s} {'headline metric':32s} {'mean':>12s} {'95% CI of mean':>26s}  paired")
    for gate in gates:
        base, headline, pairs = GATES[gate]
        rows = run_seed_sweep(run_synthetic_experiment, base, seeds)
        summary = summarize(rows, n_boot=args.n_boot, seed=0)
        paired = [paired_difference(rows, a, b, n_boot=args.n_boot, seed=0) for a, b in pairs]

        stem = f"{gate}_sweep_{len(seeds)}seeds"
        write_rows_csv(rows, args.out_dir / f"{stem}.csv")
        write_json(
            {"experiment_id": gate, "seeds": seeds, "config": base, "summary": summary, "pairs": paired},
            args.out_dir / f"{stem}.json",
        )

        s = summary.get(headline, {})
        ci = f"[{s.get('ci95', (float('nan'),) * 2)[0]:.4g}, {s.get('ci95', (float('nan'),) * 2)[1]:.4g}]"
        verdict = ""
        if paired:
            p = paired[0]
            verdict = f"{p['comparison'].split(' - ')[0][:14]}…  {'EXCL 0' if p['ci95_excludes_zero'] else 'incl 0'}"
        print(f"{gate:5s} {headline:32s} {s.get('mean', float('nan')):12.4g} {ci:>26s}  {verdict}")

    print(f"\narchived per-gate CSV + JSON -> {args.out_dir}")


if __name__ == "__main__":
    main()
