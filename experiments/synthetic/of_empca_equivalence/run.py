"""Run SYN-A: fixed-template OF vs rank-1 weighted subspace smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.synthetic import make_rank1_pulse_dataset, run_of_empca_equivalence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-traces", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--noise-kind", choices=["white", "pink", "red", "brownian", "blue", "violet"], default="pink")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/synthetic/of_empca_equivalence/summary.json")
    args = parser.parse_args()

    dataset = make_rank1_pulse_dataset(
        n_traces=args.n_traces,
        n_samples=args.n_samples,
        noise_kind=args.noise_kind,
        seed=args.seed,
    )
    summary = run_of_empca_equivalence(dataset)
    summary.update({"seed": args.seed, "noise_kind": args.noise_kind})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
