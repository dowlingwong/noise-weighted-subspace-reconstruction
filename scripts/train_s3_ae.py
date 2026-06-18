"""Train the S3 tied weighted linear autoencoder(s) and SAVE the model.

The config-driven S3 experiment (``scripts/run_experiment.py``) trains the AE
and records only metrics; the learned basis is discarded. This script trains
the same model on the same deterministic dataset and persists the basis to
``.npz`` so it can be reloaded without retraining.

Examples
--------
Train the default (direct weighted loss, L-BFGS) and save:

    uv run python scripts/train_s3_ae.py

Train a specific method/optimizer and choose the output directory:

    uv run python scripts/train_s3_ae.py --method whitened --optimizer lbfgs \
        --out-dir results/models

Train every method x optimizer combination:

    uv run python scripts/train_s3_ae.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.autoencoders import (  # noqa: E402
    save_trained_ae,
    train_weighted_linear_ae,
    train_whitened_linear_ae,
)
from noise_geometry.experiments.synthetic import make_s3_dataset  # noqa: E402

TRAINERS = {"direct": train_weighted_linear_ae, "whitened": train_whitened_linear_ae}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the S3 tied linear AE.")
    parser.add_argument("--method", choices=list(TRAINERS), default="direct")
    parser.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    parser.add_argument("--all", action="store_true", help="train all method x optimizer combos")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n-traces", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "models")
    args = parser.parse_args()

    cfg = {
        "seed": args.seed,
        "n_traces": args.n_traces,
        "n_features": args.n_features,
        "rank": args.rank,
    }
    X, weights, rank = make_s3_dataset(cfg)

    combos = (
        [(m, o) for m in TRAINERS for o in ("lbfgs", "adam")]
        if args.all
        else [(args.method, args.optimizer)]
    )

    summary = []
    for method, optimizer in combos:
        result = TRAINERS[method](
            X, weights, rank, optimizer=optimizer, seed=args.seed, max_iter=args.max_iter
        )
        name = f"S3_ae_{method}_{optimizer}_seed{args.seed}"
        path = save_trained_ae(result, args.out_dir / f"{name}.npz")
        row = {
            "model": name,
            "path": path,
            "n_iter": result.n_iter,
            "final_loss": result.final_loss,
            "optimal_loss": result.optimal_loss,
            "relative_gap": result.relative_gap,
            "max_principal_angle_deg": result.max_principal_angle_deg,
            "m_orthonormality_error": result.m_orthonormality_error,
        }
        summary.append(row)
        print(
            f"[{name}] trained {result.n_iter} iters -> {path}\n"
            f"    rel_gap={result.relative_gap:.2e}  angle={result.max_principal_angle_deg:.2e} deg  "
            f"ortho_err={result.m_orthonormality_error:.2e}"
        )

    summary_path = args.out_dir / f"S3_ae_train_summary_seed{args.seed}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
