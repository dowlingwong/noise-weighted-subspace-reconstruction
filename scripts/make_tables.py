"""Build simple CSV summary tables from metrics JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "results/metrics")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/tables/synthetic_summary.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.metrics_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        row = {"file": str(path), "experiment_id": record.get("experiment_id"), "status": record.get("status")}
        metrics = record.get("metrics", {})
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
