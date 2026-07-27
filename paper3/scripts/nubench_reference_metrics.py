#!/usr/bin/env python3
"""Recompute NuBench Table 6 metrics from released prediction artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl


PUBLISHED = {
    "cc_track": {"median_deg": 23.60, "le_1deg_pct": 2.26, "le_5deg_pct": 22.24},
    "nc_cascade": {"median_deg": 56.12, "le_1deg_pct": 0.04, "le_5deg_pct": 1.03},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pl.scan_parquet(args.predictions)
    zenith = pl.col("initial_state_zenith")
    azimuth = pl.col("initial_state_azimuth")
    truth_x = zenith.sin() * azimuth.cos()
    truth_y = zenith.sin() * azimuth.sin()
    truth_z = zenith.cos()
    prediction_norm = (
        pl.col("dir_x_pred") ** 2
        + pl.col("dir_y_pred") ** 2
        + pl.col("dir_z_pred") ** 2
    ).sqrt()
    cosine = (
        (
            truth_x * pl.col("dir_x_pred")
            + truth_y * pl.col("dir_y_pred")
            + truth_z * pl.col("dir_z_pred")
        )
        / prediction_norm
    ).clip(-1.0, 1.0)
    angle = (cosine.arccos() * (180.0 / math.pi)).alias("angle")

    aggregate = (
        frame.select("is_track", angle)
        .group_by("is_track")
        .agg(
            pl.len().alias("n"),
            pl.col("angle").median().alias("median_deg"),
            (pl.col("angle") <= 1.0).mean().mul(100).alias("le_1deg_pct"),
            (pl.col("angle") <= 5.0).mean().mul(100).alias("le_5deg_pct"),
        )
        .collect(engine="streaming")
    )
    by_label = {bool(row["is_track"]): row for row in aggregate.to_dicts()}
    computed = {
        "cc_track": {k: v for k, v in by_label[True].items() if k != "is_track"},
        "nc_cascade": {
            k: v for k, v in by_label[False].items() if k != "is_track"
        },
    }
    deltas = {
        name: {
            metric: float(computed[name][metric] - PUBLISHED[name][metric])
            for metric in PUBLISHED[name]
        }
        for name in PUBLISHED
    }
    result = {
        "source_file": args.predictions.name,
        "rows": int(sum(group["n"] for group in computed.values())),
        "published_table_6": PUBLISHED,
        "computed": computed,
        "computed_minus_published": deltas,
        "tolerance_declared_before_run": {
            "median_deg_absolute": 0.01,
            "threshold_rate_percentage_points_absolute": 0.01,
        },
        "gate_pass": all(
            abs(delta) <= 0.01 for group in deltas.values() for delta in group.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
