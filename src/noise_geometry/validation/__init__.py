"""Validation infrastructure: seed sweeps, held-out splits, bootstrap CIs."""

from .harness import (
    flatten_metrics,
    paired_difference,
    run_seed_sweep,
    summarize,
    train_test_split_indices,
    write_json,
    write_rows_csv,
)

__all__ = [
    "flatten_metrics",
    "paired_difference",
    "run_seed_sweep",
    "summarize",
    "train_test_split_indices",
    "write_json",
    "write_rows_csv",
]
