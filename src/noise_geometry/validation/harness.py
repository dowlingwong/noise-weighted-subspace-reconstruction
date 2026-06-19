"""Seed-sweep + held-out + bootstrap-CI harness (Paper 1 item 3).

Every S1-S9 acceptance gate repeats the same requirement: run many seeds,
evaluate on held-out data, and report paired uncertainty intervals. This module
provides that once, as infrastructure, so hardening an experiment becomes a
config change rather than bespoke code:

    rows = run_seed_sweep(runner, base_config, seeds)     # one row per seed
    summary = summarize(rows)                             # mean/std/68/95 CIs
    rev = paired_difference(rows, key_a, key_b)           # paired diff + sign

The per-seed rows are the archivable, one-row-per-seed source data the roadmap
requires; the summary carries bootstrap confidence intervals on the mean.
"""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Callable, Iterable, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Metric flattening
# --------------------------------------------------------------------------- #
def flatten_metrics(result: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten a (possibly nested) result dict to dotted numeric leaves.

    Strings, bools, and non-numeric values are dropped; nested dicts are
    recursed with dotted keys (e.g. ``runs.direct_lbfgs.relative_gap``).
    """
    flat: dict[str, float] = {}
    for key, value in result.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{name}."))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, Real):
            flat[name] = float(value)
        elif isinstance(value, (list, tuple)):
            # expand numeric lists into indexed columns (e.g. per-rank sweeps)
            for i, item in enumerate(value):
                if not isinstance(item, bool) and isinstance(item, Real):
                    flat[f"{name}.{i}"] = float(item)
    return flat


# --------------------------------------------------------------------------- #
# Seed sweep
# --------------------------------------------------------------------------- #
def run_seed_sweep(
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    base_config: dict[str, Any],
    seeds: Iterable[int],
) -> list[dict[str, float]]:
    """Run ``runner`` once per seed; return one flattened row per seed."""
    rows: list[dict[str, float]] = []
    for seed in seeds:
        cfg = copy.deepcopy(base_config)
        cfg["seed"] = int(seed)
        flat = flatten_metrics(runner(cfg))
        flat["seed"] = float(int(seed))
        rows.append(flat)
    return rows


# --------------------------------------------------------------------------- #
# Held-out splitting
# --------------------------------------------------------------------------- #
def train_test_split_indices(n: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, test_idx)`` for a leakage-free random split."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(n))
    n_test = max(1, int(round(test_frac * n)))
    return perm[n_test:], perm[:n_test]


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def _bootstrap_mean_ci(
    values: np.ndarray, level: float, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return (float(lo), float(hi))


def _numeric_keys(rows: Sequence[dict[str, float]]) -> list[str]:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k != "seed" and k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def summarize(
    rows: Sequence[dict[str, float]],
    keys: Sequence[str] | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Per-metric mean, std, and 68%/95% bootstrap CIs of the mean across seeds."""
    rng = np.random.default_rng(seed)
    keys = list(keys) if keys is not None else _numeric_keys(rows)
    out: dict[str, dict[str, Any]] = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows if k in r], dtype=np.float64)
        if vals.size == 0:
            continue
        out[k] = {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "ci68": _bootstrap_mean_ci(vals, 0.68, n_boot, rng),
            "ci95": _bootstrap_mean_ci(vals, 0.95, n_boot, rng),
        }
    return out


def paired_difference(
    rows: Sequence[dict[str, float]],
    key_a: str,
    key_b: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-seed paired difference ``a - b`` with bootstrap CIs and sign consistency.

    Use for two methods sharing the same per-seed test data (e.g. PCA vs weighted
    PCA). ``ci95_excludes_zero`` is the roadmap's paired-interval acceptance.
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(
        [r[key_a] - r[key_b] for r in rows if key_a in r and key_b in r], dtype=np.float64
    )
    if diffs.size == 0:
        raise ValueError(f"no rows contain both {key_a!r} and {key_b!r}")
    ci95 = _bootstrap_mean_ci(diffs, 0.95, n_boot, rng)
    return {
        "comparison": f"{key_a} - {key_b}",
        "n": int(diffs.size),
        "mean_difference": float(diffs.mean()),
        "std": float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0,
        "ci68": _bootstrap_mean_ci(diffs, 0.68, n_boot, rng),
        "ci95": ci95,
        "fraction_positive": float(np.mean(diffs > 0)),
        "ci95_excludes_zero": bool(ci95[0] > 0 or ci95[1] < 0),
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_rows_csv(rows: Sequence[dict[str, float]], path: str | Path) -> str:
    """Write one-row-per-seed CSV (``seed`` first, metric columns sorted)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted(k for k in _numeric_keys(rows))
    fieldnames = ["seed", *metric_keys]
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return str(p)


def write_json(obj: Any, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return str(p)
