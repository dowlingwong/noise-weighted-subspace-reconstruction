"""Deterministic train/val/test split helpers for Paper 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split_indices(
    n_samples: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitIndices:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    n_train = int(round(train_fraction * n_samples))
    n_val = int(round(val_fraction * n_samples))
    train = np.sort(order[:n_train])
    val = np.sort(order[n_train : n_train + n_val])
    test = np.sort(order[n_train + n_val :])
    return SplitIndices(train=train, val=val, test=test)
