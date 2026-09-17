"""R5: bounded-memory covariance accumulation.

The theorem-domain reachability question needs a full sample covariance at a
calibration size of millions of samples, so the accumulator must never hold the
draws.  This module streams row chunks and accumulates only the ``d x d`` second
moment; :func:`streaming_covariance` is checked against a direct construction on
small fixtures (tests and a control cell).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def direct_covariance(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    return (Z.T @ Z) / Z.shape[0]


def streaming_covariance(draw_chunk: Callable[[int], np.ndarray], m: int, d: int, chunk: int) -> np.ndarray:
    """Uncentred sample covariance ``X^T X / m`` without materialising ``X``.

    ``draw_chunk(n)`` returns the next ``n`` rows.  Peak extra memory is
    ``O(chunk * d + d^2)``, independent of ``m``.
    """
    m = int(m)
    if m <= 0:
        raise ValueError("m must be positive")
    acc = np.zeros((d, d), dtype=float)
    done = 0
    while done < m:
        n = min(int(chunk), m - done)
        Z = np.asarray(draw_chunk(n), dtype=float)
        if Z.shape != (n, d):
            raise ValueError(f"draw_chunk({n}) returned shape {Z.shape}, expected {(n, d)}")
        acc += Z.T @ Z
        done += n
    return acc / m


def nested_streaming_covariances(draw_chunk: Callable[[int], np.ndarray], m_grid, d: int, chunk: int) -> dict:
    """Covariances for several ``m`` that are prefixes of the *same* stream.

    The arms are nested exactly as ``pool[:m]`` would be, so a reachability
    comparison across ``m`` is not confounded by a different draw.
    """
    ms = sorted(int(x) for x in m_grid)
    acc = np.zeros((d, d), dtype=float)
    done = 0
    out = {}
    for m in ms:
        while done < m:
            n = min(int(chunk), m - done)
            Z = np.asarray(draw_chunk(n), dtype=float)
            acc += Z.T @ Z
            done += n
        out[m] = acc / m
    return out
