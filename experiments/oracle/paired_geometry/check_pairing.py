"""Step 3: prove the pairing, and measure the floor it sits on.

Two checks, and the second is the one the design was missing.

C_A  IDENTITY.  Every event in geometry A has exactly one partner in B with
     the same truth (energy, direction, and vertex up to the recentring
     delta).  If this fails the pairing is broken and nothing downstream is
     interpretable.

C_B  NULL.  Ice/PPC is seedable only to Poisson-level photon variation, so
     even "the same event in the same detector" produces different pulses.
     Re-run ONE geometry with a different --photon-seed and measure the
     representation displacement induced by photon stochasticity ALONE.

     That distribution is the denominator for every cross-geometry number.
     Without it, a geometry-induced displacement has no scale: you cannot say
     whether it is large.  Report cross-geometry displacement as a ratio to
     the 95th percentile of this null, not in raw units.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def check_identity(truth_a: np.ndarray, truth_b: np.ndarray,
                   delta: np.ndarray, atol: float = 1e-6) -> dict:
    """truth_* : structured arrays with fields x,y,z,zenith,azimuth,energy."""
    if len(truth_a) != len(truth_b):
        return {"paired": False, "reason": f"n differs: {len(truth_a)} vs {len(truth_b)}"}
    pos_a = np.stack([truth_a[k] for k in "xyz"], axis=1)
    pos_b = np.stack([truth_b[k] for k in "xyz"], axis=1)
    dpos = np.abs((pos_b - pos_a) - delta).max()
    dang = max(
        np.abs(truth_a["zenith"] - truth_b["zenith"]).max(),
        np.abs(truth_a["azimuth"] - truth_b["azimuth"]).max(),
    )
    dE = np.abs(truth_a["energy"] / truth_b["energy"] - 1.0).max()
    return {
        "paired": bool(dpos < atol and dang < atol and dE < atol),
        "max_vertex_residual_m": float(dpos),
        "max_angle_residual_rad": float(dang),
        "max_energy_relative_residual": float(dE),
    }


def photon_seed_null(displacements: np.ndarray) -> dict:
    """Summarise the same-event/same-geometry/different-photon-seed spread."""
    d = np.asarray(displacements, dtype=float)
    return {
        "n": int(d.size),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "p99": float(np.percentile(d, 99)),
        "note": "report cross-geometry displacement as a multiple of p95",
    }


def effect_over_null(cross: np.ndarray, null: np.ndarray) -> dict:
    c, n = np.asarray(cross, float), np.asarray(null, float)
    p95 = float(np.percentile(n, 95))
    # Probability a random cross-geometry event exceeds a random null event.
    auc = float((c[:, None] > n[None, :]).mean())
    return {
        "null_p95": p95,
        "median_ratio_to_null_p95": float(np.median(c) / p95) if p95 > 0 else float("inf"),
        "auc_cross_vs_null": auc,
        "verdict": (
            "geometry effect is above the photon-stochasticity floor"
            if auc > 0.75 else
            "NOT SEPARATED from photon stochasticity -- the pair is unusable"
        ),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        rng = np.random.default_rng(0)
        null = rng.gamma(2.0, 1.0, 5000)
        cross = rng.gamma(2.0, 3.0, 5000)
        print(photon_seed_null(null))
        print(effect_over_null(cross, null))
