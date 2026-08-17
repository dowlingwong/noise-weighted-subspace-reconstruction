"""Whitening dispersion of the rung-3 oracle covariance, comparable to rung 4's.

Why
---
Round 8's Task A measured the dispersion of rung 4's whitening weights on the
TIDMAD readout -- ``cv_inverse_J = 0.69`` -- and thereby refuted the paper's
near-collinearity explanation for the absent metric reversal at that rung. That
leaves a question the paper can now answer quantitatively: do the rungs where
the reversal IS observed differ from rung 4 in how strongly the metric departs
from a rescaling?

This script computes the same statistic for the rung-3 confirmatory study's
oracle covariance ``Sigma = Sigma_t (x) Sigma_c`` (Brownian 1/f^2 temporal PSD,
AR(1) channel correlation, the exact construction Block A used). Rung 4's
``Sigma`` is diagonal in the STFT band basis, so its whitening weights are the
inverse PSD values ``1/J``; the basis-free generalization is the spectrum of
``Sigma^{-1}`` -- whitening is diagonal in the eigenbasis with weights
``1/lambda_i``. For rung 4 those coincide with ``1/J``, so the numbers are
directly comparable.

The white-noise control has ``Sigma = I``: every weight equal, CV exactly 0, by
construction. It is included as the anchor of the axis.

Run from the repository root:

    PYTHONPATH=src:experiments/synthetic python scripts/rung3_whitening_range.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("results/rps2026/E6b_rung3_whitening_range.json")

# Rung 4's measured record, for the side-by-side (results/rps2026/E6_whitening_range.json).
RUNG4_RECORD = Path("results/rps2026/E6_whitening_range.json")


def dispersion(weights: np.ndarray, label: str) -> dict:
    """The E6 statistics for one vector of inverse-covariance weights."""
    w = np.asarray(weights, dtype=np.float64)
    lam = 1.0 / w  # eigenvalues of Sigma
    participation = float(w.sum() ** 2 / np.square(w).sum())
    return {
        "label": label,
        "n_weights": int(w.size),
        "lambda_min": float(lam.min()),
        "lambda_max": float(lam.max()),
        "ratio_max_min": float(lam.max() / lam.min()),
        "ratio_p99_p01": float(np.percentile(lam, 99) / np.percentile(lam, 1)),
        "cv_inverse_weights": float(w.std() / w.mean()),
        "effective_dimensions": participation,
        "effective_fraction": participation / w.size,
    }


def main() -> int:
    from nfpa_confirmatory.config import PrimaryConfig
    from nfpa_confirmatory.covariance import full_covariance

    cfg = PrimaryConfig()

    record = {"comment": (
        "Dispersion of inverse whitening weights, comparable across rungs. "
        "Rung 3: eigenvalues of the oracle Sigma = Sigma_t (x) Sigma_c "
        "(whitening is diagonal in the eigenbasis with weights 1/lambda). "
        "Rung 4: the measured 1/J over STFT bands, where Sigma is diagonal "
        "by measurement -- the same statistic in that basis."
    )}

    for regime in ("colored", "white"):
        Sigma = full_covariance(cfg, regime)["Sigma"]
        lam = np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))
        if lam.min() <= 0:
            raise SystemExit(f"{regime}: Sigma is not positive definite; construction bug")
        record[f"rung3_{regime}"] = dispersion(1.0 / lam, f"rung 3 {regime} oracle Sigma, C={cfg.C} T={cfg.T}")

    if RUNG4_RECORD.exists():
        e6 = json.loads(RUNG4_RECORD.read_text())
        record["rung4_measured"] = {
            "label": "rung 4 TIDMAD readout, 1/J excluding DC (E6, measured)",
            "cv_inverse_weights": e6["excluding_dc"]["cv_inverse_J"],
            "ratio_max_min": e6["excluding_dc"]["ratio_max_min"],
            "effective_fraction": e6["excluding_dc"]["effective_fraction"],
        }
    else:
        print(f"  note: {RUNG4_RECORD} not found; side-by-side omitted")

    for key in sorted(k for k in record if k != "comment"):
        s = record[key]
        print(f"  {s['label']}")
        print(f"    CV(inverse weights) {s['cv_inverse_weights']:.4f}"
              f"    max/min {s['ratio_max_min']:.3g}"
              f"    effective fraction {s['effective_fraction']:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\n  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
