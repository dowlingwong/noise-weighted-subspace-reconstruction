#!/usr/bin/env python3
"""P1-E10 — does the anisotropy of Sigma inside the fitted subspace predict the gain?

Predictor (frozen before the outcome is looked at, `docs/EXPERIMENT_PLAN_PAPER1.md`
P1-E10)::

    a = 1 - (tr Sigma_S / k) / lambda_max(Sigma_S),      Sigma_S = B^T Sigma B

for the rank-k fitted basis ``B`` of a channel. ``a = 0`` means Sigma is
isotropic inside the fitted subspace (nothing for the metric to exploit);
``a -> 1 - 1/k`` means one direction dominates.

Outcome: the archived same-rank fractional reduction per channel, read from
``transfer_paper/data/cresst/sensitivity/<run>/paper_facing_table.csv``
(column ``normalized_PCA_minus_EMPCA``, rows ``channel_normalized_effect``).

Reported: Spearman rho over the channels with a permutation interval
(10 000 permutations). The sign and the interval are the result, whatever they
are — a failed prediction is reported, not tuned.

Input contract for the predictor side
-------------------------------------
This script does **not** re-fit CRESST. It needs a per-channel dump of Sigma and
the fitted basis, which the CRESST runbook must produce on the cluster
(``docs/runbooks/CRESST_REMOTE_AGENT_RUNBOOK.md``: "dump per-channel PSD / Sigma
and fitted bases"). The expected file is an ``.npz`` with, for each channel
label ``L``:

    ``L__Sigma``   (p, p) covariance, or ``L__psd`` (p//2+1,) for a stationary
                   model, from which Sigma is rebuilt with
                   ``noise_geometry.noise.covariance.covariance_from_psd``
    ``L__basis``   (k, p) fitted subspace basis (rows), any scaling

Labels must match the ``label`` column of the paper-facing table (e.g. ``33_36.0``).

Usage::

    python scripts/gain_predictor.py \
        --sigma-npz /ceph/dwong/.../cresst_channel_sigma_basis.npz \
        --table transfer_paper/data/cresst/sensitivity/20260713_channel_robustness_v1/paper_facing_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "P1-E10"
PERMUTATIONS = 10_000
PERMUTATION_SEED = 20260906


def anisotropy(Sigma: np.ndarray, basis: np.ndarray) -> float:
    """``a = 1 - (tr Sigma_S / k) / lambda_max(Sigma_S)`` for ``Sigma_S = B^T Sigma B``.

    ``basis`` rows are the subspace vectors; they are orthonormalised first, so
    the predictor depends on the subspace, not on how the basis was scaled.
    """
    B = np.linalg.qr(np.asarray(basis, dtype=np.float64).T)[0]      # (p, k)
    Sigma_S = B.T @ np.asarray(Sigma, dtype=np.float64) @ B
    k = Sigma_S.shape[0]
    eig = np.linalg.eigvalsh(0.5 * (Sigma_S + Sigma_S.T))
    lam_max = float(np.max(eig))
    if lam_max <= 0:
        return float("nan")
    return float(1.0 - (float(np.trace(Sigma_S)) / k) / lam_max)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order.astype(float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def permutation_interval(x, y, *, permutations=PERMUTATIONS, seed=PERMUTATION_SEED) -> dict:
    """Null distribution of Spearman rho under label permutation."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    observed = spearman(x, y)
    rng = np.random.default_rng(seed)
    null = np.empty(int(permutations))
    for i in range(int(permutations)):
        null[i] = spearman(x, rng.permutation(y))
    lo, hi = np.percentile(null, [2.5, 97.5])
    return {
        "spearman_rho": observed,
        "n_channels": int(x.size),
        "permutations": int(permutations),
        "permutation_seed": int(seed),
        "null_ci95": [float(lo), float(hi)],
        "two_sided_p": float(np.mean(np.abs(null) >= abs(observed))),
        "outside_null_interval": bool(observed < lo or observed > hi),
    }


def read_outcomes(table: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with table.open() as fh:
        for row in csv.DictReader(fh):
            if row["row_type"] == "channel_normalized_effect":
                out[row["label"]] = float(row["normalized_PCA_minus_EMPCA"])
    return out


def read_predictors(npz: Path) -> dict[str, float]:
    data = np.load(npz, allow_pickle=False)
    labels = sorted({k.split("__")[0] for k in data.files})
    out: dict[str, float] = {}
    for label in labels:
        if f"{label}__basis" not in data:
            continue
        if f"{label}__Sigma" in data:
            Sigma = data[f"{label}__Sigma"]
        elif f"{label}__psd" in data:
            from noise_geometry.noise.covariance import covariance_from_psd
            basis = data[f"{label}__basis"]
            Sigma = covariance_from_psd(data[f"{label}__psd"], basis.shape[1])
        else:
            continue
        out[label] = anisotropy(Sigma, data[f"{label}__basis"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-npz", type=str, default=None,
                    help="per-channel Sigma/basis dump produced by the CRESST runbook")
    ap.add_argument("--table", type=str,
                    default="transfer_paper/data/cresst/sensitivity/"
                            "20260713_channel_robustness_v1/paper_facing_table.csv")
    ap.add_argument("--out", type=str, default="results/paper1/P1-E10")
    a = ap.parse_args(argv)

    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    table = Path(a.table) if Path(a.table).is_absolute() else REPO_ROOT / a.table

    record: dict = {
        "experiment_id": EXPERIMENT_ID,
        "predictor": "a = 1 - (tr Sigma_S / k) / lambda_max(Sigma_S), Sigma_S = B^T Sigma B",
        "predictor_frozen_before_outcome": True,
        "outcome_source": str(table),
        "permutations": PERMUTATIONS, "permutation_seed": PERMUTATION_SEED,
        "host": platform.node(),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if not table.exists():
        record.update({"status": "not_run", "reason": f"outcome table not found: {table}"})
    elif a.sigma_npz is None or not Path(a.sigma_npz).exists():
        outcomes = read_outcomes(table)
        record.update({
            "status": "not_run",
            "reason": ("the per-channel Sigma / fitted-basis dump is required and was not "
                       "supplied; it is produced on the cluster by the CRESST runbook step "
                       "'dump per-channel PSD / Sigma and fitted bases'. The archived "
                       "sensitivity bundle carries residuals and effects, not Sigma."),
            "sigma_npz_requested": a.sigma_npz,
            "outcomes_available": outcomes,
            "n_channels_with_outcome": len(outcomes),
        })
    else:
        outcomes = read_outcomes(table)
        predictors = read_predictors(Path(a.sigma_npz))
        labels = sorted(set(outcomes) & set(predictors))
        x = np.array([predictors[l] for l in labels])
        y = np.array([outcomes[l] for l in labels])
        stats = permutation_interval(x, y)
        with (out / "gain_predictor.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["channel", "anisotropy_a", "normalized_PCA_minus_EMPCA"])
            for l, xi, yi in zip(labels, x, y):
                w.writerow([l, xi, yi])
        record.update({"status": "complete", "channels": labels, **stats})

    (out / "summary.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2)[:1600])


if __name__ == "__main__":
    main()
