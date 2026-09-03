"""P1-E9 — Physical-variable probes: does the latent space organise by physics?

For each method (CW-PCA, NFPA, Iso-MPCA, IsoPCA) at the exactly separable
condition (alpha = 0) fit on the frozen configuration, take the latent codes
``z`` of held-out events and ask, Panda-style, whether simple linear probes
recover the planted physical variables (amplitude, rise time, decay time, and
the secondary latent):

* ``probe_r2[v]``       -- held-out R^2 of a ridge probe z -> v (train split only);
* ``max_abs_corr[v]``   -- largest |corr(z_j, v)| over latent coordinates;
* ``participation[v]``  -- effective number of latent coordinates the variable
                            occupies, 1 / sum_j w_j^2 with w_j = corr_j^2 / sum corr^2
                            (1 = one axis carries the variable, k = spread out);
* ``pairwise_r2``        -- full (k x n_variables) |corr| matrix, saved for the figure.

The comparison NFPA vs Iso-MPCA isolates the metric (same architecture,
different training geometry); CW-PCA vs IsoPCA does the same for the
unrestricted class.  The claim is only about *organisation*: whether the
representation trained in the likelihood geometry exposes the physical
degrees of freedom to a linear readout, and whether it does so on fewer axes.

Usage::

    python -m experiments.paper1.latent_probes --seeds 0 1 2 3 [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import semantic as sem
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.covariance import whiten_separable
from experiments.synthetic.nfpa_confirmatory.generator import make_dataset

from . import common

EXPERIMENT_ID = "P1-E9"
METHODS = ("CW-PCA", "NFPA", "Iso-MPCA", "IsoPCA")
VARIABLES = ("amplitude", "rise_time", "decay_time", "secondary")


def _standardise(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (Z - mu) / sd


def codes(fit, ds, X: np.ndarray) -> np.ndarray:
    """``(n, rank)`` latent codes for factored and unfactored fits alike."""
    if fit.U_c is not None:
        return sem.latent_codes(fit, ds, X)
    if fit.coord == "whitened":
        Z = whiten_separable(X - ds.train_mean, ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])
    else:
        Z = X - ds.train_mean
    flat = Z.reshape(X.shape[0], -1, order="F")
    return flat @ fit.P


def probe_metrics(z_train: np.ndarray, z_test: np.ndarray, lat_train: dict, lat_test: dict,
                  ridge: float = 1e-3) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    mu, sd = z_train.mean(0), z_train.std(0) + 1e-12
    zt, ze = _standardise(z_train, mu, sd), _standardise(z_test, mu, sd)
    k = zt.shape[1]
    out: dict[str, dict[str, float]] = {}
    corr_mat = np.zeros((k, len(VARIABLES)))
    for vi, v in enumerate(VARIABLES):
        yt, ye = np.asarray(lat_train[v], float), np.asarray(lat_test[v], float)
        ym = yt.mean()
        A = np.column_stack([zt, np.ones(len(zt))])
        G = A.T @ A + ridge * np.eye(k + 1)
        w = np.linalg.solve(G, A.T @ (yt - ym))
        pred = np.column_stack([ze, np.ones(len(ze))]) @ w + ym
        ss_res = float(np.sum((ye - pred) ** 2))
        ss_tot = float(np.sum((ye - ye.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        c = np.array([np.corrcoef(ze[:, j], ye)[0, 1] if ze[:, j].std() > 0 else 0.0 for j in range(k)])
        c = np.nan_to_num(c)
        corr_mat[:, vi] = np.abs(c)
        c2 = c ** 2
        if c2.sum() > 0:
            wj = c2 / c2.sum()
            participation = float(1.0 / np.sum(wj ** 2))
        else:
            participation = float("nan")
        out[v] = {
            "probe_r2": float(r2),
            "max_abs_corr": float(np.max(np.abs(c))),
            "participation": participation,
        }
    return out, corr_mat


def run(seeds: list[int], *, smoke: bool) -> Path:
    cfg = replace(common.smoke_config() if smoke else PRIMARY, nonseparability_alpha=0.0)
    out = common.out_dir_for(EXPERIMENT_ID, smoke=smoke)
    rows: list[dict] = []
    corr_store: dict[str, np.ndarray] = {}
    for seed in seeds:
        ds = make_dataset(seed, "colored", cfg)
        fits = common.fit_methods(ds, METHODS)
        for name, fit in fits.items():
            z_train = codes(fit, ds, ds.X_train)
            z_test = codes(fit, ds, ds.X_test)
            metrics, corr_mat = probe_metrics(z_train, z_test, ds.latents_train, ds.latents_test)
            corr_store[f"seed{seed:02d}_{name.replace('-', '_')}"] = corr_mat
            for v in VARIABLES:
                rows.append({
                    "config_hash": config_hash(cfg), "seed": seed, "method": name, "variable": v,
                    "k": z_test.shape[1], **metrics[v],
                    "runtime_seconds": fit.diag.get("runtime_seconds"),
                })
            print(f"seed {seed} {name:9s} " + "  ".join(
                f"{v}: R2={metrics[v]['probe_r2']:.3f} part={metrics[v]['participation']:.2f}" for v in VARIABLES), flush=True)
    with (out / "probes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(out / "corr_matrices.npz", **corr_store)
    # summary: mean and sd over seeds per (method, variable)
    summary: dict[str, dict] = {}
    for name in METHODS:
        summary[name] = {}
        for v in VARIABLES:
            sel = [r for r in rows if r["method"] == name and r["variable"] == v]
            for key in ("probe_r2", "max_abs_corr", "participation"):
                vals = np.array([r[key] for r in sel], float)
                summary[name][f"{v}.{key}.mean"] = float(np.nanmean(vals))
                summary[name][f"{v}.{key}.sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    common.write_manifest(out, experiment_id=EXPERIMENT_ID, cfgs=[cfg], seeds=seeds,
                         extra={"notes": "P1-E9 physical-variable probes at alpha=0; ridge probes fit on train, "
                                         "scored on test; participation = effective number of latent axes."})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = run(a.seeds, smoke=a.smoke)
    print("wrote", out)


if __name__ == "__main__":
    main()
