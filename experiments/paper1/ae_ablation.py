"""P1-E12 — tied versus untied linear autoencoders.

The only ablation the tied-adjoint claim (§5.4, Rem. E.3) needs. Four arms at
matched rank on the same data:

    (a) tied   + Sigma^{-1} loss      (``trained.train_weighted_linear_ae``)
    (b) untied + Sigma^{-1} loss      (``untied.train_untied_linear_ae``)
    (c) tied   + MSE loss
    (d) untied + MSE loss

each at two stopping rules: ``converged`` (L-BFGS to its ftol/gtol floor) and
``early`` (the same optimiser stopped at 200 iterations). Every arm is scored
with the *same* Sigma-based diagnostics, so the MSE arms are measured by how
far they land from the EMPCA subspace.

Datasets
--------
``S3``       ``configs/synthetic/s3_ae_bridge.yaml`` — colored (geometrically
             graded) diagonal noise, d = 128, rank 3; seeds 0-9.
``S4_white`` the same generator with white noise — the control where every
             arm must agree; seeds 0-4.
``planted``  the NFPA planted generator at alpha = 0, vectorised to
             d = C*T = 2048 at rank 6, with the separable Sigma^{-1} carried as
             its two Kronecker factors.

Usage::

    python -m experiments.paper1.ae_ablation --datasets S3 S4_white --seeds 0 1 ...
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory import r9_semantics as sem
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.covariance import whiten_separable
from experiments.synthetic.nfpa_confirmatory.generator import make_dataset
from noise_geometry.autoencoders.trained import train_weighted_linear_ae
from noise_geometry.autoencoders.untied import (
    SeparableMetric, obliqueness, train_untied_linear_ae, _whiten_rows,
)
from noise_geometry.experiments.synthetic import make_s3_dataset
from noise_geometry.subspace import fit_weighted_pca, principal_angles

from . import common

EXPERIMENT_ID = "P1-E12"

ARMS = {
    "a_tied_weighted":   ("tied", "weighted"),
    "b_untied_weighted": ("untied", "weighted"),
    "c_tied_mse":        ("tied", "mse"),
    "d_untied_mse":      ("untied", "mse"),
}
STOPPING = {"converged": 20000, "early": 200}

COLUMNS = ["dataset", "seed", "arm", "stopping", "angle_to_empca_max_deg", "obliqueness",
           "held_out_weighted_residual", "held_out_raw_mse", "iterations", "walltime_s",
           "tie", "loss", "rank", "n_train", "n_test", "d", "converged_flag"]


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def _split(X, frac=0.7):
    n_fit = int(round(frac * X.shape[0]))
    return X[:n_fit], X[n_fit:]


def dataset_S3(seed: int):
    X, weights, rank = make_s3_dataset({"seed": int(seed), "n_traces": 512,
                                        "n_features": 128, "rank": 3})
    Xtr, Xte = _split(X)
    return Xtr, Xte, weights, int(rank)


def dataset_S4_white(seed: int):
    """The S3 generator with a flat noise spectrum: the control."""
    from noise_geometry.experiments.synthetic import _rankk_signal
    rng = np.random.default_rng(int(seed))
    clean, _ = _rankk_signal(rng, 512, 128, 3)
    variance = np.full(128, 0.1)
    weights = 1.0 / variance
    X = clean + rng.normal(scale=np.sqrt(variance)[None, :], size=clean.shape)
    Xtr, Xte = _split(X)
    return Xtr, Xte, weights, 3


def dataset_planted(seed: int):
    """The NFPA planted generator at alpha = 0, vectorised (d = C*T, rank 6)."""
    cfg = replace(PRIMARY, nonseparability_alpha=0.0)
    ds = make_dataset(seed, "colored", cfg)
    d = cfg.C * cfg.T
    Xtr = ds.X_train.reshape(cfg.N_train, d, order="F")
    Xte = ds.X_test.reshape(cfg.N_test, d, order="F")
    metric = SeparableMetric(np.linalg.inv(ds.covariances["Sigma_c"]),
                             np.linalg.inv(ds.covariances["Sigma_t"]))
    return Xtr, Xte, metric, cfg.k_c * cfg.k_t


DATASETS = {"S3": dataset_S3, "S4_white": dataset_S4_white, "planted": dataset_planted}
DATASET_SEEDS = {"S3": list(range(10)), "S4_white": list(range(5)), "planted": list(range(3))}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _weighted_residual(X, mean, D, E, metric) -> float:
    from noise_geometry.autoencoders.untied import _apply_metric
    R = (X - mean[None, :]) - ((X - mean[None, :]) @ E.T) @ D.T
    return float(np.sum(_apply_metric(R, metric) * R) / X.shape[0])


def _raw_mse(X, mean, D, E) -> float:
    Xc = X - mean[None, :]
    R = Xc - (Xc @ E.T) @ D.T
    return float(np.mean(R ** 2))


def _empca_components(Xtr, metric, rank):
    if isinstance(metric, SeparableMetric):
        # EMPCA == PCA on metric-whitened data; done through the Kronecker
        # factors so the dense d x d metric is never formed.
        mean = Xtr.mean(axis=0)
        Zw = _whiten_rows(Xtr - mean[None, :], metric)
        _, _, vh = np.linalg.svd(Zw, full_matrices=False)
        # map back to data units: components solve <c, x>_M ; use M^{-1/2} vh
        from noise_geometry.autoencoders.untied import _sym_sqrt
        Ainv = np.linalg.inv(_sym_sqrt(metric.A))
        Binv = np.linalg.inv(_sym_sqrt(metric.B))
        C, T = metric.A.shape[0], metric.B.shape[0]
        V = vh[:rank].reshape(rank, C, T, order="F")
        back = np.einsum("ab,nbt,ts->nas", Ainv, V, Binv, optimize=True)
        return back.reshape(rank, C * T, order="F")
    return fit_weighted_pca(Xtr, metric, rank).components


def _fit_arm(arm: str, stopping: str, Xtr, metric, rank, seed: int):
    tie, loss = ARMS[arm]
    max_iter = STOPPING[stopping]
    t0 = time.time()
    if tie == "untied":
        res = train_untied_linear_ae(
            Xtr, metric, rank, loss=loss, optimizer="lbfgs", seed=seed,
            max_iter=max_iter, l2=0.0, reference_components=None)
        D, E, mean, nit, ok = res.decoder, res.encoder, res.mean, res.n_iter, res.converged
    else:
        d = Xtr.shape[1]
        w = metric if loss == "weighted" else np.ones(d)
        if loss == "weighted" and isinstance(metric, SeparableMetric):
            w = metric.dense()
        res = train_weighted_linear_ae(Xtr, w, rank, optimizer="lbfgs", seed=seed,
                                       max_iter=max_iter)
        D = res.decoder
        # the tie: encoder = D^T M, with M the metric the arm was trained under
        from noise_geometry.autoencoders.untied import _apply_metric
        E = _apply_metric(D.T, w if not isinstance(w, SeparableMetric) else w)
        mean, nit, ok = res.mean, res.n_iter, True
    return D, E, mean, nit, ok, time.time() - t0


def _cell(dataset: str, seed: int) -> list[dict]:
    Xtr, Xte, metric, rank = DATASETS[dataset](seed)
    ref = _empca_components(Xtr, metric, rank)
    rows = []
    for arm in ARMS:
        for stopping in STOPPING:
            D, E, mean, nit, ok, wall = _fit_arm(arm, stopping, Xtr, metric, rank, seed)
            ang = np.asarray(principal_angles(_whiten_rows(D.T, metric),
                                              _whiten_rows(ref, metric), degrees=True))
            rows.append({
                "dataset": dataset, "seed": int(seed), "arm": arm, "stopping": stopping,
                "angle_to_empca_max_deg": float(np.max(ang)),
                "obliqueness": obliqueness(D, E, metric),
                "held_out_weighted_residual": _weighted_residual(Xte, mean, D, E, metric),
                "held_out_raw_mse": _raw_mse(Xte, mean, D, E),
                "iterations": int(nit), "walltime_s": float(wall),
                "tie": ARMS[arm][0], "loss": ARMS[arm][1], "rank": int(rank),
                "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                "d": int(Xtr.shape[1]), "converged_flag": bool(ok),
            })
    return rows


def _load_done(out: Path) -> dict:
    done: dict = {}
    for path in sorted(out.glob("cells*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[(rec["dataset"], int(rec["seed"]))] = rec["rows"]
    return done


def run(datasets: list[str], seeds: dict[str, list[int]], *, shard: str = "",
        aggregate: bool = True, budget_s: float | None = None) -> Path:
    out = common.out_dir_for(EXPERIMENT_ID)
    jsonl = out / (f"cells.{shard}.jsonl" if shard else "cells.jsonl")
    done = _load_done(out)
    t_start = time.time()
    for dataset in datasets:
        for seed in seeds[dataset]:
            if (dataset, int(seed)) in done:
                continue
            if budget_s is not None and time.time() - t_start > budget_s:
                print(f"[{EXPERIMENT_ID}] budget reached", flush=True)
                return out
            t0 = time.time()
            rows = _cell(dataset, seed)
            done[(dataset, int(seed))] = rows
            with jsonl.open("a") as fh:
                fh.write(json.dumps(common.as_jsonable(
                    {"dataset": dataset, "seed": int(seed), "rows": rows})) + "\n")
            print(f"[{EXPERIMENT_ID}] {dataset} seed={seed} done in {time.time() - t0:.1f}s", flush=True)

    done = _load_done(out)
    want = [(d, s) for d in datasets for s in seeds[d]]
    missing = [k for k in want if k not in done]
    if missing or not aggregate:
        print(f"[{EXPERIMENT_ID}] incomplete: {len(missing)} cells missing", flush=True)
        return out

    all_rows = [r for k in want for r in done[k]]
    rc.write_rows_csv(out / "ae_ablation.csv", all_rows, COLUMNS)

    def sel(**kw):
        return [r for r in all_rows if all(r[k] == v for k, v in kw.items())]

    summary: dict = {"datasets": datasets, "seeds": seeds, "per_cell": {}}
    for dataset in datasets:
        for arm in ARMS:
            for stopping in STOPPING:
                sub = sel(dataset=dataset, arm=arm, stopping=stopping)
                if not sub:
                    continue
                summary["per_cell"][f"{dataset}|{arm}|{stopping}"] = {
                    "angle_to_empca_max_deg_mean": float(np.mean([r["angle_to_empca_max_deg"] for r in sub])),
                    "angle_to_empca_max_deg_max": float(np.max([r["angle_to_empca_max_deg"] for r in sub])),
                    "obliqueness_mean": float(np.mean([r["obliqueness"] for r in sub])),
                    "obliqueness_max": float(np.max([r["obliqueness"] for r in sub])),
                    "held_out_weighted_residual_mean": float(np.mean([r["held_out_weighted_residual"] for r in sub])),
                    "held_out_raw_mse_mean": float(np.mean([r["held_out_raw_mse"] for r in sub])),
                    "iterations_mean": float(np.mean([r["iterations"] for r in sub])),
                    "n_seeds": len(sub),
                }

    def _g(ds, arm, stop, key):
        c = summary["per_cell"].get(f"{ds}|{arm}|{stop}")
        return None if c is None else c[key]

    preds: dict = {}
    for ds in datasets:
        if _g(ds, "a_tied_weighted", "converged", "angle_to_empca_max_deg_max") is None:
            continue
        b_early = sel(dataset=ds, arm="b_untied_weighted", stopping="early")
        preds[ds] = {
            "a_converged_angle_lt_0.01deg_and_obliqueness_lt_1e-6": {
                "angle_max": _g(ds, "a_tied_weighted", "converged", "angle_to_empca_max_deg_max"),
                "obliqueness_max": _g(ds, "a_tied_weighted", "converged", "obliqueness_max"),
                "ok": bool(_g(ds, "a_tied_weighted", "converged", "angle_to_empca_max_deg_max") < 0.01
                           and _g(ds, "a_tied_weighted", "converged", "obliqueness_max") < 1e-6)},
            "b_converged_angle_lt_0.01deg": {
                "angle_max": _g(ds, "b_untied_weighted", "converged", "angle_to_empca_max_deg_max"),
                "ok": bool(_g(ds, "b_untied_weighted", "converged", "angle_to_empca_max_deg_max") < 0.01)},
            "b_early_oblique_and_rotated_in_ge_8_of_10_seeds": {
                "n_seeds_meeting": int(sum(1 for r in b_early
                                           if r["obliqueness"] > 1e-2 and r["angle_to_empca_max_deg"] > 1.0)),
                "n_seeds": len(b_early),
                "ok": bool(sum(1 for r in b_early
                               if r["obliqueness"] > 1e-2 and r["angle_to_empca_max_deg"] > 1.0)
                           >= max(1, int(round(0.8 * len(b_early)))))},
            "cd_mse_converged_angle": {
                "c_tied_mse_angle_mean": _g(ds, "c_tied_mse", "converged", "angle_to_empca_max_deg_mean"),
                "d_untied_mse_angle_mean": _g(ds, "d_untied_mse", "converged", "angle_to_empca_max_deg_mean"),
                "expected": ">30 deg on colored S3/planted, <0.01 deg on the S4 white control"},
        }
        # does (b) converged equal (a) converged?
        preds[ds]["b_equals_a_at_convergence"] = {
            "angle_gap_deg": abs((_g(ds, "b_untied_weighted", "converged", "angle_to_empca_max_deg_mean") or 0.0)
                                 - (_g(ds, "a_tied_weighted", "converged", "angle_to_empca_max_deg_mean") or 0.0)),
            "obliqueness_b": _g(ds, "b_untied_weighted", "converged", "obliqueness_max"),
        }
    summary["predictions"] = preds
    rc.write_json(out / "summary.json", common.as_jsonable(summary))
    cfgs = [replace(PRIMARY, nonseparability_alpha=0.0)]
    all_seeds = sorted({s for d in datasets for s in seeds[d]})
    common.write_manifest(out, experiment_id=EXPERIMENT_ID, cfgs=cfgs, seeds=all_seeds,
                          extra={"arms": {k: list(v) for k, v in ARMS.items()},
                                 "stopping": STOPPING, "datasets": datasets,
                                 "dataset_seeds": seeds,
                                 "walltime_s": time.time() - t_start,
                                 "host": platform.node(), "platform": platform.platform()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["S3", "S4_white"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override the per-dataset seed list")
    ap.add_argument("--shard", type=str, default="")
    ap.add_argument("--no-aggregate", action="store_true")
    ap.add_argument("--budget-s", type=float, default=None)
    a = ap.parse_args(argv)
    seeds = {d: (a.seeds if a.seeds is not None else DATASET_SEEDS[d]) for d in a.datasets}
    print(f"wrote {run(a.datasets, seeds, shard=a.shard, aggregate=not a.no_aggregate, budget_s=a.budget_s)}")


if __name__ == "__main__":
    main()
