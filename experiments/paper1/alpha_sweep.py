"""P1-E2 — cost of the separability prior: nonseparability sweep.

For alpha in a grid, fit NFPA, CW-PCA, Iso-MPCA and IsoPCA at matched rank
k_c*k_t on the colored regime and record held-out normalized chi^2, raw MSE,
the NFPA <-> CW-PCA mean principal angle and the planted projector errors.
The approximation term is the held-out chi^2 gap NFPA - CW-PCA; the metric
term is NFPA - Iso-MPCA and must keep its sign at every alpha.

Task card E2* additionally requires the *optimisation* term: for every
(alpha, seed) the NFPA fit reports, over its ``n_restarts`` ALS restarts, the
minimum and maximum held-out normalized chi^2 and the across-restart standard
deviation of the planted channel/temporal projector errors. Those restarts are
the ones ``als_fit`` already runs internally (``collect_restarts=True`` only
records them), so no extra optimisation is performed and the selected NFPA fit
is identical to ``methods.fit_nfpa``.

The run is resumable: rows are keyed by (alpha, seed) in ``rows.jsonl`` and a
cell already present is never recomputed.

Usage::

    python -m experiments.paper1.alpha_sweep --alphas 0 0.05 0.1 0.2 0.4 0.8 --seeds 0 1 ... [--smoke]
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
from experiments.synthetic.nfpa_confirmatory.generator import make_dataset
from experiments.synthetic.nfpa_confirmatory.metrics import normalized_chi2, raw_mse
from noise_geometry.subspace.angles import principal_angles
from experiments.synthetic.nfpa_confirmatory.methods import (
    MethodFit, _whiten_batch, als_fit, recon_kron, reconstruct,
)

from . import common

EXPERIMENT_ID = "P1-E2"
METHODS = ("CW-PCA", "NFPA", "Iso-MPCA", "IsoPCA")

#: Legacy columns keep their names (ground rule 4: add columns, never rename);
#: the task-card names are added alongside them.
COLUMNS = [
    "config_hash", "alpha", "seed", "method", "test_normalized_chi2", "test_raw_mse",
    "mean_angle_to_cw_pca_deg", "channel_projector_error", "temporal_projector_error",
    "joint_projector_error_whitened", "runtime_seconds",
    "raw_mse", "angle_to_cwpca_deg", "projector_err_c", "projector_err_t",
    "chi2_restart_min", "chi2_restart_max",
    "projector_err_c_restart_sd", "projector_err_t_restart_sd",
    "n_restarts_recorded", "selected_restart",
]


def _nfpa_with_restarts(ds):
    """``methods.fit_nfpa`` with every ALS restart kept for diagnostics.

    Reproduces ``fit_nfpa`` exactly (same whitening, same seed, same restart
    sequence, same selection rule); the only difference is that the discarded
    restarts are returned as well.
    """
    from experiments.synthetic.nfpa_confirmatory.methods import _whiten_batch

    t0 = time.time()
    Z = _whiten_batch(ds, ds.X_train)
    out = als_fit(Z, ds.cfg.k_c, ds.cfg.k_t, n_iter=ds.cfg.n_als_iter,
                  n_restarts=ds.cfg.n_restarts, seed=ds.seed,
                  tol=ds.cfg.als_convergence_tol, collect_restarts=True)

    def _mk(U_c, U_t, diag):
        return MethodFit(name="NFPA", metric="Sigma^{-1}", rep_class="factored",
                         coord="whitened", U_c=U_c, U_t=U_t, diag=diag)

    best = _mk(out["U_c"], out["U_t"], {
        "train_loss": out["loss"], "selected_restart": out["restart"],
        "converged": out["converged"], "loss_history": out["loss_history"],
        "runtime_seconds": time.time() - t0,
    })
    restarts = [_mk(r["U_c"], r["U_t"], {"train_loss": r["loss"], "restart": r["restart"],
                                         "converged": r["converged"]})
                for r in out["restarts"]]
    return best, restarts


def _restart_spread(ds, restarts, Z_test) -> dict:
    """chi^2 min/max and projector-error SD across all ALS restarts.

    The restart chi^2 is evaluated directly in whitened coordinates:
    ``mean((Z - recon_kron(Z, U_c, U_t))**2)`` with ``Z`` the whitened,
    train-mean-centred test set. For a ``coord="whitened"`` method this is
    exactly ``normalized_chi2(ds, X_test, reconstruct(fit, ds, X_test))`` --
    ``reconstruct`` unwhitens and ``normalized_chi2`` whitens again, an exact
    round trip up to float64 rounding (~1e-15 relative) -- but it whitens the
    test set once per cell instead of four times per restart, which is what
    makes the 10-restart diagnostic affordable at the frozen protocol.
    """
    chi2, err_c, err_t = [], [], []
    for fit in restarts:
        chi2.append(float(np.mean((Z_test - recon_kron(Z_test, fit.U_c, fit.U_t)) ** 2)))
        errs = sem.planted_errors(fit, ds)
        if errs["channel_projector_error"] is not None:
            err_c.append(errs["channel_projector_error"])
        if errs["temporal_projector_error"] is not None:
            err_t.append(errs["temporal_projector_error"])
    return {
        "chi2_restart_min": float(np.min(chi2)) if chi2 else None,
        "chi2_restart_max": float(np.max(chi2)) if chi2 else None,
        "projector_err_c_restart_sd": float(np.std(err_c, ddof=0)) if err_c else None,
        "projector_err_t_restart_sd": float(np.std(err_t, ddof=0)) if err_t else None,
        "n_restarts_recorded": len(restarts),
    }


def _cell_rows(alpha: float, seed: int, cfg) -> list[dict]:
    ds = make_dataset(seed, "colored", cfg)
    nfpa, restarts = _nfpa_with_restarts(ds)
    spread = _restart_spread(ds, restarts, _whiten_batch(ds, ds.X_test))
    fits = common.fit_methods(ds, [m for m in METHODS if m != "NFPA"])
    fits["NFPA"] = nfpa
    cw = fits["CW-PCA"]
    B_cw = sem.fitted_basis_whitened(cw, ds)  # corrected unstacking (E11), not methods.basis_whitened
    rows = []
    for name in METHODS:
        fit = fits[name]
        Xhat = reconstruct(fit, ds, ds.X_test)
        errs = sem.planted_errors(fit, ds)
        angles = (np.asarray(principal_angles(sem.fitted_basis_whitened(fit, ds).T, B_cw.T, degrees=True))
                  if name != "CW-PCA" else np.zeros(1))
        chi2 = normalized_chi2(ds, ds.X_test, Xhat)
        mse_raw = raw_mse(ds.X_test, Xhat)
        ang = float(np.mean(angles))
        row = {
            "config_hash": config_hash(cfg), "alpha": float(alpha), "seed": int(seed), "method": name,
            "test_normalized_chi2": chi2,
            "test_raw_mse": mse_raw,
            "mean_angle_to_cw_pca_deg": ang,
            "channel_projector_error": errs.get("channel_projector_error"),
            "temporal_projector_error": errs.get("temporal_projector_error"),
            "joint_projector_error_whitened": errs.get("joint_projector_error_whitened"),
            "runtime_seconds": fit.diag.get("runtime_seconds"),
            "raw_mse": mse_raw,
            "angle_to_cwpca_deg": ang,
            "projector_err_c": errs.get("channel_projector_error"),
            "projector_err_t": errs.get("temporal_projector_error"),
            "chi2_restart_min": None, "chi2_restart_max": None,
            "projector_err_c_restart_sd": None, "projector_err_t_restart_sd": None,
            "n_restarts_recorded": None,
            "selected_restart": fit.diag.get("selected_restart"),
        }
        if name == "NFPA":
            row.update(spread)
        rows.append(row)
    return rows


def _load_done(out: Path) -> dict:
    """Union of every shard file ``rows*.jsonl`` in the output directory."""
    done: dict = {}
    for path in sorted(out.glob("rows*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[(float(rec["alpha"]), int(rec["seed"]))] = rec["rows"]
    return done


def run(alphas: list[float], seeds: list[int], *, smoke: bool, budget_s: float | None = None,
        shard: str = "", aggregate: bool = True) -> Path:
    """Run the (alpha, seed) grid.

    ``shard`` names this worker's append-only row file so several workers can
    cover disjoint seed sets concurrently; ``aggregate`` writes the CSV/summary
    /manifest only when every cell of the *full* grid is present.
    """
    base = common.smoke_config() if smoke else PRIMARY
    out = common.out_dir_for(EXPERIMENT_ID, smoke=smoke)
    jsonl = out / (f"rows.{shard}.jsonl" if shard else "rows.jsonl")
    done = _load_done(out)
    cfgs = [replace(base, nonseparability_alpha=float(a)) for a in alphas]
    total = len(alphas) * len(seeds)
    t_start = time.time()
    for alpha, cfg in zip(alphas, cfgs):
        for seed in seeds:
            key = (float(alpha), int(seed))
            if key in done:
                continue
            if budget_s is not None and time.time() - t_start > budget_s:
                print(f"[P1-E2] budget reached; {len(done)}/{total} cells done", flush=True)
                return out
            t0 = time.time()
            rows = _cell_rows(alpha, seed, cfg)
            done[key] = rows
            with jsonl.open("a") as fh:
                fh.write(json.dumps(common.as_jsonable(
                    {"alpha": float(alpha), "seed": int(seed), "rows": rows})) + "\n")
            print(f"[P1-E2] alpha={alpha} seed={seed} done in {time.time() - t0:.1f}s "
                  f"({len(done)}/{total})", flush=True)

    done = _load_done(out)  # pick up rows written by concurrent shards
    missing = [(a, s) for a in alphas for s in seeds if (float(a), int(s)) not in done]
    if missing or not aggregate:
        print(f"[P1-E2] incomplete: {len(missing)} cells missing; not aggregating", flush=True)
        return out

    all_rows = [r for a in alphas for s in seeds for r in done[(float(a), int(s))]]
    rc.write_rows_csv(out / "alpha_sweep.csv", all_rows, COLUMNS)

    summary = {"alphas": alphas, "seeds": seeds, "per_alpha": {}}
    for alpha in alphas:
        sub = [r for r in all_rows if r["alpha"] == float(alpha)]
        nf = [r for r in sub if r["method"] == "NFPA"]
        metric_term = rc.paired_contrast(sub, metric="test_normalized_chi2",
                                         method_a="NFPA", method_b="Iso-MPCA")
        spread = [r["chi2_restart_max"] - r["chi2_restart_min"] for r in nf
                  if r["chi2_restart_min"] is not None]
        summary["per_alpha"][str(alpha)] = {
            "approximation_term_chi2_nfpa_minus_cwpca": rc.paired_contrast(
                sub, metric="test_normalized_chi2", method_a="NFPA", method_b="CW-PCA"),
            "metric_term_chi2_nfpa_minus_isompca": metric_term,
            "metric_term_mse_nfpa_minus_isompca": rc.paired_contrast(
                sub, metric="test_raw_mse", method_a="NFPA", method_b="Iso-MPCA"),
            "mean_angle_nfpa_to_cwpca_deg": float(np.mean([r["mean_angle_to_cw_pca_deg"] for r in sub if r["method"] == "NFPA"])),
            "mean_angle_isompca_to_cwpca_deg": float(np.mean([r["mean_angle_to_cw_pca_deg"] for r in sub if r["method"] == "Iso-MPCA"])),
            "channel_projector_error_nfpa": float(np.mean([r["channel_projector_error"] for r in sub if r["method"] == "NFPA"])),
            "channel_projector_error_isompca": float(np.mean([r["channel_projector_error"] for r in sub if r["method"] == "Iso-MPCA"])),
            "nfpa_chi2_restart_spread_mean": float(np.mean(spread)) if spread else None,
            "nfpa_chi2_restart_spread_max": float(np.max(spread)) if spread else None,
            "nfpa_projector_err_c_restart_sd_mean": float(np.mean([r["projector_err_c_restart_sd"] for r in nf])) if nf else None,
            "nfpa_projector_err_t_restart_sd_mean": float(np.mean([r["projector_err_t_restart_sd"] for r in nf])) if nf else None,
            "restart_spread_over_abs_metric_term": (
                float(np.mean(spread) / abs(metric_term["mean_difference"]))
                if spread and metric_term.get("identified") and metric_term["mean_difference"] != 0.0 else None),
        }

    per = summary["per_alpha"]
    order = [str(a) for a in alphas]
    approx = [per[a]["approximation_term_chi2_nfpa_minus_cwpca"] for a in order]
    summary["predictions"] = {
        "a_alpha0_gap_interval_includes_le_zero": bool(approx[0]["ci95_low"] <= 0.0),
        "b_approximation_term_non_decreasing_overlapping": [
            {"from": order[i], "to": order[i + 1],
             "mean_from": approx[i]["mean_difference"], "mean_to": approx[i + 1]["mean_difference"],
             "ok": bool(approx[i + 1]["mean_difference"] >= approx[i]["mean_difference"]
                        or approx[i + 1]["ci95_high"] >= approx[i]["ci95_low"])}
            for i in range(len(order) - 1)],
        "c_metric_term_negative_excludes_zero": {
            a: {"mean": per[a]["metric_term_chi2_nfpa_minus_isompca"]["mean_difference"],
                "ci95_low": per[a]["metric_term_chi2_nfpa_minus_isompca"]["ci95_low"],
                "ci95_high": per[a]["metric_term_chi2_nfpa_minus_isompca"]["ci95_high"],
                "ok": bool(per[a]["metric_term_chi2_nfpa_minus_isompca"]["ci95_high"] < 0.0)}
            for a in order},
        "d_restart_spread_le_10pct_of_metric_term": {
            a: {"ratio": per[a]["restart_spread_over_abs_metric_term"],
                "ok": bool(per[a]["restart_spread_over_abs_metric_term"] is not None
                           and per[a]["restart_spread_over_abs_metric_term"] <= 0.10)}
            for a in order},
    }
    rc.write_json(out / "summary.json", common.as_jsonable(summary))
    common.write_manifest(out, experiment_id=EXPERIMENT_ID, cfgs=cfgs, seeds=seeds,
                          extra={"methods": list(METHODS),
                                 "walltime_s": time.time() - t_start,
                                 "host": platform.node(),
                                 "platform": platform.platform()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="stop after this many seconds of new cells (the run is resumable)")
    ap.add_argument("--shard", type=str, default="",
                    help="name of this worker's append-only row file (rows.<shard>.jsonl)")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="compute cells only; leave the CSV/summary to the aggregating pass")
    a = ap.parse_args(argv)
    print(f"wrote {run(a.alphas, a.seeds, smoke=a.smoke, budget_s=a.budget_s, shard=a.shard, aggregate=not a.no_aggregate)}")


if __name__ == "__main__":
    main()
