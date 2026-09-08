"""P1-E11 — excited support: does the data or the initialisation fix T_S^perp?

Turns Prop. 7.3 (``MS/06_loss_geometry.tex``, ``prop:excited-support``) into a
measured statement. Two training supports are compared at the frozen protocol:

* ``S_full``       = {amplitude, rise_time, decay_time, secondary}
* ``S_restricted`` = {amplitude, rise_time}, with ``decay_time`` and
  ``secondary`` held at their distribution means (both 0.0) in the *training*
  draw only. Validation and test keep the full support, so the unexcited
  coordinates are present at test time.

Per (support, seed, method) we record

(i)   ``angle_to_TS_max``    — largest principal angle between the learned
      whitened subspace and the excited tangent span T_S, computed with
      ``oracle_cov.tangent.tangent_geometry`` at 64 training latents;
(ii)  ``complement_spread``  — mean pairwise largest principal angle, across the
      R initialisations, of the learned subspace's component orthogonal to T_S;
(iii) ``held_out_chi2``      — normalized chi^2 on the S_full test set.

NFPA restarts are obtained exactly as the task card specifies: ``als_fit`` with
``n_restarts=1`` and ``seed = 1000*seed + r``. CW-PCA is a closed-form
eigen-solve with no initialisation, so it contributes a single solution per
seed and its complement spread is identically zero by construction; it is
recorded, but P-b is decided by NFPA.

Usage::

    python -m experiments.paper1.excited_support --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import generator as gen
from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory import r9_semantics as sem
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.methods import (
    MethodFit, _whiten_batch, als_fit, fit_cw_pca, recon_kron, reconstruct,
)
from experiments.synthetic.nfpa_confirmatory.metrics import normalized_chi2
from experiments.oracle.oracle_cov import generator as ogen
from experiments.oracle.oracle_cov.tangent import tangent_geometry
from noise_geometry.subspace.angles import principal_angles

from . import common

EXPERIMENT_ID = "P1-E11"

SUPPORTS = {
    "S_full": ("amplitude", "rise_time", "decay_time", "secondary"),
    "S_restricted": ("amplitude", "rise_time"),
}
#: Latents held at their distribution means in the restricted *training* draw.
HELD_AT_MEAN = {"decay_time": 0.0, "secondary": 0.0}

COLUMNS = ["support", "seed", "method", "restart", "angle_to_TS_max", "held_out_chi2",
           "n_train", "dim_TS", "dim_learned", "train_loss", "walltime_s",
           # diagnostics (added after the smoke wiring run; no threshold or
           # prediction of the task card is affected)
           "angle_to_TS_min", "angle_to_TS_median", "angle_to_planted6_max",
           "ts_data_var_min", "ts_data_var_max", "noise_floor_var"]
SPREAD_COLUMNS = ["support", "seed", "method", "n_train", "complement_spread_deg",
                  "n_restarts", "dim_complement"]


@contextlib.contextmanager
def _restricted_training_latents(n_train: int):
    """Hold the unexcited latents at their means in the training draw only.

    Implemented as the task card specifies: the output of ``sample_latents`` is
    overridden before ``build_coefficients``. The RNG stream is untouched (the
    original sampler is still called with the same ``n``), so the noise draws,
    the validation set and the test set are bit-identical to the S_full run.
    """
    original = gen.sample_latents

    def patched(rng, n):
        latents = original(rng, n)
        if n == n_train:
            for name, value in HELD_AT_MEAN.items():
                latents[name] = np.full_like(latents[name], value)
        return latents

    gen.sample_latents = patched
    try:
        yield
    finally:
        gen.sample_latents = original


def _training_set(seed: int, cfg, *, restricted: bool):
    if not restricted:
        return ogen.make_training_set(seed, cfg, alpha=0.0)
    with _restricted_training_latents(cfg.N_train):
        return ogen.make_training_set(seed, cfg, alpha=0.0)


def _nfpa_restart(Z, cfg, seed: int, r: int) -> MethodFit:
    """One ALS initialisation, seeded exactly as the card prescribes."""
    out = als_fit(Z, cfg.k_c, cfg.k_t, n_iter=cfg.n_als_iter, n_restarts=1,
                  seed=1000 * seed + r, tol=cfg.als_convergence_tol)
    return MethodFit(name="NFPA", metric="Sigma^{-1}", rep_class="factored", coord="whitened",
                     U_c=out["U_c"], U_t=out["U_t"],
                     diag={"train_loss": out["loss"], "restart": r, "converged": out["converged"]})


def _chi2_whitened(Z: np.ndarray, fit: MethodFit) -> float:
    """Held-out normalized chi^2, evaluated directly in whitened coordinates.

    For a ``coord="whitened"`` method this equals
    ``normalized_chi2(ds, X, reconstruct(fit, ds, X))`` exactly:
    ``reconstruct`` unwhitens and ``normalized_chi2`` whitens again, an exact
    round trip. Doing it here whitens the test set once per cell instead of
    four times per initialisation.
    """
    if fit.U_c is not None:
        Zhat = recon_kron(Z, fit.U_c, fit.U_t)
    else:
        n = Z.shape[0]
        flat = Z.reshape(n, -1, order="F") @ (fit.P @ fit.P.T)
        Zhat = flat.reshape(Z.shape, order="F")
    return float(np.mean((Z - Zhat) ** 2))


def _complement_basis(B: np.ndarray, T_S: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the part of ``span(B)`` orthogonal to ``span(T_S)``."""
    R = B - T_S @ (T_S.T @ B)
    q, s, _ = np.linalg.svd(R, full_matrices=False)
    keep = int(np.sum(s > 1e-8 * max(s[0], 1e-300)))
    return q[:, :keep]


def _pairwise_spread_deg(bases: list[np.ndarray]) -> tuple[float | None, int]:
    usable = [b for b in bases if b.shape[1] > 0]
    if len(usable) < 2:
        return (0.0 if len(usable) <= 1 else None), (usable[0].shape[1] if usable else 0)
    vals = []
    for a, b in combinations(usable, 2):
        ang = np.asarray(principal_angles(a.T, b.T, degrees=True))
        vals.append(float(np.max(ang)))
    return float(np.mean(vals)), usable[0].shape[1]


def _cell(support: str, seed: int, n_train: int, n_restarts: int, base) -> tuple[list[dict], list[dict]]:
    cfg = replace(base, N_train=int(n_train), nonseparability_alpha=0.0)
    restricted = support == "S_restricted"
    ts = _training_set(seed, cfg, restricted=restricted)
    ds = ts.ds
    geom = tangent_geometry(ts, SUPPORTS[support])
    T_S = geom.T_S
    Z = _whiten_batch(ds, ds.X_train)
    Z_test = _whiten_batch(ds, ds.X_test)
    planted6 = sem.planted_joint_basis_whitened(ds)

    # How strongly the training data actually excites each T_S direction, and
    # the Marchenko-Pastur upper edge of the isotropic whitened noise: a T_S
    # direction whose data variance sits below this edge is not recoverable at
    # this N by *any* second-moment method. Diagnostic only.
    Zf = Z.reshape(ds.cfg.N_train, -1, order="F")
    ts_var = np.var(Zf @ T_S, axis=0, ddof=1)
    d_amb = Zf.shape[1]
    noise_floor = float((1.0 + np.sqrt(d_amb / float(ds.cfg.N_train))) ** 2)

    def _diag(B):
        a = np.asarray(principal_angles(B.T, T_S.T, degrees=True))
        return {
            "angle_to_TS_max": float(np.max(a)),
            "angle_to_TS_min": float(np.min(a)),
            "angle_to_TS_median": float(np.median(a)),
            "angle_to_planted6_max": float(np.max(np.asarray(
                principal_angles(B.T, planted6.T, degrees=True)))),
            "ts_data_var_min": float(np.min(ts_var)),
            "ts_data_var_max": float(np.max(ts_var)),
            "noise_floor_var": noise_floor,
        }

    rows, comp = [], {"NFPA": [], "CW-PCA": []}
    # --- CW-PCA: closed form, one solution per seed -------------------------
    t0 = time.time()
    cw = fit_cw_pca(ds)
    B = sem.fitted_basis_whitened(cw, ds)
    row = {"support": support, "seed": seed, "method": "CW-PCA", "restart": 0,
           "held_out_chi2": _chi2_whitened(Z_test, cw),
           "n_train": int(n_train), "dim_TS": int(T_S.shape[1]), "dim_learned": int(B.shape[1]),
           "train_loss": None, "walltime_s": time.time() - t0}
    row.update(_diag(B))
    rows.append(row)
    comp["CW-PCA"].append(_complement_basis(B, T_S))

    # --- NFPA: R initialisations -------------------------------------------
    for r in range(n_restarts):
        t0 = time.time()
        fit = _nfpa_restart(Z, cfg, seed, r)
        B = sem.fitted_basis_whitened(fit, ds)
        row = {"support": support, "seed": seed, "method": "NFPA", "restart": r,
               "held_out_chi2": _chi2_whitened(Z_test, fit),
               "n_train": int(n_train), "dim_TS": int(T_S.shape[1]), "dim_learned": int(B.shape[1]),
               "train_loss": fit.diag["train_loss"], "walltime_s": time.time() - t0}
        row.update(_diag(B))
        rows.append(row)
        comp["NFPA"].append(_complement_basis(B, T_S))

    spread_rows = []
    for method, bases in comp.items():
        spread, dim = _pairwise_spread_deg(bases)
        spread_rows.append({"support": support, "seed": seed, "method": method,
                            "n_train": int(n_train),
                            "complement_spread_deg": spread,
                            "n_restarts": len(bases), "dim_complement": dim})
    return rows, spread_rows


def _load_done(out: Path) -> dict:
    done: dict = {}
    for path in sorted(out.glob("cells*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[(rec["support"], int(rec["seed"]), int(rec["n_train"]))] = rec
    return done


def _bootstrap_gap(per_seed: list[float]) -> dict:
    return rc.bootstrap_paired(per_seed)


def run(seeds: list[int], *, smoke: bool, restarts: int, restarts_4x: int,
        shard: str = "", aggregate: bool = True, budget_s: float | None = None) -> Path:
    base = common.smoke_config() if smoke else PRIMARY
    out = common.out_dir_for(EXPERIMENT_ID, smoke=smoke)
    jsonl = out / (f"cells.{shard}.jsonl" if shard else "cells.jsonl")
    done = _load_done(out)
    n1, n4 = base.N_train, 4 * base.N_train
    plan = ([(s, n1, restarts) for s in seeds] + [(s, n4, restarts_4x) for s in seeds])
    t_start = time.time()
    for seed, n_train, R in plan:
        for support in SUPPORTS:
            key = (support, int(seed), int(n_train))
            if key in done:
                continue
            if budget_s is not None and time.time() - t_start > budget_s:
                print(f"[{EXPERIMENT_ID}] budget reached", flush=True)
                return out
            t0 = time.time()
            rows, spread_rows = _cell(support, seed, n_train, R, base)
            rec = {"support": support, "seed": int(seed), "n_train": int(n_train),
                   "restarts": R, "rows": rows, "spread": spread_rows}
            done[key] = rec
            with jsonl.open("a") as fh:
                fh.write(json.dumps(common.as_jsonable(rec)) + "\n")
            print(f"[{EXPERIMENT_ID}] {support} seed={seed} n_train={n_train} "
                  f"done in {time.time() - t0:.1f}s", flush=True)

    done = _load_done(out)
    want = [(sup, s, n) for s in seeds for n in (n1, n4) for sup in SUPPORTS]
    missing = [k for k in want if k not in done]
    if missing or not aggregate:
        print(f"[{EXPERIMENT_ID}] incomplete: {len(missing)} cells missing", flush=True)
        return out

    all_rows = [r for k in want for r in done[k]["rows"]]
    all_spread = [r for k in want for r in done[k]["spread"]]
    cfgs = [replace(base, N_train=n1, nonseparability_alpha=0.0),
            replace(base, N_train=n4, nonseparability_alpha=0.0)]
    rc.write_rows_csv(out / "excited_support.csv", all_rows, COLUMNS)
    rc.write_rows_csv(out / "complement_spread.csv", all_spread, SPREAD_COLUMNS)

    def sel(rows, **kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    summary: dict = {"seeds": seeds, "restarts": restarts, "restarts_4x": restarts_4x,
                     "n_train_primary": n1, "n_train_4x": n4, "per_cell": {}}
    for sup in SUPPORTS:
        for meth in ("NFPA", "CW-PCA"):
            sub = sel(all_rows, support=sup, method=meth, n_train=n1)
            spr = [r["complement_spread_deg"] for r in sel(all_spread, support=sup, method=meth, n_train=n1)
                   if r["complement_spread_deg"] is not None]
            summary["per_cell"][f"{sup}|{meth}"] = {
                "angle_to_TS_max_mean": float(np.mean([r["angle_to_TS_max"] for r in sub])),
                "angle_to_TS_max_max": float(np.max([r["angle_to_TS_max"] for r in sub])),
                "angle_to_TS_min_mean": float(np.mean([r["angle_to_TS_min"] for r in sub])),
                "angle_to_TS_median_mean": float(np.mean([r["angle_to_TS_median"] for r in sub])),
                "angle_to_planted6_max_mean": float(np.mean([r["angle_to_planted6_max"] for r in sub])),
                "ts_data_var_min_mean": float(np.mean([r["ts_data_var_min"] for r in sub])),
                "ts_data_var_max_mean": float(np.mean([r["ts_data_var_max"] for r in sub])),
                "noise_floor_var": float(sub[0]["noise_floor_var"]),
                "complement_spread_deg_mean": float(np.mean(spr)) if spr else None,
                "complement_spread_deg_min": float(np.min(spr)) if spr else None,
                "complement_spread_deg_max": float(np.max(spr)) if spr else None,
                "dim_TS": int(sub[0]["dim_TS"]), "dim_learned": int(sub[0]["dim_learned"]),
            }

    # --- P-c: held-out chi2 penalty of training on the restricted support ---
    def best_chi2(sup, meth, n_train, seed):
        sub = sel(all_rows, support=sup, method=meth, n_train=n_train, seed=seed)
        # the deployed model is the restart with the lowest training loss;
        # CW-PCA has a single closed-form solution.
        if meth == "NFPA":
            sub = sorted(sub, key=lambda r: r["train_loss"])
        return sub[0]["held_out_chi2"]

    pc = {}
    for meth in ("NFPA", "CW-PCA"):
        for n_train, label in ((n1, "n_train_1x"), (n4, "n_train_4x")):
            gaps = [best_chi2("S_restricted", meth, n_train, s) - best_chi2("S_full", meth, n_train, s)
                    for s in seeds]
            pc[f"{meth}|{label}"] = _bootstrap_gap(gaps)
        g1 = pc[f"{meth}|n_train_1x"]["mean_difference"]
        g4 = pc[f"{meth}|n_train_4x"]["mean_difference"]
        pc[f"{meth}|shrinkage_fraction"] = None if g1 == 0 else float((g1 - g4) / abs(g1))
    summary["p_c_held_out_chi2_gap"] = pc

    summary["predictions"] = {
        "P_a_angle_to_TS_lt_5deg": {
            k: {"max": v["angle_to_TS_max_max"], "ok": bool(v["angle_to_TS_max_max"] < 5.0)}
            for k, v in summary["per_cell"].items()},
        "P_b_complement_spread": {
            "S_restricted_NFPA_gt_30deg": {
                "mean": summary["per_cell"]["S_restricted|NFPA"]["complement_spread_deg_mean"],
                "ok": bool((summary["per_cell"]["S_restricted|NFPA"]["complement_spread_deg_mean"] or 0.0) > 30.0)},
            "S_full_NFPA_lt_5deg": {
                "mean": summary["per_cell"]["S_full|NFPA"]["complement_spread_deg_mean"],
                "ok": bool((summary["per_cell"]["S_full|NFPA"]["complement_spread_deg_mean"]
                            if summary["per_cell"]["S_full|NFPA"]["complement_spread_deg_mean"] is not None
                            else 1e9) < 5.0)},
        },
        "P_c_gap_positive_and_not_reduced_by_data": {
            meth: {
                "gap_1x_mean": pc[f"{meth}|n_train_1x"]["mean_difference"],
                "gap_1x_ci95": [pc[f"{meth}|n_train_1x"]["ci95_low"], pc[f"{meth}|n_train_1x"]["ci95_high"]],
                "excludes_zero": pc[f"{meth}|n_train_1x"]["excludes_zero"],
                "gap_4x_mean": pc[f"{meth}|n_train_4x"]["mean_difference"],
                "shrinkage_fraction": pc[f"{meth}|shrinkage_fraction"],
                "ok": bool(pc[f"{meth}|n_train_1x"]["excludes_zero"]
                           and pc[f"{meth}|n_train_1x"]["mean_difference"] > 0
                           and (pc[f"{meth}|shrinkage_fraction"] is not None
                                and pc[f"{meth}|shrinkage_fraction"] <= 0.10)),
            } for meth in ("NFPA", "CW-PCA")},
    }
    rc.write_json(out / "summary.json", common.as_jsonable(summary))
    common.write_manifest(out, experiment_id=EXPERIMENT_ID, cfgs=cfgs, seeds=seeds,
                          extra={"supports": {k: list(v) for k, v in SUPPORTS.items()},
                                 "held_at_mean": HELD_AT_MEAN,
                                 "restarts": restarts, "restarts_4x": restarts_4x,
                                 "walltime_s": time.time() - t_start,
                                 "host": platform.node(), "platform": platform.platform()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--restarts-4x", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=str, default="")
    ap.add_argument("--no-aggregate", action="store_true")
    ap.add_argument("--budget-s", type=float, default=None)
    a = ap.parse_args(argv)
    print(f"wrote {run(a.seeds, smoke=a.smoke, restarts=a.restarts, restarts_4x=a.restarts_4x, shard=a.shard, aggregate=not a.no_aggregate, budget_s=a.budget_s)}")


if __name__ == "__main__":
    main()
