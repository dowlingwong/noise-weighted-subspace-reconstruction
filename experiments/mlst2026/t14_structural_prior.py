"""T14 - structural-prior tradeoff: finite-sample class gain vs approximation cost.

Protocol: MS ``plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md`` T14 and
``plan/mlst_protocol_2026-09-10.json`` key ``t14``.

The contrast is NFPA (the separable Kronecker class, ``k_c x k_t``) against
CW-PCA (the unrestricted comparator at matched rank ``k_c * k_t``, fitted by
**thin SVD** in whitened coordinates -- ``methods.fit_cw_pca``).  The prediction
is a crossover in ``N``: the restricted class should win at small ``N`` where
its lower variance dominates, and lose at large ``N`` where its approximation
error (nonzero for ``alpha > 0``) dominates.

Budget, derived rather than assumed
-----------------------------------
The protocol JSON carries ``fits_per_cell: 40``.  That number is *not* trusted
here: the runner derives the count from the actual seeds, methods and restarts
it will execute, and refuses to start if the derived total exceeds the declared
cap.  With the protocol's ``nfpa_restarts = 3`` and 10 seeds a cell is
``10 x (3 NFPA restarts + 1 CW-PCA) = 40`` fits, which reproduces the declared
40 -- but note the repository's own ``PRIMARY.n_restarts`` is **10**, which
would make a cell 110 fits.  The runner therefore overrides ``n_restarts`` to
the protocol's 3 explicitly, from the common configuration, and records both.

Declared before any confirmation is run
---------------------------------------
*Crossover statistic.*  For a cell, the paired per-seed difference
``d = chi2_test(NFPA) - chi2_test(CW-PCA)`` on held-out data (both methods see
the same data within a seed).  ``d < 0`` means the structural prior helps.
The cell statistic is the seed mean ``dbar(N)`` with a paired bootstrap
interval over seeds.

*Crossover.*  Along an ``(alpha, SNR, Sigma)`` line in increasing ``N``, a
crossover is an adjacent pair ``(N_i, N_{i+1})`` with ``dbar(N_i) < 0 <
dbar(N_{i+1})`` -- the predicted direction only.  A sign change in the other
direction is recorded as a *reverse crossing* and never triggers confirmation.
A crossing counts as *resolved* only when both endpoints' 95 % paired intervals
exclude zero; otherwise it is recorded as *ambiguous* and does not trigger
confirmation.

*Multiple crossings.*  If several resolved crossings occur on one line, the one
at the **smallest** ``N_i`` is selected.  If several lines produce crossings,
the line is chosen in the declared stage order (stage 1 before stage 2) and
within a stage by ascending ``alpha`` then ascending ``N_i``.

*Boundary neighbours.*  Confirmation covers the crossover ``N`` and its two
``N``-neighbours.  At a grid boundary there is only one neighbour, so the
declared fallback takes the next-nearest point on the available side:
``N = 200 -> {200, 400, 800}``; ``N = 3200 -> {800, 1600, 3200}``; otherwise
``{N_{i-1}, N_i, N_{i+1}}``.  Three ``N`` values x two covariance arms = 6
cells, matching the declared maximum.

*No crossing.*  If stages 1-2 produce no resolved crossing in the predicted
direction, stage 3 does **not** run, the claim stays conditional, and the null
is reported as the result.  A null crossover is a reportable outcome, not a
reason to widen the search.

*Seeds.*  Exploration 0-9, confirmation 10-19, smoke 900-909 -- pairwise
disjoint, checked at start-up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash  # noqa: E402
from experiments.synthetic.nfpa_confirmatory.generator import make_dataset  # noqa: E402
from experiments.synthetic.nfpa_confirmatory.metrics import normalized_chi2  # noqa: E402
from experiments.synthetic.nfpa_confirmatory.covariance import _sqrt_inv  # noqa: E402
from experiments.synthetic.nfpa_confirmatory.methods import (  # noqa: E402
    MethodFit, _whiten_batch, als_fit, fit_cw_pca, recon_kron, reconstruct,
)
from experiments.mlst2026.provenance import RunRecorder, write_json  # noqa: E402

# --- frozen protocol constants ------------------------------------------------
N_GRID = (200, 400, 800, 1600, 3200)
ALPHA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)
SNR_GRID = (3, 10)
COV_ARMS = ("oracle", "estimated-512")
NFPA_RESTARTS = 3          # protocol value; PRIMARY.n_restarts is 10
N_CALIBRATION = 512
SEEDS_EXPLORATION = tuple(range(10))
SEEDS_CONFIRMATION = tuple(range(10, 20))
SEEDS_SMOKE = tuple(range(900, 910))
FIT_CAP_HARD = 1500
FIT_CAP_PLANNED = 1240
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260914

STAGES = {
    1: {"kind": "exploration", "alpha": (0.0, 0.4, 0.8), "snr": (10,),
        "cov": ("oracle",), "N": N_GRID, "seeds": SEEDS_EXPLORATION},
    2: {"kind": "exploration", "alpha": (0.0, 0.8), "snr": (3,),
        "cov": ("oracle",), "N": N_GRID, "seeds": SEEDS_EXPLORATION},
}


def cell_id(N, alpha, snr, cov) -> str:
    return f"N{N}-a{alpha}-snr{snr}-{cov}"


def fits_per_cell(n_seeds: int, n_restarts: int) -> int:
    """Derived, never taken from the config: seeds x (NFPA restarts + comparator)."""
    return int(n_seeds) * (int(n_restarts) + 1)


def stage_cells(stage: int) -> list[tuple]:
    s = STAGES[stage]
    return [(N, a, snr, cov) for a in s["alpha"] for snr in s["snr"]
            for cov in s["cov"] for N in s["N"]]


def derived_budget() -> dict:
    per = fits_per_cell(len(SEEDS_EXPLORATION), NFPA_RESTARTS)
    s1, s2 = len(stage_cells(1)), len(stage_cells(2))
    s3 = 6
    total = (s1 + s2 + s3) * per
    return {"fits_per_cell_derived": per,
            "fits_per_cell_in_config": 40,
            "repository_PRIMARY_n_restarts": PRIMARY.n_restarts,
            "protocol_nfpa_restarts": NFPA_RESTARTS,
            "stage1_cells": s1, "stage2_cells": s2, "stage3_cells_max": s3,
            "stage1_fits": s1 * per, "stage2_fits": s2 * per,
            "stage3_fits_max": s3 * per, "total_planned": total,
            "planned_cap": FIT_CAP_PLANNED, "hard_cap": FIT_CAP_HARD,
            "within_planned_cap": total <= FIT_CAP_PLANNED,
            "within_hard_cap": total <= FIT_CAP_HARD}


# --- covariance arms ----------------------------------------------------------
def estimated_covariances(ds, n_calibration: int, seed: int) -> dict:
    """Separable (flip-flop) covariance estimate from noise-only traces.

    The scale is fixed from the calibration traces themselves -- ``Sigma_c`` is
    identified by ``trace/C = 1`` and ``Sigma_t`` carries the total variance -- so
    no part of the truth is used.
    """
    from experiments.synthetic.nfpa_confirmatory.generator import generate_noise
    rng = np.random.default_rng(BOOTSTRAP_SEED + 7919 * seed)
    cfg = ds.cfg
    Z = generate_noise(rng, int(n_calibration), cfg, ds.covariances)
    n = Z.shape[0]
    Sc = np.einsum("nct,ndt->cd", Z, Z) / (n * cfg.T)
    St = np.einsum("nct,ncs->ts", Z, Z) / (n * cfg.C)
    Sc = Sc / (np.trace(Sc) / cfg.C)
    St = St * (float(np.mean(Z ** 2)) / (np.trace(St) / cfg.T))
    Sc += 1e-10 * np.eye(cfg.C)
    St += 1e-10 * np.eye(cfg.T)
    return {"Sigma_c": Sc, "Sigma_t": St}


def apply_cov_arm(ds, arm: str, seed: int):
    if arm == "oracle":
        return ds, {"arm": "oracle", "n_calibration": None}
    if arm == "estimated-512":
        import copy
        ds2 = copy.copy(ds)
        ds2.covariances = estimated_covariances(ds, N_CALIBRATION, seed)
        ds2.train_mean = ds.train_mean
        return ds2, {"arm": arm, "n_calibration": N_CALIBRATION}
    raise ValueError(arm)


# --- fast evaluation path -----------------------------------------------------
# `covariance.whiten_separable` contracts three operands with a single
# `np.einsum` and no `optimize=`, which falls into the naive O(n C^2 T^2) path:
# 8.4e9 elementary operations for one (2000, 8, 256) batch, about 24 s. It is
# called four times per evaluation. That function is on the archival path of
# earlier runs and is deliberately NOT modified here; this module does the same
# algebra as two BLAS matmuls instead, and `tests/test_mlst_t14_structural.py`
# pins the two to agree to machine precision.


def fast_whiten(X: np.ndarray, Wc: np.ndarray, Wt: np.ndarray) -> np.ndarray:
    """``Z = Wc X Wt`` for ``X`` of shape ``(n, C, T)``, BLAS-backed."""
    return np.matmul(np.matmul(Wc, X), Wt)


def whitened_chi2(Z: np.ndarray, Zhat: np.ndarray) -> float:
    """Normalised whitened residual, computed directly in whitened coordinates.

    The archival route reconstructs into measurement space and re-whitens, which
    re-subtracts ``train_mean`` and leaves a spurious ``W mu`` term in the
    residual. Evaluating in whitened coordinates avoids the round trip entirely
    and is the quantity the protocol defines. The test module checks that the
    two agree to within the size of that term on this configuration.
    """
    return float(np.mean((Z - Zhat) ** 2))


def cw_pca_projector(Z_train: np.ndarray, rank: int) -> np.ndarray:
    """Unrestricted comparator: thin SVD in whitened coordinates."""
    F = Z_train.reshape(Z_train.shape[0], -1, order="F")
    _, _, vh = np.linalg.svd(F, full_matrices=False)
    return vh[:rank].T


# --- one seed of one cell -----------------------------------------------------
def run_seed(N: int, alpha: float, snr: float, cov: str, seed: int) -> dict:
    cfg = replace(PRIMARY, N_train=int(N), nonseparability_alpha=float(alpha),
                  target_snr=float(snr), n_restarts=NFPA_RESTARTS)
    ds = make_dataset(seed, "colored", cfg)
    ds, cov_label = apply_cov_arm(ds, cov, seed)

    Wc = _sqrt_inv(ds.covariances["Sigma_c"])
    Wt = _sqrt_inv(ds.covariances["Sigma_t"])
    Z = fast_whiten(ds.X_train - ds.train_mean, Wc, Wt)
    Zte = fast_whiten(ds.X_test - ds.train_mean, Wc, Wt)

    t0 = time.time()
    out = als_fit(Z, cfg.k_c, cfg.k_t, n_iter=cfg.n_als_iter,
                  n_restarts=NFPA_RESTARTS, seed=seed,
                  tol=cfg.als_convergence_tol, collect_restarts=True)
    t_nfpa = time.time() - t0

    t0 = time.time()
    rank = cfg.k_c * cfg.k_t
    P = cw_pca_projector(Z, rank)
    t_cw = time.time() - t0

    chi2 = {"NFPA": whitened_chi2(Zte, recon_kron(Zte, out["U_c"], out["U_t"]))}
    Fte = Zte.reshape(Zte.shape[0], -1, order="F")
    chi2["CW-PCA"] = whitened_chi2(Fte, Fte @ (P @ P.T))

    # optimisation diagnostics: restart spread and excess over the best restart
    losses = np.array([r["loss"] for r in out["restarts"]], dtype=float)
    restart_chi2 = np.array([
        whitened_chi2(Zte, recon_kron(Zte, r["U_c"], r["U_t"]))
        for r in out["restarts"]])

    return {
        "cell": cell_id(N, alpha, snr, cov), "N": int(N), "alpha": float(alpha),
        "snr": float(snr), "cov_arm": cov, "seed": int(seed), **cov_label,
        "chi2_nfpa": chi2["NFPA"], "chi2_cw_pca": chi2["CW-PCA"],
        "d_nfpa_minus_cwpca": chi2["NFPA"] - chi2["CW-PCA"],
        "n_restarts": int(NFPA_RESTARTS),
        "restart_train_loss_min": float(losses.min()),
        "restart_train_loss_max": float(losses.max()),
        "restart_train_loss_spread": float(losses.max() - losses.min()),
        "excess_objective_over_best_restart": float(out["loss"] - losses.min()),
        "restart_chi2_min": float(restart_chi2.min()),
        "restart_chi2_max": float(restart_chi2.max()),
        "restart_chi2_sd": float(restart_chi2.std(ddof=1)) if restart_chi2.size > 1 else 0.0,
        "selected_restart": int(out["restart"]),
        "converged": bool(out["converged"]),
        "n_converged_restarts": int(sum(r["converged"] for r in out["restarts"])),
        "runtime_nfpa_s": t_nfpa, "runtime_cwpca_s": t_cw,
        "fits_used": NFPA_RESTARTS + 1,
        "config_hash": config_hash(cfg),
    }


def _task(args):
    return run_seed(*args)


def run_tasks(tasks, workers: int) -> list[dict]:
    """Execute (N, alpha, snr, cov, seed) tasks, optionally in parallel.

    Every task is fully determined by its seed, so the results do not depend on
    the number of workers; parallelism is a scheduling choice only. Each worker
    is pinned to one BLAS thread, because the matrices here are small enough
    that thread contention costs more than it saves.
    """
    if workers <= 1:
        return [run_seed(*t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_task, tasks, chunksize=1))


def paired_interval(d: np.ndarray, seed: int = BOOTSTRAP_SEED,
                    reps: int = BOOTSTRAP_REPLICATES) -> dict:
    """Paired bootstrap over seeds of the mean of ``d``."""
    rng = np.random.default_rng(seed)
    n = d.size
    means = d[rng.integers(0, n, size=(reps, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_seeds": int(n)}


def summarise_cells(rows: list[dict]) -> dict:
    out = {}
    for cid in sorted({r["cell"] for r in rows}):
        sub = sorted([r for r in rows if r["cell"] == cid], key=lambda r: r["seed"])
        d = np.array([r["d_nfpa_minus_cwpca"] for r in sub])
        out[cid] = {
            "N": sub[0]["N"], "alpha": sub[0]["alpha"], "snr": sub[0]["snr"],
            "cov_arm": sub[0]["cov_arm"], "n_seeds": len(sub),
            "d_bar": float(d.mean()), **paired_interval(d),
            "chi2_nfpa_mean": float(np.mean([r["chi2_nfpa"] for r in sub])),
            "chi2_cw_pca_mean": float(np.mean([r["chi2_cw_pca"] for r in sub])),
            "restart_chi2_sd_mean": float(np.mean([r["restart_chi2_sd"] for r in sub])),
            "excess_objective_mean": float(np.mean([r["excess_objective_over_best_restart"] for r in sub])),
            "converged_fraction": float(np.mean([r["n_converged_restarts"] / r["n_restarts"] for r in sub])),
            "runtime_total_s": float(np.sum([r["runtime_nfpa_s"] + r["runtime_cwpca_s"] for r in sub])),
        }
    return out


def find_crossovers(summary: dict) -> dict:
    """Apply the declared crossover rules. Returns every crossing found."""
    lines = {}
    for cid, v in summary.items():
        lines.setdefault((v["alpha"], v["snr"], v["cov_arm"]), []).append((v["N"], cid, v))
    found, reverse, ambiguous = [], [], []
    for key, pts in sorted(lines.items()):
        pts.sort(key=lambda p: p[0])
        for i in range(len(pts) - 1):
            (N0, c0, v0), (N1, c1, v1) = pts[i], pts[i + 1]
            if v0["d_bar"] < 0 < v1["d_bar"]:
                rec = {"alpha": key[0], "snr": key[1], "cov_arm": key[2],
                       "N_lower": N0, "N_upper": N1, "cell_lower": c0,
                       "cell_upper": c1, "d_lower": v0["d_bar"], "d_upper": v1["d_bar"],
                       "lower_resolved": v0["excludes_zero"],
                       "upper_resolved": v1["excludes_zero"]}
                (found if (v0["excludes_zero"] and v1["excludes_zero"])
                 else ambiguous).append(rec)
            elif v0["d_bar"] > 0 > v1["d_bar"]:
                reverse.append({"alpha": key[0], "snr": key[1], "cov_arm": key[2],
                                "N_lower": N0, "N_upper": N1,
                                "d_lower": v0["d_bar"], "d_upper": v1["d_bar"]})
    selected = None
    if found:
        selected = sorted(found, key=lambda r: (r["alpha"], r["N_lower"]))[0]
    return {"resolved_crossings": found, "ambiguous_crossings": ambiguous,
            "reverse_crossings": reverse, "selected": selected,
            "triggers_confirmation": selected is not None}


def confirmation_cells(selected: dict) -> list[tuple]:
    """Declared boundary-neighbour rule -> exactly three N values x two arms."""
    N = selected["N_lower"]
    i = N_GRID.index(N)
    if i == 0:
        Ns = list(N_GRID[0:3])
    elif i == len(N_GRID) - 1:
        Ns = list(N_GRID[-3:])
    else:
        Ns = list(N_GRID[i - 1:i + 2])
    return [(n, selected["alpha"], selected["snr"], cov)
            for n in Ns for cov in COV_ARMS]


# --- CLI ----------------------------------------------------------------------
def _check_seed_disjointness() -> None:
    a, b, c = set(SEEDS_EXPLORATION), set(SEEDS_CONFIRMATION), set(SEEDS_SMOKE)
    if a & b or a & c or b & c:
        raise SystemExit("smoke, exploration and confirmation seeds must be disjoint")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["benchmark", "1", "2", "confirm", "budget"])
    ap.add_argument("--out")
    ap.add_argument("--exploration", nargs="*", default=[],
                    help="stage-1/2 result JSONs, for --stage confirm")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes; results are seed-determined "
                         "and do not depend on this value")
    args = ap.parse_args(argv)
    _check_seed_disjointness()
    budget = derived_budget()

    if args.stage == "budget":
        print(json.dumps(budget, indent=2))
        return 0

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory {out_dir}")
    rec = RunRecorder(run_id=f"T14_structural_prior_stage_{args.stage}",
                      protocol_refs={"programme": "plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md#T14",
                                     "config": "plan/mlst_protocol_2026-09-10.json#t14"})
    rec.add_input("runner_source", Path(__file__).resolve())
    rec.add_seed("exploration_seeds", list(SEEDS_EXPLORATION))
    rec.add_seed("confirmation_seeds", list(SEEDS_CONFIRMATION))
    rec.add_seed("smoke_seeds", list(SEEDS_SMOKE))
    rec.add_seed("bootstrap_seed", BOOTSTRAP_SEED)
    payload: dict = {"stage": args.stage, "smoke": bool(args.smoke),
                     "budget": budget, "n_grid": list(N_GRID)}

    if args.stage == "benchmark":
        # one small and one large representative cell, as required before scaling
        rows, bench = [], {}
        for N in (N_GRID[0], N_GRID[-1]):
            t0 = time.time()
            r = run_seed(N, 0.4, 10, "oracle", SEEDS_SMOKE[0])
            rows.append(r)
            bench[f"N={N}"] = {
                "wall_seconds_one_seed": round(time.time() - t0, 3),
                "fits": r["fits_used"],
                "projected_cell_seconds": round((time.time() - t0) * len(SEEDS_EXPLORATION), 1),
            }
            print(f"  [benchmark N={N}] {bench[f'N={N}']['wall_seconds_one_seed']:.2f} s/seed "
                  f"-> {bench[f'N={N}']['projected_cell_seconds']:.0f} s/cell "
                  f"(chi2 NFPA {r['chi2_nfpa']:.5f} vs CW-PCA {r['chi2_cw_pca']:.5f}, "
                  f"d={r['d_nfpa_minus_cwpca']:+.5f})")
        mean_cell = np.mean([b["projected_cell_seconds"] for b in bench.values()])
        bench["projected_stage1_plus_2_minutes"] = round(
            mean_cell * (budget["stage1_cells"] + budget["stage2_cells"]) / 60, 1)
        payload["benchmark"] = bench
        payload["rows"] = rows

    elif args.stage in ("1", "2"):
        stage = int(args.stage)
        seeds = SEEDS_SMOKE[:2] if args.smoke else STAGES[stage]["seeds"]
        cells = stage_cells(stage)
        tasks = [(N, a, snr, cov, seed) for (N, a, snr, cov) in cells for seed in seeds]
        planned = len(tasks) * (NFPA_RESTARTS + 1)
        if planned > FIT_CAP_HARD:
            raise SystemExit(f"planned {planned} fits exceeds hard cap {FIT_CAP_HARD}")
        t0 = time.time()
        rows = run_tasks(tasks, args.workers)
        used = sum(r["fits_used"] for r in rows)
        if used > FIT_CAP_HARD:
            raise SystemExit(f"hard fit cap {FIT_CAP_HARD} exceeded ({used})")
        summary = summarise_cells(rows)
        for (N, a, snr, cov) in cells:
            v = summary[cell_id(N, a, snr, cov)]
            print(f"  [stage {stage} {cell_id(N, a, snr, cov):<28}] "
                  f"d={v['d_bar']:+.6f} CI[{v['ci95_lo']:+.5f},{v['ci95_hi']:+.5f}] "
                  f"{'*' if v['excludes_zero'] else ' '} "
                  f"chi2 NFPA {v['chi2_nfpa_mean']:.5f} CW {v['chi2_cw_pca_mean']:.5f}")
        print(f"  stage {stage}: {used} fits in {time.time() - t0:.0f} s")
        payload["rows"] = rows
        payload["cells"] = summary
        payload["fits_used"] = used
        payload["crossovers_this_stage"] = find_crossovers(payload["cells"])

    elif args.stage == "confirm":
        merged: dict = {}
        for p in args.exploration:
            rec.add_input("exploration", Path(p))
            merged.update(json.loads(Path(p).read_text())["cells"])
        cross = find_crossovers(merged)
        payload["exploration_crossovers"] = cross
        if not cross["triggers_confirmation"]:
            payload["confirmation"] = {
                "run": False,
                "reason": "no resolved crossing in the predicted direction was "
                          "found in stages 1-2; the declared rule does not run "
                          "the confirmation stage and the claim stays conditional",
                "resolved_crossings": cross["resolved_crossings"],
                "ambiguous_crossings": cross["ambiguous_crossings"],
                "reverse_crossings": cross["reverse_crossings"],
                "fits_used": 0}
            print("  [confirm] no resolved crossing -> confirmation NOT run "
                  "(null crossover reported)")
        else:
            cells = confirmation_cells(cross["selected"])
            tasks = [(N, a, snr, cov, seed) for (N, a, snr, cov) in cells
                     for seed in SEEDS_CONFIRMATION]
            rows = run_tasks(tasks, args.workers)
            used = sum(r["fits_used"] for r in rows)
            if used > FIT_CAP_HARD:
                raise SystemExit(f"hard fit cap {FIT_CAP_HARD} exceeded ({used})")
            summary = summarise_cells(rows)
            for (N, a, snr, cov) in cells:
                v = summary[cell_id(N, a, snr, cov)]
                print(f"  [confirm {cell_id(N, a, snr, cov):<28}] d={v['d_bar']:+.6f} "
                      f"CI[{v['ci95_lo']:+.5f},{v['ci95_hi']:+.5f}] {'*' if v['excludes_zero'] else ' '}")
            payload["confirmation"] = {
                "run": True, "selected_crossing": cross["selected"],
                "cells_confirmed": [cell_id(*c) for c in cells],
                "seeds": list(SEEDS_CONFIRMATION), "fits_used": used,
                "cells": summary,
                "crossover_reproduced": find_crossovers(summary)}
            payload["rows"] = rows

    out_dir.mkdir(parents=True, exist_ok=True)
    payload["provenance"] = rec.finalize()
    digest = write_json(out_dir / f"t14_stage_{args.stage}.json", payload)
    (out_dir / "RESULTS_SHA256").write_text(f"{digest}  t14_stage_{args.stage}.json\n")
    print(f"[T14/{args.stage}] wrote {out_dir}/t14_stage_{args.stage}.json sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
