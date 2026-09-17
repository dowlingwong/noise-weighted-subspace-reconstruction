"""R5: S_full theorem-domain reachability.

For each calibration size ``m`` the full sample covariance ``S_full`` is formed
by the bounded-memory accumulator; the covariance error ``E``, its operator norm
``eta`` and the domain class follow the archived T19-S definitions.  In-domain
realisations get the 44 eta^2 remainder-bound check and a first-order agreement
ratio; out-of-domain realisations are **kept and reported**, never discarded or
redrawn.

Nothing here re-scores the archived T19 prediction; the package answers only the
reachability question and leaves "not reachable" as a valid result.
"""
from __future__ import annotations

import numpy as np

from experiments.mlst2026.successor_20260913.t19s import observables as O
from experiments.mlst2026.successor_20260913.t19s import synthetic as S

from . import covariance as cov

ETA_IN_DOMAIN = O.ETA_IN_DOMAIN
ETA_STRESS = O.ETA_STRESS


def _sigma_cholesky(g):
    return np.linalg.cholesky(g.Sigma)


def reachability_records(seed: int, g, *, m_grid, n_train: int, chunk: int, n_seeds: int):
    """One record per (seed, m).  A fresh stream per seed; nested arms within a seed."""
    L = _sigma_cholesky(g)
    out = []
    for s in range(int(n_seeds)):
        parent_seed = int(seed)
        stream_seed = parent_seed * 10_000 + s
        rng = np.random.default_rng(stream_seed)
        # one stream per seed, consumed in nested m order: the covariance for each
        # larger m is a proven prefix-extension of the previous one
        covs = cov.nested_streaming_covariances(lambda n: rng.standard_normal((n, g.d)) @ L.T,
                                                m_grid, g.d, chunk)
        for m in sorted(covs):
            S_full = covs[m]
            E = S.E_from_sigma_hat(g, S_full)
            eps_op = float(np.linalg.norm(E, 2))
            X, _ = S.draw_training(g, rng, int(n_train))     # independent training draw
            A_hat, lam, Q = S.sample_operator(g, X)
            P_hat, _, _ = S.fit_subspace(X, S_full, g.k, floor=False)
            eta = O.eta_of(eps_op, lam, g.k)
            mc = O.covariance_model_subspace(A_hat, E, g.k)
            rec = {"seed": int(s), "parent_seed": parent_seed, "stream_seed": stream_seed,
                   "m": int(m), "eta": float(eta), "eps_op": eps_op,
                   "lambda_min_IplusE": (None if not mc["valid"] else float(mc["lambda_min_IplusE"])),
                   "in_domain": bool(mc["valid"] and eta <= ETA_IN_DOMAIN),
                   "domain_class": O.domain_class(eta, None if not mc["valid"] else mc["lambda_min_IplusE"]),
                   "estimator_arm": "S_full"}
            if rec["in_domain"]:
                V = S.whitened_basis(g, P_hat)
                Eq = Q.T @ E @ Q
                X1, _ = O.first_order_graph(lam, Eq, g.k, "covariance")
                obs = O.observables(V, Q, lam, g.k, X1)
                rec["bound"] = O.bound_record(obs["projector_matrix"]["residual_op"], eta, applies=True)
                rec["first_order_projector_ratio_op"] = obs["projector_matrix"]["ratio_op"]
                rec["first_order_graph_residual_fro"] = obs["graph"]["residual_fro"]
            out.append(rec)
    return out


def summarise(records, m_grid) -> dict:
    rows = {}
    for m in sorted(int(x) for x in m_grid):
        rs = [r for r in records if r["m"] == m]
        etas = [r["eta"] for r in rs]
        in_dom = [r for r in rs if r["in_domain"]]
        viol = [r["bound"]["violation"] for r in in_dom if "bound" in r]
        ratios = [r["first_order_projector_ratio_op"] for r in in_dom if r.get("first_order_projector_ratio_op") is not None]
        rows[str(m)] = {
            "n": len(rs), "n_in_domain": len(in_dom),
            "fraction_reaching_domain": (len(in_dom) / len(rs)) if rs else None,
            "eta_median": float(np.median(etas)) if etas else None,
            "eta_min": float(np.min(etas)) if etas else None, "eta_max": float(np.max(etas)) if etas else None,
            "eta_distribution": sorted(float(x) for x in etas),
            "conditional_in_domain_violations": int(sum(v > 0 for v in viol)),
            "max_violation_over_eta2": (float(max(v / (e ** 2) for v, e in zip(viol, [r["eta"] for r in in_dom]) if e > 0))
                                        if in_dom else None),
            "first_order_ratio_median": float(np.median(ratios)) if ratios else None,
            "first_order_ratio_min": float(np.min(ratios)) if ratios else None,
            "first_order_ratio_max": float(np.max(ratios)) if ratios else None,
        }
    return {"by_m": rows,
            "theorem_domain_reachable": any(r["n_in_domain"] > 0 for r in rows.values()),
            "never_discarded": "every drawn realisation is retained, including out-of-domain ones"}


def fit_eta_scaling(summary: dict) -> dict:
    """Fit log(median eta) vs log(m); the archived extrapolation was eta ~ m^-1/2."""
    pts = [(float(m), r["eta_median"]) for m, r in summary["by_m"].items() if r["eta_median"]]
    if len(pts) < 2:
        return {"slope": None, "intercept": None, "n_points": len(pts)}
    x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
    b, a = np.polyfit(x, y, 1)
    return {"slope": float(b), "intercept": float(a), "n_points": len(pts),
            "expected_slope_from_archived_extrapolation": -0.5}
