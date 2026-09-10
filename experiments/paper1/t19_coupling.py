"""T19 — subspace coupling, spectral separation and conditioning (Part II protocol).

Which covariance perturbations rotate the learned subspace, which destabilise
inversion, and does the first-order displacement predict the former?

Setting: generator G1 (:mod:`experiments.paper1.g1`), rank-2 covariance-weighted PCA
fitted on 512 training traces whitened by an *assumed* covariance ``Sigma_hat``.
The training draw is held fixed per seed across every covariance arm (paired
design).  In the eigenbasis ``Q`` of the population whitened operator
``A = Sigma^{-1/2} R_x Sigma^{-1/2}`` (eigenvalues ``lam``, retained set ``i < k``),
with ``E = Sigma^{-1/2}(Sigma_hat - Sigma)Sigma^{-1/2}`` and ``D`` the training-sample
perturbation of ``A``, the registered first-order prediction is

    T_ji ~= (D_ji - lam_j E_ji) / (lam_i - lam_j),   i < k <= j,

where ``T = C_disc C_ret^{-1}`` is the displacement of the fitted subspace expressed
in ``Q``.  The covariance part alone is measured as the difference between the
perturbed fit and the paired oracle fit on the same training draw, and predicted
by ``-lam_j E_ji / (lam_i - lam_j)``.

Families (protocol II.T19): P diagonal order-preserving; R diagonal order-reversing;
C pure coupling (off-diagonal block only); F full random symmetric; M structured
misspecification (correlation length x2, x1/2; slope -1.5, -0.5); S empirical
in-domain (m = 2d, 4d, 8d); S- empirical near-singular (m = d/2, d), floored and
reported as a separate regime; K calibration-selected shrinkage (m = d/2, d, 2d);
and the training-sample arm (oracle covariance, N = 128, 512, 2048).

Deviations from the Part II text, decided after a one-seed smoke run at the first
design and BEFORE the full run (recorded in the manifest and the report):

1. Families P/R/C/F are built in the eigenbasis of the *sample* operator A+D that
   the fit diagonalises (primary); the population-basis versions are kept as
   secondary arms (Ppop/Cpop/Fpop at eps 0.1, 0.4) because they measure the D x E
   cross term, which is linear in eps and would otherwise mask the eps^2 residual.
2. The 10 Sep T15 calibration estimator could not be identified from the report
   (its eps scaling matches neither the full sample covariance nor a Toeplitz nor a
   periodogram estimate), so Part II defines its estimators explicitly: S = full
   sample covariance at m in {16d, 32d, 64d} (measured in-domain), S- = the same at
   {d/2, d} (near-singular, floored), K = calibration-selected shrinkage at
   {d/2, d, 2d}, and Sstat = the stationary (Toeplitz) estimate at {d, 8d}.

Usage::

    python -m experiments.paper1.t19_coupling --seeds 0 1 2 3 4 5 6 7 8 9

Outputs ``results/mlst_2026-09-11/t19/{rows.jsonl, summary.json, manifest.json}``.
Fit count is derived from the arm list and refused above the cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.paper1 import g1 as G

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "mlst_2026-09-11" / "t19"
K = G.RANK
FIT_CAP = 400
ARMS_EXPECTED = 38
EPS_GRID = [0.05, 0.1, 0.2, 0.4]
FLOOR_FRACTION = 1e-6
SHRINK_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------

def arm_list() -> list[dict]:
    arms: list[dict] = []
    # designed families in the eigenbasis of the SAMPLE operator A+D (primary)
    arms += [{"id": f"P-eps{e}", "family": "P", "eps": e, "basis": "sample"} for e in EPS_GRID]
    arms += [{"id": f"R-eps{e}", "family": "R", "eps": e, "basis": "sample"} for e in (0.2, 0.4, 0.8)]
    arms += [{"id": f"C-eps{e}", "family": "C", "eps": e, "basis": "sample"} for e in EPS_GRID]
    arms += [{"id": f"F-eps{e}", "family": "F", "eps": e, "basis": "sample"} for e in EPS_GRID]
    # the same families in the POPULATION eigenbasis (secondary: measures the D x E cross term)
    arms += [{"id": f"{f}pop-eps{e}", "family": f, "eps": e, "basis": "population"} for f in ("P", "C", "F") for e in (0.1, 0.4)]
    arms += [{"id": "M-corrlength-x2", "family": "M", "f_knee": G.F_KNEE / 2, "exponent": -1.0},
             {"id": "M-corrlength-x0.5", "family": "M", "f_knee": G.F_KNEE * 2, "exponent": -1.0},
             {"id": "M-slope-1.5", "family": "M", "f_knee": G.F_KNEE, "exponent": -1.5},
             {"id": "M-slope-0.5", "family": "M", "f_knee": G.F_KNEE, "exponent": -0.5}]
    # full sample covariance: in-domain sizes (measured), near-singular sizes, shrinkage
    arms += [{"id": f"S-m{m}", "family": "S", "m": m} for m in (4096, 8192, 16384)]
    arms += [{"id": f"S--m{m}", "family": "S-", "m": m} for m in (128, 256)]
    arms += [{"id": f"K-m{m}", "family": "K", "m": m} for m in (128, 256, 512)]
    # stationary (Toeplitz, lag-averaged) estimator: the structured calibration a physicist fits
    arms += [{"id": f"Sstat-m{m}", "family": "Sstat", "m": m} for m in (256, 2048)]
    arms += [{"id": f"D-N{N}", "family": "D", "N": N} for N in (128, 512, 2048)]
    ids = [a["id"] for a in arms]
    assert len(ids) == len(set(ids))
    return arms


# ----------------------------------------------------------------------------
# linear algebra helpers
# ----------------------------------------------------------------------------

def orth(V: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(V)
    return np.asarray(Q)


def sin_theta_max(U: np.ndarray, V: np.ndarray) -> float:
    """Largest principal-angle sine between the column spaces of orthonormal U, V."""
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.sqrt(max(0.0, 1.0 - s.min() ** 2)))


def displacement(V_whitened: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """T = C_disc C_ret^{-1} for a basis V expressed in the eigenbasis Q."""
    C = Q.T @ V_whitened
    return C[K:, :] @ np.linalg.inv(C[:K, :])


def first_order(Dq: np.ndarray, Eq: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predicted total and covariance-only displacements (rows j >= K, cols i < K)."""
    d = len(lam)
    denom = lam[:K][None, :] - lam[K:][:, None]          # (d-K, K): lam_i - lam_j
    total = (Dq[K:, :K] - lam[K:][:, None] * Eq[K:, :K]) / denom
    cov_only = (-lam[K:][:, None] * Eq[K:, :K]) / denom
    return total, cov_only


def blocks(Mq: np.ndarray) -> dict:
    return {"11": float(np.linalg.norm(Mq[:K, :K], 2)), "21": float(np.linalg.norm(Mq[K:, :K], 2)),
            "22": float(np.linalg.norm(Mq[K:, K:], 2))}


def perturbed_spectrum(A: np.ndarray, Qref: np.ndarray, E: np.ndarray) -> dict:
    """Spectrum of A(I+E)^{-1} via the similar symmetric operator; retained-set check vs Qref[:, :K]."""
    IE = np.eye(G.D) + E
    vals, vecs = np.linalg.eigh(0.5 * (IE + IE.T))
    lam_min = float(vals.min())
    if lam_min <= 0:
        return {"lambda_min_IplusE": lam_min, "separation": None, "retained_set_preserved": None,
                "retained_angle_sin": None}
    S_isqrt = (vecs / np.sqrt(vals)[None, :]) @ vecs.T
    S_sqrt = (vecs * np.sqrt(vals)[None, :]) @ vecs.T
    B = S_isqrt @ A @ S_isqrt
    mu, U = np.linalg.eigh(0.5 * (B + B.T))
    order = np.argsort(mu)[::-1]
    mu, U = mu[order], U[:, order]
    # invariant subspace of A(I+E)^{-1} is (I+E)^{1/2} times that of B
    V = orth(S_sqrt @ U[:, :K])
    sep = float(min(abs(mu[i] - mu[j]) for i in range(K) for j in range(K, len(mu))))
    ang = sin_theta_max(V, Qref[:, :K])
    return {"lambda_min_IplusE": lam_min, "separation": sep,
            "retained_set_preserved": bool(ang < 0.5), "retained_angle_sin": ang,
            "mu_retained": [float(x) for x in mu[:K]], "mu_next": float(mu[K])}


# ----------------------------------------------------------------------------
# covariance arms
# ----------------------------------------------------------------------------

def make_E_family(Qb: np.ndarray, family: str, eps: float, rng: np.random.Generator,
                  lam_b: np.ndarray | None = None) -> np.ndarray:
    """Perturbation E = Qb E_q Qb^T built in the given eigenbasis Qb (eigenvalues lam_b).

    Family P draws random signs and then enforces the protocol's ordering constraint:
    the retained eigenvalues lam_i/(1+e_i) must stay above every discarded one, which
    with the sample spectrum (discarded bulk at the Marchenko-Pastur edge) is not
    automatic at eps = 0.4.  Retained coordinates get -eps (raising them); a discarded
    coordinate whose +/-eps draw would overtake the retained set is flipped to +eps.
    """
    d = Qb.shape[0]
    Eq = np.zeros((d, d))
    if family == "P":
        assert lam_b is not None
        signs = rng.choice([-1.0, 1.0], size=d)
        e = eps * signs
        e[:K] = -eps
        floor_ret = min(lam_b[:K] / (1.0 + e[:K]))
        for j in range(K, d):
            if lam_b[j] / (1.0 + e[j]) >= floor_ret:
                e[j] = +eps
        assert max(lam_b[K:] / (1.0 + e[K:])) < floor_ret, "P: ordering could not be preserved"
        Eq[np.diag_indices(d)] = e
    elif family == "R":
        Eq[K - 1, K - 1] = +eps
        Eq[K, K] = -eps
    elif family == "C":
        Bblk = rng.standard_normal((d - K, K))
        Bblk *= eps / np.linalg.norm(Bblk, 2)
        Eq[K:, :K] = Bblk
        Eq[:K, K:] = Bblk.T
    elif family == "F":
        Gm = rng.standard_normal((d, d))
        Sm = 0.5 * (Gm + Gm.T)
        Eq = Sm * (eps / np.linalg.norm(Sm, 2))
    else:
        raise ValueError(family)
    return Qb @ Eq @ Qb.T


def sigma_hat_from_E(g: G.G1, E: np.ndarray) -> np.ndarray:
    S_sqrt = G.sym_sqrt(g.Sigma)
    return S_sqrt @ (np.eye(G.D) + E) @ S_sqrt


def E_from_sigma_hat(g: G.G1, Sigma_hat: np.ndarray) -> np.ndarray:
    return g.Sigma_isqrt @ (Sigma_hat - g.Sigma) @ g.Sigma_isqrt


def empirical_cov(noise: np.ndarray) -> np.ndarray:
    """Noise-only calibration: mean known to be zero, so the uncentred MLE."""
    return noise.T @ noise / noise.shape[0]


def toeplitz_cov(noise: np.ndarray) -> np.ndarray:
    """Stationary (Toeplitz) estimate: the sample covariance averaged along each lag."""
    S = empirical_cov(noise)
    d = S.shape[0]
    c = np.array([np.mean(np.diagonal(S, offset=l)) for l in range(d)])
    idx = np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
    return c[idx]


def gaussian_nll(S: np.ndarray, X: np.ndarray) -> float:
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    vals = np.clip(vals, 1e-12, None)
    Z = X @ vecs
    return float(0.5 * (np.sum(Z**2 / vals[None, :]) / X.shape[0] + np.sum(np.log(vals))))


def shrinkage_select(noise: np.ndarray) -> tuple[np.ndarray, dict]:
    half = noise.shape[0] // 2
    A_, B_ = noise[:half], noise[half:]
    S_A = empirical_cov(A_)
    target = np.trace(S_A) / G.D * np.eye(G.D)
    nll = {g_: gaussian_nll((1 - g_) * S_A + g_ * target, B_) for g_ in SHRINK_GRID}
    gamma = min(nll, key=nll.get)
    S_all = empirical_cov(noise)
    S_hat = (1 - gamma) * S_all + gamma * (np.trace(S_all) / G.D) * np.eye(G.D)
    return S_hat, {"gamma": gamma, "nll_by_gamma": {str(k): v for k, v in nll.items()}}


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------

def fit_subspace(X: np.ndarray, Sigma_hat: np.ndarray, *, floor: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rank-K CW-PCA under the assumed covariance.

    Returns the reconstruction basis in detector coordinates ``P`` (d, K), the
    encoder ``Sigma_hat^{-1/2}`` used, and the whitening diagnostics.
    """
    floor_val = FLOOR_FRACTION * np.trace(Sigma_hat) / G.D if floor else None
    Si, diag = G.sym_inv_sqrt(Sigma_hat, floor=floor_val)
    Xc = X - X.mean(axis=0, keepdims=True)
    Z = Xc @ Si                                   # whitened rows
    _, _, vh = np.linalg.svd(Z, full_matrices=False)
    W = vh[:K].T                                  # (d, K) in Sigma_hat-whitened coords
    # v2 (after the 11 Sep audit): the decoder is the exact inverse of the encoder that was
    # used, floored or not.  v1 used sqrt(Sigma_hat) of the UNFLOORED covariance when the floor
    # was active, i.e. an encoder/decoder pair from two different covariances; that suppressed
    # the floored directions and changed both the subspace and the probe coordinates.
    P = np.linalg.solve(Si, W)
    if diag["floor_applied"]:
        # the covariance the fit actually used (eigenvalues floored), for effective diagnostics
        Sigma_eff = np.linalg.inv(Si @ Si)
        diag["sigma_eff_trace_over_d"] = float(np.trace(Sigma_eff) / G.D)
        diag["decoder"] = "solve(Si, W) [v2, consistent]"
        return P, Si, diag, Sigma_eff
    diag["decoder"] = "solve(Si, W) [v2, consistent]"
    return P, Si, diag, None


def probe_scores(P: np.ndarray, Si: np.ndarray, X_tr: np.ndarray, a_tr: np.ndarray,
                 X_te: np.ndarray, a_te: np.ndarray) -> dict:
    """Linear probe from the K-dim code to the two amplitudes; R^2 and RMSE on test."""
    W = orth(Si @ P)                              # encoder basis in Sigma_hat-whitened coords
    mu = X_tr.mean(axis=0, keepdims=True)
    C_tr = (X_tr - mu) @ Si @ W
    C_te = (X_te - mu) @ Si @ W
    Xd = np.column_stack([C_tr, np.ones(len(C_tr))])
    beta, *_ = np.linalg.lstsq(Xd, a_tr, rcond=None)
    pred = np.column_stack([C_te, np.ones(len(C_te))]) @ beta
    resid = a_te - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    r2 = float(1.0 - np.sum(resid**2) / np.sum((a_te - a_te.mean(axis=0))**2))
    return {"amplitude_rmse": rmse, "probe_r2": r2}


# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------

def run_seed(g: G.G1, seed: int, arms: list[dict]) -> list[dict]:
    rows: list[dict] = []
    rng_train = np.random.default_rng(1_000 + seed)
    rng_test = np.random.default_rng(2_000 + seed)
    rng_cal = np.random.default_rng(3_000 + seed)
    rng_fam = np.random.default_rng(4_000 + seed)

    X_full, a_full = G.draw_training(g, rng_train, n=2048)          # nested draw; first 512 = the fixed training set
    X_tr, a_tr = X_full[:G.N_TRAIN], a_full[:G.N_TRAIN]
    X_te, a_te = G.draw_training(g, rng_test, n=G.N_TEST)
    cal_pool = G.draw_noise(g, rng_cal, 16384)                     # nested calibration pool (largest S arm)

    Q, lam = g.Q, g.lam
    planted = g.planted_whitened

    # oracle fit on the fixed training draw; sample operator and its eigenbasis
    P_or, Si_or, _, _ = fit_subspace(X_tr, g.Sigma, floor=False)
    V_or = orth(g.Sigma_isqrt @ P_or)
    Xc = X_tr - X_tr.mean(axis=0, keepdims=True)
    A_hat = g.Sigma_isqrt @ (Xc.T @ Xc / G.N_TRAIN) @ g.Sigma_isqrt
    A_hat = 0.5 * (A_hat + A_hat.T)
    lam_hat, Q_hat = np.linalg.eigh(A_hat)
    order = np.argsort(lam_hat)[::-1]
    lam_hat, Q_hat = lam_hat[order], Q_hat[:, order]
    Dq = Q.T @ (A_hat - g.A_pop) @ Q
    T_or_pop = displacement(V_or, Q)
    thr_sample = float((lam_hat[K - 1] - lam_hat[K]) / (lam_hat[K - 1] + lam_hat[K]))

    for arm in arms:
        t0 = time.time()
        fam = arm["family"]
        floor = False
        extra: dict = {}
        if fam in ("P", "R", "C", "F"):
            Qb, lb = (Q_hat, lam_hat) if arm["basis"] == "sample" else (Q, lam)
            E = make_E_family(Qb, fam, arm["eps"], rng_fam, lam_b=lb)
            Sigma_hat = sigma_hat_from_E(g, E)
        elif fam == "M":
            Sigma_hat = G.covariance(exponent=arm["exponent"], f_knee=arm["f_knee"])
            E = E_from_sigma_hat(g, Sigma_hat)
        elif fam in ("S", "S-"):
            Sigma_hat = empirical_cov(cal_pool[: arm["m"]])
            E = E_from_sigma_hat(g, Sigma_hat)
            floor = fam == "S-"
        elif fam == "Sstat":
            Sigma_hat = toeplitz_cov(cal_pool[: arm["m"]])
            E = E_from_sigma_hat(g, Sigma_hat)
        elif fam == "K":
            Sigma_hat, extra = shrinkage_select(cal_pool[: arm["m"]])
            E = E_from_sigma_hat(g, Sigma_hat)
        elif fam == "D":
            Sigma_hat = g.Sigma
            E = np.zeros((G.D, G.D))
        else:
            raise ValueError(fam)

        if fam == "D":
            N = arm["N"]
            X_use, a_use = X_full[:N], a_full[:N]
        else:
            X_use, a_use = X_tr, a_tr

        P, Si, wdiag, Sigma_eff = fit_subspace(X_use, Sigma_hat, floor=floor)
        V = orth(g.Sigma_isqrt @ P)
        if Sigma_eff is not None:
            # effective (floored) covariance actually used by the fit: E_eff and its spectrum
            E_eff = E_from_sigma_hat(g, Sigma_eff)
            extra = dict(extra or {})
            extra["effective"] = {"eps_op": float(np.linalg.norm(E_eff, 2)), "eps_fro": float(np.linalg.norm(E_eff, "fro")),
                                  "spectrum": perturbed_spectrum(A_hat, Q_hat, E_eff),
                                  "E_blocks_hat": blocks(Q_hat.T @ E_eff @ Q_hat)}
        Eq_pop = Q.T @ E @ Q
        Eq_hat = Q_hat.T @ E @ Q_hat
        row_pred: dict = {}
        if fam == "D":
            XcN = X_use - X_use.mean(axis=0, keepdims=True)
            A_hat_N = g.Sigma_isqrt @ (XcN.T @ XcN / N) @ g.Sigma_isqrt
            DqN = Q.T @ (A_hat_N - g.A_pop) @ Q
            T_meas = displacement(V, Q)
            pred_total, _ = first_order(DqN, Eq_pop, lam)
            row_pred = {"T_meas_pop_fro": float(np.linalg.norm(T_meas)), "T_pred_pop_fro": float(np.linalg.norm(pred_total)),
                        "resid_pop_fro": float(np.linalg.norm(T_meas - pred_total)),
                        "ratio_pop": float(np.linalg.norm(T_meas) / np.linalg.norm(pred_total)) if np.linalg.norm(pred_total) > 0 else None,
                        "D_blocks": blocks(DqN)}
            spec = perturbed_spectrum(g.A_pop, Q, E)
        else:
            # PRIMARY: covariance-only displacement in the sample eigenbasis (oracle fit has zero displacement there)
            T_cov_hat = displacement(V, Q_hat)
            _, pred_cov_hat = first_order(np.zeros_like(Eq_hat), Eq_hat, lam_hat)
            # SECONDARY: population eigenbasis, D and E separate
            T_meas_pop = displacement(V, Q)
            pred_total_pop, pred_cov_pop = first_order(Dq, Eq_pop, lam)
            T_cov_pop = T_meas_pop - T_or_pop
            n = lambda M: float(np.linalg.norm(M))
            row_pred = {
                "T_cov_hat_fro": n(T_cov_hat), "T_cov_hat_pred_fro": n(pred_cov_hat),
                "resid_cov_hat_fro": n(T_cov_hat - pred_cov_hat),
                "ratio_cov_hat": n(T_cov_hat) / n(pred_cov_hat) if n(pred_cov_hat) > 0 else None,
                "T_meas_pop_fro": n(T_meas_pop), "T_pred_pop_fro": n(pred_total_pop), "resid_pop_fro": n(T_meas_pop - pred_total_pop),
                "ratio_pop": n(T_meas_pop) / n(pred_total_pop) if n(pred_total_pop) > 0 else None,
                "T_cov_pop_fro": n(T_cov_pop), "T_cov_pop_pred_fro": n(pred_cov_pop), "resid_cov_pop_fro": n(T_cov_pop - pred_cov_pop),
                "ratio_cov_pop": n(T_cov_pop) / n(pred_cov_pop) if n(pred_cov_pop) > 0 else None,
                "D_blocks": blocks(Dq),
            }
            spec = perturbed_spectrum(A_hat, Q_hat, E)
            spec["population"] = perturbed_spectrum(g.A_pop, Q, E)
        row = {
            "seed": seed, "arm": arm["id"], "family": fam, "basis": arm.get("basis"),
            "sample_reversal_threshold": thr_sample, "lam_hat_top3": [float(x) for x in lam_hat[:3]],
            "eps_op": float(np.linalg.norm(E, 2)), "eps_fro": float(np.linalg.norm(E, "fro")),
            "E_blocks_hat": blocks(Eq_hat), "E_blocks_pop": blocks(Eq_pop),
            "sin_theta_to_oracle": sin_theta_max(V, V_or) if fam != "D" else None,
            "sin_theta_to_planted": sin_theta_max(V, planted),
            "whitening": wdiag, "spectrum": spec,
            "in_theorem_domain": bool(np.linalg.norm(E, 2) <= 0.5),
            "n_train": int(X_use.shape[0]),
            "arm_params": {k: v for k, v in arm.items() if k not in ("id", "family")},
            "extra": extra,
            "walltime_s": time.time() - t0,
        }
        row.update(row_pred)
        row.update(probe_scores(P, Si, X_use, a_use, X_te, a_te))
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    def med(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return float(np.median(v)) if v else None
    out: dict = {"n_rows": len(rows), "arms": {}}
    for arm in sorted({r["arm"] for r in rows}):
        ra = [r for r in rows if r["arm"] == arm]
        out["arms"][arm] = {
            "family": ra[0]["family"], "basis": ra[0]["basis"], "n": len(ra),
            "eps_op_median": med(ra, "eps_op"),
            "E21_hat_median": float(np.median([r["E_blocks_hat"]["21"] for r in ra])),
            "sin_theta_to_oracle_median": med(ra, "sin_theta_to_oracle"),
            "sin_theta_to_planted_median": med(ra, "sin_theta_to_planted"),
            "ratio_cov_hat_median": med(ra, "ratio_cov_hat"), "resid_cov_hat_median": med(ra, "resid_cov_hat_fro"),
            "ratio_cov_pop_median": med(ra, "ratio_cov_pop"), "ratio_pop_median": med(ra, "ratio_pop"),
            "resid_pop_median": med(ra, "resid_pop_fro"),
            "lambda_min_IplusE_median": float(np.median([r["spectrum"]["lambda_min_IplusE"] for r in ra])),
            "retained_set_preserved_frac": float(np.mean([bool(r["spectrum"]["retained_set_preserved"]) for r in ra if r["spectrum"]["retained_set_preserved"] is not None])) if any(r["spectrum"]["retained_set_preserved"] is not None for r in ra) else None,
            "floor_applied_frac": float(np.mean([r["whitening"]["floor_applied"] for r in ra])),
            "amplitude_rmse_median": med(ra, "amplitude_rmse"), "probe_r2_median": med(ra, "probe_r2"),
            "in_domain_all": all(r["in_theorem_domain"] for r in ra),
        }
    # PRIMARY endpoint: sample-basis designed families C, F plus M, S, Sstat, K at eps_op <= 0.2
    # v2: boundary tolerance 1e-9 (nominal eps = 0.2 rows computed as 0.2000000000000001 were
    # dropped by the v1 filter); designed (C, F) and estimator (M, S, Sstat, K) families reported separately
    prim = [r for r in rows if r["family"] in ("C", "F", "M", "S", "Sstat", "K") and r.get("basis") != "population"
            and r["eps_op"] <= 0.2 + 1e-9 and r.get("ratio_cov_hat") is not None]
    frac = float(np.mean([0.5 <= r["ratio_cov_hat"] <= 2.0 for r in prim])) if prim else None
    prim_designed = [r for r in prim if r["family"] in ("C", "F")]
    prim_estim = [r for r in prim if r["family"] not in ("C", "F")]
    slopes = {}
    for fam in ("C", "F"):
        xs, ys = [], []
        for e in EPS_GRID:
            rr = [r for r in rows if r["family"] == fam and r["basis"] == "sample" and r["arm_params"].get("eps") == e]
            if rr:
                xs.append(np.log(e)); ys.append(np.log(np.median([r["resid_cov_hat_fro"] for r in rr])))
        if len(xs) >= 2:
            slopes[fam] = float(np.polyfit(xs, ys, 1)[0])
    P_rows = [r for r in rows if r["family"] == "P" and r["basis"] == "sample"]
    R_rows = [r for r in rows if r["family"] == "R"]
    out["primary"] = {
        "ratio_cov_hat_in_band_fraction_eps_le_0.2": frac, "n_primary_fits": len(prim),
        "designed_CF": {"n": len(prim_designed), "in_band_fraction": float(np.mean([0.5 <= r["ratio_cov_hat"] <= 2.0 for r in prim_designed])) if prim_designed else None},
        "estimator_families": {"n": len(prim_estim), "arms": sorted({r["arm"] for r in prim_estim}),
                               "in_band_fraction": float(np.mean([0.5 <= r["ratio_cov_hat"] <= 2.0 for r in prim_estim])) if prim_estim else None},
        "eps_filter": "eps_op <= 0.2 + 1e-9 (v2; v1 used <= 0.2 and dropped 10 nominal-0.2 C/F rows at 0.2000000000000001)",
        "in_band_by_arm": {arm: float(np.mean([0.5 <= r["ratio_cov_hat"] <= 2.0 for r in prim if r["arm"] == arm])) for arm in sorted({r["arm"] for r in prim})},
        "residual_loglog_slope_sample_basis": slopes,
        "P_sample_max_sin_theta_to_oracle": float(max(r["sin_theta_to_oracle"] for r in P_rows)) if P_rows else None,
        "R_by_arm": {arm: {"sin_theta_to_oracle_median": float(np.median([r["sin_theta_to_oracle"] for r in R_rows if r["arm"] == arm])),
                           "retained_set_preserved_frac": float(np.mean([bool(r["spectrum"]["retained_set_preserved"]) for r in R_rows if r["arm"] == arm]))}
                     for arm in sorted({r["arm"] for r in R_rows})},
        "S_minus_lambda_min_lt_0.05_frac": float(np.mean([r["spectrum"]["lambda_min_IplusE"] < 0.05 for r in rows if r["family"] == "S-"])) if any(r["family"] == "S-" for r in rows) else None,
        "S_minus_effective": {arm: {"n_floored": int(sum(1 for r in rows if r["arm"] == arm and r["whitening"]["floor_applied"])),
                                    "eps_op_raw_median": med([r for r in rows if r["arm"] == arm], "eps_op"),
                                    "eps_op_effective_median": float(np.median([r["extra"]["effective"]["eps_op"] for r in rows if r["arm"] == arm and r.get("extra", {}).get("effective")])) if any(r.get("extra", {}).get("effective") for r in rows if r["arm"] == arm) else None,
                                    "lambda_min_effective_median": float(np.median([r["extra"]["effective"]["spectrum"]["lambda_min_IplusE"] for r in rows if r["arm"] == arm and r.get("extra", {}).get("effective")])) if any(r.get("extra", {}).get("effective") for r in rows if r["arm"] == arm) else None,
                                    "sin_theta_to_oracle_median": med([r for r in rows if r["arm"] == arm], "sin_theta_to_oracle"),
                                    "probe_r2_median": med([r for r in rows if r["arm"] == arm], "probe_r2")}
                              for arm in sorted({r["arm"] for r in rows if r["family"] == "S-"})},
        "K_lambda_min_median": {arm: float(np.median([r["spectrum"]["lambda_min_IplusE"] for r in rows if r["arm"] == arm])) for arm in sorted({r["arm"] for r in rows if r["family"] == "K"})},
    }
    return out


def provenance(argv: list[str]) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return "unavailable"
    src = Path(__file__).read_bytes()
    g1src = (Path(__file__).parent / "g1.py").read_bytes()
    return {"code_commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tree_dirty": bool(git("status", "--porcelain")),
            "runner_sha256": hashlib.sha256(src).hexdigest(), "g1_sha256": hashlib.sha256(g1src).hexdigest(),
            "g1_note": "local re-implementation of the 10 Sep G1 (a7125ea not available locally); equivalence checked against recorded cond/lambda_1/Delta_lambda",
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "thread_limits": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "command": argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--cap", type=int, default=FIT_CAP)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="restrict to these arm ids (v2 re-computation of the floored S- rows; results go to --out)")
    args = ap.parse_args(argv)
    arms = arm_list()
    assert len(arms) == ARMS_EXPECTED, f"{len(arms)} arms != {ARMS_EXPECTED} declared in the protocol JSON"
    if args.arms:
        arms = [a for a in arms if a["id"] in set(args.arms)]
        assert len(arms) == len(set(args.arms)), f"unknown arm id in {args.arms}"
        if args.out == OUT_DIR:
            print("refusing: --arms requires an explicit --out (never overwrite the frozen archive)", file=sys.stderr)
            return 4
    n_fits = len(arms) * len(args.seeds)
    if n_fits > args.cap:
        print(f"refusing: {n_fits} fits exceed cap {args.cap}", file=sys.stderr)
        return 2
    g = G.make_g1()
    check = G.check_against_recorded(g)
    if not check["matches_recorded"]:
        print(f"refusing: local G1 does not match the recorded generator: {check}", file=sys.stderr)
        return 3
    thr = (g.lam[K - 1] - g.lam[K]) / (g.lam[K - 1] + g.lam[K])
    args.out.mkdir(parents=True, exist_ok=True)
    prov = provenance(sys.argv)
    prov["g1_check"] = check
    prov["family_R_reversal_threshold_eps"] = float(thr)
    prov["n_fits_planned"] = n_fits
    rows: list[dict] = []
    t0 = time.time()
    with open(args.out / "rows.jsonl", "w") as fh:
        for s in args.seeds:
            for r in run_seed(g, s, arms):
                fh.write(json.dumps(r) + "\n"); rows.append(r)
            print(f"seed {s} done ({len(rows)} rows, {time.time()-t0:.1f}s)", flush=True)
    summ = summarise(rows)
    summ["family_R_reversal_threshold_eps"] = float(thr)
    summ["g1_check"] = check
    (args.out / "summary.json").write_text(json.dumps(summ, indent=1))
    prov["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prov["wall_seconds"] = time.time() - t0
    prov["n_fits_done"] = len(rows)
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.name != "manifest.json"}
    prov["runner_version"] = "v2 (11 Sep audit): consistent floored decoder, effective-covariance diagnostics, tolerant eps filter"
    (args.out / "manifest.json").write_text(json.dumps({"task": "T19", "provenance": prov, "arms": [a["id"] for a in arms],
                                                          "seeds": args.seeds, "outputs": outputs}, indent=1))
    print(json.dumps(summ["primary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
