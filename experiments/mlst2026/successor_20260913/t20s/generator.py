"""T20-S analytic generator: a whitened clean-signal second moment K(theta) with a
SPECIFIED spectrum and a geometry knob theta that moves the top-k subspace off the
product class without touching the eigenvalues.

Construction (all in whitened coordinates on R^{CT}, Fortran vec index c + C t,
the convention of ``kron_basis`` and of the archived T20 code):

    K(theta) = V(theta) diag(mu) V(theta)^T,     V(theta) = V_sep expm(theta A)

* ``mu`` is a fixed eigenvalue vector: a strictly decreasing top-k block
  ``mu_1 > ... > mu_k``, a rank-k gap ``Delta_k = mu_k - mu_{k+1}`` and a
  geometric tail ``mu_{k+1} > ... > mu_d > 0``.  It never depends on theta.
* ``V_sep`` is an orthonormal basis of R^d whose first k columns are
  ``kron_basis(U_c0, U_t0)`` for seeded, frozen ``U_c0 (C x k_c)``, ``U_t0
  (T x k_t)``: the top-k eigenspace at theta = 0 is EXACTLY a product subspace,
  so the approximation cost a(0) = 0 exactly.
* ``A`` is a fixed skew-symmetric generator (seeded, frozen, spectral norm 1)
  whose only nonzero blocks couple the top-k block with the tail block, so
  ``expm(theta A)`` rotates top-k directions into non-product directions.
  ``expm(theta A)`` is orthogonal, hence the spectrum of K(theta) is ``mu`` for
  every theta (checked to 1e-10 by :func:`invariants`).
* Data: z_i = s_i + n_i with s_i ~ N(0, K), n_i ~ N(0, I) i.i.d. Gaussian, so
  z_i ~ N(0, K + I).  The top-k eigenvectors of K + I are those of K.

This is a NEW synthetic family.  It is distinct from the historical
pulse-family population of T14/T20 (colored-noise pulses, Monte-Carlo covariance
proxy) and certifies nothing about that proxy.

Quantities from MLST Prop. 3.6 implemented here:

    R_K(P)          = tr K - tr(P K)                       (population risk)
    P*_U            = top-k eigenprojector of K            (unrestricted optimum)
    Delta_k         = mu_k - mu_{k+1}
    cap(theta)      = min{k, sum_{i<=k_c} lambda_i(M_c), sum_{i<=k_t} lambda_i(M_t)}
                      with M_c = tr_t P*_U, M_t = tr_c P*_U (partial traces)
    lower bound     a >= Delta_k (k - cap)  for every product projector
    best-found a    R_K(P_F^best) - R_K(P*_U), P_F^best = best of population ALS
                    restarts (an UPPER bound on a; never a certificate)

Nothing here launches work on import.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.paper1.t20_population_accounting import als_population, risk  # noqa: E402  (audited, imported unchanged)
from experiments.synthetic.nfpa_confirmatory.generator import kron_basis  # noqa: E402  (audited, imported unchanged)

__all__ = ["Geometry", "spectrum", "make_geometry", "K_of_theta", "top_k_basis", "partial_trace_cap",
           "population_reference", "invariants", "draw_data", "risk", "kron_basis"]

EIG_TOL = 1e-10


# --------------------------------------------------------------------------------------
# spectrum
# --------------------------------------------------------------------------------------
def spectrum(spec: dict, d: int, k: int) -> np.ndarray:
    """Eigenvalue vector ``mu`` (length d, strictly decreasing) from the protocol block::

        {"top": [mu_1, ..., mu_k], "gap": Delta_k, "tail_min": mu_d}

    The tail is geometric from ``mu_k - gap`` down to ``tail_min`` over d - k entries.
    """
    top = np.asarray(spec["top"], dtype=float)
    if top.size != k:
        raise ValueError(f"spectrum.top must have k={k} entries, got {top.size}")
    if np.any(np.diff(top) >= 0):
        raise ValueError("spectrum.top must be strictly decreasing")
    gap = float(spec["gap"])
    tail_max = float(top[-1]) - gap
    tail_min = float(spec["tail_min"])
    if not (gap > 0 and tail_max > tail_min > 0):
        raise ValueError("need gap > 0 and mu_k - gap > tail_min > 0")
    n_tail = d - k
    if n_tail < 1:
        raise ValueError("d must exceed k")
    if n_tail == 1:
        tail = np.array([tail_max])
    else:
        tail = np.exp(np.linspace(np.log(tail_max), np.log(tail_min), n_tail))
    mu = np.concatenate([top, tail])
    if np.any(np.diff(mu) >= 0):
        raise ValueError("mu is not strictly decreasing")
    return mu


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
class Geometry:
    """Frozen ingredients of K(theta) for one (C, T, k_c, k_t, spectrum, seed)."""

    def __init__(self, C: int, T: int, k_c: int, k_t: int, mu: np.ndarray, V_sep: np.ndarray, A: np.ndarray,
                 U_c0: np.ndarray, U_t0: np.ndarray, seed: int):
        self.C, self.T, self.k_c, self.k_t = int(C), int(T), int(k_c), int(k_t)
        self.k = self.k_c * self.k_t
        self.d = self.C * self.T
        self.mu = np.asarray(mu, dtype=float)
        self.V_sep = V_sep
        self.A = A
        self.U_c0, self.U_t0 = U_c0, U_t0
        self.seed = int(seed)

    @property
    def Delta_k(self) -> float:
        return float(self.mu[self.k - 1] - self.mu[self.k])

    def V(self, theta: float) -> np.ndarray:
        if theta == 0.0:
            return self.V_sep.copy()
        return self.V_sep @ expm(float(theta) * self.A)

    def K(self, theta: float) -> np.ndarray:
        V = self.V(theta)
        K = (V * self.mu[None, :]) @ V.T
        return 0.5 * (K + K.T)


def make_geometry(*, C: int, T: int, k_c: int, k_t: int, spec: dict, seed: int) -> Geometry:
    """Build the frozen geometry.  ``seed`` fixes U_c0, U_t0, the completion of
    V_sep and the skew generator A; it is a design constant of the protocol, not a
    sampling seed."""
    d, k = C * T, k_c * k_t
    mu = spectrum(spec, d, k)
    rng = np.random.default_rng([int(seed), C, T, k_c, k_t])
    U_c0 = np.linalg.qr(rng.standard_normal((C, k_c)))[0]
    U_t0 = np.linalg.qr(rng.standard_normal((T, k_t)))[0]
    B = kron_basis(U_c0, U_t0)                                      # (d, k) exactly product
    # complete to an orthonormal basis: project a random block off span(B), then QR
    R = rng.standard_normal((d, d - k))
    R = R - B @ (B.T @ R)
    Qr = np.linalg.qr(R)[0]
    V_sep = np.concatenate([B, Qr], axis=1)
    # skew generator coupling the top-k block to the tail block only
    A = np.zeros((d, d))
    Gblk = rng.standard_normal((k, d - k))
    A[:k, k:] = Gblk
    A[k:, :k] = -Gblk.T
    A /= np.linalg.norm(A, 2)                                        # spectral norm 1: theta in radians
    return Geometry(C, T, k_c, k_t, mu, V_sep, A, U_c0, U_t0, seed)


def K_of_theta(g: Geometry, theta: float) -> np.ndarray:
    return g.K(theta)


def top_k_basis(K: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenbasis (d, k) and the full descending eigenvalue vector."""
    w, U = np.linalg.eigh(K)
    order = np.argsort(w)[::-1]
    return U[:, order[:k]], w[order]


def partial_trace_cap(B_U: np.ndarray, C: int, T: int, k_c: int, k_t: int) -> dict:
    """cap = min{k, sum_{i<=k_c} lambda_i(M_c), sum_{i<=k_t} lambda_i(M_t)} for
    P*_U = B_U B_U^T (Fortran vec index c + C t, as in the archived 'bounds' stage)."""
    k = k_c * k_t
    P = B_U @ B_U.T
    P4 = P.reshape(C, T, C, T, order="F")
    M_c = np.einsum("ctdt->cd", P4)
    M_t = np.einsum("ctcs->ts", P4)
    ec = np.sort(np.linalg.eigvalsh(0.5 * (M_c + M_c.T)))[::-1]
    et = np.sort(np.linalg.eigvalsh(0.5 * (M_t + M_t.T)))[::-1]
    cap = min(float(k), float(ec[:k_c].sum()), float(et[:k_t].sum()))
    return {"k": k, "sum_top_kc_M_c": float(ec[:k_c].sum()), "sum_top_kt_M_t": float(et[:k_t].sum()), "cap": cap,
            "k_minus_cap": float(k - cap)}


# --------------------------------------------------------------------------------------
# population reference (the algebraic part of a theta): solves, no fits
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _population_reference_cached(key: tuple) -> dict:
    C, T, k_c, k_t, spec_items, seed, theta, R_pop, n_iter, tol = key
    spec = dict(spec_items)
    spec["top"] = list(spec["top"])
    g = make_geometry(C=C, T=T, k_c=k_c, k_t=k_t, spec=spec, seed=seed)
    return _population_reference(g, theta, R_pop=R_pop, n_iter=n_iter, tol=tol)


def population_reference(*, C: int, T: int, k_c: int, k_t: int, spec: dict, seed: int, theta: float, R_pop: int,
                         n_iter: int = 400, tol: float = 1e-13) -> dict:
    """Deterministic population quantities for one theta (cached per process).
    Solves only; charged in the algebraic cells of the matrix."""
    key = (int(C), int(T), int(k_c), int(k_t), tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in spec.items())),
           int(seed), float(theta), int(R_pop), int(n_iter), float(tol))
    return _population_reference_cached(key)


def _population_reference(g: Geometry, theta: float, *, R_pop: int, n_iter: int, tol: float) -> dict:
    K = g.K(theta)
    B_U, w = top_k_basis(K, g.k)
    R_U_star = risk(K, B_U)
    ptc = partial_trace_cap(B_U, g.C, g.T, g.k_c, g.k_t)
    Delta = g.Delta_k
    lower = float(Delta * ptc["k_minus_cap"])
    if theta == 0.0:
        # exact: at theta = 0 the top-k eigenspace IS the product subspace by construction
        lower = 0.0 if abs(lower) < 1e-9 else lower
    pop = als_population(K, g.C, g.T, g.k_c, g.k_t, n_iter=n_iter, n_restarts=R_pop, seed=g.seed + 17, tol=tol)
    B_F = kron_basis(pop["best"]["U_c"], pop["best"]["U_t"])
    R_F_best = risk(K, B_F)
    a_best = float(R_F_best - R_U_star)
    # the population-ALS captured energy tr(P_F K) per init, converted to risk spread
    return {
        "theta": float(theta), "eigenvalues_max_abs_dev": float(np.max(np.abs(w - g.mu))),
        "trace_K": float(np.trace(K)), "trace_mu": float(g.mu.sum()), "d": g.d, "k": g.k,
        "numerical_rank": int(np.sum(w > 1e-12 * w[0])), "snr_per_coordinate": float(np.trace(K) / g.d),
        "Delta_k": Delta, "partial_trace": ptc, "a_lower_bound": lower, "a_best_found": a_best,
        "R_U_star": float(R_U_star), "R_F_best_found": float(R_F_best),
        "pop_als": {"n_inits": int(pop["n_inits"]), "best_init": str(pop["best_init"]),
                    "restart_spread_captured": float(pop["restart_spread"]),
                    "hosvd_captured": float(pop["hosvd_captured"]), "best_captured": float(pop["best_captured"])},
        "B_U_star": B_U, "B_F_best": B_F, "K": K,
    }


def invariants(g: Geometry, thetas: list[float], *, R_pop: int, n_iter: int = 400, tol: float = 1e-13,
               spec: dict | None = None) -> dict:
    """Verify and record, per theta and before any fitting: spectrum fixed to
    EIG_TOL, trace/rank/SNR fixed, and the nonseparability measure k - cap
    (with the lower bound and the best-found a) changing with theta.  A theta at
    which the constraint fails is recorded and marked ``excluded``."""
    per = {}
    for th in thetas:
        r = population_reference(C=g.C, T=g.T, k_c=g.k_c, k_t=g.k_t, spec=spec, seed=g.seed, theta=th, R_pop=R_pop,
                                 n_iter=n_iter, tol=tol)
        per[th] = {k: v for k, v in r.items() if k not in ("B_U_star", "B_F_best", "K")}
    base = per[thetas[0]]
    problems = {}
    for th, r in per.items():
        p = []
        if r["eigenvalues_max_abs_dev"] > EIG_TOL:
            p.append(f"spectrum deviates by {r['eigenvalues_max_abs_dev']:.2e} > {EIG_TOL}")
        if abs(r["trace_K"] - base["trace_K"]) > 1e-8 * max(1.0, base["trace_K"]):
            p.append("trace changed")
        if r["numerical_rank"] != base["numerical_rank"]:
            p.append("rank changed")
        if abs(r["snr_per_coordinate"] - base["snr_per_coordinate"]) > 1e-8:
            p.append("SNR changed")
        if abs(r["Delta_k"] - base["Delta_k"]) > 1e-12:
            p.append("gap changed")
        if r["a_lower_bound"] > r["a_best_found"] + 1e-9:
            p.append("lower bound exceeds best-found a (bug)")
        if r["a_best_found"] < -1e-9:
            p.append("best-found a negative (bug)")
        if th == 0.0 and (abs(r["a_lower_bound"]) > 1e-9 or r["a_best_found"] > 1e-8):
            p.append("theta = 0 control failed: a must be 0")
        problems[th] = p
    # nonseparability must move with theta: k - cap pairwise distinct and increasing from theta = 0
    kmc = {th: per[th]["partial_trace"]["k_minus_cap"] for th in thetas}
    moves = True
    for i, th in enumerate(thetas):
        for th2 in thetas[i + 1:]:
            if abs(kmc[th] - kmc[th2]) < 1e-6:
                moves = False
                problems[th2].append(f"k - cap does not move between theta={th} and theta={th2}")
    excluded = [th for th in thetas if problems[th]]
    return {"per_theta": per, "problems": {str(th): v for th, v in problems.items()}, "nonseparability_moves": moves,
            "excluded_thetas": excluded, "usable_thetas": [th for th in thetas if th not in excluded],
            "eig_tol": EIG_TOL}


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def draw_data(g: Geometry, theta: float, N: int, seed_seq) -> np.ndarray:
    """N i.i.d. draws z = s + n, s ~ N(0, K(theta)), n ~ N(0, I) in whitened
    coordinates; returned as (N, d) with Fortran vec index c + C t."""
    rng = np.random.default_rng(seed_seq)
    V = g.V(theta)
    L = V * np.sqrt(g.mu + 1.0)[None, :]           # (K + I)^{1/2} up to an orthogonal factor
    G = rng.standard_normal((int(N), g.d))
    return G @ L.T
