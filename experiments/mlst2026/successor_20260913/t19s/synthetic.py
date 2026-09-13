"""T19-S synthetic part: the G1-shaped generator at any dimension, the rank-k
covariance-weighted PCA estimator, and one paired cell (one training draw,
one oracle fit, every arm on the same draw).

At ``d = 256`` the generator reproduces ``experiments.paper1.g1.make_g1`` bit for
bit (tested), and the estimator reproduces ``t19_coupling.fit_subspace`` (tested).
Smaller ``d`` (the fixture) keeps the same PSD shape relative to the grid
(``f_knee = 4/d``), the same rank, SNR and basis seed, and is labelled
``g1_like``; it is never the production generator.

Accounting inside one paired cell (see SCHEMA.md):

    fits   = 1 oracle + n_eps(C) + n_eps(F) + 2 n_m (S_full, S_toeplitz) + (n_N - 1) (D arm)
    solves = 1 (sample-operator eigendecomposition) + one model-identity solve per
             covariance-error fit (engineered and estimated), which checks that the
             estimator returned the invariant subspace of A_hat (I+E)^{-1}
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.paper1 import g1 as G
from experiments.paper1 import t19_coupling as T19

from ..common import fitcount
from . import observables as O

ESTIMATED_FAMILIES = ("S_full", "S_toeplitz")


@dataclass
class Generator:
    d: int
    k: int
    Sigma: np.ndarray
    Sigma_isqrt: np.ndarray
    B: np.ndarray
    A_pop: np.ndarray
    lam: np.ndarray
    Q: np.ndarray
    label: str

    @property
    def planted_whitened(self) -> np.ndarray:
        W = self.Sigma_isqrt @ self.B
        return O.orth(W)


def make_generator(d: int, k: int = G.RANK) -> Generator:
    """G1 at d = 256 (identical to ``g1.make_g1``); G1-like otherwise."""
    if k != G.RANK:
        raise ValueError("the frozen generator is rank 2")
    Sigma = G.covariance(d=d, f_knee=4.0 / d)
    Si, _ = G.sym_inv_sqrt(Sigma)
    B = G.signal_basis(d=d, rank=k, seed=G.BASIS_SEED)
    A = Si @ (B * G.AMPLITUDE_SD[None, :] ** 2) @ B.T @ Si + np.eye(d)
    A = 0.5 * (A + A.T)
    lam, Q = np.linalg.eigh(A)
    order = np.argsort(lam)[::-1]
    return Generator(d=d, k=k, Sigma=Sigma, Sigma_isqrt=Si, B=B, A_pop=A, lam=lam[order], Q=Q[:, order],
                     label="g1_d256" if d == G.D else f"g1_like_d{d}")


def draw_noise(g: Generator, rng: np.random.Generator, n: int) -> np.ndarray:
    L = np.linalg.cholesky(g.Sigma)
    return rng.standard_normal((n, g.d)) @ L.T


def draw_training(g: Generator, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    a = rng.standard_normal((n, g.k)) * G.AMPLITUDE_SD[None, :]
    noise = draw_noise(g, rng, n)
    return a @ g.B.T + noise, a


@fitcount.counted("cw_pca")
def fit_subspace(X: np.ndarray, Sigma_hat: np.ndarray, k: int, *, floor: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rank-k CW-PCA under the assumed covariance; identical to
    ``t19_coupling.fit_subspace`` (v2, consistent decoder) for any d.  One fit.

    Instrumented: this is the package's only estimator invocation on drawn data,
    so a paired cell's charged fit count is the number of times this ran."""
    d = X.shape[1]
    floor_val = T19.FLOOR_FRACTION * np.trace(Sigma_hat) / d if floor else None
    Si, diag = G.sym_inv_sqrt(Sigma_hat, floor=floor_val)
    Xc = X - X.mean(axis=0, keepdims=True)
    Z = Xc @ Si
    _, _, vh = np.linalg.svd(Z, full_matrices=False)
    W = vh[:k].T
    P = np.linalg.solve(Si, W)
    return P, Si, diag


def whitened_basis(g: Generator, P: np.ndarray) -> np.ndarray:
    return O.orth(g.Sigma_isqrt @ P)


def sample_operator(g: Generator, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A_hat = Sigma^{-1/2} (Xc^T Xc / N) Sigma^{-1/2}, its eigenvalues (desc) and eigenbasis."""
    Xc = X - X.mean(axis=0, keepdims=True)
    A_hat = g.Sigma_isqrt @ (Xc.T @ Xc / X.shape[0]) @ g.Sigma_isqrt
    A_hat = 0.5 * (A_hat + A_hat.T)
    lam, Q = np.linalg.eigh(A_hat)
    order = np.argsort(lam)[::-1]
    return A_hat, lam[order], Q[:, order]


def sigma_hat_from_E(g: Generator, E: np.ndarray) -> np.ndarray:
    S_sqrt = G.sym_sqrt(g.Sigma)
    return S_sqrt @ (np.eye(g.d) + E) @ S_sqrt


def E_from_sigma_hat(g: Generator, Sigma_hat: np.ndarray) -> np.ndarray:
    return g.Sigma_isqrt @ (Sigma_hat - g.Sigma) @ g.Sigma_isqrt


def toeplitz_from_cov(Sm: np.ndarray) -> np.ndarray:
    """Stationary (Toeplitz) estimate from a sample covariance: the same lag
    average as ``t19_coupling.toeplitz_cov`` (tested equal), without needing
    the raw draws."""
    d = Sm.shape[0]
    c = np.array([np.mean(np.diagonal(Sm, offset=l)) for l in range(d)])
    idx = np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
    return c[idx]


CALIBRATION_CHUNK = 8192


def calibration_covariances(g: Generator, rng: np.random.Generator, m_grid: list[int], *, chunk: int = CALIBRATION_CHUNK) -> dict[int, dict]:
    """Noise-only calibration: the nested pool is streamed in row chunks and
    the uncentred second moment accumulated, so memory is O(d^2) whatever m.
    For every m in ``m_grid`` (a prefix of the same stream, so the arms are
    nested exactly as ``pool[:m]`` would be) returns the full sample covariance
    ``S_full = X^T X / m`` (identical to ``t19_coupling.empirical_cov`` up to
    summation order) and its Toeplitz lag average."""
    L = np.linalg.cholesky(g.Sigma)
    acc = np.zeros((g.d, g.d))
    done = 0
    out: dict[int, dict] = {}
    for m in sorted(int(x) for x in m_grid):
        while done < m:
            n = min(chunk, m - done)
            Z = rng.standard_normal((n, g.d)) @ L.T
            acc += Z.T @ Z
            done += n
        S_full = acc / m
        out[m] = {"S_full": S_full, "S_toeplitz": toeplitz_from_cov(S_full)}
    return out


def effective_E(Si_used: np.ndarray, g: Generator) -> np.ndarray:
    """E of the covariance the fit actually used (after a floor): Sigma_eff = (Si Si)^{-1}."""
    Sigma_eff = np.linalg.inv(Si_used @ Si_used)
    return E_from_sigma_hat(g, Sigma_eff)


# -----------------------------------------------------------------------------
# one paired cell
# -----------------------------------------------------------------------------
def plan_counts(params: dict) -> tuple[int, int]:
    """Fits and solves a paired cell will charge, derived from its params."""
    n_eps = len(params["eps_grid"])
    n_m = len(params["m_over_d_grid"])
    n_N = len(params["n_train_ratio_grid"])
    fits = 1 + 2 * n_eps + 2 * n_m + (n_N - 1)
    solves = 1 + 2 * n_eps + 2 * n_m
    return fits, solves


def _fit_record(*, arm: str, source: str, family: str, basis: str, g: Generator, V: np.ndarray, Q: np.ndarray,
                lam: np.ndarray, k: int, X1: np.ndarray, identifiable: bool, eps_op: float, eps_fro: float,
                eta_sample: float | None, eta_population: float | None, eta_used: float, lambda_min: float | None,
                bound_applies: bool, model_check: dict | None, V_or: np.ndarray, planted: np.ndarray,
                fit_index: int, floor_diag: dict, extra: dict) -> dict:
    obs = O.observables(V, Q, lam, k, X1)
    cls = O.domain_class(eta_used, lambda_min)
    rec = {"arm": arm, "source": source, "family": family, "basis": basis, "fit_index": int(fit_index),
           "eps_op": float(eps_op), "eps_fro": float(eps_fro), "lambda_min_IplusE": lambda_min,
           "eta_sample": eta_sample, "eta_population": eta_population, "eta_used_for_classification": float(eta_used),
           "eta_used_is": "sample" if basis == "sample" else "population",
           "domain_class": cls, "in_domain": bool(cls == "in_domain"),
           "first_order_identifiable": bool(identifiable), "observables": obs,
           "bound": O.bound_record(obs["projector_matrix"]["residual_op"], eta_used, applies=bound_applies),
           "eigensolver_residual": model_check["eigensolver_residual"] if model_check else None,
           "model_identity_sin_theta": model_check["sin_theta"] if model_check else None,
           "perturbed_separation": model_check["separation"] if model_check else None,
           "sin_theta_to_oracle": float(O.sin_theta_max(V, V_or)), "sin_theta_to_planted": float(O.sin_theta_max(V, planted)),
           "floor_applied": bool(floor_diag.get("floor_applied", False)), "floor_value": floor_diag.get("floor_value")}
    rec.update(extra)
    return rec


def run_paired_cell(params: dict, seed: int, *, selection_sha256: str | None) -> tuple[dict, int, int]:
    """Execute one paired cell.  ``params`` carries d, k, n_train, eps_grid,
    m_over_d_grid, n_train_ratio_grid and the seed offsets; ``seed`` is the
    training draw (the independent unit)."""
    d, k, N = int(params["d"]), int(params["k"]), int(params["n_train"])
    g = make_generator(d, k)
    off = params["rng_offsets"]
    rng_train = np.random.default_rng(off["train"] + seed)
    rng_cal = np.random.default_rng(off["calibration"] + seed)
    rng_fam = np.random.default_rng(off["direction"] + seed)
    N_grid = [int(round(N * r)) for r in params["n_train_ratio_grid"]]
    assert N in N_grid, "the oracle N must be one of the D-arm sizes"
    m_grid = [int(round(d * r)) for r in params["m_over_d_grid"]]
    eps_grid = [float(e) for e in params["eps_grid"]]

    X_full, _ = draw_training(g, rng_train, n=max(N_grid))      # nested: first N rows = the fixed training set
    X_tr = X_full[:N]
    calib = calibration_covariances(g, rng_cal, m_grid)          # nested calibration pool, streamed
    planted = g.planted_whitened
    fits = 0
    solves = 0
    records: list[dict] = []

    # oracle fit (charged) and the sample operator (one solve)
    P_or, _, _ = fit_subspace(X_tr, g.Sigma, k, floor=False); fits += 1
    V_or = whitened_basis(g, P_or)
    A_hat, lam_hat, Q_hat = sample_operator(g, X_tr); solves += 1
    spec_sample = O.spectral_summary(lam_hat, k)
    spec_pop = O.spectral_summary(g.lam, k)
    oracle_graph_sample = float(np.linalg.norm(O.graph_of(V_or, Q_hat, k)))
    Dq = g.Q.T @ (A_hat - g.A_pop) @ g.Q
    X1_or_pop, _ = O.first_order_graph(g.lam, Dq, k, "additive")
    oracle = {"n_fits": 1, "graph_in_sample_basis_fro": oracle_graph_sample,
              "graph_in_population_basis_fro": float(np.linalg.norm(O.graph_of(V_or, g.Q, k))),
              "graph_first_order_population_fro": float(np.linalg.norm(X1_or_pop)),
              "sin_theta_to_planted": float(O.sin_theta_max(V_or, planted)),
              "eta_D_population": O.eta_of(float(np.linalg.norm(A_hat - g.A_pop, 2)), g.lam, k)}

    # (i) engineered E in the SAMPLE eigenbasis, direction fixed along the eps path
    for fam, kind in (("C", "pure_coupling"), ("F", "generic")):
        E0 = O.make_direction(kind, Q_hat, k, rng_fam, lam_b=lam_hat)
        for eps in eps_grid:
            E = eps * E0
            Sigma_hat = sigma_hat_from_E(g, E)
            P, Si, diag = fit_subspace(X_tr, Sigma_hat, k, floor=False); fits += 1
            V = whitened_basis(g, P)
            Eq_hat = Q_hat.T @ E @ Q_hat
            X1, ident = O.first_order_graph(lam_hat, Eq_hat, k, "covariance")
            eps_op = float(np.linalg.norm(E, 2))
            mc = O.covariance_model_subspace(A_hat, E, k); solves += 1
            model_check = {"eigensolver_residual": mc["eigensolver_residual"], "separation": mc["separation"],
                           "sin_theta": float(O.sin_theta_max(V, mc["V"])) if mc["valid"] else None}
            records.append(_fit_record(
                arm=f"{fam}-eps{eps:g}", source="engineered", family=fam, basis="sample", g=g, V=V, Q=Q_hat, lam=lam_hat, k=k,
                X1=X1, identifiable=ident, eps_op=eps_op, eps_fro=float(np.linalg.norm(E)),
                eta_sample=O.eta_of(eps_op, lam_hat, k), eta_population=O.eta_of(eps_op, g.lam, k),
                eta_used=O.eta_of(eps_op, lam_hat, k), lambda_min=mc["lambda_min_IplusE"], bound_applies=True,
                model_check=model_check, V_or=V_or, planted=planted, fit_index=fits, floor_diag=diag,
                extra={"eps_nominal": eps, "eps_realized_minus_nominal": float(eps_op - eps), "direction_shared_along_path": True,
                       "E_blocks_sample": T19.blocks(Eq_hat) if k == T19.K else None}))

    # (ii) independently estimated covariance from m noise-only draws (population-basis error)
    for m in m_grid:
        for fam in ESTIMATED_FAMILIES:
            Sigma_hat = calib[m][fam]
            E = E_from_sigma_hat(g, Sigma_hat)
            P, Si, diag = fit_subspace(X_tr, Sigma_hat, k, floor=True); fits += 1
            V = whitened_basis(g, P)
            E_used = effective_E(Si, g) if diag["floor_applied"] else E
            Eq_hat = Q_hat.T @ E_used @ Q_hat
            X1, ident = O.first_order_graph(lam_hat, Eq_hat, k, "covariance")
            eps_op = float(np.linalg.norm(E_used, 2))
            mc = O.covariance_model_subspace(A_hat, E_used, k); solves += 1
            model_check = {"eigensolver_residual": mc["eigensolver_residual"], "separation": mc["separation"],
                           "sin_theta": float(O.sin_theta_max(V, mc["V"])) if mc["valid"] else None}
            records.append(_fit_record(
                arm=f"{fam}-m{m}", source="estimated", family=fam, basis="sample", g=g, V=V, Q=Q_hat, lam=lam_hat, k=k,
                X1=X1, identifiable=ident, eps_op=eps_op, eps_fro=float(np.linalg.norm(E_used)),
                eta_sample=O.eta_of(eps_op, lam_hat, k), eta_population=O.eta_of(eps_op, g.lam, k),
                eta_used=O.eta_of(eps_op, lam_hat, k), lambda_min=mc["lambda_min_IplusE"], bound_applies=True,
                model_check=model_check, V_or=V_or, planted=planted, fit_index=fits, floor_diag=diag,
                extra={"m": int(m), "m_over_d": float(m / d), "E_defined_with": "population Sigma (not engineered)",
                       "eps_op_raw": float(np.linalg.norm(E, 2)), "E_blocks_sample": T19.blocks(Eq_hat) if k == T19.K else None,
                       "E_blocks_population": T19.blocks(g.Q.T @ E_used @ g.Q) if k == T19.K else None}))

    # (iii) training-sample error D with E = 0 (population basis, ADDITIVE perturbation of A_pop)
    for n_use in N_grid:
        if n_use == N:
            continue
        X_use = X_full[:n_use]
        P, Si, diag = fit_subspace(X_use, g.Sigma, k, floor=False); fits += 1
        V = whitened_basis(g, P)
        A_N, _, _ = sample_operator(g, X_use)   # diagnostic algebra (D itself), not a subspace solve
        Dn = A_N - g.A_pop
        DqN = g.Q.T @ Dn @ g.Q
        X1, ident = O.first_order_graph(g.lam, DqN, k, "additive")
        eps_D = float(np.linalg.norm(Dn, 2))
        eta_D = O.eta_of(eps_D, g.lam, k)
        records.append(_fit_record(
            arm=f"D-N{n_use}", source="training_sample", family="D", basis="population", g=g, V=V, Q=g.Q, lam=g.lam, k=k,
            X1=X1, identifiable=ident, eps_op=eps_D, eps_fro=float(np.linalg.norm(Dn)),
            eta_sample=None, eta_population=eta_D, eta_used=eta_D, lambda_min=None, bound_applies=False,
            model_check=None, V_or=V_or, planted=planted, fit_index=fits, floor_diag=diag,
            extra={"N": int(n_use), "eta_D": eta_D, "perturbation_model": "additive (A_pop + D); the 44 eta^2 bound is stated for the covariance model and is not asserted here",
                   "D_blocks_population": T19.blocks(DqN) if k == T19.K else None}))

    result = {
        "kind": "paired", "d": d, "k": k, "model": "covariance", "generator": g.label,
        "generator_check": {**G.check_against_recorded(g), "reference_applies": bool(d == G.D),
                            "note": "recorded cond/lambda_1/Delta_lambda are for d = 256 only" if d != G.D else "G1 (10 Sep) recorded values"},
        "seed": int(seed), "n_train": N, "selection_sha256": selection_sha256,
        "spectrum_sample": spec_sample, "spectrum_population": spec_pop,
        "basis_notes": {"engineered": "E built in the SAMPLE eigenbasis Q_hat of A_hat; eta, gaps, first order use the sample spectrum",
                        "estimated": "E = Sigma^{-1/2}(Sigma_hat - Sigma)Sigma^{-1/2} with the POPULATION Sigma; analysed in the sample eigenbasis; eta from the sample spectrum",
                        "training_sample": "D = A_hat_N - A_pop in the POPULATION eigenbasis; eta_D from the population spectrum"},
        "oracle": oracle, "records": records,
        "summary": {"n_records": len(records), "n_in_domain": int(sum(r["in_domain"] for r in records)),
                    "domain_counts": {c: int(sum(r["domain_class"] == c for r in records)) for c in ("in_domain", "intermediate", "stress", "beyond", "indefinite")}},
        "checks": {
            "oracle_graph_in_sample_basis_lt_1e-8": bool(oracle_graph_sample < 1e-8),
            "engineered_eps_realized_equals_nominal": bool(all(abs(r["eps_realized_minus_nominal"]) <= 1e-9 * max(1.0, r["eps_nominal"]) for r in records if r["source"] == "engineered")),
            "model_identity_sin_theta_lt_1e-6": bool(all((r["model_identity_sin_theta"] is None) or (r["model_identity_sin_theta"] < 1e-6) for r in records if r["source"] != "training_sample" and not r["floor_applied"])),
            "projector_identity_lt_1e-10": bool(all(r["observables"]["identity_abs_error"] < 1e-10 for r in records)),
        },
    }
    return result, fits, solves
