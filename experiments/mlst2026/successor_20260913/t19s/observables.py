"""T19-S scientific core: models, observables, first-order predictions, families.

Everything here is deterministic linear algebra with an explicit retained rank
``k``.  No data is drawn and nothing is fitted in this module; the synthetic
generator and the estimator live in :mod:`.synthetic`.

Model (the paper's, MLST Prop 3.4).  With ``A`` symmetric (eigenbasis ``Q``,
eigenvalues ``lam`` descending, retained set ``i < k``) and a covariance error
``E`` (``Sigma_hat = Sigma^{1/2}(I+E)Sigma^{1/2}``), the estimator fitted with the
wrong covariance returns the top-``k`` invariant subspace of ``A (I+E)^{-1}`` in
true-whitened coordinates.  In the eigenbasis, with ``Ehat = Q^T E Q``:

    graph          X = C_disc C_ret^{-1},  C = Q^T V             (T19 archive observable)
    first order    (X1)_ji = -lam_j Ehat_ji / (lam_i - lam_j),  i < k <= j
    projector      dP = C C^T - P0,  delta P = [[0, X1^T], [X1, 0]]
    bound          ||P_hat - P - delta P||_op <= 44 eta^2,  eta = eps lam_1 / delta <= 1/10,
                   eps = ||E||_op, delta = lam_k - lam_{k+1}
    identity       ||P(X) - P(0)||_F^2 = 2 tr[S (I+S)^{-1}],  S = X^T X   (exact)

Control model (NOT the paper's): ``A + eps G`` additive, first order
``(X1)_ji = G_ji / (lam_i - lam_j)``.  It is kept only as a labelled control.

Predeclared leading residual orders per (family, observable), covariance model:

    family          graph  projector_matrix  projector_norm  angle
    generic           2          2                2            2
    pure_coupling     3          2                3            3
    block_diagonal   exact zero displacement (absolute branch; ratio undefined)
    zero             exact identity (absolute branch; ratio undefined)
    2-D example      pure coupling with k = 1: 3, 2, 3, 3 and closed forms

The additive control has the same integer orders (the parity argument does
not depend on the model); it is reported separately and never as the paper's
model.
"""
from __future__ import annotations

import numpy as np

from experiments.paper1 import t19_coupling as T19

BOUND_CONSTANT = 44.0
ETA_IN_DOMAIN = 0.1
ETA_STRESS = (0.2, 1.3)
ABSOLUTE_BRANCH_THRESHOLD = 1e-12   # predicted first-order norm below this -> ratio undefined (null)
SLOPE_FLOOR = 1e-13                 # residuals at or below this are numerical zero and excluded from slopes
NUMERICAL_FLOOR = 1e-12             # a bound violation at or below this is floating-point noise (eta = 0 -> bound 0)

OBSERVABLES = ("graph", "projector_matrix", "projector_norm", "angle")

# integer orders declared before any measurement (see module docstring)
PREDECLARED_ORDERS = {
    "generic": {"graph": 2, "projector_matrix": 2, "projector_norm": 2, "angle": 2},
    "pure_coupling": {"graph": 3, "projector_matrix": 2, "projector_norm": 3, "angle": 3},
    "two_d_example": {"graph": 3, "projector_matrix": 2, "projector_norm": 3, "angle": 3},
}
EXACT_FAMILIES = ("block_diagonal", "zero")   # first-order prediction identically zero -> absolute branch


def sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


orth = T19.orth
sin_theta_max = T19.sin_theta_max


# -----------------------------------------------------------------------------
# graph, first order, projector
# -----------------------------------------------------------------------------
def graph_of(V: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    """T = C_disc C_ret^{-1} for an orthonormal basis V expressed in Q (generic k;
    equals ``t19_coupling.displacement`` for k = 2)."""
    C = Q.T @ V
    return C[k:, :] @ np.linalg.inv(C[:k, :])


def first_order_graph(lam: np.ndarray, Eq: np.ndarray, k: int, model: str = "covariance") -> tuple[np.ndarray, bool]:
    """First-order graph for the perturbation ``Eq`` (already scaled by eps) in
    the eigenbasis.  Returns ``(X1, identifiable)``; ``identifiable`` is False
    when a retained/discarded eigenvalue pair ties (then X1 is NaN and must be
    reported, never divided through)."""
    lam = np.asarray(lam, dtype=float)
    denom = lam[:k][None, :] - lam[k:][:, None]           # (d-k, k): lam_i - lam_j
    blk = np.asarray(Eq)[k:, :k]
    identifiable = bool(np.min(np.abs(denom)) > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        if model == "covariance":
            X1 = (-lam[k:][:, None] * blk) / denom
        elif model == "additive":
            X1 = blk / denom
        else:
            raise ValueError(model)
    return X1, identifiable


def first_order_projector(X1: np.ndarray, d: int, k: int) -> np.ndarray:
    dP1 = np.zeros((d, d))
    dP1[k:, :k] = X1
    dP1[:k, k:] = X1.T
    return dP1


def projector_identity_error(X: np.ndarray, dP_fro: float) -> float:
    """|  ||dP||_F^2 - 2 tr[S (I+S)^{-1}] |  with S = X^T X (exact identity)."""
    S = X.T @ X
    k = S.shape[0]
    rhs = 2.0 * np.trace(S @ np.linalg.inv(np.eye(k) + S))
    return float(abs(dP_fro ** 2 - rhs))


# -----------------------------------------------------------------------------
# the two models: perturbed top-k invariant subspaces
# -----------------------------------------------------------------------------
def covariance_model_subspace(A: np.ndarray, E: np.ndarray, k: int) -> dict:
    """Top-k invariant subspace of ``A (I+E)^{-1}`` via the similar symmetric
    operator ``(I+E)^{-1/2} A (I+E)^{-1/2}`` (same construction as
    ``t19_coupling.perturbed_spectrum``).  One *solve*.

    Returns V (orthonormal, d x k), the perturbed eigenvalues ``mu`` (descending),
    the eigensolver residual ``max_i ||M v_i - mu_i v_i||_2`` on the unnormalised
    eigenvectors of the non-symmetric ``M``, ``lambda_min(I+E)`` and the perturbed
    cutoff separation ``mu_k - mu_{k+1}``.
    """
    d = A.shape[0]
    IE = np.eye(d) + E
    vals, vecs = np.linalg.eigh(sym(IE))
    lam_min = float(vals.min())
    if lam_min <= 0.0:
        return {"valid": False, "lambda_min_IplusE": lam_min, "V": None, "mu": None,
                "eigensolver_residual": None, "separation": None}
    S_isqrt = (vecs / np.sqrt(vals)[None, :]) @ vecs.T
    S_sqrt = (vecs * np.sqrt(vals)[None, :]) @ vecs.T
    B = sym(S_isqrt @ A @ S_isqrt)
    mu, U = np.linalg.eigh(B)
    order = np.argsort(mu)[::-1]
    mu, U = mu[order], U[:, order]
    Vraw = S_sqrt @ U[:, :k]
    Vraw = Vraw / np.linalg.norm(Vraw, axis=0, keepdims=True)
    M = np.linalg.solve(IE.T, A.T).T                       # A (I+E)^{-1}
    res = M @ Vraw - Vraw * mu[:k][None, :]
    resid = float(np.max(np.linalg.norm(res, axis=0)))
    return {"valid": True, "lambda_min_IplusE": lam_min, "V": orth(Vraw), "mu": mu,
            "eigensolver_residual": resid, "separation": float(mu[k - 1] - mu[k])}


def additive_model_subspace(A: np.ndarray, Gm: np.ndarray, k: int) -> dict:
    """Top-k eigenspace of ``A + G`` (CONTROL model, not the paper's).  One solve."""
    Mtot = sym(A + Gm)
    w, V = np.linalg.eigh(Mtot)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    Vk = V[:, :k]
    res = Mtot @ Vk - Vk * w[:k][None, :]
    return {"valid": True, "lambda_min_IplusE": None, "V": Vk, "mu": w,
            "eigensolver_residual": float(np.max(np.linalg.norm(res, axis=0))), "separation": float(w[k - 1] - w[k])}


MODELS = {"covariance": covariance_model_subspace, "additive": additive_model_subspace}


# -----------------------------------------------------------------------------
# observables of one perturbed subspace against its first-order prediction
# -----------------------------------------------------------------------------
def _ratio(meas: float, pred: float) -> float | None:
    return float(meas / pred) if pred > ABSOLUTE_BRANCH_THRESHOLD else None


def observables(V: np.ndarray, Q: np.ndarray, lam: np.ndarray, k: int, X1: np.ndarray) -> dict:
    """All four observables, measured and first-order, with residuals and
    ratios.  Ratios are ``None`` (absolute branch) when the first-order
    prediction is numerically zero; the residual is then the absolute
    displacement and is compared with the bound directly.

    graph:            ||X - X1||_F
    projector_matrix: ||dP - delta P||_op and _F                (theorem's quantity is _op)
    projector_norm:   | ||dP||_F - sqrt(2)||X1||_F |
    angle:            || sin Theta - sigma(X1) ||_2,  sin Theta = singular values of Q_disc^T V
    """
    d = Q.shape[0]
    C = Q.T @ V
    X = C[k:, :] @ np.linalg.inv(C[:k, :])
    P0 = np.zeros((d, d))
    P0[:k, :k] = np.eye(k)
    dP = C @ C.T - P0
    dP1 = first_order_projector(X1, d, k)
    R = dP - dP1
    # graph
    g_meas, g_pred, g_res = np.linalg.norm(X), np.linalg.norm(X1), np.linalg.norm(X - X1)
    # projector matrix
    pm_meas_op, pm_pred_op = np.linalg.norm(dP, 2), np.linalg.norm(dP1, 2)
    pm_meas_fro, pm_pred_fro = np.linalg.norm(dP), np.linalg.norm(dP1)
    pm_res_op, pm_res_fro = np.linalg.norm(R, 2), np.linalg.norm(R)
    # projector Frobenius norm (scalar)
    n_meas, n_pred = pm_meas_fro, np.sqrt(2.0) * g_pred
    # principal angles
    s_meas = np.sort(np.linalg.svd(C[k:, :], compute_uv=False))[::-1]
    s_pred = np.sort(np.linalg.svd(X1, compute_uv=False))[::-1]
    a_res = np.linalg.norm(s_meas - s_pred)
    absolute_branch = bool(g_pred <= ABSOLUTE_BRANCH_THRESHOLD)
    return {
        "graph": {"measured_fro": float(g_meas), "predicted_fro": float(g_pred), "residual_fro": float(g_res),
                  "ratio": _ratio(g_meas, g_pred)},
        "projector_matrix": {"measured_op": float(pm_meas_op), "predicted_op": float(pm_pred_op),
                             "residual_op": float(pm_res_op), "measured_fro": float(pm_meas_fro),
                             "predicted_fro": float(pm_pred_fro), "residual_fro": float(pm_res_fro),
                             "ratio_op": _ratio(pm_meas_op, pm_pred_op)},
        "projector_norm": {"measured": float(n_meas), "predicted": float(n_pred), "residual": float(abs(n_meas - n_pred)),
                           "ratio": _ratio(n_meas, n_pred)},
        "angle": {"measured_sin": [float(x) for x in s_meas], "predicted_sin": [float(x) for x in s_pred],
                  "measured_sin_max": float(s_meas[0]), "predicted_sin_max": float(s_pred[0]),
                  "residual": float(a_res), "ratio": _ratio(s_meas[0], s_pred[0])},
        "identity_abs_error": projector_identity_error(X, float(pm_meas_fro)),
        "absolute_branch": absolute_branch,
    }


def residual_of(obs: dict, name: str) -> float:
    """The residual used for order estimation of each observable."""
    return {"graph": obs["graph"]["residual_fro"], "projector_matrix": obs["projector_matrix"]["residual_op"],
            "projector_norm": obs["projector_norm"]["residual"], "angle": obs["angle"]["residual"]}[name]


# -----------------------------------------------------------------------------
# eta, domain classification, bound
# -----------------------------------------------------------------------------
def spectral_summary(lam: np.ndarray, k: int) -> dict:
    lam = np.asarray(lam, dtype=float)
    delta = float(lam[k - 1] - lam[k])
    cross = lam[:k][None, :] - lam[k:][:, None]
    return {"lambda_1": float(lam[0]), "lambda_k": float(lam[k - 1]), "lambda_k_plus_1": float(lam[k]),
            "delta": delta, "lambda1_over_delta": float(lam[0] / delta) if delta > 0 else None,
            "retained_internal_gaps": [float(lam[i] - lam[i + 1]) for i in range(k - 1)],
            "cross_gap_min": float(np.min(np.abs(cross))), "cross_gap_max": float(np.max(np.abs(cross))),
            "top": [float(x) for x in lam[: k + 2]]}


def eta_of(eps_op: float, lam: np.ndarray, k: int) -> float:
    lam = np.asarray(lam, dtype=float)
    return float(eps_op * lam[0] / (lam[k - 1] - lam[k]))


def domain_class(eta: float, lambda_min_IplusE: float | None) -> str:
    if lambda_min_IplusE is not None and lambda_min_IplusE <= 0.0:
        return "indefinite"
    if eta <= ETA_IN_DOMAIN:
        return "in_domain"
    if eta < ETA_STRESS[0]:
        return "intermediate"
    if eta <= ETA_STRESS[1]:
        return "stress"
    return "beyond"


def bound_record(residual_op: float, eta: float, *, applies: bool) -> dict:
    """Absolute remainder bound of Prop 3.4: violation = max(0, ||R||_op - 44 eta^2)."""
    bound = BOUND_CONSTANT * eta ** 2
    violation = float(max(0.0, residual_op - bound))
    rec = {"applies": bool(applies), "constant": BOUND_CONSTANT, "eta": float(eta), "value": float(bound),
           "residual_op": float(residual_op), "violation": violation,
           "satisfied": bool(violation <= NUMERICAL_FLOOR), "numerical_floor": NUMERICAL_FLOOR,
           "remainder_over_eta2": float(residual_op / eta ** 2) if eta > 0 else None,
           "in_domain": bool(eta <= ETA_IN_DOMAIN)}
    return rec


# -----------------------------------------------------------------------------
# perturbation directions (unit operator norm) in a given eigenbasis
# -----------------------------------------------------------------------------
def make_direction(kind: str, Qb: np.ndarray, k: int, rng: np.random.Generator, lam_b: np.ndarray | None = None) -> np.ndarray:
    """Unit-operator-norm symmetric direction ``E0 = Qb Eq Qb^T``.

    ``pure_coupling`` and ``generic`` reuse the audited ``t19_coupling.make_E_family``
    (families C and F, eps = 1) and therefore require k == 2; ``block_diagonal``
    and ``zero`` are built here.
    """
    d = Qb.shape[0]
    if kind == "pure_coupling":
        assert k == T19.K, "make_E_family is frozen at rank 2"
        return T19.make_E_family(Qb, "C", 1.0, rng, lam_b=lam_b)
    if kind == "generic":
        assert k == T19.K, "make_E_family is frozen at rank 2"
        return T19.make_E_family(Qb, "F", 1.0, rng, lam_b=lam_b)
    Eq = np.zeros((d, d))
    if kind == "block_diagonal":
        Eq[:k, :k] = sym(rng.standard_normal((k, k)))
        Eq[k:, k:] = sym(rng.standard_normal((d - k, d - k)))
        Eq /= np.linalg.norm(Eq, 2)
    elif kind == "zero":
        pass
    else:
        raise ValueError(kind)
    return Qb @ Eq @ Qb.T


# -----------------------------------------------------------------------------
# the 2-D regression example (review finding S1)
# -----------------------------------------------------------------------------
TWO_D_A = np.diag([2.0, 1.0])
TWO_D_E0 = np.array([[0.0, 1.0], [1.0, 0.0]])
TWO_D_REFERENCE = {0.01: 1.414143, 0.005: 1.414196, 0.0025: 1.414209}   # ||P_hat-P-dP||_F / eps^2


def two_d_closed_form(eps: float) -> dict:
    """A = diag(2,1), E = eps [[0,1],[1,0]], k = 1: graph coordinate
    t = -2 eps / (1 + sqrt(1 + 8 eps^2)), first order X1 = -eps, and the exact
    projector remainder built from t."""
    t = -2.0 * eps / (1.0 + np.sqrt(1.0 + 8.0 * eps ** 2))
    Ph = np.array([[1.0, t], [t, t * t]]) / (1.0 + t * t)
    R = Ph - np.diag([1.0, 0.0]) - np.array([[0.0, -eps], [-eps, 0.0]])
    return {"t": float(t), "X1": -float(eps), "projector_matrix_residual_fro": float(np.linalg.norm(R)),
            "projector_matrix_residual_over_eps2": float(np.linalg.norm(R) / eps ** 2)}


# -----------------------------------------------------------------------------
# slopes along a fixed perturbation path
# -----------------------------------------------------------------------------
def slope_loglog(eps: list[float], y: list[float], *, min_points: int = 3) -> dict:
    """Paired log-log OLS slope along ONE fixed direction (the points share the
    direction, so this identifies the order cleanly).  Points at the numerical
    floor are excluded and reported."""
    e = np.asarray(eps, dtype=float)
    v = np.asarray(y, dtype=float)
    m = np.isfinite(v) & (v > SLOPE_FLOOR) & (e > 0)
    n = int(m.sum())
    if n < min_points:
        return {"slope": None, "n_points": n, "n_excluded_floor": int((~m).sum()), "min_points": min_points}
    s, b = np.polyfit(np.log(e[m]), np.log(v[m]), 1)
    return {"slope": float(s), "intercept": float(b), "n_points": n, "n_excluded_floor": int((~m).sum()), "min_points": min_points}
