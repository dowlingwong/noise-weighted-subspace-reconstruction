"""ZC estimators: thin wrappers on the T25 recipe plus CV class selection, the N*-rule and SHR.

All maps are affine with the fitted mean: x_hat = A x + b, A = t25.map_for(B, W), b = (I - A) mean,
mean and centred second moment from t25.science.moments (divisor N).  Records are rows.

* fit_U  -- t25.science.unrestricted(W, S, k): W-whitened top-k eigenspace of S (EMPCA/CW-PCA form).
* fit_F  -- t25.science.structured(W, S, starts, max_iter, tol) with t25.science.starts (HOSVD + 3
            random starts, 4 restarts; best training objective selected): NFPA, the T25 recipe.
* cv_select -- 5-fold CV over {U, F}; score = mean held-out observed weighted residual
            (x - A x - b)^T W (x - A x - b) over all N held-out records (= mean of the 5 fold means,
            because every registered N is a multiple of 5); the chosen class is refitted on all N
            (the refit is the full-data fit of that class, identical code and start key).
            Ties (exact equality) go to U.
* nstar_rule -- product iff N < N*_pred(theta); N*_pred = None (theta = 0, a = 0) -> product.
* fit_shr -- top-k eigenspace of K_lambda = (1 - lambda) S_w + lambda NKP(S_w), S_w = W^{1/2} S W^{1/2}
            (whitened centred sample second moment in the supplied metric), mapped back with
            W^{-1/2} exactly as t25.unrestricted; lambda chosen on the lambda grid by the same folds and
            score (ties -> smallest lambda), then refitted on all N.  lambda = 0 is bitwise fit_U.
            lambda = 1 is NOT NFPA: the top-k eigenvectors of A_t (x) A_c are products a_i (x) b_j
            ranked by eigenvalue product, which need not form a k_c x k_t grid.  SHR is a third
            estimator, not an interpolation path between the hard classes.  NKP in whitened
            coordinates is meaningful only when W^{1/2} is itself Kronecker (oracle and Kronecker-
            estimated metrics), which is why SHR is not run on the full-estimate arms.
"""
from __future__ import annotations

import numpy as np

from experiments.mlst2026.x3_nonlinear_metric_20261005 import guard as _guard

_guard.assert_direct()                     # accepted T25 bytes before importing them

from experiments.mlst2026.t25_production_20260930 import science as t25   # noqa: E402
from . import nkp                                                            # noqa: E402


# ------------------------------------------------------------------------------ helpers
def affine(B: np.ndarray, W: np.ndarray, mean: np.ndarray):
    A = t25.map_for(B, W)
    return A, (np.eye(len(A)) - A) @ mean


def heldout_residuals(A, b, xh, W) -> np.ndarray:
    """Per-record observed weighted residual (x - A x - b)^T W (x - A x - b) for rows of xh."""
    R = xh - xh @ A.T - b
    return np.einsum('ni,ij,nj->n', R, W, R)


def folds(N: int, key: tuple, n_folds: int = 5) -> list[np.ndarray]:
    """Deterministic fold assignment: permutation from t25.rng(*key); record perm[i] -> fold i mod n_folds.
    Returns the held-out index arrays (sorted)."""
    perm = t25.rng(*key).permutation(N)
    lab = np.empty(N, int)
    lab[perm] = np.arange(N) % n_folds
    return [np.flatnonzero(lab == j) for j in range(n_folds)]


def _complement(N, idx):
    m = np.ones(N, bool)
    m[idx] = False
    return np.flatnonzero(m)


# ------------------------------------------------------------------------------ hard classes
def _U(mean, S, W, k):
    B = t25.unrestricted(W, S, k)
    A, b = affine(B, W, mean)
    return {'cls': 'U', 'B': B, 'A': A, 'b': b, 'mean': mean}


def _F(mean, S, W, dims, start_key, recipe):
    C, T, kc, kt = dims
    initial = t25.starts(S, C, T, kc, kt, start_key)
    B, diag, factors = t25.structured(W, S, initial, recipe['max_iter'], recipe['gradient_tolerance'])
    A, b = affine(B, W, mean)
    restarts = [{'objective': r['objective'], 'gradient_norm': r['gradient_norm'], 'status': r['status'],
                 'iterations': len(r['trajectory']) - 1, 'accepted_steps': len(r['accepted_steps'])}
                for r in diag['restarts']]
    return {'cls': 'F', 'B': B, 'A': A, 'b': b, 'mean': mean, 'factors': factors,
            'selected': diag['selected'], 'restarts': restarts,
            'selected_stationary': restarts[diag['selected']]['status'] == 'stationary'}


def fit_U(x: np.ndarray, W: np.ndarray, dims: tuple) -> dict:
    _, mean, S = t25.moments(x, len(x))
    return _U(mean, S, W, dims[2] * dims[3])


def fit_F(x: np.ndarray, W: np.ndarray, dims: tuple, start_key: tuple, recipe: dict) -> dict:
    _, mean, S = t25.moments(x, len(x))
    return _F(mean, S, W, dims, start_key, recipe)


# ------------------------------------------------------------------------------ CV select
def cv_select(x, W, dims, fold_idx, recipe, start_key, full=None) -> dict:
    """5-fold CV over {U, F}; returns the chosen full-data fit and diagnostics.  `full` may carry the
    already computed full-data fits {'U': ..., 'F': ...} (same code, same start key)."""
    N = len(x)
    k = dims[2] * dims[3]
    res = {'U': np.empty(N), 'F': np.empty(N)}
    fold_scores = {'U': [], 'F': []}
    fold_F = []
    for j, h in enumerate(fold_idx):
        tr = _complement(N, h)
        _, mean, S = t25.moments(x[tr], len(tr))
        fits = {'U': _U(mean, S, W, k), 'F': _F(mean, S, W, dims, tuple(start_key) + ('cv-fold', j), recipe)}
        for c in ('U', 'F'):
            r = heldout_residuals(fits[c]['A'], fits[c]['b'], x[h], W)
            res[c][h] = r
            fold_scores[c].append(float(r.mean()))
        fold_F.append({'selected': fits['F']['selected'], 'selected_stationary': fits['F']['selected_stationary'],
                       'statuses': [r['status'] for r in fits['F']['restarts']],
                       'iterations': [r['iterations'] for r in fits['F']['restarts']]})
    mean_scores = {c: float(res[c].mean()) for c in ('U', 'F')}
    choice = 'F' if mean_scores['F'] < mean_scores['U'] else 'U'
    if full is None:
        full = {'U': fit_U(x, W, dims)} if choice == 'U' else {'F': fit_F(x, W, dims, start_key, recipe)}
    return {'choice': choice, 'fit': full[choice], 'fold_scores': fold_scores, 'mean_scores': mean_scores,
            'fold_F': fold_F, 'fold_F_nonstationary': int(sum(not f['selected_stationary'] for f in fold_F))}


# ------------------------------------------------------------------------------ N*-rule
def nstar_rule(theta: float, N: int, table: dict) -> dict:
    """table: {repr(float theta): N*_pred or None}.  Product iff N < N*_pred; None (theta = 0) -> product."""
    key = repr(float(theta))
    if key not in table:
        raise KeyError(f'no N*_pred for theta={theta}')
    ns = table[key]
    choice = 'F' if ns is None or N < ns else 'U'
    return {'choice': choice, 'N_star_pred': ns}


# ------------------------------------------------------------------------------ SHR
def _shr_basis(Sw, NK, lam, wi, k):
    if lam == 0:
        K = Sw                              # bitwise the unrestricted computation
    elif lam == 1:
        K = NK
    else:
        K = (1 - lam) * Sw + lam * NK
    return t25.qr(wi @ t25.top(K, k))


def fit_shr(x, W, dims, fold_idx, lambdas, return_path: bool = False) -> dict:
    C, T, kc, kt = dims
    k = kc * kt
    N = len(x)
    lambdas = [float(l) for l in lambdas]
    ws, wi = t25.powers(W)
    res = np.empty((len(lambdas), N))
    fold_scores = [[] for _ in lambdas]
    nkp_folds = []
    for h in fold_idx:
        tr = _complement(N, h)
        _, mean, S = t25.moments(x[tr], len(tr))
        Sw = ws @ S @ ws
        NK, info = nkp.nkp_matrix(Sw, C, T)
        nkp_folds.append(info)
        for i, lam in enumerate(lambdas):
            A, b = affine(_shr_basis(Sw, NK, lam, wi, k), W, mean)
            r = heldout_residuals(A, b, x[h], W)
            res[i, h] = r
            fold_scores[i].append(float(r.mean()))
    mean_scores = res.mean(1)
    i_hat = int(np.argmin(mean_scores))                     # first minimum = smallest lambda on ties
    _, mean, S = t25.moments(x, N)
    Sw = ws @ S @ ws
    NK, info = nkp.nkp_matrix(Sw, C, T)
    path = {}
    todo = range(len(lambdas)) if return_path else [i_hat]
    for i in todo:
        B = _shr_basis(Sw, NK, lambdas[i], wi, k)
        A, b = affine(B, W, mean)
        path[i] = {'cls': 'SHR', 'lambda': lambdas[i], 'B': B, 'A': A, 'b': b, 'mean': mean}
    return {'lambda_hat': lambdas[i_hat], 'index_hat': i_hat, 'fit': path[i_hat], 'lambdas': lambdas,
            'fold_scores': fold_scores, 'mean_scores': [float(v) for v in mean_scores],
            'nkp_full': info, 'nkp_folds': nkp_folds, 'path': path if return_path else None}
