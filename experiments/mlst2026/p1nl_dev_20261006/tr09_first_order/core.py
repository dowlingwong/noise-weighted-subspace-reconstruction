"""TR09 first-order (local, leading-order in 1/N) fitted excess of the product and unrestricted classes
under second-moment-matched sparse latents.  Development computation; see
MS plan/onthefly/TR09_FIRST_ORDER_NOTE.md for the derivation, assumptions and limits.

Everything is in TRUE-whitened coordinates under the oracle metric.  T25's physical product class
{range(U_t (x) U_c)} is mapped onto the whitened product class by the separable L^{-1}, and the
Sigma^{-1}-orthogonal projector becomes the Euclidean orthogonal projector, so the oracle-arm risks are
reproduced exactly here.

Notation: M = K + I (population whitened data covariance), eigenpairs (ell_a = mu_a + 1, v_a);
f_S(P) = tr(P S); product projectors P = P_t (x) P_c (np.kron(Pt, Pc), Fortran vec c + C t).

Graph chart at (U_c, U_t): P_c(X) = B (B^T B)^{-1} B^T with B = U_c + U_c_perp X, X of shape
(C - k_c, k_c), and likewise P_t(Y).  To second order
    P_c(X) = P_c + D1(X) + D2(X),  D1 = Up X U^T + U X^T Up^T,  D2 = Up X X^T Up^T - U X^T X U^T.
Coordinates xi = (vec X, vec Y) (column-major vec).  For a symmetric S:
    gradient g_S(xi=0) and Hessian of f_S in xi, computed from the partial traces
    A_c(S) = tr_t[(P_t (x) I) S],  A_t(S) = tr_c[(I (x) P_c) S].
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# basic helpers
# --------------------------------------------------------------------------------------
def orth(B):
    q, r = np.linalg.qr(B)
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)


def complement(U):
    q, _ = np.linalg.qr(np.concatenate([U, np.eye(len(U))], 1))
    return q[:, U.shape[1]:len(U)]


def proj(U):
    return U @ U.T


def graph_projector(U, Up, X):
    B = U + Up @ X
    return B @ np.linalg.solve(B.T @ B, B.T)


def reshape4(S, C, T):
    """S4[c, t, d, s] = S[c + C t, d + C s]."""
    return S.reshape(C, T, C, T, order='F')


def product_value(S4, Pt, Pc):
    return float(np.einsum('ts,cd,ctds->', Pt, Pc, S4))


def partial_c(S4, Pt):
    return np.einsum('ts,ctds->cd', Pt, S4)


def partial_t(S4, Pc):
    return np.einsum('cd,ctds->ts', Pc, S4)


# --------------------------------------------------------------------------------------
# local model of f_S at a product point
# --------------------------------------------------------------------------------------
class ProductPoint:
    def __init__(self, Uc, Ut):
        self.Uc, self.Ut = orth(Uc), orth(Ut)
        self.Ucp, self.Utp = complement(self.Uc), complement(self.Ut)
        self.C, self.kc = self.Uc.shape
        self.T, self.kt = self.Ut.shape
        self.nc = (self.C - self.kc) * self.kc
        self.nt = (self.T - self.kt) * self.kt
        self.n = self.nc + self.nt

    @property
    def Pc(self):
        return proj(self.Uc)

    @property
    def Pt(self):
        return proj(self.Ut)

    def P(self):
        return np.kron(self.Pt, self.Pc)

    def split(self, xi):
        X = xi[:self.nc].reshape(self.C - self.kc, self.kc, order='F')
        Y = xi[self.nc:].reshape(self.T - self.kt, self.kt, order='F')
        return X, Y

    def exact(self, S4, xi):
        """f_S at chart coordinates xi (exact, no expansion)."""
        X, Y = self.split(xi)
        return product_value(S4, graph_projector(self.Ut, self.Utp, Y), graph_projector(self.Uc, self.Ucp, X))

    def moved(self, xi):
        X, Y = self.split(xi)
        return ProductPoint(orth(self.Uc + self.Ucp @ X), orth(self.Ut + self.Utp @ Y))

    def gradient(self, S4):
        Ac, At = partial_c(S4, self.Pt), partial_t(S4, self.Pc)
        gc = 2 * self.Ucp.T @ Ac @ self.Uc
        gt = 2 * self.Utp.T @ At @ self.Ut
        return np.concatenate([gc.ravel(order='F'), gt.ravel(order='F')])

    def hessian(self, S4):
        """Second derivative matrix of f_S in xi at xi = 0 (symmetric, n x n)."""
        Ac, At = partial_c(S4, self.Pt), partial_t(S4, self.Pc)
        Hcc = 2 * (np.kron(np.eye(self.kc), self.Ucp.T @ Ac @ self.Ucp) - np.kron(self.Uc.T @ Ac @ self.Uc, np.eye(self.C - self.kc)))
        Htt = 2 * (np.kron(np.eye(self.kt), self.Utp.T @ At @ self.Utp) - np.kron(self.Ut.T @ At @ self.Ut, np.eye(self.T - self.kt)))
        # cross block: d^2 f / dX[i,j] dY[u,v] = tr((D1t(E_uv) (x) D1c(E_ij)) S)
        Qc = np.concatenate([self.Uc, self.Ucp], 1)
        Qt = np.concatenate([self.Ut, self.Utp], 1)
        R = np.einsum('ctds,ca,tb,de,sf->abef', S4, Qc, Qt, Qc, Qt, optimize=True)   # rotated tensor
        kc, kt = self.kc, self.kt
        par_c, per_c = slice(0, kc), slice(kc, self.C)
        par_t, per_t = slice(0, kt), slice(kt, self.T)
        # tr((A (x) B) S) = sum A[t,s] B[c,d] R[d,s,c,t] (rotated coordinates; R symmetric under (ab)<->(ef))
        # B = E_{i,j} + E_{j,i} with i in perp_c (index into X rows), j in par_c; A = E_{u,v} + E_{v,u}, u perp_t, v par_t
        T1 = R[par_c, par_t, per_c, per_t]      # R[d=j, s=v, c=i, t=u]  -> [j, v, i, u]
        T2 = R[per_c, par_t, par_c, per_t]      # R[d=i, s=v, c=j, t=u]  -> [i, v, j, u]
        T3 = R[par_c, per_t, per_c, par_t]      # R[d=j, s=u, c=i, t=v]  -> [j, u, i, v]
        T4 = R[per_c, per_t, par_c, par_t]      # R[d=i, s=u, c=j, t=v]  -> [i, u, j, v]
        cross = (T1.transpose(2, 0, 3, 1) + T2.transpose(0, 2, 3, 1)
                 + T3.transpose(2, 0, 1, 3) + T4.transpose(0, 2, 1, 3))           # [i, j, u, v]
        # row i + rc j (column-major vec of X), column u + rt v (column-major vec of Y)
        Hct = cross.transpose(1, 0, 3, 2).reshape(self.nc, self.nt)
        H = np.zeros((self.n, self.n))
        H[:self.nc, :self.nc] = Hcc
        H[self.nc:, self.nc:] = Htt
        H[:self.nc, self.nc:] = Hct
        H[self.nc:, :self.nc] = Hct.T
        return H

    def tangent_matrices(self):
        """G_alpha = dP/dxi_alpha at xi = 0 (n x D x D), for unit coordinate directions."""
        D = self.C * self.T
        G = np.empty((self.n, D, D))
        Pc, Pt = self.Pc, self.Pt
        for a in range(self.n):
            e = np.zeros(self.n)
            e[a] = 1.0
            X, Y = self.split(e)
            D1c = self.Ucp @ X @ self.Uc.T
            D1c = D1c + D1c.T
            D1t = self.Utp @ Y @ self.Ut.T
            D1t = D1t + D1t.T
            G[a] = np.kron(Pt, D1c) + np.kron(D1t, Pc)
        return G


# --------------------------------------------------------------------------------------
# maximisation (Newton in the graph chart, with backtracking on the exact objective)
# --------------------------------------------------------------------------------------
def refine(S, C, T, point, tol=1e-11, max_iter=200):
    """Local maximiser of f_S over product projectors, started at ``point``.  Returns
    (point, info).  Uses Newton steps when -Hessian is positive definite, gradient steps otherwise."""
    S4 = reshape4(S, C, T)
    scale = float(np.trace(S))
    info = {'iterations': 0, 'status': 'iteration_limit', 'newton_steps': 0}
    for it in range(max_iter):
        g = point.gradient(S4)
        gn = float(np.linalg.norm(g)) / scale
        info.update(iterations=it, gradient_norm=gn)
        if gn <= tol:
            info['status'] = 'stationary'
            break
        H = -point.hessian(S4)
        f0 = point.exact(S4, np.zeros(point.n))
        ev = np.linalg.eigvalsh(H)
        if ev[0] > 1e-12 * max(abs(ev[-1]), 1.0):
            step = np.linalg.solve(H, g)
            info['newton_steps'] += 1
        else:                                   # shifted (Levenberg) step away from saddles
            step = np.linalg.solve(H + (abs(ev[0]) + 1e-3 * max(abs(ev[-1]), 1.0)) * np.eye(len(g)), g)
        t = 1.0
        for _ in range(40):
            if point.exact(S4, t * step) >= f0 - 1e-15 * scale:
                break
            t *= 0.5
        point = point.moved(t * step)
    H = -point.hessian(S4)
    info['min_eig_negH'] = float(np.linalg.eigvalsh(H)[0])
    info['value'] = point.exact(S4, np.zeros(point.n))
    return point, info


def product_starts(S, C, T, kc, kt, gen, n_random=8, extra=()):
    S4 = reshape4(S, C, T)
    top = lambda A, k: np.linalg.eigh((A + A.T) / 2)[1][:, -k:]      # noqa: E731
    starts = [('hosvd', top(np.einsum('ctdt->cd', S4), kc), top(np.einsum('ctcs->ts', S4), kt))]
    starts += [(name, Uc, Ut) for name, Uc, Ut in extra]
    for i in range(n_random):
        starts.append((f'random{i}', orth(gen.normal(size=(C, kc))), orth(gen.normal(size=(T, kt)))))
    return starts


def population_product_optimum(model, gen, n_random=8):
    """Best product maximiser of tr(P M) found from HOSVD, the theta = 0 product factors and random starts
    (an upper bound on the approximation cost a; never a certificate).  Reports every distinct local
    maximum so that non-uniqueness is visible."""
    M, C, T, kc, kt = model.M, model.C, model.T, model.kc, model.kt
    # theta = 0 product factors of the geometry (exact eigenspace at theta = 0)
    extra = [('theta0', model.g.U_c0, model.g.U_t0)]
    found = []
    for name, Uc, Ut in product_starts(M, C, T, kc, kt, gen, n_random, extra):
        pt, info = refine(M, C, T, ProductPoint(Uc, Ut))
        found.append((name, pt, info))
    vals = np.array([f[2]['value'] for f in found])
    best = int(np.argmax(vals))
    Pbest = found[best][1].P()
    distinct = []
    for name, pt, info in found:
        dist = float(np.linalg.norm(pt.P() - Pbest, 'fro'))
        distinct.append({'start': name, 'value': info['value'], 'gap_to_best': float(vals[best] - info['value']),
                         'projector_distance_to_best': dist, 'status': info['status'],
                         'min_eig_negH': info['min_eig_negH'], 'gradient_norm': info['gradient_norm']})
    return found[best][1], distinct


# --------------------------------------------------------------------------------------
# leading-order coefficients
# --------------------------------------------------------------------------------------
def unrestricted_coefficient(ell, k):
    """c_U = sum_{i<k<=j} ell_i ell_j / (ell_i - ell_j): N E[Delta_U] -> c_U (any latent law with
    independent eigen-coordinates; Gaussian or not)."""
    i, j = ell[:k, None], ell[None, k:]
    return float(np.sum(i * j / (i - j)))


def first_order(model, point):
    """Leading-order coefficients at the population product maximiser ``point``.

    Returns H (= minus the Hessian of f_M), the Gaussian coefficient c_F = 1/2 tr(H^-1 Sigma_g) with
    Sigma_g[a, b] = 2 tr(G_a M G_b M), the geometry weights w_a = 1/2 h_a^T H^-1 h_a with
    h_a[alpha] = v_a^T G_alpha v_a, and the gradient norm at the point (a stationarity check)."""
    M, C, T = model.M, model.C, model.T
    M4 = reshape4(M, C, T)
    grad = point.gradient(M4)
    H = -point.hessian(M4)
    eig = np.linalg.eigvalsh(H)
    G = point.tangent_matrices()                       # n x D x D (standard coordinates)
    V = model.V
    Gt = np.einsum('ai,nab,bj->nij', V, G, V, optimize=True)       # eigenbasis
    ell = model.ell
    W = np.sqrt(np.outer(ell, ell))
    F = (Gt * W[None]).reshape(point.n, -1)
    Sigma_g = 2 * F @ F.T
    Hinv = np.linalg.inv(H)
    h = np.einsum('naa->na', Gt)                       # n x D
    w = 0.5 * np.einsum('na,nm,ma->a', h, Hinv, h)
    cF = 0.5 * float(np.trace(Hinv @ Sigma_g))
    return {'H': H, 'H_min_eig': float(eig[0]), 'H_max_eig': float(eig[-1]), 'gradient_norm': float(np.linalg.norm(grad)),
            'cF_gaussian': cF, 'w': w, 'Sigma_g': Sigma_g, 'h': h, 'Hinv': Hinv}


def kurtosis_term(model, w):
    """K(p) = sum_a lambda_a^2 (kappa_a - 3) w_a; for Bernoulli-Gaussian sparse latents
    lambda^2 (kappa - 3) = 3 lambda^2 (1/p - 1)."""
    return float(np.sum(model.mu ** 2 * (model.kappa - 3.0) * w))


def summary(model, point, fo):
    k = model.k
    PU = proj(model.V[:, :k])
    PF = point.P()
    a = float(np.trace(PU @ model.M) - np.trace(PF @ model.M))
    cU = unrestricted_coefficient(model.ell, k)
    Kp = kurtosis_term(model, fo['w'])
    off_U = float(np.trace((np.eye(model.d) - PU) @ model.M))
    off_F = float(np.trace((np.eye(model.d) - PF) @ model.M))
    # E D(N) = a + B/N + o(1/N),  B = (cF + K) - cU + (off_F - off_U) = cF + K - cU + a
    B = fo['cF_gaussian'] + Kp - cU + (off_F - off_U)
    out = {'theta': model.theta, 'p': model.p, 'a': a, 'cU': cU, 'cF_gaussian': fo['cF_gaussian'], 'K': Kp,
           'offset_difference': off_F - off_U, 'B': B,
           'N_star_leading_order': (-B / a) if (a > 0 and B < 0) else None,
           'w_sparse': [float(x) for x in fo['w'][:model.n_sparse]],
           'w_tail_sum': float(np.sum(fo['w'][model.n_sparse:])),
           'H_min_eig': fo['H_min_eig'], 'gradient_norm_at_optimum': fo['gradient_norm']}
    return out


def unrestricted_kurtosis_second_order(model):
    """PARTIAL second-order (1/N^2) kurtosis coefficient of the unrestricted excess: the eigenvalue
    renormalisation terms only.  Expanding Delta_U ~ sum_{i in R, j in D} E_ij^2 g / (g + delta)^2 with
    g = ell_i - ell_j and delta = E_ii - E_jj, to second order in delta, and keeping the kappa-dependent
    parts of E[E_ij^2 delta] and E[E_ij^2] E[delta^2] gives

        d_kappa = sum_{i in R, j in D} lambda_i^2 (kappa_i - 3) ell_j / g^2 * (3 ell_i / g - 2)   (+ i <-> j terms
        for sparse j; none here, the discarded coordinates are Gaussian).

    Not a complete second-order expansion (terms coupling three or more coordinates, including rotations
    inside the retained block, are omitted); the TR09 note compares it with the development Monte Carlo."""
    ell, lam, kap, k = model.ell, model.mu, model.kappa, model.k
    i = np.arange(k)[:, None]
    j = np.arange(k, model.d)[None, :]
    g = ell[i] - ell[j]
    term = lam[i] ** 2 * (kap[i] - 3.0) * ell[j] / g ** 2 * (3.0 * ell[i] / g - 2.0)
    return float(term.sum())


def unrestricted_coefficient_general(Aeig, kappa, ell, k):
    """Leading-order c_U when the latents are independent along the columns of a square root s = V Aeig z
    (Aeig Aeig^T = diag(mu), z independent with E z^4 = kappa).  In K's eigenbasis the coordinates are then
    uncorrelated but not independent, and Var(E_ij) = [ell_i ell_j + sum_m (kappa_m - 3) Aeig_im^2 Aeig_jm^2] / N,
    so the leading-order unrestricted excess depends on the fourth moments unless Aeig is diagonal
    (latents along the eigenbasis)."""
    i, j = slice(0, k), slice(k, len(ell))
    cum = np.einsum('m,im,jm->ij', kappa - 3.0, Aeig[i] ** 2, Aeig[j] ** 2)
    num = np.outer(ell[i], ell[j]) + cum
    return float(np.sum(num / (ell[i][:, None] - ell[j][None, :])))
