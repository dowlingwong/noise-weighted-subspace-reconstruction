"""D3 population (pre-fit, data-free given Sigma_ref and the signal model) quantities, certificate and N*_pred.

For one subject and theta: R_s = signal second moment (physical, rank k = 6), Sigma_ref the reference metric
(Ledoit-Wolf full estimate of the run 1-2 noise; review F1/(a)), K = Sigma_ref^{-1/2} R_s Sigma_ref^{-1/2},
M = K + I (the data covariance in Sigma_ref-whitened coordinates if the measured noise has covariance Sigma_ref),
ell = eig(M).  The fitted product class is NFPA in measurement coordinates under the metric Sigma_ref^{-1} (as
T25): in whitened coordinates the subspaces range(A (U_t (x) U_c)), A = Sigma_ref^{-1/2}.

* **general path** (default): a_best by Newton in the graph chart of the physical factors (HOSVD + theta = 0
  factors + random starts), the Hessian by complex-step differentiation of the analytic gradient,
  Sigma_g = 2 tr(G_a M G_b M) by a low-rank formula (G_a = Z_a Q^T + Q Z_a^T), c_F = 1/2 tr(H^{-1} Sigma_g)
  (chart-invariant at a nondegenerate maximum), c_U by the pinned ``core.unrestricted_coefficient``;
  B = c_F - c_U + a, N* = (c_U - c_F)/a - 1 (Supplement S4.3).  For a Kronecker Sigma_ref it reproduces the
  pinned Y19/TR09 path to <= 3e-12 (tests).
* **certificate B_dagger** (``b_dagger``): the paper's partial-trace certificate is defined for the whitened
  product class; for a non-separable Sigma_ref it is replaced by the metric-corrected Kronecker-whitened bound
  of the independent review (proof sketch in ``b_dagger``).  Exactly 0 at theta = 0.
* **pinned path** (``pinned_row``): ``y19_20261012.predictions.constants`` unchanged, used only for the
  Kronecker-reference sensitivity (``--crosscheck``); TR09 ``first_order`` holds n x d x d tensors.
* **theta_s rule** (``theta_for_target``): per subject, the population-only 1-D root search that places
  N*_pred,s(theta) at the frozen targets N*_A = floor(P/5), N*_B = max(5 n_min, N*_A / 2) (review (c)).

Commands (EXP root, PYTHONPATH=src):
    python -m experiments.mlst2026.meg_z_20261013.population --derived DIR --subjects sub-01,sub-02,sub-03,sub-04
        --out OUT/predictions_dev.json [--protocol PATH] [--gates OUT/gates.json] [--jobs 4] [--crosscheck]
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import common, estimators, signal
from .common import symmetric
from experiments.mlst2026.y19_20261012 import predictions as y19pred          # runs the T25 byte guard first
from experiments.mlst2026.p1nl_dev_20261006.tr09_first_order import core     # noqa: E402
from experiments.mlst2026.successor_20260913.t20s.generator import partial_trace_cap   # noqa: E402

A_ZERO_TOL = 1e-9


def spd_powers(a: np.ndarray):
    w, q = np.linalg.eigh(symmetric(a))
    if w[0] <= 0 or not np.isfinite(w).all():
        raise np.linalg.LinAlgError('matrix not SPD')
    return symmetric((q * np.sqrt(w)) @ q.T), symmetric((q / np.sqrt(w)) @ q.T)


def round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


# ----------------------------------------------------------------------------------------------- pinned path
class KroneckerPopulationModel:
    """Duck-typed model for TR09 ``core`` and Y19 ``predictions.constants`` (Gaussian latents, p = 1)."""

    def __init__(self, K, C, T, kc, kt, theta, Uc0, Ut0):
        w, V = np.linalg.eigh(symmetric(K))
        order = np.argsort(w)[::-1]
        self.mu = w[order].copy()
        self.V = signal.fix_signs(V[:, order])
        self.C, self.T, self.kc, self.kt = int(C), int(T), int(kc), int(kt)
        self.k, self.d = self.kc * self.kt, self.C * self.T
        self.theta, self.p, self.n_sparse = float(theta), 1.0, self.k
        Kr = (self.V * self.mu) @ self.V.T
        self.K = symmetric(Kr)
        self.M = self.K + np.eye(self.d)
        self.ell = self.mu + 1.0
        self.g = SimpleNamespace(U_c0=Uc0, U_t0=Ut0)

    @property
    def kappa(self) -> np.ndarray:
        return np.full(self.d, 3.0)


def pinned_row(R_s, St, Sc, A_templates, B1, theta, kc, kt, gen, n_random) -> dict:
    C, T = Sc.shape[0], St.shape[0]
    _, ict = spd_powers(St)
    _, icc = spd_powers(Sc)
    Li = np.kron(ict, icc)                                   # Sigma_ref^{-1/2}, separable
    K = symmetric(Li @ R_s @ Li)
    model = KroneckerPopulationModel(K, C, T, kc, kt, theta, signal.orth(icc @ A_templates), signal.orth(ict @ B1))
    row = y19pred.constants(model, gen, n_random)
    row['path'] = 'pinned: y19_20261012.predictions.constants (TR09 core) in Kronecker-whitened coordinates'
    row['theta'] = float(theta)
    return row


# ----------------------------------------------------------------------------------------------- general path
class GeneralProductClass:
    """Physical product subspaces range(U_t (x) U_c) mapped by A into whitened coordinates; objective tr(P M)."""

    def __init__(self, A, M, C, T, kc, kt):
        self.A, self.M = np.asarray(A, float), np.asarray(M, float)
        self.C, self.T, self.kc, self.kt = int(C), int(T), int(kc), int(kt)
        self.k = self.kc * self.kt
        self.nc, self.nt = (self.C - self.kc) * self.kc, (self.T - self.kt) * self.kt
        self.n = self.nc + self.nt

    def chart(self, Uc, Ut):
        Uc, Ut = core.orth(Uc), core.orth(Ut)
        return SimpleNamespace(Uc=Uc, Ut=Ut, Ucp=core.complement(Uc), Utp=core.complement(Ut))

    def split(self, xi):
        X = xi[:self.nc].reshape(self.C - self.kc, self.kc, order='F')
        Y = xi[self.nc:].reshape(self.T - self.kt, self.kt, order='F')
        return X, Y

    def basis(self, pt, xi=None):
        if xi is None:
            return self.A @ np.kron(pt.Ut, pt.Uc)
        X, Y = self.split(xi)
        return self.A @ np.kron(pt.Ut + pt.Utp @ Y, pt.Uc + pt.Ucp @ X)

    def value(self, pt, xi=None) -> float:
        B = self.basis(pt, xi)
        return float(np.trace(np.linalg.solve(B.T @ B, B.T @ self.M @ B)))

    def _grad(self, pt, xi):
        """Analytic chart gradient of f = tr((B^T B)^{-1} B^T M B); plain transposes only (complex-step safe)."""
        X, Y = self.split(xi)
        Ucx, Uty = pt.Uc + pt.Ucp @ X, pt.Ut + pt.Utp @ Y
        B = self.A @ np.kron(Uty, Ucx)
        Gi = np.linalg.inv(B.T @ B)
        MB = self.M @ B
        Gamma = MB @ Gi - B @ (Gi @ (B.T @ MB) @ Gi)          # (I - P) M B G^{-1}
        Phi4 = (self.A.T @ Gamma).reshape(self.C, self.T, self.kc, self.kt, order='F')
        gc = pt.Ucp.T @ np.einsum('ctab,tb->ca', Phi4, Uty)
        gt = pt.Utp.T @ np.einsum('ctab,ca->tb', Phi4, Ucx)
        return 2.0 * np.concatenate([gc.ravel(order='F'), gt.ravel(order='F')])

    def gradient(self, pt) -> np.ndarray:
        return np.real(self._grad(pt, np.zeros(self.n)))

    def hessian(self, pt, h: float = 1e-20) -> np.ndarray:
        H = np.empty((self.n, self.n))
        for b in range(self.n):
            e = np.zeros(self.n, complex)
            e[b] = 1j * h
            H[:, b] = np.imag(self._grad(pt, e)) / h
        return symmetric(H)

    def moved(self, pt, xi):
        X, Y = self.split(xi)
        return self.chart(pt.Uc + pt.Ucp @ X, pt.Ut + pt.Utp @ Y)

    def refine(self, pt, tol: float = 1e-11, max_iter: int = 200):
        """Newton ascent in the graph chart with backtracking on the exact objective (as TR09 ``refine``)."""
        scale = float(np.trace(self.M))
        info = {'iterations': 0, 'status': 'iteration_limit', 'newton_steps': 0}
        for it in range(max_iter):
            g = self.gradient(pt)
            gn = float(np.linalg.norm(g)) / scale
            info.update(iterations=it, gradient_norm=gn)
            if gn <= tol:
                info['status'] = 'stationary'
                break
            Hn = -self.hessian(pt)
            f0 = self.value(pt)
            ev = np.linalg.eigvalsh(Hn)
            if ev[0] > 1e-12 * max(abs(ev[-1]), 1.0):
                step = np.linalg.solve(Hn, g)
                info['newton_steps'] += 1
            else:
                step = np.linalg.solve(Hn + (abs(ev[0]) + 1e-3 * max(abs(ev[-1]), 1.0)) * np.eye(self.n), g)
            t = 1.0
            for _ in range(40):
                if self.value(pt, t * step) >= f0 - 1e-15 * scale:
                    break
                t *= 0.5
            pt = self.moved(pt, t * step)
        info['min_eig_negH'] = float(np.linalg.eigvalsh(-self.hessian(pt))[0])
        info['value'] = self.value(pt)
        return pt, info

    def tangents(self, pt):
        """Z_a = (I - Q Q^T) dB_a R^{-1} (n x d x k) with B = A (U_t (x) U_c) = Q R, so that G_a = Z_a Q^T + Q Z_a^T."""
        B = self.basis(pt)
        Q, R = np.linalg.qr(B)
        Ri = np.linalg.inv(R)
        Z = np.empty((self.n, B.shape[0], self.k))
        for a in range(self.n):
            e = np.zeros(self.n)
            e[a] = 1.0
            X, Y = self.split(e)
            dB = self.A @ (np.kron(pt.Ut, pt.Ucp @ X) + np.kron(pt.Utp @ Y, pt.Uc))
            dB = dB - Q @ (Q.T @ dB)
            Z[a] = dB @ Ri
        return Q, Z

    def first_order(self, pt) -> dict:
        H = -self.hessian(pt)
        Q, Z = self.tangents(pt)
        M = self.M
        MQ = M @ Q
        Qm = Q.T @ MQ
        MZ = np.einsum('de,aek->adk', M, Z, optimize=True)
        N = np.einsum('dk,adj->akj', Q, MZ, optimize=True)                    # N_a = Q^T M Z_a
        t1 = np.einsum('aij,bji->ab', N, N, optimize=True)                    # tr(N_a N_b)
        P = np.einsum('bdi,adj->abij', Z, MZ, optimize=True)                  # Z_b^T M Z_a
        t2 = np.einsum('ji,abij->ab', Qm, P, optimize=True)                   # tr(Qm Z_b^T M Z_a)
        Sg = symmetric(2.0 * (2.0 * t1 + 2.0 * t2))
        ev = np.linalg.eigvalsh(H)
        Hinv = np.linalg.inv(H)
        return {'H': H, 'Hinv': Hinv, 'Sigma_g': Sg, 'H_min_eig': float(ev[0]), 'H_max_eig': float(ev[-1]),
                'gradient_norm': float(np.linalg.norm(self.gradient(pt))), 'cF_gaussian': 0.5 * float(np.trace(Hinv @ Sg))}


def general_row(R_s, ref_cov, C, T, kc, kt, theta, A_templates, B1, gen, n_random, B_cert=None) -> dict:
    """S4.3 quantities for the physical product class under the (possibly non-separable) metric ref_cov^{-1}."""
    _, A = spd_powers(ref_cov)
    K = symmetric(A @ R_s @ A)
    d, k = C * T, kc * kt
    w = np.sort(np.linalg.eigvalsh(K))[::-1]
    M = K + np.eye(d)
    cls = GeneralProductClass(A, M, C, T, kc, kt)
    Sphys = symmetric(R_s + ref_cov)
    S4 = Sphys.reshape(C, T, C, T, order='F')
    top = lambda X, kk: np.linalg.eigh(symmetric(X))[1][:, -kk:]          # noqa: E731
    starts = [('hosvd', top(np.einsum('ctdt->cd', S4), kc), top(np.einsum('ctcs->ts', S4), kt)),
              ('theta0', A_templates, B1)]
    for i in range(n_random):
        starts.append((f'random{i}', core.orth(gen.normal(size=(C, kc))), core.orth(gen.normal(size=(T, kt)))))
    found = [(name, *cls.refine(cls.chart(Uc, Ut))) for name, Uc, Ut in starts]
    vals = np.array([f[2]['value'] for f in found])
    best = int(np.argmax(vals))
    pt = found[best][1]
    fo = cls.first_order(pt)
    a = float(np.sum(w[:k]) + k - vals[best])                              # tr(P_U M) - tr(P_F M)
    ell = w + 1.0
    cU = core.unrestricted_coefficient(ell, k)
    cF = fo['cF_gaussian']
    B = cF - cU + a
    nonsep = a > A_ZERO_TOL
    row = {'theta': float(theta), 'a_best': a, 'cU': cU, 'cF': cF, 'B': B, 'B_cert': B_cert,
           'Delta_k': float(w[k - 1] - w[k]), 'trace_K': float(np.sum(w)),
           'H_min_eig': fo['H_min_eig'], 'H_max_eig': fo['H_max_eig'], 'gradient_norm_at_optimum': fo['gradient_norm'],
           'n_starts': len(found), 'distinct_population_maxima': sorted({round(float(v), 9) for v in vals}),
           'all_starts_reach_best': bool(np.max(vals[best] - vals) < 1e-9),
           'starts_reaching_best': int(np.sum(vals[best] - vals < 1e-9)),
           'start_statuses': sorted({f[2]['status'] for f in found}),
           'path': 'general: physical product class under a non-separable metric (complex-step Hessian)'}
    row['N_star_pred'] = (cU - cF) / a - 1 if nonsep and B < 0 else None
    row['N_star_interval'] = ([(cU - cF) / a - 1, (cU - cF) / B_cert - 1]
                              if nonsep and B < 0 and B_cert is not None and B_cert > A_ZERO_TOL else None)
    row['checks'] = {'cU_gt_cF': bool(cU > cF), 'a_pos': bool(nonsep), 'B_neg': bool(B < 0),
                     'H_pos_def': bool(fo['H_min_eig'] > 0), 'gap_pos': bool(row['Delta_k'] > 0)}
    return row


# ----------------------------------------------------------------------------------------------- certificate
def b_dagger(R_s, ref_cov, St, Sc, kc, kt, zero_tol: float = 1e-10) -> dict:
    """Metric-corrected Kronecker-whitened certificate (independent review, recommendation (a)):

        B_dagger = lambda_min(Sigma_K^{1/2} Sigma_ref^{-1} Sigma_K^{1/2}) * B_cert(K_G),
        K_G = Sigma_K^{-1/2} R_s Sigma_K^{-1/2},  Sigma_K = S_t (x) S_c (flip-flop fit of the same run 1-2 data).

    Proof sketch.  rank R_s = k, so the unrestricted population optimum has zero approximation risk in any metric
    and a = min over physical product subspaces S of R_W(S), W = Sigma_ref^{-1}, where R_W(S) = E min_{s in S}
    ||x_sig - s||_W^2 (the W-orthogonal projection is the W-nearest point of S).  For every vector z,
    ||z||_W^2 = z^T Sigma_ref^{-1} z >= lambda z^T Sigma_K^{-1} z with lambda = lambda_min(Sigma_K^{1/2} Sigma_ref^{-1}
    Sigma_K^{1/2}), hence R_W(S) >= lambda R_{Sigma_K^{-1}}(S).  In the coordinates y = Sigma_K^{-1/2} x the
    subspace Sigma_K^{-1/2} S is a (whitened) product subspace because Sigma_K^{-1/2} = S_t^{-1/2} (x) S_c^{-1/2}
    is separable, and R_{Sigma_K^{-1}}(S) = tr K_G - tr(P K_G) with P its orthogonal projector.  The paper's
    Proposition (03_restriction.tex, eq. mlst-restriction-bound) applied to K_G (whose top-k risk is 0) gives
    tr K_G - tr(P K_G) >= B_cert(K_G).  So a >= lambda B_cert(K_G) = B_dagger.  At theta = 0 the signal span is
    the physical product span(B_1) (x) span{a_1, a_2}, so the top-k range of K_G is a product subspace and
    B_cert(K_G) = 0 (eq. mlst-zero-bound); floating-point residues |B_cert(K_G)| <= zero_tol x max(1, tr K_G)
    (scale-relative cut, delta review D5) are set to exactly 0; the raw value is stored as ``B_cert_KG_raw``.  For Sigma_ref = Sigma_K, lambda = 1 and B_dagger is the paper's
    certificate."""
    ct, ict = spd_powers(St)
    cc, icc = spd_powers(Sc)
    half, ihalf = np.kron(ct, cc), np.kron(ict, icc)
    lam = float(np.linalg.eigvalsh(symmetric(half @ np.linalg.inv(symmetric(ref_cov)) @ half))[0])
    m = KroneckerPopulationModel(symmetric(ihalf @ R_s @ ihalf), Sc.shape[0], St.shape[0], kc, kt, 0.0, None, None)
    cert = y19pred.certificate(m)
    raw = float(cert['B_cert'])
    cut = zero_tol * max(1.0, float(np.trace(m.K)))
    bc = 0.0 if abs(raw) <= cut else raw
    return {'B_dagger': lam * bc, 'lambda_min': lam, 'B_cert_KG': bc, 'B_cert_KG_raw': raw, 'zero_cut': cut,
            'Delta_k_KG': cert['Delta_k'],
            'partial_trace_KG': cert['partial_trace']}


def kron_certificate(R_s, St, Sc, kc, kt) -> dict:
    """The paper's certificate (Y19 ``certificate``) in Kronecker-whitened coordinates."""
    _, ict = spd_powers(St)
    _, icc = spd_powers(Sc)
    Li = np.kron(ict, icc)
    m = KroneckerPopulationModel(symmetric(Li @ R_s @ Li), Sc.shape[0], St.shape[0], kc, kt, 0.0, None, None)
    return y19pred.certificate(m)


# ----------------------------------------------------------------------------------------------- per subject
def reference_for(sd, proto: dict) -> dict:
    rm = proto['reference_metric']
    return estimators.reference_metric(sd.ref, sd.C, sd.T, rm['kind'], rm.get('flipflop'))


def kronecker_fit(sd, proto: dict) -> tuple[np.ndarray, np.ndarray]:
    St, Sc, _ = estimators.kronecker_flipflop_from_S(sd.ref.S, sd.C, sd.T, **(proto['reference_metric'].get('flipflop') or {}))
    return St, Sc


def restart_agreement(row: dict, fraction: float) -> bool:
    return bool(row['all_starts_reach_best'] or row['starts_reaching_best'] >= fraction * row['n_starts'])


class SubjectPopulation:
    """Population quantities of one subject at any theta (fixed latent-variance scale, general path)."""

    def __init__(self, sd, proto: dict, namespace: str):
        self.sd, self.proto, self.ns = sd, proto, namespace
        self.sc, self.pc = proto['signal'], proto['population']
        self.kc, self.kt = int(self.sc['kc']), int(self.sc['kt'])
        self.ref = reference_for(sd, proto)
        self.St, self.Sc = (self.ref['St'], self.ref['Sc']) if 'St' in self.ref else kronecker_fit(sd, proto)

    def row(self, theta: float, n_random: int, stream: str, crosscheck: bool = False) -> dict:
        m, info = signal.build(self.sd.evoked, self.sd.sfreq, self.ref['cov'], theta, self.sc)
        R_s = m.second_moment()
        bd = b_dagger(R_s, self.ref['cov'], self.St, self.Sc, self.kc, self.kt)
        r = general_row(R_s, self.ref['cov'], self.sd.C, self.sd.T, self.kc, self.kt, theta, m.A, m.B1,
                        common.rng(self.ns, self.sd.subject, stream, float(theta)), n_random, B_cert=bd['B_dagger'])
        r['B_dagger'] = bd
        r['restart_agreement'] = restart_agreement(r, float(self.pc.get('restart_agreement_fraction', 0.75)))
        r['signal'] = info
        if crosscheck and self.ref['kind'] == 'kronecker_flipflop':
            pin = pinned_row(R_s, self.St, self.Sc, m.A, m.B1, theta, self.kc, self.kt,
                             common.rng(self.ns, self.sd.subject, stream + '-pinned', float(theta)), n_random)
            r['crosscheck_pinned'] = {q: {'pinned': pin[q], 'general': r[q], 'abs_diff': abs(pin[q] - r[q])}
                                      for q in ('a_best', 'cU', 'cF')}
            r['crosscheck_pinned']['B_cert'] = {'pinned': pin['B_cert'], 'B_dagger': bd['B_dagger']}
        return r


def theta_for_target(sp: SubjectPopulation, target: float, rule: dict, n_random_final: int, label: str) -> dict:
    """Population-only root search: theta_s in the bracket with N*_pred,s(theta_s) = target (brentq on
    log N*).  N* undefined because a ~ 0 counts as +infinity; N* <= 0 (a > 0 and B >= 0: the product class never
    pays) counts as -infinity (delta review D6).  Final constants at theta_s with all starts."""
    from scipy.optimize import brentq
    lo, hi = (float(x) for x in rule['theta_bracket'])
    nr = int(rule['search_random_starts'])
    trace = []

    def g(th):
        r = sp.row(th, nr, f'theta-search-{label}')
        n = r['N_star_pred']
        trace.append({'theta': th, 'N_star_pred': n, 'a_best': r['a_best'], 'B': r['B']})
        if n is not None and n > 0:
            return float(np.log(n) - np.log(target))
        if r['a_best'] > A_ZERO_TOL and (r['B'] >= 0 or (n is not None and n <= 0)):
            return -50.0
        return 50.0

    g_lo, g_hi = g(lo), g(hi)
    if not (g_lo > 0 > g_hi):
        return {'label': label, 'target': target, 'root': False, 'search': trace,
                'reason': 'no sign change of log N*(theta) - log target in the bracket'}
    th = float(brentq(g, lo, hi, xtol=float(rule['xtol']), maxiter=100))
    final = sp.row(th, n_random_final, f'theta-final-{label}')
    return {'label': label, 'target': target, 'root': True, 'theta': th, 'row': final, 'search': trace,
            'N_star': final['N_star_pred'], 'restart_agreement': final['restart_agreement']}


def theta_targets(P: int, n_min: int) -> dict:
    a = int(np.floor(P / 5))
    return {'P': int(P), 'n_min': int(n_min), 'N_star_A': float(a), 'N_star_B': float(max(5 * n_min, a / 2))}


def choose_m_hi(dev_calibration_pools: dict, cal: dict) -> dict:
    need = float(cal['factor']) * max(cal['candidates_hi'])
    ok = all(v >= need for v in dev_calibration_pools.values())
    return {'m_lo': int(cal['m_lo']), 'm_hi': int(max(cal['candidates_hi']) if ok else min(cal['candidates_hi'])),
            'rule': cal['rule'], 'threshold': need, 'dev_calibration_pools': dev_calibration_pools}


def subject_predictions(sd, proto: dict, namespace: str, targets: dict, m_grid: dict, crosscheck: bool = False) -> dict:
    sp = SubjectPopulation(sd, proto, namespace)
    n_random = int(proto['population']['n_random_starts'])
    rows = [sp.row(th, n_random, 'population', crosscheck) for th in proto['signal']['theta']]
    geoms = {'control': {'label': 'control', 'theta': 0.0, 'grid': control_grid(proto, targets['P'])}}
    reasons = []
    for lab, key in (('A', 'N_star_A'), ('B', 'N_star_B')):
        res = theta_for_target(sp, targets[key], proto['theta_rule'], n_random, lab)
        if res['root']:
            if res['N_star'] is None:
                res['root'] = False
                reasons.append(f'{lab}: N* undefined at the final theta_s')
            else:
                res['factors'] = list(proto['grid']['factors'])
                res['grid'] = [round_half_up(f * res['N_star']) for f in res['factors']]
                if not res['restart_agreement']:
                    reasons.append(f'{lab}: no restart agreement at theta_s')
        else:
            reasons.append(f'{lab}: {res["reason"]}')
        geoms[lab] = res
    pools = {p: int(len(sd.windows[p])) for p in sd.windows}
    need_N = max([max(g['grid']) for g in geoms.values() if g.get('grid')] or [0])
    if pools['training'] < need_N:
        reasons.append(f'training pool {pools["training"]} < largest grid N {need_N}')
    if pools['calibration'] < m_grid['m_hi']:
        reasons.append(f'calibration pool {pools["calibration"]} < m_hi {m_grid["m_hi"]}')
    if pools['evaluation'] < int(proto['n_eval']):
        reasons.append(f'evaluation pool {pools["evaluation"]} < n_eval {proto["n_eval"]}')
    return {'subject': sd.subject, 'C': sd.C, 'T': sd.T, 'd': sd.d, 'reference_kind': sp.ref['kind'],
            'reference_info': sp.ref['info'], 'rows': rows, 'pz1': pz1_subject(rows, float(proto['population']['zero_tol'])),
            'geometries': geoms, 'pools': pools, 'eligible': not reasons, 'exclusion_reasons': reasons}


def control_grid(proto: dict, P: int) -> list:
    g = proto['grid']
    return [n for n in g['control_absolute'] if g['n_min'] <= n <= P]


def pz1_subject(rows: list, tol: float = 1e-9) -> dict:
    """PZ1 for one subject: B_dagger exactly 0 at theta = 0 and strictly increasing over the fixed theta grid;
    a_best(theta = 0) = 0 (tolerance tol x max(1, tr K)) as a consistency check."""
    rows = sorted(rows, key=lambda r: r['theta'])
    v = [r.get('B_cert') for r in rows]
    if any(x is None for x in v):
        return {'quantity': 'B_dagger', 'values': v, 'met': False, 'note': 'certificate missing'}
    scale = max(1.0, float(rows[0].get('trace_K', 1.0)))
    zero = rows[0]['theta'] == 0.0 and v[0] == 0.0
    a0 = rows[0]['theta'] == 0.0 and abs(rows[0]['a_best']) <= tol * scale
    inc = all(b > a for a, b in zip(v, v[1:]))
    return {'quantity': 'B_dagger', 'values': v, 'thetas': [r['theta'] for r in rows], 'zero_at_theta0': bool(zero),
            'a_best_zero_at_theta0': bool(a0), 'strictly_increasing': bool(inc), 'met': bool(zero and inc and a0)}


def provenance() -> dict:
    root = common.EXP_ROOT
    files = {str(p.relative_to(root)): common.sha_file(p) for p in sorted(common.PACKAGE.glob('*.py'))}
    for rel in ('experiments/mlst2026/y19_20261012/predictions.py', 'experiments/mlst2026/p1nl_dev_20261006/tr09_first_order/core.py',
                'experiments/mlst2026/successor_20260913/t20s/generator.py', 'experiments/mlst2026/t25_production_20260930/science.py'):
        files[rel] = common.sha_file(root / rel)
    return {'files': files, 'python': platform.python_version(), 'numpy': np.__version__}


def study_constants(proto: dict, derived: Path) -> tuple[dict, dict]:
    """theta_rule targets (P) and the calibration m grid: from the frozen protocol, or (development) from the
    derived files of ALL dev subjects (pools only; no signal statistic is used)."""
    import h5py
    tr, cal = proto['theta_rule'], proto['calibration']
    if tr.get('P') is not None and cal.get('m_hi') is not None:
        return theta_targets(int(tr['P']), int(tr['n_min'])), {'m_lo': int(cal['m_lo']), 'm_hi': int(cal['m_hi']), 'rule': 'frozen'}
    if proto.get('frozen'):
        raise SystemExit('a frozen protocol must fix theta_rule.P and calibration.m_hi')
    pools = {}
    for s in proto['subjects']['dev']:
        f = Path(derived) / f'{s}.h5'
        if not f.is_file():
            raise SystemExit(f'P and m_hi need every dev subject; {f} missing')
        with h5py.File(f, 'r') as h:
            pools[s] = {p: int(h['windows'][p]['data'].shape[0]) for p in ('training', 'calibration', 'evaluation')}
    t = theta_targets(min(v['training'] for v in pools.values()), int(tr['n_min']))
    t['dev_pools'] = pools
    return t, choose_m_hi({s: v['calibration'] for s, v in pools.items()}, cal)


def _worker(args):
    path, proto, ns, targets, m_grid, crosscheck = args
    from .preprocess import load_subject
    import time as _t
    t0 = _t.time()
    out = subject_predictions(load_subject(path), proto, ns, targets, m_grid, crosscheck)
    out['seconds'] = _t.time() - t0
    return out


def main(argv=None):
    from .protocol import load_protocol
    from .preprocess import refuse_unfrozen_confirmation
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', required=True)
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--protocol')
    ap.add_argument('--gates', help='gates.json: excluded subjects are skipped (E1)')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--crosscheck', action='store_true', help='Kronecker reference only: also run the pinned Y19/TR09 path')
    a = ap.parse_args(argv)
    proto = load_protocol(a.protocol)
    subs = a.subjects.split(',')
    refuse_unfrozen_confirmation(subs, proto, a.protocol)
    excluded = []
    if a.gates:
        g = json.loads(Path(a.gates).read_text())
        excluded = [s for s in subs if g['subjects'].get(s, {}).get('excluded')]
    ns = proto['namespace']
    targets, m_grid = study_constants(proto, Path(a.derived))
    out = {'status': 'DEV predictions (dev subjects only; not registered)' if not proto.get('frozen') else 'REGISTERED',
           'protocol_sha256': common.sha_json(proto), 'reference_kind': proto['reference_metric']['kind'],
           'namespace': ns, 'theta_rule': targets, 'm_grid': m_grid, 'subjects': {}, 'excluded_by_gates': excluded,
           'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    todo = [(str(Path(a.derived) / f'{s}.h5'), proto, ns, targets, m_grid, a.crosscheck) for s in subs if s not in excluded]
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.jobs) as p:
            res = p.map(_worker, todo)
    else:
        res = [_worker(t) for t in todo]
    for r in res:
        out['subjects'][r['subject']] = r
        print(r['subject'], 'eligible', r['eligible'], r['exclusion_reasons'],
              {k: (g.get('theta'), g.get('N_star')) for k, g in r['geometries'].items()}, file=sys.stderr)
    out['pz1'] = {'per_subject': {s: v['pz1']['met'] for s, v in out['subjects'].items()},
                  'met': bool(out['subjects']) and all(v['pz1']['met'] for v in out['subjects'].values())}
    out['provenance'] = provenance()
    common.atomic_json(a.out, out)
    print(json.dumps({'pz1': out['pz1'], 'theta_rule': {k: v for k, v in targets.items() if k != 'dev_pools'},
                      'm_grid': {k: v for k, v in m_grid.items() if k != 'dev_calibration_pools'}}, indent=1))


if __name__ == '__main__':
    main()
