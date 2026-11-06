"""X-OPT kernel: strengthened refits of frozen HCP Rnoise learner rows.

Post hoc optimizer-sensitivity analysis of the registered once-only HCP Rnoise
confirmation run (OSF https://osf.io/n2rcu/).  The learner is unchanged: every
kernel function is imported from the frozen EXP tree at commit 419ffeba (the kit
that produced the registered FITS.jsonl).  This module adds only optimizer
routes and selection bookkeeping.

Selection discipline (fixed before any result is computed):
  * every route is ranked by the TRAINING objective alone;
  * "best" = largest training objective = smallest training risk
    tr(W S) - tr((B^T W B)^-1 B^T W S W B);
  * no clean-risk quantity is an input to selection.  The frozen route necessarily
    scores its own fit (that is the kit's code path and it yields the stored
    contrast); the strengthened routes are ranked without any clean-risk value, and
    the winner's clean risk is evaluated afterwards.
"""
from __future__ import annotations

import json
import time

import numpy as np

from experiments.mlst2026.meg_z_20261013.estimators import arm_estimate
from experiments.mlst2026.meg_z_20261013.fit import Evaluation, metric
from experiments.mlst2026.meg_z_20261013.protocol import PACKAGE as OLD_PACKAGE
from experiments.mlst2026.hcp_rnoise_20261019 import controls, learning, windows
from experiments.mlst2026.t25_production_20260930 import science as t25

# Frozen recipe (protocol_dev.json learners) and the strengthened settings.
S0_MAX_ITER, S0_TOL = 6000, 1e-8
S1_MAX_ITER, S1_TOL = 60000, 1e-11      # tol/1000, max_iter x10
S2_EXTRA_STARTS = 16                    # 1 HOSVD + 15 random starts under S1 settings
S3_METHOD = 'BFGS'                      # independent algorithm, local product-Grassmannian chart
TIE_REL = 1e-12                         # objective tie band for the "numerically tied" set

_capture = {}


def learners_config():
    return json.loads((OLD_PACKAGE / 'protocol_dev.json').read_text())['learners']


def _recorder_structured(orig):
    def rec(W, S, initial, max_iter=150, tol=1e-8):
        out = orig(W, S, initial, max_iter, tol)
        _capture['structured'] = dict(W=W, S=S, initial=initial, max_iter=max_iter, tol=tol,
                                      B=out[0], diag=out[1], bases=out[2])
        return out
    return rec


def _recorder_moments(orig):
    def rec(training, N):
        out = orig(training, N)
        _capture['moments'] = dict(x=out[0], mu=out[1], S=out[2])
        return out
    return rec


def assemble(roles, model, Sref, N, m, arm, date_index, replicate):
    """The pre-fit assembly of `learning.fit_cell` (frozen lines 15-31), unchanged.

    Builds no clean-risk object, so a timing run can exercise the optimizer only.
    """
    C, T = model.C, model.T
    gen, _ = controls.rng(8, 100, date_index, replicate)
    train, train_indices = windows.sample_blocks(*roles['train'], N, gen)
    clean_train, _ = model.draw(N, gen)
    if arm == 'reference':
        cov = Sref
    elif arm == 'isotropic':
        cov = np.eye(C * T)
    else:
        cal_gen, _ = controls.rng(8, 102, date_index, replicate)
        calibration, _ = windows.sample_blocks(*roles['cal'], m, cal_gen)
        cov, _ = arm_estimate(calibration, C, T, arm)
    Wm = metric(cov)
    metric_scale = float(np.trace(Wm) / (C * T))
    Wm = Wm / metric_scale
    _, mu, S = t25.moments(clean_train + train, N)
    key = ('hcp-rnoise', date_index, replicate, float(model.theta), N)
    initial = t25.starts(S, C, T, 2, 3, key)
    return dict(S=S, mu=mu, W=Wm, metric_scale=metric_scale, initial=initial, key=key,
                sampled_training_indices=train_indices.tolist(), C=C, T=T)


def evaluation_for(roles, model, Sref, date_index, replicate):
    """The fixed evaluation set of `learning.fit_cell` (frozen lines 19-25), unchanged."""
    eval_gen, _ = controls.rng(8, 101, date_index, replicate)
    clean_eval, alpha_eval = model.draw(len(roles['eval'][0]), eval_gen)
    return Evaluation(clean_eval, roles['eval'][0], alpha_eval, model.templates(),
                      model.variances, metric(Sref), model.second_moment())


def frozen_route(roles, model, Sref, N, m, arm, date_index, replicate):
    """Run the frozen learner unchanged, capturing the metric, moment and factor bases it used."""
    orig_s, orig_m = t25.structured, t25.moments
    t25.structured, t25.moments = _recorder_structured(orig_s), _recorder_moments(orig_m)
    try:
        frozen = learning.fit_cell(roles, model, Sref, N, m, arm, date_index, replicate)
    finally:
        t25.structured, t25.moments = orig_s, orig_m
    cap_s = _capture.pop('structured')
    cap_m = _capture.pop('moments')
    return frozen, cap_s, cap_m


def _independent_algorithm(W, S, H, u0, v0, C, T, maxiter=300):
    """Quasi-Newton on a local chart of the product Grassmannian (independent of NFPA/Armijo)."""
    from scipy.optimize import minimize
    scale = float(np.trace(W @ S))
    Uperp = np.linalg.svd(np.eye(C) - u0 @ u0.T)[0][:, :C - 2]
    Vperp = np.linalg.svd(np.eye(T) - v0 @ v0.T)[0][:, :T - 3]
    dimA, dimB = (C - 2) * 2, (T - 3) * 3

    def unpack(x):
        return (t25.qr(u0 + Uperp @ x[:dimA].reshape(C - 2, 2)),
                t25.qr(v0 + Vperp @ x[dimA:].reshape(T - 3, 3)))

    def neg(x):
        u, v = unpack(x)
        return -t25.objective_gradient(u, v, W, S, H)[0] / scale

    t0 = time.time()
    res = minimize(neg, np.zeros(dimA + dimB), method=S3_METHOD, jac='3-point',
                   options={'gtol': 1e-12, 'maxiter': maxiter, 'disp': False})
    u, v = unpack(res.x)
    val = t25.objective_gradient(u, v, W, S, H)[0]
    _, gc, gt = t25.objective_gradient(u, v, W, S, H)
    gcn = (gc - u @ (u.T @ gc)) / scale
    gtn = (gt - v @ (v.T @ gt)) / scale
    gn = float(np.sqrt(np.sum(gcn * gcn) + np.sum(gtn * gtn)))
    return u, v, float(val), dict(status=('stationary' if gn <= S1_TOL else 's3_gradient_floor'),
                                  gradient_norm=gn, iterations=int(res.nit), nfev=int(res.nfev),
                                  method=S3_METHOD, seconds=time.time() - t0,
                                  message=str(res.message)[:200])


def strengthened(cap_s, cap_m, C, T, key, *, with_s3=True):
    """Run S1, S2 (and optionally S3) from the captured frozen route; return candidates."""
    W, S = cap_s['W'], cap_s['S']
    H = W @ S @ W
    diag0, bases0 = cap_s['diag'], cap_s['bases']
    sel = int(diag0['selected'])
    u0, v0 = bases0[sel]
    out = {'route_seconds': {}}
    out['S0'] = dict(route='S0', objective=float(diag0['restarts'][sel]['objective']),
                     status=diag0['restarts'][sel]['status'],
                     gradient_norm=float(diag0['restarts'][sel]['gradient_norm']),
                     iterations=len(diag0['restarts'][sel]['trajectory']) - 1)
    candidates = [dict(out['S0'], u=u0, v=v0)]

    t0 = time.time()
    _, diag1, bases1 = t25.structured(W, S, [(u0, v0)], S1_MAX_ITER, S1_TOL)
    out['route_seconds']['S1'] = time.time() - t0
    r1 = diag1['restarts'][0]
    out['S1'] = dict(route='S1', objective=float(r1['objective']), status=r1['status'],
                     gradient_norm=float(r1['gradient_norm']),
                     iterations=len(r1['trajectory']) - 1)
    candidates.append(dict(out['S1'], u=bases1[0][0], v=bases1[0][1]))

    t0 = time.time()
    s2_key = list(key) + ['xopt-s2']
    gen = t25.rng(*s2_key)
    s4 = S.reshape(C, T, C, T, order='F')
    extra = [(t25.top(np.einsum('ctdt->cd', s4), 2), t25.top(np.einsum('ctcs->ts', s4), 3))]
    for _ in range(S2_EXTRA_STARTS - 1):
        extra.append((t25.qr(gen.normal(size=(C, 2))), t25.qr(gen.normal(size=(T, 3)))))
    _, diag2, bases2 = t25.structured(W, S, extra, S1_MAX_ITER, S1_TOL)
    out['route_seconds']['S2'] = time.time() - t0
    out['s2_seed_key'] = s2_key
    out['S2'] = [dict(route=f'S2_{i}', objective=float(r['objective']), status=r['status'],
                      gradient_norm=float(r['gradient_norm']),
                      iterations=len(r['trajectory']) - 1) for i, r in enumerate(diag2['restarts'])]
    for i, (r, (u, v)) in enumerate(zip(diag2['restarts'], bases2)):
        candidates.append(dict(out['S2'][i], u=u, v=v))

    if with_s3:
        best = max(candidates, key=lambda c: c['objective'])
        t0 = time.time()
        u3, v3, obj3, diag3 = _independent_algorithm(W, S, H, best['u'], best['v'], C, T)
        diag3['seconds'] = time.time() - t0
        out['route_seconds']['S3'] = diag3['seconds']
        out['S3'] = dict(route='S3', objective=obj3, **diag3)
        candidates.append(dict(out['S3'], u=u3, v=v3))

    win = max(candidates, key=lambda c: c['objective'])
    risk0 = float(np.trace(W @ S))
    out['training_risk_at_best'] = risk0 - win['objective']
    out['selected_route'] = win['route']
    out['objective'] = float(win['objective'])
    out['selected_status'] = win['status']
    out['selected_gradient_norm'] = float(win['gradient_norm'])
    out['selected_iterations'] = int(win['iterations'])
    out['candidates'] = [{k: v for k, v in c.items() if k not in ('u', 'v')} for c in candidates]
    tied = [c for c in candidates if c['objective'] >= win['objective'] * (1.0 - TIE_REL)] \
        if win['objective'] > 0 else [win]
    out['n_tied'] = len(tied)
    return out, win['u'], win['v'], tied


def clean_scores(ev, u, v, W, mu):
    """Clean-risk scores of one fixed fit (called only after selection)."""
    A = t25.map_for(np.kron(v, u), W)
    b = (np.eye(len(mu)) - A) @ mu
    return A, b, ev.scores(A, b)


def flags_for(win_status, win_grad, tied_scores, D, spread, frozen_tol=1e-8, s1_tol=1e-11):
    """Frozen flag vocabulary, recomputed for the strengthened selection.

    Reported at two gradient tolerances: the frozen recipe tolerance 1e-8 (directly
    comparable with the registered ``optimization_limited`` flag) and the S1
    tolerance 1e-11 (the strengthened stationarity criterion).  The Armijo
    floating-point floor sits near 1e-8, so the 1e-11 criterion is generally not
    reachable and is reported only to make that explicit.
    """
    out = {}
    for tag, tol in (('at_frozen_tol', frozen_tol), ('at_s1_tol', s1_tol)):
        status = 'stationary' if win_grad <= tol else win_status
        refined = (status == 'iteration_limit'
                   or (status in ('line_search_stopped', 's3_gradient_floor') and win_grad > 1e-6)
                   or spread >= abs(D))
        t25_rule = (status != 'stationary') or spread >= abs(D)
        out[tag] = dict(optimization_limited=bool(refined),
                        optimization_limited_t25_rule=bool(t25_rule),
                        status_at_tolerance=status, tolerance=float(tol))
    out['selected_status'] = win_status
    out['selected_gradient_norm'] = float(win_grad)
    out['tied_spread'] = float(spread)
    return out
