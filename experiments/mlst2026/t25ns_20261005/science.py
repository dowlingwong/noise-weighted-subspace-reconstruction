"""X2 (T25-NS) science: the T25 fit under a nonseparable physical noise covariance.

The fit is the accepted T25 production code, imported read-only after guard.py
has checked its bytes against the accepted closure (identity 01f2f82f...).  What
X2 adds: the covariance constructor, the grid and namespaces, the population-limit
references (estimator limits and clean-risk penalties), the E1f extrapolated-floor
contrast, the cross-rho contrasts and the registered prediction rules.

Indexing is T25's throughout: vec index c + C*t, Sigma_sep = St (x) Sc with the
TIME factor first.  The shared-line term is therefore S_line (x) g g^T in code;
the preparedness note's "(g g^T) (x) S_line" names the same operator in
channel-first notation.
"""
from __future__ import annotations

from . import guard

guard.assert_direct()   # fail closed before any T25 module is imported

import numpy as np                                                    # noqa: E402
from scipy.stats import t as student_t                                # noqa: E402
from experiments.mlst2026.t25_production_20260930 import science as t25  # noqa: E402
from experiments.mlst2026.t25_production_20260930.science import (    # noqa: E402,F401
    rng, digest, symmetric, powers, qr, top, geometry, regularize, estimate, map_for,
    objective_gradient, starts, structured, unrestricted, risks, representation, moments,
    response, profile)

ARMS = ('full', 'kron')
CLASSES = ('restricted', 'unrestricted')


def variants(c):
    return ['oracle'] + [f'{arm}_{m}' for m in c['m'] for arm in ARMS]


# --------------------------------------------------------------------------------------
# covariance
# --------------------------------------------------------------------------------------
def gain_vector(c):
    """Positive channel gains of the shared line, ||g||^2 = C exactly by construction."""
    C = c['C']
    g = 1.0 + c['gain']['amplitude'] * np.cos(2 * np.pi * np.arange(C) / C)
    return g * np.sqrt(C / np.dot(g, g))


def line_kernel(c):
    """S_line[i, j] = cos(2 pi f0 |i-j|) exp(-|i-j| / ell): unit diagonal, Toeplitz,
    strictly positive definite (its spectral density is a sum of two Poisson kernels)."""
    T = c['T']
    lag = np.abs(np.arange(T)[:, None] - np.arange(T)).astype(float)
    return np.cos(2 * np.pi * c['line']['f0'] * lag) * np.exp(-lag / c['line']['ell'])


def separable_part(c):
    return t25.covariance(c)                      # St (x) Sc, the accepted T25 constructor


def shared_line_part(c):
    g = gain_vector(c)
    return np.kron(line_kernel(c), np.outer(g, g))   # time (x) channel in vec index c + C t


def covariance(c, rho):
    """Sigma_rho = c_rho [(1 - rho) St(x)Sc + rho S_line(x)g g^T], c_rho matching the T25
    trace.  No special case: at rho = 0 every floating-point step is exact, so the result
    is bitwise the T25 covariance (tested)."""
    sep = separable_part(c)
    mix = (1.0 - rho) * sep + rho * shared_line_part(c)
    return (np.trace(sep) / np.trace(mix)) * mix


def trace_scale(c, rho):
    sep = separable_part(c)
    return float(np.trace(sep) / np.trace((1.0 - rho) * sep + rho * shared_line_part(c)))


def nearest_kronecker(A, C, T):
    """Van Loan--Pitsianis: Frobenius-nearest Bt (x) Cc (Bt: T x T, Cc: C x C) under the
    c + C t index, by a rank-one SVD of the rearrangement R[(t,t'),(c,c')]."""
    R = np.asarray(A).reshape(T, C, T, C).transpose(0, 2, 1, 3).reshape(T * T, C * C)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    Bt = (np.sqrt(s[0]) * U[:, 0]).reshape(T, T)
    Cc = (np.sqrt(s[0]) * Vt[0]).reshape(C, C)
    rel = float(np.sqrt(max(float(np.sum(s[1:] ** 2)), 0.0)) / np.linalg.norm(A, 'fro'))
    return Bt, Cc, rel


def moment_limits(sigma, C, T):
    """Population (m -> infinity) limits of the two T25 estimators, shrinkage and floors
    included: regularize(Sigma) and regularize(Tr_c Sigma / C) (x) regularize(Tr_t Sigma /(T v))."""
    d = C * T
    z4 = sigma.reshape(T, C, T, C)                 # [t, c, t', c'] = Sigma[c + C t, c' + C t']
    v = float(np.trace(sigma) / d)
    ac = np.einsum('tctd->cd', z4) / T / v
    at = np.einsum('tcsc->ts', z4) / C
    full, _ = regularize(sigma)
    sc, _ = regularize(ac)
    st, _ = regularize(at)
    return {'full': full, 'kron': np.kron(st, sc), 'kron_unshrunk': np.kron(at, ac)}


def population_reference(c, rho):
    sigma = covariance(c, rho)
    C, T = c['C'], c['T']
    w = np.linalg.eigvalsh(sigma)
    sc = c['rho_c'] ** np.abs(np.arange(C)[:, None] - np.arange(C))
    st = c['rho_t'] ** np.abs(np.arange(T)[:, None] - np.arange(T))
    bound = (1.0 - rho) * trace_scale(c, rho) * np.linalg.eigvalsh(st)[0] * np.linalg.eigvalsh(sc)[0]
    if not np.isfinite(w).all() or w[0] <= 0 or w[0] < bound * (1 - 1e-9):
        raise ValueError('Sigma_rho is not SPD or violates its proved eigenvalue bound')
    L, Li = powers(sigma)
    limits = moment_limits(sigma, C, T)
    out = {'rho': rho, 'trace': float(np.trace(sigma)), 'trace_scale_c_rho': trace_scale(c, rho),
           'lambda_min': float(w[0]), 'lambda_min_lower_bound': float(bound),
           'condition_number': float(w[-1] / w[0]),
           'line_kernel_lambda_min': float(np.linalg.eigvalsh(line_kernel(c))[0]),
           'gain_norm_squared': float(np.dot(gain_vector(c), gain_vector(c))),
           'nearest_kronecker_relative_distance': nearest_kronecker(sigma, C, T)[2]}
    for name in ('full', 'kron', 'kron_unshrunk'):
        e = Li @ limits[name] @ Li - np.eye(C * T)
        out[f'{name}_limit_whitened_error_op'] = float(np.linalg.norm(e, 2))
        out[f'{name}_limit_relative_frobenius'] = float(
            np.linalg.norm(limits[name] - sigma, 'fro') / np.linalg.norm(sigma, 'fro'))
    return out


def population_penalties(c):
    """Deterministic m, N -> infinity clean-risk penalties relative to the oracle in the common metric
    Sigma_rho^-1, both classes, for the population limits of the full and Kronecker estimators
    (shrinkage and floors included) and for the unshrunk Kronecker limit (its separability part alone).
    The population second moment of x is C_s + Sigma (zero mean, so no intercept).  No seed is drawn;
    the random product starts use a fixed key that names no namespace."""
    g = geometry(c)
    rows = []
    for rho in [0.0] + list(c['rho']):
        sigma = covariance(c, rho)
        L, Li = powers(sigma)
        W0 = Li @ Li
        lim = moment_limits(sigma, c['C'], c['T'])
        metrics = {'oracle': sigma, 'full_limit': lim['full'], 'kron_limit': lim['kron'],
                   'kron_unshrunk_limit': lim['kron_unshrunk']}
        for theta in c['theta']:
            Cs = L @ g.K(theta) @ L
            S = Cs + sigma
            initial = starts(S, g.C, g.T, g.k_c, g.k_t, ('population-reference', rho, theta))
            risk, status = {}, {}
            for name, cov in metrics.items():
                W = np.linalg.inv(cov)
                Au = map_for(unrestricted(W, S, g.k), W)
                Br, diag, _ = structured(W, S, initial, c['max_iter'], c['gradient_tolerance'])
                Ar = map_for(Br, W)
                zero = np.zeros(g.d)
                risk[name] = {'restricted': risks(Ar, zero, Cs, sigma, W0)['clean'],
                              'unrestricted': risks(Au, zero, Cs, sigma, W0)['clean']}
                status[name] = diag['restarts'][diag['selected']]['status']
            row = {'rho': rho, 'theta': theta, 'oracle_risk': risk['oracle'], 'restricted_fit_status': status}
            for name in metrics:
                if name != 'oracle':
                    row['penalty_' + name] = {cl: risk[name][cl] - risk['oracle'][cl] for cl in CLASSES}
            rows.append(row)
    return rows


def population_references(c):
    g = geometry(c)
    spectra = []
    for theta in c['theta']:
        err = float(np.max(np.abs(np.linalg.eigvalsh(g.K(theta)) - np.sort(g.mu))))
        if err > 1e-10:
            raise ValueError('registered spectrum invariant failed')
        spectra.append({'theta': theta, 'spectrum_max_error': err})
    rows = [population_reference(c, rho) for rho in [0.0] + list(c['rho'])]
    for r in rows:
        if abs(r['trace_scale_c_rho'] - 1.0) > 1e-12:
            raise ValueError('trace scale departs from 1 although ||g||^2 = C and S_line has unit diagonal')
    return {'covariance': rows, 'whitened_spectrum': spectra, 'population_penalties': population_penalties(c),
            'note': 'whitened clean covariance K(theta) is rho-independent; references are population '
                    'quantities computed before any seed is drawn'}


# --------------------------------------------------------------------------------------
# data and fit
# --------------------------------------------------------------------------------------
def make_data(c, seed, rho, theta):
    """T25's draw with Sigma_rho.  Streams are keyed by (namespace, seed[, theta]) only:
    both rho levels share the standard-normal draws (common random numbers), so the
    true-whitened data L^-1 x are the same at every rho."""
    g = geometry(c)
    sigma = covariance(c, rho)
    L, Li = powers(sigma)
    key = (c['namespace'], seed)
    calibration = rng(*key, 'calibration').normal(size=(max(c['m']), g.d)) @ L
    clean = rng(*key, theta, 'signal').normal(size=(max(c['N']), g.d)) @ (g.V(theta) * np.sqrt(g.mu)).T @ L
    noise = rng(*key, theta, 'noise').normal(size=(max(c['N']), g.d)) @ L
    return g, sigma, L, Li, calibration, clean + noise


def covariance_estimates(c, pool, g):
    estimates, diag = {}, {}
    for m in c['m']:
        covs, d = estimate(pool[:m], g.C, g.T)
        for name, cov in covs.items():
            estimates[f'{name}_{m}'] = cov
        diag[str(m)] = d
    return estimates, diag


def fit_cell(c, seed, rho, theta, N):
    g, sigma, L, Li, pool, training = make_data(c, seed, rho, theta)
    x, mu, S = moments(training, N)
    K = g.K(theta)
    Cs = L @ K @ L
    W0 = Li @ Li
    truth = top(K, g.k)
    estimated, covdiag = covariance_estimates(c, pool, g)
    estimates = {'oracle': sigma, **estimated}
    initial = starts(S, g.C, g.T, g.k_c, g.k_t, (c['namespace'], seed, theta, N))
    sample = Li @ S @ Li
    results = {}
    arrays = {'S': S, 'mean': mu}
    for name in variants(c):
        cov = estimates[name]
        W = np.linalg.inv(cov)
        if name != 'oracle':
            arrays[name + '_cov'] = cov
        B, diagnostic, factors = structured(W, S, initial, c['max_iter'], c['gradient_tolerance'])
        Bu = unrestricted(W, S, g.k)
        for cls, basis in (('restricted', B), ('unrestricted', Bu)):
            tag = name + '__' + cls
            A = map_for(basis, W)
            b = (np.eye(g.d) - A) @ mu
            arrays[tag + '_B'] = basis
            row = risks(A, b, Cs, sigma, W0) | representation(basis, Li, truth)
            row['training_loss'] = float(np.trace(W @ S) - np.trace(
                np.linalg.solve(basis.T @ W @ basis, basis.T @ W @ S @ W @ basis)))
            if cls == 'restricted':
                row['optimization'] = diagnostic
                rr = []
                for i, (uc, ut) in enumerate(factors):
                    arrays[tag + f'_uc_{i}'] = uc
                    arrays[tag + f'_ut_{i}'] = ut
                    ar = map_for(np.kron(ut, uc), W)
                    rr.append(risks(ar, (np.eye(g.d) - ar) @ mu, Cs, sigma, W0)['clean'])
                row['restart_clean_risks'] = rr
                row['restart_risk_spread'] = float(max(rr) - min(rr))
            results[tag] = row
        if name != 'oracle':
            results[name + '__domain'] = response(sample, Li @ cov @ Li - np.eye(g.d), g.k)[0]
    record = {'seed': seed, 'rho': rho, 'theta': theta, 'N': N, 'training_sha': digest(x),
              'calibration_sha': digest(pool), 'sigma_sha': digest(sigma), 'covariance': covdiag,
              'methods': results}
    return record, arrays


# --------------------------------------------------------------------------------------
# contrasts, inference and registered predictions
# --------------------------------------------------------------------------------------
def contrast_values(methods, ms):
    """31 contrasts per (rho, theta, N) location; negative D favours structure."""
    V = ['oracle'] + [f'{arm}_{m}' for m in ms for arm in ARMS]
    R = lambda v, cl: methods[v + '__' + cl]['clean']                       # noqa: E731
    D = {v: R(v, 'restricted') - R(v, 'unrestricted') for v in V}
    out = {'class/' + v: D[v] for v in V}
    for v in V[1:]:
        for cl in CLASSES:
            out[f'metric/{v}/{cl}'] = R(v, cl) - R('oracle', cl)
        out['interaction/' + v] = D[v] - D['oracle']
    for m in ms:
        out[f'covariance_class/{m}'] = D[f'kron_{m}'] - D[f'full_{m}']
    for arm in ARMS:
        for a, b in zip(ms, ms[1:]):
            out[f'calibration_class/{arm}/{a}-{b}'] = D[f'{arm}_{b}'] - D[f'{arm}_{a}']
            for cl in CLASSES:
                out[f'metric_calibration/{arm}/{cl}/{a}-{b}'] = R(f'{arm}_{b}', cl) - R(f'{arm}_{a}', cl)
    top_m, lo_m = max(ms), min(ms)
    for cl in CLASSES:
        out[f'estimator/{top_m}/{cl}'] = R(f'full_{top_m}', cl) - R(f'kron_{top_m}', cl)
    r = top_m / lo_m
    for arm in ARMS:
        for cl in CLASSES:
            hi = R(f'{arm}_{top_m}', cl) - R('oracle', cl)
            lo = R(f'{arm}_{lo_m}', cl) - R('oracle', cl)
            out[f'floor_extrapolated/{arm}/{cl}'] = (r * hi - lo) / (r - 1)   # penalty = F + c/m
    return out


def cross_rho_values(low, high, ms):
    """Seed-paired change of the Kronecker penalty at the largest m from the lower to the
    higher rho (common random numbers)."""
    top_m = max(ms)

    def pen(meth, cl):
        return meth[f'kron_{top_m}__{cl}']['clean'] - meth['oracle__' + cl]['clean']
    return {f'rho_change/kron_{top_m}/{cl}': pen(high, cl) - pen(low, cl) for cl in CLASSES}


def involved_metrics(name, ms):
    """Metrics whose RESTRICTED fit enters a contrast (unrestricted fits are closed form)."""
    parts = name.split('/')
    kind = parts[0]
    if kind == 'class':
        return {parts[1]}
    if kind == 'metric':
        return {parts[1], 'oracle'} if parts[2] == 'restricted' else set()
    if kind == 'interaction':
        return {parts[1], 'oracle'}
    if kind == 'covariance_class':
        return {f'full_{parts[1]}', f'kron_{parts[1]}'}
    if kind == 'calibration_class':
        a, b = parts[2].split('-')
        return {f'{parts[1]}_{a}', f'{parts[1]}_{b}'}
    if kind == 'metric_calibration':
        a, b = parts[3].split('-')
        return {f'{parts[1]}_{a}', f'{parts[1]}_{b}'} if parts[2] == 'restricted' else set()
    if kind == 'estimator':
        return {f'full_{parts[1]}', f'kron_{parts[1]}'} if parts[2] == 'restricted' else set()
    if kind == 'rho_change':
        return {parts[1], 'oracle'} if parts[2] == 'restricted' else set()
    if kind == 'floor_extrapolated':
        return ({f'{parts[1]}_{m}' for m in ms} | {'oracle'}) if parts[2] == 'restricted' else set()
    raise ValueError('unknown contrast ' + name)


def optimization_limited(record, metric):
    """T25's registered rule: selected restart nonstationary, or restart clean-risk spread
    at least the absolute class contrast."""
    rr = record['methods'][metric + '__restricted']
    uu = record['methods'][metric + '__unrestricted']
    opt = rr['optimization']
    return (opt['restarts'][opt['selected']]['status'] != 'stationary'
            or rr['restart_risk_spread'] >= abs(rr['clean'] - uu['clean']))


def sign(row):
    return '-' if row['ci'][1] < 0 else '+' if row['ci'][0] > 0 else '?'


def cross_label(lo, hi):
    return f'{hi}-{lo}'


def predictions(c, rows):
    ms, rhos = c['m'], sorted(c['rho'])
    lo_m, top_m, lo_r, hi_r = min(ms), max(ms), rhos[0], rhos[-1]
    cross = cross_label(lo_r, hi_r)
    get = {(r['rho'], r['theta'], r['N'], r['contrast']): r for r in rows}
    p1 = []
    for th in c['theta']:
        for n in c['N']:
            for cl in CLASSES:
                a = get[(cross, th, n, f'rho_change/kron_{top_m}/{cl}')]
                b = get[(hi_r, th, n, f'metric_calibration/kron/{cl}/{lo_m}-{top_m}')]
                f = get[(hi_r, th, n, f'floor_extrapolated/kron/{cl}')]
                flagged = a['optimization_limited'] or b['optimization_limited'] or f['optimization_limited']
                holds = sign(a) == '+' and sign(b) != '-'
                p1.append({'theta': th, 'N': n, 'class': cl, 'penalty_rise_with_rho': sign(a),
                           'penalty_change_with_m_at_high_rho': sign(b), 'extrapolated_floor_at_high_rho': sign(f),
                           'holds': holds, 'contradicts': sign(a) == '-' or sign(b) == '-',
                           'optimization_limited': flagged,
                           'floor_wording_allowed': holds and sign(f) == '+' and not flagged})
    if not all(x['holds'] for x in p1):          # floor wording only under P1 HOLDS (protocol section 8)
        for x in p1:
            x['floor_wording_allowed'] = False
    p2 = []
    for r in rhos:
        for th in c['theta']:
            for n in c['N']:
                for cl in CLASSES:
                    row = get[(r, th, n, f'metric_calibration/full/{cl}/{lo_m}-{top_m}')]
                    p2.append({'rho': r, 'theta': th, 'N': n, 'class': cl, 'sign': sign(row),
                               'optimization_limited': row['optimization_limited']})
    p3 = []
    for th in c['theta']:
        for n in c['N']:
            for m in ms:
                row = get[(hi_r, th, n, f'interaction/kron_{m}')]
                p3.append({'theta': th, 'N': n, 'm': m, 'sign': sign(row),
                           'optimization_limited': row['optimization_limited']})
    e2 = []
    for r in rhos:
        for th in c['theta']:
            for n in c['N']:
                for cl in CLASSES:
                    row = get[(r, th, n, f'estimator/{top_m}/{cl}')]
                    e2.append({'rho': r, 'theta': th, 'N': n, 'class': cl, 'sign': sign(row),
                               'optimization_limited': row['optimization_limited']})
    floor = []
    for r in rhos:
        for th in c['theta']:
            for n in c['N']:
                for arm in ARMS:
                    for cl in CLASSES:
                        row = get[(r, th, n, f'floor_extrapolated/{arm}/{cl}')]
                        floor.append({'rho': r, 'theta': th, 'N': n, 'estimator': arm, 'class': cl,
                                      'sign': sign(row), 'optimization_limited': row['optimization_limited']})
    interaction = {}
    for r in rhos:
        for v in variants(c)[1:]:
            sel = [get[(r, th, n, f'interaction/{v}')] for th in c['theta'] for n in c['N']]
            interaction[f'{r}/{v}'] = {s: sum(sign(x) == s for x in sel) for s in '-+?'} | \
                {'optimization_limited': sum(x['optimization_limited'] for x in sel), 'locations': len(sel)}

    def verdict(holds, contradicts):
        return 'HOLDS' if all(holds) else 'FAILS' if any(contradicts) else 'UNDECIDED'
    return {
        'P1': {'rule': 'HOLDS if at every (theta, N, class) the Kronecker penalty at the largest m rises from '
                       'the lower to the higher rho (interval above zero) and its change from the smallest to '
                       'the largest m at the higher rho is not decided negative; FAILS if any location has a '
                       'decided fall with rho or a decided fall with m; UNDECIDED otherwise. A HOLDS verdict '
                       'means "rises with rho and no decided decrease with calibration on this grid"; it does '
                       'not by itself establish a floor.',
               'verdict': verdict([x['holds'] for x in p1], [x['contradicts'] for x in p1]),
               'counts': {'holds': sum(x['holds'] for x in p1), 'contradicts': sum(x['contradicts'] for x in p1),
                          'optimization_limited': sum(x['optimization_limited'] for x in p1),
                          'floor_wording_allowed': sum(x['floor_wording_allowed'] for x in p1), 'locations': len(p1)},
               'floor_wording_rule': 'the word "floor" may be used only if P1 HOLDS, and only for locations where '
                                     'the 1/m-extrapolated Kronecker penalty at the higher rho is decided positive '
                                     'and no involved restricted fit is optimization-limited; the separability '
                                     'and shrinkage shares are then read from the population-limit penalties',
               'locations': p1},
        'P2': {'rule': 'HOLDS if the full-estimate penalty change from the smallest to the largest m is decided '
                       'negative at every (rho, theta, N, class); FAILS if any is decided positive; UNDECIDED '
                       'otherwise',
               'verdict': verdict([x['sign'] == '-' for x in p2], [x['sign'] == '+' for x in p2]),
               'counts': {s: sum(x['sign'] == s for x in p2) for s in '-+?'},
               'locations': p2},
        'P3': {'rule': 'nondirectional: sign of the Kronecker interaction at the higher rho, reported only',
               'counts': {s: sum(x['sign'] == s for x in p3) for s in '-+?'}, 'locations': p3},
        'E2': {'rule': 'full minus Kronecker clean risk at the largest m, per location, no direction registered',
               'counts': {s: sum(x['sign'] == s for x in e2) for s in '-+?'}, 'locations': e2},
        'E1_floor': {'rule': 'secondary, nondirectional: 1/m-extrapolated penalty F = (r P(m_max) - P(m_min))/(r - 1), '
                             'r = m_max/m_min, per (rho, theta, N, estimator, class)',
                     'locations': floor},
        'E3_interaction_counts': {'rule': 'nondirectional counts of decided signs of I_v over (theta, N) per (rho, v)',
                                  'counts': interaction},
        'attribution': 'verdicts use the fitted-learner contrasts at every location; a floor or covariance '
                       'mechanism is not attributed at locations flagged optimization_limited'}


def reduce_records(c, records):
    expected = {(s, r, th, n) for s in c['seeds'] for r in c['rho'] for th in c['theta'] for n in c['N']}
    got = [(r['seed'], r['rho'], r['theta'], r['N']) for r in records]
    if set(got) != expected or len(got) != len(set(got)):
        raise ValueError('missing/duplicate cells')
    by = {(r['seed'], r['rho'], r['theta'], r['N']): r for r in records}
    ms, rhos = c['m'], sorted(c['rho'])
    groups, flags = {}, {}
    for key0 in sorted(by):
        r = by[key0]
        for name, val in contrast_values(r['methods'], ms).items():
            key = (r['rho'], r['theta'], r['N'], name)
            groups.setdefault(key, []).append(val)
            flags[key] = flags.get(key, 0) + int(any(optimization_limited(r, v) for v in involved_metrics(name, ms)))
    for s in c['seeds']:
        for th in c['theta']:
            for n in c['N']:
                for lo, hi in zip(rhos, rhos[1:]):
                    a, b = by[(s, lo, th, n)], by[(s, hi, th, n)]
                    for name, val in cross_rho_values(a['methods'], b['methods'], ms).items():
                        key = (cross_label(lo, hi), th, n, name)
                        groups.setdefault(key, []).append(val)
                        lim = any(optimization_limited(rec, v) for rec in (a, b) for v in involved_metrics(name, ms))
                        flags[key] = flags.get(key, 0) + int(lim)
    family = len(groups)
    rows = []
    for key in sorted(groups, key=lambda k: (str(k[0]), k[1], k[2], k[3])):
        v = np.asarray(groups[key], dtype=float)
        sd = float(v.std(ddof=1))
        half = float(student_t.ppf(1 - .05 / (2 * family), len(v) - 1) * sd / np.sqrt(len(v)))
        mean = float(v.mean())
        rows.append({'rho': key[0], 'theta': key[1], 'N': key[2], 'contrast': key[3], 'mean': mean,
                     'ci': [mean - half, mean + half], 'seeds': len(v), 'zero_standard_error': sd == 0,
                     'optimization_limited_seeds': flags[key], 'optimization_limited': flags[key] > 0})
    profiles = {}
    for rho in rhos:
        for th in c['theta']:
            for v in variants(c):
                rr = sorted([r for r in rows if r['rho'] == rho and r['theta'] == th and r['contrast'] == 'class/' + v],
                            key=lambda r: r['N'])
                profiles[f'{rho}/{th}/{v}'] = profile(rr)
    limits = [{'seed': r['seed'], 'rho': r['rho'], 'theta': r['theta'], 'N': r['N'], 'metric': v,
               'optimization_limited': optimization_limited(r, v)} for r in records for v in variants(c)]
    domains = [{'rho': r['rho'], 'estimator': k.split('__')[0], 'domain': d['domain'], 'eta': d.get('eta'),
                'epsilon': d.get('epsilon')}
               for r in records for k, d in r['methods'].items() if k.endswith('__domain')]
    census = {k: sum(1 for d in domains if d['domain'] == k) for k in ('in_domain', 'out_of_domain', 'undefined_gap')}
    if sum(census.values()) != len(records) * (len(variants(c)) - 1) or len(domains) != sum(census.values()):
        raise ValueError('domain census incomplete')
    errors = {}
    for rho in rhos:
        for est in variants(c)[1:]:
            eta = [d['eta'] for d in domains if d['rho'] == rho and d['estimator'] == est and d['eta'] is not None]
            eps = [d['epsilon'] for d in domains if d['rho'] == rho and d['estimator'] == est]
            errors[f'{rho}/{est}'] = {'median_eta': float(np.median(eta)) if eta else None,
                                      'min_eta': float(np.min(eta)) if eta else None,
                                      'median_epsilon': float(np.median(eps)), 'n': len(eps)}
    return {'family_size': family, 'interval': 'approximate Bonferroni Student-t over independent seeds',
            'zero_SE_rule': 'degenerate observed-seed interval; not proof of population certainty; zero is undecided',
            'mechanism_rule': 'learner contrasts retained; no class/covariance mechanism attribution where an '
                              'involved selected fit is nonstationary or its restart risk spread reaches the '
                              'absolute class contrast (T25 rule)',
            'contrasts': rows, 'profiles': profiles, 'optimization_limits': limits,
            'domain_census': census, 'covariance_error_summary': errors,
            'predictions': predictions(c, rows)}
