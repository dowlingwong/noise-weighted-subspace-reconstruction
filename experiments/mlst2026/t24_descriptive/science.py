"""T24-D: descriptive post-whitening moment diagnostic on CRESST noise triggers.

Closed design (plan/T24D_DESCRIPTIVE_PROTOCOL_2026-09-30.md). Units: the seven
historical CRESST release-test channels (historically exposed by design). Selection,
chronological split, preprocessing, noise metric and rank-6 weighted-PCA basis are
the historical validation settings, reproduced with the same library functions and
fitted on calibration rows only. Statistics are the reviewed T24 core (Mardia b1/b2
after within-block restandardization). The iid-Gaussian reference is a SCALE
reference for the same (N, p): it is not a null distribution for these temporally
dependent, selected records, and no p value, test or classification is produced.
"""
from __future__ import annotations
import hashlib
import numpy as np
from experiments.mlst2026.t24_cumulant import core
from noise_geometry.noise import covariance_from_psd, estimate_psd_ensemble, inverse_covariance, regularize_covariance
from noise_geometry.subspace import fit_weighted_pca
from noise_geometry.cresst.splits import time_disjoint_split

CONFIG = {
    'version': 't24d/1',
    'split': 'test',
    'units': ['33/36', '33/37', '35/26', '35/27', '35/56', '35/57', '35/58'],
    'width': 512,
    'baseline_fraction': 0.2,
    'calibration_fraction': 0.7,
    'covariance_model': 'psd_stationary median-PSD circulant',
    'covariance_shrinkage': 0.1,
    'retained_rank': 6,
    'residual_dimension': 32,
    'reference_draws': 999,
    'reference_quantiles': [0.05, 0.95],
    'noise_selection': 'noise flag (historical); clean flag not required',
    'pulse_selection': 'clean and not noise (historical); calibration segment only',
    'seed_namespace': 'paper1-t24d-v1',
}


def rng(*key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256('/'.join(map(str, key)).encode()).digest()[:8], 'little'))


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def preprocess(x, fraction):
    """Historical _prep_traces with max_features=width: per-trace baseline subtraction."""
    x = np.asarray(x, dtype=np.float64)
    b = max(1, int(x.shape[1] * float(fraction)))
    return x - x[:, :b].mean(axis=1, keepdims=True)


def whitening(noise_calibration, shrinkage):
    """Historical psd_stationary metric; SYMMETRIC square root as whitening transform.
    The circulant metric has degenerate eigenvalue pairs, so V*sqrt(L) (the
    fit_weighted_pca convention) is fixed only up to roundoff rotations inside each
    pair; V*sqrt(L)*V^T is unique. The retained span is identical either way."""
    _, psd = estimate_psd_ensemble(noise_calibration, average='median')
    cov = regularize_covariance(covariance_from_psd(psd, noise_calibration.shape[1]), shrinkage=shrinkage)
    metric = inverse_covariance(cov)
    vals, vecs = np.linalg.eigh(0.5 * (metric + metric.T))
    vals = np.clip(vals, np.finfo(float).eps, None)
    return metric, (vecs * np.sqrt(vals)[None, :]) @ vecs.T


def bases(pulse_calibration, metric, transform, rank, residual_dim, unit, namespace):
    """Whitened orthonormal retained span (historical EMPCA) and a fixed seeded
    orthonormal residual block inside its orthogonal complement."""
    fit = fit_weighted_pca(pulse_calibration, metric, rank)
    retained = fit.components @ transform            # whitened span (orthonormalized below)
    q, _ = np.linalg.qr(retained.T)
    retained = q.T
    g = rng(namespace, unit, 'residual').normal(size=(transform.shape[1], residual_dim))
    g -= retained.T @ (retained @ g)
    residual, r = np.linalg.qr(g)
    if np.min(np.abs(np.diag(r))) < 1e-8:
        raise ValueError('residual basis rank loss')
    return retained, residual


def reference(n, p, draws, quantiles, *key):
    """iid-Gaussian scale reference for b1/b2 at the same (N, p)."""
    g = rng(*key)
    b1 = np.empty(draws); b2 = np.empty(draws)
    for i in range(draws):
        s = core.statistics(g.normal(size=(n, p)))
        b1[i], b2[i] = s['b1'], s['b2']
    q = lambda v: [float(x) for x in np.quantile(v, quantiles)]
    return {'draws': draws, 'b1_mean': float(b1.mean()), 'b1_quantiles': q(b1), 'b2_mean': float(b2.mean()),
            'b2_quantiles': q(b2), 'b2_exact_mean_iid': p * (p + 2) * (n - 1) / (n + 1),
            'scope': 'scale reference for iid Gaussian rows of this (N, p); not a null for dependent/selected records'}


def describe(y, unit, block, c):
    n, p = y.shape
    s = core.statistics(y)
    ref = reference(n, p, c['reference_draws'], c['reference_quantiles'], c['seed_namespace'], unit, block, 'reference')
    half = n // 2
    halves = []
    for name, part in (('early', y[:half]), ('late', y[half:])):
        try:
            hs = core.statistics(part)
            hr = reference(len(part), p, c['reference_draws'], c['reference_quantiles'], c['seed_namespace'], unit, block, name)
            halves.append({'half': name, 'N': len(part), 'b2': hs['b2'], 'b2_reference_mean': hr['b2_mean'],
                           'b2_reference_quantiles': hr['b2_quantiles']})
        except ValueError as e:
            halves.append({'half': name, 'N': len(part), 'status': 'unavailable', 'reason': str(e)})
    return {'N': n, 'p': p, 'b1': s['b1'], 'b2': s['b2'], 'b2_ratio_to_reference_mean': s['b2'] / ref['b2_mean'],
            'marginal_skewness': s['marginal_skewness'], 'marginal_excess_kurtosis': s['marginal_excess'],
            'condition': s['condition'], 'reference': ref, 'chronological_halves': halves}


def unit_arrays(x, f, unit, c):
    """All row selection and fitting for one unit; evaluation rows enter only the projection."""
    run, ch = (float(v) for v in unit.split('/'))
    g = f[(f.run.astype(float) == run) & (f.channel.astype(float) == ch)]
    noise = g[g.noise]
    pulses = g[g.clean & ~g.noise]
    ncal, neval = time_disjoint_split(noise.hours.to_numpy(float), c['calibration_fraction'])
    pcal, _ = time_disjoint_split(pulses.hours.to_numpy(float), c['calibration_fraction'])
    ids = lambda frame, mask: frame.release_row_index.to_numpy(np.int64)[mask]
    noise_cal, noise_eval, pulse_cal = ids(noise, ncal), ids(noise, neval), ids(pulses, pcal)
    order = np.argsort(f.hours.to_numpy(float)[noise_eval], kind='stable')
    noise_eval = noise_eval[order]
    Xn_cal = preprocess(x[noise_cal], c['baseline_fraction'])
    Xp_cal = preprocess(x[pulse_cal], c['baseline_fraction'])
    Xn_eval = preprocess(x[noise_eval], c['baseline_fraction'])
    metric, transform = whitening(Xn_cal, c['covariance_shrinkage'])
    retained, residual = bases(Xp_cal, metric, transform, c['retained_rank'], c['residual_dimension'], unit, c['seed_namespace'])
    w = Xn_eval @ transform
    return {'noise_calibration_ids': noise_cal, 'pulse_calibration_ids': pulse_cal, 'noise_evaluation_ids': noise_eval,
            'evaluation_hours': f.hours.to_numpy(float)[noise_eval], 'transform': transform,
            'retained_basis': retained, 'residual_basis': residual,
            'retained': w @ retained.T, 'residual': w @ residual}


def unit_summary(arrays, unit, c, row_counts):
    ev = arrays['evaluation_hours']
    return {'unit': unit, 'counts': row_counts,
            'noise_calibration': int(len(arrays['noise_calibration_ids'])), 'pulse_calibration': int(len(arrays['pulse_calibration_ids'])),
            'noise_evaluation': int(len(arrays['noise_evaluation_ids'])), 'evaluation_hours': [float(ev.min()), float(ev.max())],
            'retained': describe(arrays['retained'], unit, 'retained', c),
            'residual': describe(arrays['residual'], unit, 'residual', c)}
