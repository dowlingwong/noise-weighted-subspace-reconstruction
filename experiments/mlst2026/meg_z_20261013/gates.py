"""D2 noise-qualification gates G1-G5, the window-dependence statistic and exclusion rule E1 (runbook §6.3 as
amended by the independent review megz_kit_review_20261013_01).  Thresholds are read from the protocol's
"gates" block (dev defaults; frozen at D4 from the dev subjects only).

G1 rank/SPD        (i) raw data: empty processing history (not MaxFiltered; preprocess refuses otherwise);
                   (ii) condition number of the SPATIAL partial trace tr_t Sigma_ref / T < max_condition_spatial
                   (1e6; detects SSS-type rank loss); (iii) full cond(Sigma_ref) < max_condition_full (1e10;
                   float64 safety).  The temporal condition is reported.  On failure of (ii) or (iii) the
                   noisiest selected channel is dropped and the subject re-preprocessed and re-gated ONCE,
                   automatically (``gate_subject_with_retest``), before E1 decides (F13).
G2 stationarity    evaluation (run 6) noise whitened by Sigma_ref: the mean whitened eigenvalue tr(Sigma_ref^-1 S)/d
                   and the trace ratio tr S / tr Sigma_ref inside the band, AND the spread: max whitened eigenvalue
                   <= r x the Marchenko-Pastur upper edge (1 + sqrt(d/n))^2 and min >= MP lower edge / r
                   (r = spread_ratio_max, set from dev only; F10).
G3 locked mean     EFFECT-SIZE gate (F3): for each partition, alignment (tiling slot after the preceding onset,
                   ``position``; slot before the next onset, ``rposition``) and slot with >= min_windows windows,
                   the debiased whitened energy of the positional mean, E = mu^T W mu - tr(W S_pos)/n_pos
                   (the next-onset alignment is accurate to within one window: slots are tiled from the
                   preceding onset; adequate for an effect-size probe)
                   (W = Sigma_ref^{-1}), must stay below effect_size_max_fraction_trK x trace_K_target.
                   Significance (max-t, Hotelling) is computed on GAP MEANS (one vector per inter-stimulus gap,
                   which handles within-gap dependence) and reported, not gating.  Disclosure: the windows are
                   inter-stimulus activity and may contain responses to the circle / fixation-cross transitions.
DEP                descriptive (F9): whitened lag-one cross-covariance of consecutive windows of one gap,
                   ||C_1||_F / ||C_0||_F and tr C_1 / tr C_0 (training partition).
G4 separability    descriptive: whitened error of the Kronecker MLE against the Ledoit-Wolf full reference,
                   nearest-Kronecker relative distances.
G5 Gaussianity     descriptive: Mardia b1/b2 after standardisation (T24 core ``statistics``) on the training pool.
E1                 a subject failing G1-G3 (after the automatic G1 retest) is excluded; with the budget rule
                   (runner / predictions) fewer than min_confirmation_subjects included confirmation subjects ->
                   stop (qualification failure; no replacement).

Command (EXP root, PYTHONPATH=src):
    python -m experiments.mlst2026.meg_z_20261013.gates --derived DIR --subjects sub-01,... --out OUT/gates.json
        [--bids-root RAW --data-manifest DATA_MANIFEST.json]   (needed for the automatic G1 retest)
Confirmation subjects are refused unless the protocol is frozen (F7).
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from scipy import stats

from . import common, estimators
from .common import symmetric
from experiments.mlst2026.t24_cumulant import core as t24core


def _powers(a):
    w, q = np.linalg.eigh(symmetric(a))
    return (q * np.sqrt(w)) @ q.T, (q / np.sqrt(w)) @ q.T


def factor_conditions(cov: np.ndarray, C: int, T: int) -> dict:
    S4 = np.asarray(cov).reshape(C, T, C, T, order='F')
    out = {}
    for name, m in (('spatial', np.einsum('ctdt->cd', S4) / T), ('temporal', np.einsum('ctcs->ts', S4) / C)):
        w = np.linalg.eigvalsh(symmetric(m))
        out[f'condition_{name}'] = float(w[-1] / w[0]) if w[0] > 0 else float('inf')
    return out


def g1_spd(cov: np.ndarray, C: int, T: int, cfg: dict, proc_history_empty: bool = True) -> dict:
    w = np.linalg.eigvalsh(symmetric(cov))
    cond = float(w[-1] / w[0]) if w[0] > 0 else float('inf')
    out = {'condition': cond, 'min_eig': float(w[0]), 'max_eig': float(w[-1]), **factor_conditions(cov, C, T),
           'proc_history_empty': bool(proc_history_empty)}
    out['checks'] = {'raw_not_maxfiltered': bool(proc_history_empty),
                     'spatial_condition': bool(out['condition_spatial'] < float(cfg['max_condition_spatial'])),
                     'full_condition': bool(w[0] > 0 and cond < float(cfg['max_condition_full']))}
    out['pass'] = bool(all(out['checks'].values()))
    return out


def noisiest_channel(cov: np.ndarray, C: int, T: int, channels: list) -> dict:
    v = np.diag(cov).reshape(T, C).mean(0)          # index c + C t -> [t, c]
    i = int(np.argmax(v))
    return {'channel': channels[i], 'index': i, 'mean_variance': float(v[i])}


def g2_stationarity(ref_cov: np.ndarray, X: np.ndarray, band, spread_ratio_max: float | None = None) -> dict:
    st = estimators.moment_stats(X)
    d = len(ref_cov)
    _, li = _powers(ref_cov)
    ev = np.linalg.eigvalsh(symmetric(li @ st.S @ li))
    mean_eig = float(np.mean(ev))
    ratio = float(np.trace(st.S) / np.trace(ref_cov))
    gamma = d / st.n
    mp_hi, mp_lo = (1 + np.sqrt(gamma)) ** 2, max((1 - np.sqrt(gamma)) ** 2, 0.0)
    out = {'mean_whitened_eigenvalue': mean_eig, 'trace_ratio': ratio,
           'geometric_mean_eigenvalue': float(np.exp(np.mean(np.log(np.maximum(ev, 1e-300))))),
           'min_whitened_eigenvalue': float(ev[0]), 'max_whitened_eigenvalue': float(ev[-1]), 'n': st.n, 'd': d,
           'mp_edges': [float(mp_lo), float(mp_hi)], 'max_over_mp_upper': float(ev[-1] / mp_hi),
           'mp_lower_over_min': float(mp_lo / ev[0]) if ev[0] > 0 else float('inf'), 'band': list(band)}
    checks = {'mean_in_band': bool(band[0] <= mean_eig <= band[1]), 'trace_ratio_in_band': bool(band[0] <= ratio <= band[1])}
    if spread_ratio_max is not None:
        checks['spread'] = bool(out['max_over_mp_upper'] <= spread_ratio_max and out['mp_lower_over_min'] <= spread_ratio_max)
        out['spread_ratio_max'] = float(spread_ratio_max)
    out['checks'] = checks
    out['pass'] = bool(all(checks.values()))
    return out


def locked_mean_tests(X: np.ndarray) -> dict:
    """p-values for H0: E x = 0 (records X, n x d): max-t (Bonferroni over d) and Hotelling T^2 if n > d + 1."""
    X = np.asarray(X, float)
    n, d = X.shape
    out = {'n': n, 'd': d, 'max_t_bonferroni': None, 'hotelling_if_n_gt_d': None}
    if n < 3:
        return out
    m = X.mean(0)
    s = X.std(0, ddof=1)
    t = m / (s / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), n - 1)
    out['max_abs_t'] = float(np.max(np.abs(t)))
    out['max_t_bonferroni'] = float(min(1.0, d * float(np.min(p))))
    if n > d + 1:
        S = np.cov(X, rowvar=False)
        T2 = float(n * m @ np.linalg.solve(S, m))
        out['hotelling_T2'] = T2
        out['hotelling_if_n_gt_d'] = float(stats.f.sf((n - d) / (d * (n - 1)) * T2, d, n - d))
    return out


def positional_energy(X: np.ndarray, W: np.ndarray) -> dict:
    """Debiased whitened energy of the mean of the rows of X: mu^T W mu - tr(W S)/n (S divisor n - 1)."""
    n = len(X)
    mu = X.mean(0)
    S = np.cov(X, rowvar=False) if n > 1 else np.zeros((X.shape[1],) * 2)
    raw = float(mu @ W @ mu)
    return {'n': n, 'energy_raw': raw, 'energy_debiased': raw - float(np.sum(W * S)) / n}


def gap_means(X: np.ndarray, gap: np.ndarray, run: np.ndarray) -> np.ndarray:
    keys = np.stack([run, gap], 1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    return np.stack([X[inv == g].mean(0) for g in range(inv.max() + 1)])


def g3_locked_mean(sd, W: np.ndarray, cfg: dict, trace_K_target: float) -> dict:
    thr = float(cfg['effect_size_max_fraction_trK']) * float(trace_K_target)
    rows, worst = [], -np.inf
    sig = []
    for part in cfg['partitions']:
        X = sd.records(part)
        meta = sd.window_meta[part]
        for al in cfg['alignments']:
            pos = np.asarray(meta[al])
            for k in np.unique(pos):
                sel = pos == k
                if sel.sum() < int(cfg['min_windows_per_position']):
                    continue
                e = positional_energy(X[sel], W)
                e.update(partition=part, alignment=al, slot=int(k), fraction_of_trK=e['energy_debiased'] / trace_K_target)
                rows.append(e)
                worst = max(worst, e['energy_debiased'])
        gm = gap_means(X, np.asarray(meta['gap']), np.asarray(meta['run']))
        t = locked_mean_tests(gm)
        t.update(partition=part, unit='gap means')
        sig.append(t)
    return {'threshold_energy': thr, 'max_debiased_energy': float(worst) if rows else None,
            'max_fraction_of_trK': float(worst / trace_K_target) if rows else None, 'positions': rows,
            'significance_on_gap_means_reported_not_gating': sig,
            'pass': bool(rows) and bool(worst < thr),
            'disclosure': 'inter-stimulus windows may contain circle / fixation-cross responses'}


def lag_one_dependence(sd, W: np.ndarray, part: str = 'training') -> dict:
    """Whitened lag-one cross-covariance between consecutive windows of one gap (F9; descriptive)."""
    X = sd.records(part)
    m = sd.window_meta[part]
    _, li = _powers(np.linalg.inv(W))
    Z = (X - X.mean(0)) @ li
    order = np.lexsort((np.asarray(m['start']), np.asarray(m['gap']), np.asarray(m['run'])))
    a, b = [], []
    for i, j in zip(order, order[1:]):
        if m['run'][i] == m['run'][j] and m['gap'][i] == m['gap'][j] and m['start'][j] - m['start'][i] == sd.T:
            a.append(i)
            b.append(j)
    if len(a) < 3:
        return {'pairs': len(a), 'status': 'unavailable'}
    Za, Zb = Z[a], Z[b]
    C1 = Za.T @ Zb / len(a)
    C0 = 0.5 * (Za.T @ Za + Zb.T @ Zb) / len(a)
    return {'pairs': len(a), 'partition': part, 'frobenius_ratio': float(np.linalg.norm(C1) / np.linalg.norm(C0)),
            'trace_ratio': float(np.trace(C1) / np.trace(C0)),
            'note': 'contiguous windows are kept (budget); this dependence is a prestated candidate reason for a PZ2 miss'}


def g4_separability(sd) -> dict:
    lw, lwi = estimators.ledoit_wolf_from_stats(sd.ref)
    St, Sc, ffi = estimators.kronecker_flipflop_from_S(sd.ref.S, sd.C, sd.T)
    kron = np.kron(St, Sc)
    return {'kron_vs_lw_whitened': estimators.whitened_error(kron, lw),
            'lw_vs_kron_whitened': estimators.whitened_error(lw, kron),
            'nkp_relative_distance_lw': estimators.nearest_kronecker(lw, sd.C, sd.T)[2],
            'nkp_relative_distance_sample': estimators.nearest_kronecker(sd.ref.S, sd.C, sd.T)[2],
            'lw_shrinkage': lwi['shrinkage'], 'lw_n_eff': lwi['n_eff'], 'flipflop': ffi, 'n_reference': sd.ref.n}


def g5_mardia(X: np.ndarray) -> dict:
    n, p = X.shape
    if n < p + 2:
        return {'status': 'unavailable', 'reason': f'n = {n} < d + 2 = {p + 2}'}
    s = t24core.statistics(X)
    e_b2 = p * (p + 2) * (n - 1) / (n + 1)
    return {'status': 'available', 'n': n, 'p': p, 'b1': s['b1'], 'b2': s['b2'], 'b2_gaussian_expectation': e_b2,
            'b2_ratio': s['b2'] / e_b2, 'b2_z': (s['b2'] - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n),
            'mean_marginal_excess_kurtosis': float(np.mean(s['marginal_excess']))}


def subject_gates(sd, proto: dict) -> dict:
    gc = proto['gates']
    rm = proto['reference_metric']
    ref = estimators.reference_metric(sd.ref, sd.C, sd.T, rm['kind'], rm.get('flipflop'))
    W = np.linalg.inv(symmetric(ref['cov']))
    out = {'subject': sd.subject, 'C': sd.C, 'T': sd.T, 'reference_kind': ref['kind'], 'channels': sd.channels,
           'drop_selected': sd.attrs.get('drop_selected', [])}
    out['G1'] = g1_spd(ref['cov'], sd.C, sd.T, gc['G1'], bool(sd.attrs.get('proc_history_empty', False)))
    if not out['G1']['pass']:
        out['G1']['retest_drop'] = noisiest_channel(ref['cov'], sd.C, sd.T, sd.channels)
    g2 = gc['G2']
    out['G2'] = g2_stationarity(ref['cov'], sd.records(g2['partition']), g2['band'], g2.get('spread_ratio_max'))
    out['G3'] = g3_locked_mean(sd, W, gc['G3'], float(proto['signal']['trace_K_target']))
    out['dependence'] = lag_one_dependence(sd, W)
    out['G4'] = g4_separability(sd)
    out['G5'] = g5_mardia(sd.records(gc['G5']['partition']))
    failed = [g for g in gc['E1']['exclude_if_failed'] if not out[g]['pass']]
    out['failed'] = failed
    out['excluded'] = bool(failed)
    return out


def gate_subject_with_retest(derived, subject: str, proto: dict, bids_root=None, data_manifest=None, runs_loader=None) -> dict:
    """Gates with the prespecified automatic G1 retest (F13): if G1 fails and no channel has been dropped yet,
    re-preprocess once without the noisiest selected channel (overwriting the derived file and its manifest
    entry; the first attempt's gates are kept in the record) and gate again; E1 then decides."""
    from .preprocess import load_subject, process_subject, preprocess_config, update_derived_manifest
    derived = Path(derived)
    sd = load_subject(derived / f'{subject}.h5', cache=False)
    r = subject_gates(sd, proto)
    if r['G1']['pass'] or sd.attrs.get('drop_selected'):
        r['G1_retest'] = None
        return r
    if bids_root is None:
        r['G1_retest'] = {'status': 'NOT RUN: --bids-root needed for the automatic retest'}
        return r
    drop = r['G1']['retest_drop']['channel']
    s = process_subject(bids_root, subject, preprocess_config(proto), derived, drop_selected=[drop],
                        data_manifest=data_manifest, runs_loader=runs_loader)
    update_derived_manifest(derived, [s])
    r2 = subject_gates(load_subject(derived / f'{subject}.h5', cache=False), proto)
    r2['G1_retest'] = {'status': 'done', 'dropped': drop, 'first_attempt': {k: r[k] for k in ('G1', 'G2', 'G3', 'failed')}}
    return r2


def exclusion(gates_by_subject: dict, proto: dict, mode: str, extra_excluded: dict | None = None) -> dict:
    """E1: drop subjects failing G1-G3 (after the retest) or excluded by the budget/population rule
    (``extra_excluded``: subject -> reasons); in confirmation mode stop below the minimum."""
    subs = proto['subjects']['confirmation'] if mode == 'confirmation' else proto['subjects']['dev']
    extra = extra_excluded or {}
    present = [s for s in subs if s in gates_by_subject]
    excluded = [s for s in present if gates_by_subject[s]['excluded'] or s in extra]
    included = [s for s in present if s not in excluded]
    missing = [s for s in subs if s not in gates_by_subject]
    stop = mode == 'confirmation' and len(included) < int(proto['gates']['E1']['min_confirmation_subjects'])
    return {'mode': mode, 'included': included, 'excluded': excluded,
            'reasons': {s: (gates_by_subject[s].get('failed', []) + list(extra.get(s, []))) for s in excluded},
            'missing_gates': missing, 'stop': bool(stop),
            'reason': 'fewer than the minimum number of qualified confirmation subjects (E1)' if stop else None}


def main(argv=None):
    from .preprocess import refuse_unfrozen_confirmation
    from .protocol import load_protocol
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', required=True)
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--protocol')
    ap.add_argument('--bids-root')
    ap.add_argument('--data-manifest')
    a = ap.parse_args(argv)
    proto = load_protocol(a.protocol)
    subs = a.subjects.split(',')
    refuse_unfrozen_confirmation(subs, proto, a.protocol)
    dm = json.loads(Path(a.data_manifest).read_text()) if a.data_manifest else None
    out = {'protocol_sha256': common.sha_json(proto), 'gates_config': proto['gates'], 'subjects': {},
           'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for s in subs:
        r = gate_subject_with_retest(a.derived, s, proto, a.bids_root, dm)
        man = json.loads((Path(a.derived) / 'DERIVED_MANIFEST.json').read_text())
        r['derived_file_sha256'] = man['subjects'][s]['file_sha256']
        out['subjects'][s] = r
        print(s, 'failed:', r['failed'], 'G1 cond %.3g spatial %.3g temporal %.3g' % (
            r['G1']['condition'], r['G1']['condition_spatial'], r['G1']['condition_temporal']),
            'G2 %.3f/%.3f spread %.2f' % (r['G2']['mean_whitened_eigenvalue'], r['G2']['trace_ratio'], r['G2']['max_over_mp_upper']),
            'G3 max fraction %s' % r['G3']['max_fraction_of_trK'], 'retest', (r.get('G1_retest') or {}).get('status'))
    for mode in ('dev', 'confirmation'):
        out['E1_' + mode] = exclusion(out['subjects'], proto, mode)
    common.atomic_json(a.out, out)


if __name__ == '__main__':
    main()
