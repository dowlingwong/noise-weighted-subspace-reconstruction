"""Y19 population (data-free) quantities and predictions; writes predictions.json and protocol.json.

No record is sampled here and no learner is fitted.  Everything is computed from the generator's
population K(phi) with the S14.3 first-order constants, through the TR09 first-order code that
produced the T25 weight table (EXP p1nl_dev_20261006/tr09_first_order/core.py, imported unchanged,
SHA-256 pinned in protocol.json):

    c_U  = sum_{i<=k<j} ell_i ell_j / (ell_i - ell_j)                (core.unrestricted_coefficient)
    c_F  = 1/2 tr(H^{-1} Sigma_g), Sigma_g = 2 tr(G_a M G_b M)          (core.first_order)
    a    = R_K(P*_F,best) - R_K(P*_U), P*_F,best = best of >= 50 Newton restarts
           (core.population_product_optimum; HOSVD + phi=0 factors + random starts)
    B    = c_F - c_U + a  (fitted-mean affine convention; core.summary, Gaussian latents K = 0)
    N*   = -B/a = (c_U - c_F)/a - 1
    interval [ (c_U - c_F)/a_best - 1 , (c_U - c_F)/B_cert - 1 ],
    B_cert = Delta_k (k - min{k, s_c, s_t})  (03_restriction.tex Eq. mlst-zero-bound;
             t20s.generator.partial_trace_cap, the T25/T20-S implementation).

Commands (EXP root, PYTHONPATH=src, PYTHONDONTWRITEBYTECODE=1):
    python -m experiments.mlst2026.y19_20261012.predictions write     # predictions.json + protocol.json
    python -m experiments.mlst2026.y19_20261012.predictions check     # recompute and compare, no write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import generator as gen                                                           # runs the T25 byte guard
from experiments.mlst2026.successor_20260913.t20s.generator import partial_trace_cap    # noqa: E402
from experiments.mlst2026.p1nl_dev_20261006.tr09_first_order import core                # noqa: E402
from experiments.mlst2026.t25_production_20260930 import science as t25                 # noqa: E402

PACKAGE = Path(__file__).resolve().parent
EXP_ROOT = PACKAGE.parents[2]
NAMESPACE = 'y19-confirm-20261012'
A_ZERO_TOL = 1e-9

#: G2 generator design (registered; see generator.py for the construction).
GENERATOR = {
    'family': gen.FAMILY,
    'C': 6, 'T': 64, 'kc': 2, 'kt': 3,
    'rho_c': 0.4, 'rho_t': 0.7,
    'spectrum': {'top_scale': 24.0, 'gap': 2.0,
                 'law': 'lambda_i = top_scale/i (i <= k); lambda_j = (top_scale/k - gap)(k+1)/j (j > k)'},
    'pulses': {'onset': 16.0, 'rise_decay': [[1.0, 6.0], [2.5, 15.0], [5.0, 35.0]], 'weights': [1.0, 0.7, 0.45]},
    'channel_patterns': {'forms': ['exp(-(c-c0)^2/8)', '(c-c0)/c0'], 'weights': [1.0, 0.6]},
    'delays': 'delta_c = c - (C-1)/2 samples per unit phi',
    'geometry_key': NAMESPACE + '/geometry',
}

#: phi-selection rule (population quantities only), fixed before any fit.
PHI_SCAN = {'grid': [round(0.01 * i, 2) for i in range(1, 51)], 'n_random': 6,
            'rule': 'phi_A = smallest grid phi with N*_pred <= 3000; phi_B = largest grid phi with N*_pred >= 1200; '
                    'require N*_A / N*_B >= 2 and a > 0, B < 0, H > 0, c_U > c_F at both',
            'upper': 3000.0, 'lower': 1200.0, 'min_ratio': 2.0}
#: Registered phi values (the output of the phi-selection rule on 12 Oct 2026; re-derived by `write`/`check`).
REGISTERED_PHI = {'A': 0.12, 'B': 0.20}
N_RANDOM_FINAL = 62            # + HOSVD + phi=0 factors = 64 Newton restarts (>= 50)
FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]
N_SEEDS = 24


def seeds() -> list[int]:
    return [gen.seed_int(NAMESPACE, 'seed', i) % 10 ** 12 for i in range(N_SEEDS)]


def round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


# --------------------------------------------------------------------------------------------
def certificate(model) -> dict:
    w, U = np.linalg.eigh(model.K)
    order = np.argsort(w)[::-1]
    w = w[order]
    Uk = U[:, order[:model.k]]
    pt = partial_trace_cap(Uk, model.C, model.T, model.kc, model.kt)
    gap = float(w[model.k - 1] - w[model.k])
    B = gap * pt['k_minus_cap']
    # separability residual of P*_U: distance to the product of its partial-trace eigenprojectors
    P = Uk @ Uk.T
    P4 = P.reshape(model.C, model.T, model.C, model.T, order='F')
    Uc = np.linalg.eigh(np.einsum('ctdt->cd', P4))[1][:, -model.kc:]
    Ut = np.linalg.eigh(np.einsum('ctcs->ts', P4))[1][:, -model.kt:]
    sep = float(np.linalg.norm(P - np.kron(Ut @ Ut.T, Uc @ Uc.T)))
    return {'B_cert': float(B), 'Delta_k': gap, 'partial_trace': pt,
            'spectrum_max_error': float(np.max(np.abs(w - np.sort(model.mu)[::-1]))),
            'separability_residual_fro': sep, 'trace_K': float(np.trace(model.K))}


def constants(model, rng, n_random: int) -> dict:
    """All S14.3 population quantities for one model (G2 or T25)."""
    pt, dist = core.population_product_optimum(model, rng, n_random=n_random)
    fo = core.first_order(model, pt)
    s = core.summary(model, pt, fo)
    cert = certificate(model)
    a, cU, cF, B = s['a'], s['cU'], s['cF_gaussian'], s['B']
    Hinv, Sg = fo['Hinv'], fo['Sigma_g']
    ell, k = model.ell, model.k
    w = np.outer(ell[:k], ell[k:]) / (ell[:k, None] - ell[None, k:])
    row = {'phi': model.theta, 'a_best': a, 'cU': cU, 'cF': cF, 'B': B, 'B_formula_cF_minus_cU_plus_a': cF - cU + a,
           'B_cert': cert['B_cert'], 'Delta_k': cert['Delta_k'], 'partial_trace': cert['partial_trace'],
           'spectrum_max_error': cert['spectrum_max_error'], 'trace_K': cert['trace_K'],
           'separability_residual_fro': cert['separability_residual_fro'],
           'H_min_eig': fo['H_min_eig'], 'H_max_eig': fo['H_max_eig'], 'gradient_norm_at_optimum': fo['gradient_norm'],
           'n_starts': len(dist), 'distinct_population_maxima': sorted({round(d['value'], 9) for d in dist}),
           'all_starts_reach_best': bool(max(d['gap_to_best'] for d in dist) < 1e-9),
           'starts_reaching_best': int(sum(d['gap_to_best'] < 1e-9 for d in dist)),
           'start_statuses': sorted({d['status'] for d in dist}),
           'power_heuristic_first_order_sd': {
               'N_sd_Delta_U': float(np.sqrt(2 * np.sum(w ** 2))),
               'N_sd_Delta_F': float(np.sqrt(0.5 * np.trace(Hinv @ Sg @ Hinv @ Sg))),
               'note': 'first-order SDs of the subspace excesses times N (Gaussian); no contrast is simulated'}}
    nonsep = a > A_ZERO_TOL
    row['N_star_pred'] = (cU - cF) / a - 1 if nonsep and B < 0 else None
    row['N_star_interval'] = ([(cU - cF) / a - 1, (cU - cF) / cert['B_cert'] - 1]
                              if nonsep and B < 0 and cert['B_cert'] > A_ZERO_TOL else None)
    row['checks'] = {'cU_gt_cF': bool(cU > cF), 'a_pos': bool(nonsep), 'B_neg': bool(B < 0),
                     'H_pos_def': bool(fo['H_min_eig'] > 0), 'gap_pos': bool(cert['Delta_k'] > 0),
                     'cert_le_a': bool(cert['B_cert'] <= a + 1e-9),
                     'formula_B_matches': bool(abs(B - (cF - cU + a)) <= 1e-9 * abs(cU))}
    return row


def g2_constants(g, phi: float, n_random: int, stream: str) -> dict:
    m = gen.PopulationModel(g, phi)
    row = constants(m, t25.rng(NAMESPACE, stream, float(phi)), n_random)
    # the physical top-k signal subspace equals the template span (construction check)
    Bt = g.templates(phi)
    Pt = Bt @ np.linalg.pinv(Bt)
    LV = g.L @ m.V[:, :m.k]
    row['template_span_residual'] = float(np.linalg.norm(LV - Pt @ LV) / np.linalg.norm(LV))
    _, w0 = g.top_basis(phi)
    row['template_K0_eigenvalues'] = [float(x) for x in w0]
    return row


def grid(n_star: float) -> list[int]:
    return [round_half_up(n_star * f) for f in FACTORS]


def phi_scan(g) -> dict:
    rows = []
    for phi in PHI_SCAN['grid']:
        r = g2_constants(g, phi, PHI_SCAN['n_random'], 'phi-scan')
        rows.append({k: r[k] for k in ('phi', 'a_best', 'B_cert', 'cU', 'cF', 'N_star_pred', 'H_min_eig', 'all_starts_reach_best')})
    ok = [r for r in rows if r['N_star_pred'] is not None]
    phi_A = min(r['phi'] for r in ok if r['N_star_pred'] <= PHI_SCAN['upper'])
    phi_B = max(r['phi'] for r in ok if r['N_star_pred'] >= PHI_SCAN['lower'])
    return {'rows': rows, 'phi_A': phi_A, 'phi_B': phi_B, **{k: v for k, v in PHI_SCAN.items() if k != 'grid'},
            'grid': PHI_SCAN['grid']}


def provenance() -> dict:
    files = {str(p.relative_to(EXP_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [PACKAGE / 'generator.py', PACKAGE / 'predictions.py',
                       EXP_ROOT / 'experiments/mlst2026/p1nl_dev_20261006/tr09_first_order/core.py']}
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=EXP_ROOT, text=True, stderr=subprocess.DEVNULL,
                                       env={'GIT_OPTIONAL_LOCKS': '0', 'PATH': '/usr/bin:/bin'}).strip()
    except Exception:          # noqa: BLE001
        head = None
    return {'files': files, 'EXP_HEAD_at_computation': head, 'python': platform.python_version(), 'numpy': np.__version__}


def compute(scan: bool = True) -> dict:
    """Population quantities at the control and the two registered geometries (plus the phi scan)."""
    g = gen.build(GENERATOR)
    out = {'note': 'population (data-free) quantities only; no record sampled, no learner fitted',
           'namespace': NAMESPACE, 'generator': GENERATOR,
           'spectrum_summary': {'top_k': [float(x) for x in g.mu[:g.k]], 'lambda_k_plus_1': float(g.mu[g.k]),
                                'Delta_k': g.Delta_k, 'tail_min': float(g.mu[-1]), 'trace_K': float(g.mu.sum()),
                                'top_k_energy': float(g.mu[:g.k].sum()), 'snr_per_coordinate': float(g.mu.sum() / g.d),
                                'T25_reference': {'top_k': [12, 11, 10, 9, 8, 7], 'Delta_k': 4.0, 'trace_K': 206.604,
                                                  'top_k_energy': 57.0, 'snr_per_coordinate': 0.807}}}
    if scan:
        out['phi_selection'] = phi_scan(g)
    rows = {lab: g2_constants(g, phi, N_RANDOM_FINAL, 'population') for lab, phi in
            (('control', 0.0), ('A', REGISTERED_PHI['A']), ('B', REGISTERED_PHI['B']))}
    out['geometries'] = rows
    nA, nB = rows['A']['N_star_pred'], rows['B']['N_star_pred']
    out['grids'] = {'A': grid(nA), 'B': grid(nB)}
    out['grids']['control'] = sorted(set(out['grids']['A']) | set(out['grids']['B']))
    out['ratio_N_star_A_over_B'] = nA / nB
    c = rows['control']
    out['all_checks_pass'] = bool(all(all(rows[lab]['checks'].values()) for lab in ('A', 'B'))
                                  and c['checks']['gap_pos'] and c['checks']['H_pos_def'] and c['checks']['cU_gt_cF']
                                  and abs(c['a_best']) < A_ZERO_TOL and c['B_cert'] < 1e-12
                                  and c['separability_residual_fro'] < 1e-12)
    out['provenance'] = provenance()
    return out


def build_protocol(pred: dict, dependencies: dict) -> dict:
    labels = ['control', 'A', 'B']
    geoms = [{'label': lab, 'phi': pred['geometries'][lab]['phi'],
              'N_star_pred': pred['geometries'][lab]['N_star_pred'],
              'N_star_interval': pred['geometries'][lab]['N_star_interval'],
              'grid': pred['grids'][lab]} for lab in labels]
    family = sum(len(gm['grid']) for gm in geoms)
    return {
        'study': 'Y19', 'version': 'y19-v1', 'evidence_role': 'CONFIRMATION', 'namespace': NAMESPACE,
        'generator': pred['generator'], 'metric': 'oracle Sigma^{-1}',
        'geometries': geoms, 'factors': FACTORS, 'seeds': seeds(),
        'learners': {
            'records': 'y19 generator.draw (t25.science.make_data streams, oracle arm, no calibration pool); one pool of '
                       'max(grid) records per (geometry, seed); smaller N are prefixes',
            'moments': 't25.science.moments (fitted mean, centred, divisor N)',
            'unrestricted': 't25.science.unrestricted (EMPCA/CW-PCA population form: Sigma^{-1}-whitened top-k)',
            'product': 't25.science.structured (NFPA, physical product Grassmannian, Sigma^{-1} metric); starts = '
                       't25.science.starts (HOSVD + 3 random; key (namespace, seed, phi, N))',
            'affine_map': 't25.science.map_for(B, Sigma^{-1}); intercept b = (I - A) mean',
            'scoring': 't25.science.risks(A, b, L K L, Sigma, Sigma^{-1})["clean"]: exact expected clean risk on an independent '
                       'record (closed form; evaluation size infinite, no evaluation sampling error)',
            'max_iter': 6000, 'gradient_tolerance': 1e-8, 'restarts': 4},
        'contrast': 'D = clean risk(product) - clean risk(unrestricted), both affine with fitted mean, per seed',
        'optimization_limited': 'record flagged if the selected NFPA restart is not stationary or the restart clean-risk '
                                'spread >= |D| (t25.science.reduce_records rule); a cell (geometry, N) is optimization-limited '
                                'if >= 1 of its records is flagged; flagged records and cells are kept, never dropped',
        'inference': {'unit': 'seed-level mean of D over 24 seeds', 'interval': 'approximate Bonferroni Student-t',
                      'alpha': 0.05, 'df': N_SEEDS - 1, 'family_size': family,
                      'family': 'every registered (geometry, N) class contrast: 5 (A) + 5 (B) + union grid (control)',
                      'decided_negative': 'upper limit < 0', 'decided_positive': 'lower limit > 0',
                      'zero_SE': 'a cell whose seed SD is exactly 0 is undecided'},
        'reading_rule': {
            'nonseparable': 'MET iff decided negative at N*/4 and N*/2 AND decided positive at 2N* and 4N* (N* cell '
                            'unconstrained); NOT MET otherwise, with the observed sign pattern reported',
            'overall': 'MET iff both nonseparable geometries (A and B) are MET',
            'control': 'MET iff no decided positive contrast at any control N',
            'optimization_limited': 'reported with each reading; if a deciding cell is optimization-limited the reading '
                                    'carries that flag'},
        'secondary_descriptive': [
            'seed-mean contrast and interval at the N* cell of A and B (first order predicts approximately 0); OLS fit of '
            'seed-mean D(N) = alpha + beta/N over each 5-point grid, N*_fit = -beta/alpha',
            'OLS fit of N * mean Delta_U(N) = c + e/N over all registered cells, Delta_U = R_K(fitted unrestricted projector) '
            '- R_K(P*_U) (the t25 "signal" term minus sum_{j>k} lambda_j); compare c with c_U'],
        'exclusions': 'none', 'deviations_after_freeze': 'none permitted; a correction requires a new namespace',
        'dependencies': dependencies,
        'budget': {'threads_per_worker': 1, 'workers_default': 4, 'records': family * N_SEEDS},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['write', 'check'])
    a = ap.parse_args(argv)
    from . import runner                       # load every module a run loads before measuring dependencies
    pred = compute(scan=True)
    sel = pred['phi_selection']
    if (sel['phi_A'], sel['phi_B']) != (REGISTERED_PHI['A'], REGISTERED_PHI['B']):
        raise SystemExit(f'phi selection gave {sel["phi_A"]}, {sel["phi_B"]}; registered {REGISTERED_PHI}')
    if not pred['all_checks_pass'] or pred['ratio_N_star_A_over_B'] < PHI_SCAN['min_ratio']:
        raise SystemExit('population checks failed')
    if a.command == 'write':
        proto = build_protocol(pred, runner.dependency_snapshot())
        (PACKAGE / 'predictions.json').write_text(json.dumps(pred, indent=1, allow_nan=False) + '\n')
        (PACKAGE / 'protocol.json').write_text(json.dumps(proto, indent=1, allow_nan=False) + '\n')
    else:
        saved = json.loads((PACKAGE / 'predictions.json').read_text())
        print(json.dumps(runner.compare_predictions(saved, pred), indent=1))
    for lab in ('control', 'A', 'B'):
        r = pred['geometries'][lab]
        print(lab, r['phi'], 'a', r['a_best'], 'Bcert', r['B_cert'], 'cU', r['cU'], 'cF', r['cF'], 'N*', r['N_star_pred'],
              'interval', r['N_star_interval'], 'grid', pred['grids'][lab], file=sys.stderr)


if __name__ == '__main__':
    main()
