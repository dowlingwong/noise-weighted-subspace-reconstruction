"""meg_z runner (Y19 pattern): kit identity, fail-closed preflight, resumable per-record JSON, reading engine.

Commands (EXP root; PYTHONPATH=src; PYTHONDONTWRITEBYTECODE=1; OPENBLAS/OMP/MKL_NUM_THREADS=1 for `run`):
    python -m experiments.mlst2026.meg_z_20261013.runner verify-kit
    python -m experiments.mlst2026.meg_z_20261013.runner write-dependencies     # kit preparation only
    python -m experiments.mlst2026.meg_z_20261013.runner write-kit-hashes       # kit preparation only
    python -m experiments.mlst2026.meg_z_20261013.runner smoke --out OUT        # synthetic fixture end to end
    python -m experiments.mlst2026.meg_z_20261013.runner preflight --derived DIR --predictions P --gates G --subjects dev
    python -m experiments.mlst2026.meg_z_20261013.runner run --out OUT --derived DIR --predictions P --gates G
                                                         --subjects dev [--resamples R] [--workers 4]
    python -m experiments.mlst2026.meg_z_20261013.runner summarize --out OUT

Tasks are (subject, geometry, resample) groups; the geometries of a subject are the theta = 0 control (absolute
grid) and the population-rule geometries A and B (theta_A,s, theta_B,s; relative grids {1/4..4} N*_s) taken
from the predictions file.  Each group draws one training pool of max(N) records (fresh Gaussian latents +
training-run noise windows without replacement), fits every N (prefixes) under every metric arm and writes
OUT/records/<subject>/<geometry>/r<rr>_N<N>.json atomically; an existing record is never recomputed.  `run`
refuses confirmation subjects unless --confirmation is given AND the kit holds a frozen protocol.json whose
SHA-256 equals --protocol-sha256 (and whose namespace, predictions and kit hashes match).  Confirmation
subjects failing the gates or the budget/population rule are excluded (E1); fewer than 10 remaining -> stop.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from . import common, estimators, fit, population, signal              # fit runs the T25 byte guard
from .common import PACKAGE, EXP_ROOT, sha_file, sha_json, atomic_json, theta_tag
from .preprocess import load_subject, preprocess_config
from .protocol import load_protocol, is_frozen, PROTOCOL_DEV, PROTOCOL_FROZEN
from experiments.mlst2026.x3_nonlinear_metric_20261005 import guard     # noqa: E402

KIT_FILE = PACKAGE / 'KIT_SHA256.json'
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')


class KitMismatch(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- kit identity
def kit_files(package: Path = PACKAGE) -> list:
    kit_file = package / 'KIT_SHA256.json'
    return sorted(p for p in package.rglob('*') if p.is_file() and p != kit_file and '__pycache__' not in p.parts
                  and p.suffix != '.pyc' and '.pytest_cache' not in p.parts)


def kit_hashes(package: Path = PACKAGE) -> dict:
    files = {str(p.relative_to(package)): sha_file(p) for p in kit_files(package)}
    return {'package': 'experiments/mlst2026/' + package.name, 'files': files, 'kit_identity': sha_json(files)}


def write_kit_hashes() -> dict:
    h = kit_hashes()
    KIT_FILE.write_text(json.dumps(h, indent=1) + '\n')
    return h


def verify_kit(package: Path = PACKAGE) -> dict:
    kf = package / 'KIT_SHA256.json'
    if not kf.is_file():
        raise KitMismatch('KIT_SHA256.json missing')
    want = json.loads(kf.read_text())
    got = kit_hashes(package)
    if want['files'] != got['files']:
        diff = sorted(set(want['files'].items()) ^ set(got['files'].items()))
        raise KitMismatch(f'kit files differ from KIT_SHA256.json: {diff[:6]}')
    if want['kit_identity'] != got['kit_identity']:
        raise KitMismatch('kit identity differs')
    return got


def loaded_repo_modules() -> list:
    own = str(PACKAGE.relative_to(EXP_ROOT)) + '/'
    return [r for r in guard.loaded_repo_modules() if not r.startswith(own) and Path(r).name != 'conftest.py']


def import_everything() -> None:
    from . import data_manifest, gates, preprocess, secondary_d8, fixture  # noqa: F401


def dependency_snapshot() -> dict:
    """SHA-256 of every in-repository module outside this package loaded after importing the whole kit;
    T25-closure members are also checked against the accepted closure."""
    import_everything()
    closure = guard.load_closure()
    files = {r: sha_file(EXP_ROOT / r) for r in loaded_repo_modules()}
    in_closure = sorted(r for r in files if r in closure['EXP_files'])
    guard.assert_files(in_closure, closure)
    return {'accepted_t25_identity': guard.ACCEPTED_T25_IDENTITY, 'files': files, 'in_accepted_t25_closure': in_closure,
            'outside_closure': sorted(set(files) - set(in_closure))}


def write_dependencies() -> dict:
    proto = json.loads(PROTOCOL_DEV.read_text())
    proto['dependencies'] = dependency_snapshot()
    PROTOCOL_DEV.write_text(json.dumps(proto, indent=1) + '\n')
    return proto['dependencies']


def assert_dependencies(proto: dict) -> dict:
    dep = proto.get('dependencies')
    if not dep:
        raise KitMismatch('protocol has no pinned dependencies')
    for rel, h in dep['files'].items():
        got = sha_file(EXP_ROOT / rel)
        if got != h:
            raise KitMismatch(f'dependency {rel} differs ({got[:12]} != {h[:12]})')
    import_everything()
    for rel in loaded_repo_modules():
        if rel not in dep['files']:
            raise KitMismatch(f'unpinned in-repository module loaded: {rel}')
    guard.assert_repo_origin()
    guard.assert_files(dep['in_accepted_t25_closure'])
    return {'checked': len(dep['files'])}


def environment() -> dict:
    import scipy
    mods = {}
    for m in ('h5py', 'mne', 'openneuro'):
        try:
            mods[m] = __import__(m).__version__
        except Exception:          # noqa: BLE001
            mods[m] = None
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, **mods,
            'platform': platform.platform(), 'host': socket.gethostname(),
            'threads': {v: os.environ.get(v) for v in THREAD_VARS}}


# ----------------------------------------------------------------------------------------------- reading engine
GEOMS = ('control', 'A', 'B')


def interval(values, F: int, alpha: float) -> dict:
    """Approximate simultaneous Bonferroni Student-t interval over subjects (df = n - 1)."""
    v = np.asarray(values, float)
    n = len(v)
    mean = float(v.mean()) if n else float('nan')
    if n < 2:
        return {'n': n, 'mean': mean, 'sd': None, 'ci': None, 'half_width': None, 'zero_SE': False, 'decision': 'undecided'}
    sd = float(v.std(ddof=1))
    half = float(student_t.ppf(1 - alpha / (2 * F), n - 1) * sd / np.sqrt(n))
    zero = bool(sd == 0)
    lo, hi = mean - half, mean + half
    dec = 'undecided' if zero else 'negative' if hi < 0 else 'positive' if lo > 0 else 'undecided'
    return {'n': n, 'mean': mean, 'sd': sd, 'ci': [lo, hi], 'half_width': half, 'zero_SE': zero, 'decision': dec}


def n_cells(pred: dict, subjects: list) -> dict:
    s0 = subjects[0]
    return {g: len(pred['subjects'][s0]['geometries'][g]['grid']) for g in GEOMS}


def family_size(cells_per_geometry: dict, rr: dict) -> int:
    """Registered family: ref-arm cells of every geometry (PZ2 + PZ3) + PZ4 (2 estimators x 2 m) + PZ5 (2 classes)."""
    pz4 = 4 if rr['PZ4'].get('aggregation', 'cell_average') == 'cell_average' else \
        4 * sum(cells_per_geometry[g] for g in rr['PZ4'].get('geometries', ['A', 'B']))
    return int(sum(cells_per_geometry.values())) + pz4 + 2


def subject_means(records: list, subjects: list, primary: str = 'clean_ref') -> dict:
    """{(arm, geometry, cell): {subject: {mean_D, mean_R_U, mean_R_F, N, resamples, optimization_limited}}}."""
    acc = {}
    for r in records:
        if r['subject'] not in subjects:
            continue
        for arm, a in r['arms'].items():
            key = (arm, r['geometry'], r['cell'])
            acc.setdefault(key, {}).setdefault(r['subject'], []).append(
                (a['D'], a['U'][primary], a['F'][primary], a['optimization_limited'], r['N']))
    out = {}
    for key, per in acc.items():
        out[key] = {s: {'mean_D': float(np.mean([x[0] for x in v])), 'mean_R_U': float(np.mean([x[1] for x in v])),
                        'mean_R_F': float(np.mean([x[2] for x in v])), 'resamples': len(v),
                        'optimization_limited': int(sum(x[3] for x in v)), 'N': v[0][4]} for s, v in per.items()}
    return out


def cell_table(sm: dict, F: int, alpha: float) -> dict:
    out = {}
    for key, per in sm.items():
        row = interval([v['mean_D'] for v in per.values()], F, alpha)
        row.update(arm=key[0], geometry=key[1], cell=key[2], N={s: v['N'] for s, v in per.items()},
                   optimization_limited_records=int(sum(v['optimization_limited'] for v in per.values())),
                   resamples={s: v['resamples'] for s, v in per.items()})
        out[key] = row
    return out


def classify_factor(f: float) -> str:
    """PZ2 point-factor classification by the registered grid factor of the cell (delta review D1): the
    relative grid is round(f N*_s), so the cell with f <= 1/2 is 'below' and f >= 2 'above' for every subject by
    construction (comparing the rounded N with the unrounded N* would drop the 1/2 and 2 cells by rounding)."""
    return 'below' if f <= 0.5 else 'above' if f >= 2.0 else 'inside'


def readings(cells: dict, pred: dict, subjects: list, rr: dict, F: int, alpha: float, sm: dict) -> dict:
    out = {}
    pz1 = {s: pred['subjects'][s]['pz1'] for s in subjects}
    out['PZ1'] = {'reading': 'MET' if subjects and all(v['met'] for v in pz1.values()) else 'NOT MET',
                  'per_subject': {s: v['met'] for s, v in pz1.items()}, 'quantity': 'B_dagger'}
    arm = rr['PZ2'].get('arm', 'ref')
    per_geo = {}
    for g in rr['PZ2'].get('geometries', ['A', 'B']):
        ns = {s: pred['subjects'][s]['geometries'][g].get('N_star') for s in subjects}
        agree = all(pred['subjects'][s]['geometries'][g].get('restart_agreement') for s in subjects)
        facs = {tuple(pred['subjects'][s]['geometries'][g]['factors']) for s in subjects}
        if len(facs) != 1:
            raise ValueError(f'geometry {g}: subjects carry different grid factors {facs}')
        factors = list(facs.pop())
        rows = []
        for i in range(len(pred['subjects'][subjects[0]]['geometries'][g]['grid'])):
            c = cells.get((arm, g, i))
            if c is None:
                continue
            rows.append({'cell': i, 'factor': factors[i], 'N': c['N'], 'class': classify_factor(factors[i]), 'decision': c['decision'],
                         'mean': c['mean'], 'ci': c['ci'], 'optimization_limited_records': c['optimization_limited_records']})
        below = [r for r in rows if r['class'] == 'below']
        above = [r for r in rows if r['class'] == 'above']
        applicable = bool(below and above and agree)
        met = applicable and all(r['decision'] == 'negative' for r in below) and all(r['decision'] == 'positive' for r in above)
        bd = {s: pred['subjects'][s]['geometries'][g].get('row', {}).get('N_star_interval') for s in subjects}
        per_geo[g] = {'applicable': applicable, 'restart_agreement_all_subjects': bool(agree),
                      'reading': ('MET' if met else 'NOT MET') if applicable else 'NOT APPLICABLE', 'cells': rows,
                      'pattern': ' '.join({'negative': '-', 'positive': '+'}.get(r['decision'], '?') for r in rows),
                      'N_star': ns, 'B_dagger_interval_descriptive': bd,
                      'flag_optimization_limited': [r['cell'] for r in below + above if r['optimization_limited_records']]}
    # delta review D2: MET iff every registered geometry is applicable and MET; NOT APPLICABLE iff none is
    # applicable; otherwise NOT MET (with the reason)
    app = [g for g, v in per_geo.items() if v['applicable']]
    if not app:
        reading, why = 'NOT APPLICABLE', 'no geometry applicable'
    elif len(app) < len(per_geo):
        reading, why = 'NOT MET', f'geometries not applicable: {sorted(set(per_geo) - set(app))}'
    elif all(per_geo[g]['reading'] == 'MET' for g in app):
        reading, why = 'MET', None
    else:
        reading, why = 'NOT MET', f'geometries NOT MET: {[g for g in app if per_geo[g]["reading"] != "MET"]}'
    out['PZ2'] = {'reference': 'point_factors (registered grid factor)', 'per_geometry': per_geo, 'reading': reading,
                  'reason': why}
    pz3 = []
    for a in rr['PZ3'].get('arms', ['ref']):
        g = rr['PZ3'].get('geometry', 'control')
        for i in range(len(pred['subjects'][subjects[0]]['geometries'][g]['grid'])):
            c = cells.get((a, g, i))
            if c is not None:
                pz3.append({'arm': a, 'cell': i, 'N': c['N'], 'decision': c['decision'], 'mean': c['mean'], 'ci': c['ci']})
    out['PZ3'] = {'cells': pz3, 'reading': 'NOT APPLICABLE' if not pz3 else
                  ('NOT MET' if any(c['decision'] == 'positive' for c in pz3) else 'MET')}
    out['PZ4'] = read_pz4(sm, pred, subjects, rr['PZ4'], F, alpha)
    out['PZ5'] = read_pz5(sm, subjects, F, alpha)
    out['descriptive_iso_class_contrast'] = [{'geometry': k[1], 'cell': k[2], 'decision': c['decision'], 'mean': c['mean']}
                                             for k, c in sorted(cells.items()) if k[0] == 'iso']
    return out


def read_pz4(sm: dict, pred: dict, subjects: list, cfg: dict, F: int, alpha: float) -> dict:
    s0 = subjects[0]
    keys = [(g, i) for g in cfg.get('geometries', ['A', 'B']) for i in range(len(pred['subjects'][s0]['geometries'][g]['grid']))]
    res, ok_all = {}, True
    for est in cfg.get('estimators', ['lw', 'kron']):
        per_m = {}
        for lab in ('lo', 'hi'):
            arm = f'{est}_{lab}'
            if cfg.get('aggregation', 'cell_average') == 'cell_average':
                vals = []
                for s in subjects:
                    d = [sm[(arm, g, i)][s]['mean_D'] - sm[('ref', g, i)][s]['mean_D'] for g, i in keys
                         if s in sm.get((arm, g, i), {}) and s in sm.get(('ref', g, i), {})]
                    if d:
                        vals.append(float(np.mean(d)))
                per_m[lab] = interval(vals, F, alpha)
            else:
                cs = []
                for g, i in keys:
                    if (arm, g, i) in sm:
                        cs.append(interval([sm[(arm, g, i)][s]['mean_D'] - sm[('ref', g, i)][s]['mean_D'] for s in subjects
                                            if s in sm[(arm, g, i)] and s in sm[('ref', g, i)]], F, alpha))
                per_m[lab] = {'cells': cs, 'decision': 'negative' if cs and all(c['decision'] == 'negative' for c in cs) else 'mixed',
                              'mean': float(np.mean([c['mean'] for c in cs])) if cs else float('nan')}
        ok = per_m['lo']['decision'] == 'negative' and abs(per_m['hi']['mean']) < abs(per_m['lo']['mean'])
        ok_all &= ok
        res[est] = {'m': per_m, 'met': bool(ok)}
    return {'aggregation': cfg.get('aggregation', 'cell_average'), 'estimators': res,
            'reading': 'MET' if res and ok_all else 'NOT MET', 'note': cfg.get('note')}


def read_pz5(sm: dict, subjects: list, F: int, alpha: float) -> dict:
    """Isotropic negative control (review (e)): R_iso - R_ref per class, averaged per subject over all cells."""
    keys = sorted({(k[1], k[2]) for k in sm if k[0] == 'iso'})
    res = {}
    for cls in ('U', 'F'):
        vals = []
        for s in subjects:
            d = [sm[('iso', g, i)][s][f'mean_R_{cls}'] - sm[('ref', g, i)][s][f'mean_R_{cls}'] for g, i in keys
                 if s in sm[('iso', g, i)] and s in sm.get(('ref', g, i), {})]
            if d:
                vals.append(float(np.mean(d)))
        res[cls] = interval(vals, F, alpha)
    met = all(res[c]['decision'] == 'positive' for c in ('U', 'F'))
    return {'per_class': res, 'reading': 'NOT APPLICABLE' if not keys else ('MET' if met else 'NOT MET')}


# ----------------------------------------------------------------------------------------------- execution
def record_path(out: Path, sub: str, geometry: str, r: int, N: int) -> Path:
    return Path(out) / 'records' / sub / geometry / f'r{r:02d}_N{N}.json'


_REF = {}


def subject_context(h5: str, proto: dict):
    key = (str(h5), sha_json(proto['reference_metric']))
    if key not in _REF:
        sd = load_subject(h5)
        ref = population.reference_for(sd, proto)
        _REF[key] = (sd, ref, fit.metric(ref['cov']))
    return _REF[key]


def m_value(arm: dict, m_grid: dict) -> int:
    m = arm['m']
    return int(m_grid['m_' + m]) if isinstance(m, str) else int(m)


def arm_metrics(sd, proto: dict, m_grid: dict, W0: np.ndarray, namespace: str, r: int) -> tuple[dict, dict]:
    cal = sd.records('calibration')
    perm = common.rng(namespace, sd.subject, 'calibration', r).permutation(len(cal))
    W, diag = {}, {}
    for arm in proto['arms']:
        e = arm['estimator']
        if e == 'reference':
            W[arm['name']] = W0
        elif e == 'identity':
            W[arm['name']] = np.eye(sd.d)
        else:
            m = m_value(arm, m_grid)
            if m > len(cal):
                raise ValueError(f'{sd.subject}: calibration pool {len(cal)} < m = {m}')
            cov, info = estimators.arm_estimate(cal[perm[:m]], sd.C, sd.T, e, proto['reference_metric'].get('flipflop'))
            W[arm['name']] = fit.metric(cov)
            diag[arm['name']] = {k: v for k, v in info.items() if not isinstance(v, np.ndarray)}
            diag[arm['name']].update(m=m, whitened_error_vs_ref=estimators.whitened_error(cov, np.linalg.inv(W0)))
    return W, diag


def run_group(task: dict) -> dict:
    proto, sub, geo, theta, r, Ns, out, h5, ns, m_grid = (task[k] for k in (
        'proto', 'subject', 'geometry', 'theta', 'r', 'Ns', 'out', 'h5', 'namespace', 'm_grid'))
    out = Path(out)
    todo = [(i, N) for i, N in enumerate(Ns) if not record_path(out, sub, geo, r, N).is_file()]
    if not todo:
        return {'subject': sub, 'geometry': geo, 'r': r, 'written': 0}
    sd, ref, W0 = subject_context(h5, proto)
    sc = proto['signal']
    kc, kt = int(sc['kc']), int(sc['kt'])
    model, sinfo = signal.build(sd.evoked, sd.sfreq, ref['cov'], theta, sc)
    R_s, Bs = model.second_moment(), model.templates()
    n_eval = int(proto['n_eval'])
    ev_pool = sd.records('evaluation')
    if n_eval > len(ev_pool):
        raise ValueError(f'{sub}: evaluation pool {len(ev_pool)} < n_eval = {n_eval}')
    ev_idx = common.rng(ns, sub, 'eval-noise').permutation(len(ev_pool))[:n_eval]
    S_ev, al_ev = model.draw(n_eval, common.rng(ns, sub, geo, float(theta), 'eval-latents'))
    ev = fit.Evaluation(S_ev, ev_pool[ev_idx], al_ev, Bs, model.variances, W0, R_s)
    W, wdiag = arm_metrics(sd, proto, m_grid, W0, ns, r)
    tr = sd.records('training')
    Nmax = max(Ns)
    if Nmax > len(tr):
        raise ValueError(f'{sub}: training pool {len(tr)} < N = {Nmax}')
    perm = common.rng(ns, sub, 'training', r).permutation(len(tr))[:Nmax]
    S_tr, _ = model.draw(Nmax, common.rng(ns, sub, geo, float(theta), r, 'latents'))
    X = S_tr + tr[perm]
    pool_sha = common.sha_array(X)
    for i, N in todo:
        t0 = time.time()
        arms = {}
        for arm in proto['arms']:
            f = fit.fit_pair(X, N, W[arm['name']], sd.C, sd.T, kc, kt, proto['learners'], (ns, sub, geo, float(theta), r, N))
            arms[arm['name']] = fit.score_fit(f, ev, proto['scores']['primary'], proto['learners'].get('flag_rule'))
        rec = {'subject': sub, 'geometry': geo, 'theta': float(theta), 'r': r, 'cell': i, 'N': N, 'arms': arms,
               'pool_sha': pool_sha, 'training_noise_index_sha': common.sha_array(perm[:N]),
               'eval_noise_index_sha': common.sha_array(ev_idx), 'metric_diagnostics': wdiag, 'signal': sinfo,
               'seconds': time.time() - t0, 'namespace': ns}
        atomic_json(record_path(out, sub, geo, r, N), rec)
        with (out / 'RUN_LOG.txt').open('a') as fh:
            fh.write(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {sub} {geo} theta={theta:.4f} r{r:02d} '
                     f'N={N} {rec["seconds"]:.1f}s D_ref={arms.get("ref", {}).get("D", float("nan")):.4g}\n')
    return {'subject': sub, 'geometry': geo, 'r': r, 'written': len(todo)}


def tasks(proto: dict, pred: dict, subjects: list, derived: Path, out: Path, R: int, ns: str) -> list:
    tl = []
    for s in subjects:
        for g in GEOMS:
            geo = pred['subjects'][s]['geometries'][g]
            for r in range(R):
                tl.append({'proto': proto, 'subject': s, 'geometry': g, 'theta': float(geo['theta']), 'r': r,
                           'Ns': list(geo['grid']), 'out': str(out), 'h5': str(Path(derived) / f'{s}.h5'),
                           'namespace': ns, 'm_grid': pred['m_grid']})
    return sorted(tl, key=lambda t_: -max(t_['Ns']))


def execute(tl: list, workers: int) -> list:
    if workers <= 1:
        return [run_group(t_) for t_ in tl]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=4) as p:
        return list(p.imap_unordered(run_group, tl))


def load_records(out: Path) -> list:
    return [json.loads(p.read_text()) for p in sorted((Path(out) / 'records').rglob('*.json'))]


# ----------------------------------------------------------------------------------------------- summary
def summarize(out: Path, partial: bool = False) -> dict:
    out = Path(out)
    start = json.loads((out / 'START.json').read_text())
    proto, pred, _ = (json.loads((out / n).read_text()) for n in ('protocol_used.json', 'predictions_used.json', 'gates_used.json'))
    recs = load_records(out)
    status = 'COMPLETE' if len(recs) == start['expected_records'] else ('INCOMPLETE - not a reading' if partial else None)
    if status is None:
        raise ValueError(f'{len(recs)} of {start["expected_records"]} records present; use --partial for a diagnostic summary')
    e1 = start['E1']
    included = list(e1['included'])
    rr, inf = proto['reading_rules'], proto['inference']
    F = family_size(n_cells(pred, included), rr) if included else 0
    sm = subject_means(recs, included, proto['scores']['primary'])
    cells = cell_table(sm, F, float(inf['alpha'])) if included else {}
    rd = readings(cells, pred, included, rr, F, float(inf['alpha']), sm) if included and not e1['stop'] else None
    summ = {'status': status if not e1['stop'] else 'STOPPED (E1): ' + str(e1['reason']), 'mode': start['mode'],
            'subjects_included': included, 'subjects_excluded': e1['excluded'], 'exclusion_reasons': e1.get('reasons'),
            'family_size': F, 'interval': f'approximate Bonferroni Student-t over subjects, df = n - 1, alpha = {inf["alpha"]}',
            'readings': rd, 'cells': [dict(v, key=list(k)) for k, v in sorted(cells.items())], 'records': len(recs),
            'optimization_limited_records': int(sum(a['optimization_limited'] for r_ in recs for a in r_['arms'].values())),
            'timing': timing(recs), 'optimization': optimization_report(recs, proto) if recs else None,
            'optimization_limited_records_t25_rule': int(sum(a.get('optimization_limited_t25_rule', False)
                                                             for r_ in recs for a in r_['arms'].values())),
            'protocol_sha256': sha_json(proto)}
    if start['mode'] == 'dev' and included:
        summ['dev_power'] = dev_power(cells, pred, included, proto, F)
    atomic_json(out / ('summary.json' if status == 'COMPLETE' else 'summary_partial.json'), summ)
    return summ


def optimization_report(recs: list, proto: dict) -> dict:
    """Per arm (delta review): selected-restart stop-status counts, gradient-norm quantiles of selected restarts
    that ended line_search_stopped, flag counts under the refined and the T25 rule, and the prespecified D4
    decision on the iteration cap (raise max_iter to 12,000 iff iteration_limit occurs on > 1% of selected
    restarts)."""
    out, sel_all = {}, []
    for k in sorted({a for r in recs for a in r['arms']}):
        st, gls = {}, []
        for r in recs:
            a = r['arms'][k]
            sel = a['restarts'][a['selected']]
            st[sel['status']] = st.get(sel['status'], 0) + 1
            sel_all.append(sel['status'])
            if sel['status'] == 'line_search_stopped':
                gls.append(sel['gradient_norm'])
        out[k] = {'selected_status_counts': st,
                  'line_search_stopped_gradient_quantiles': (dict(zip(['min', 'q50', 'q90', 'max'],
                                                                      [float(x) for x in np.quantile(gls, [0, .5, .9, 1])]))
                                                             if gls else None),
                  'flags_refined_rule': int(sum(r['arms'][k]['optimization_limited'] for r in recs)),
                  'flags_t25_rule': int(sum(r['arms'][k].get('optimization_limited_t25_rule', False) for r in recs))}
    frac = float(np.mean([s_ == 'iteration_limit' for s_ in sel_all])) if sel_all else 0.0
    rule = proto['learners'].get('max_iter_rule', {'threshold_fraction': 0.01, 'raised_max_iter': 12000})
    return {'per_arm': out, 'iteration_limit_fraction': frac,
            'max_iter_decision': (f'raise max_iter to {rule["raised_max_iter"]}' if frac > rule['threshold_fraction']
                                  else f'keep max_iter = {proto["learners"]["max_iter"]}'),
            'rule': rule}


def timing(recs: list) -> dict:
    if not recs:
        return {}
    fits = [a['seconds_fit'] for r in recs for a in r['arms'].values()]
    by_arm = {}
    for r in recs:
        for k, a in r['arms'].items():
            by_arm.setdefault(k, []).append(a['restarts'][a['selected']]['status'] != 'stationary')
    return {'records': len(recs), 'mean_seconds_per_record': float(np.mean([r['seconds'] for r in recs])),
            'mean_seconds_per_fit_pair': float(np.mean(fits)), 'max_seconds_per_fit_pair': float(np.max(fits)),
            'total_cpu_seconds': float(np.sum([r['seconds'] for r in recs])),
            'restart_spread_median': float(np.median([a['restart_spread'] for r in recs for a in r['arms'].values()])),
            'nonstationary_selected': int(sum(sum(v) for v in by_arm.values())),
            'nonstationary_selected_by_arm': {k: int(sum(v)) for k, v in by_arm.items()},
            'optimization_limited_by_arm': {k: int(sum(r['arms'][k]['optimization_limited'] for r in recs)) for k in by_arm}}


def dev_power(cells: dict, pred: dict, subjects: list, proto: dict, F: int) -> dict:
    """Power rule (runbook §6.5) from the dev subjects: projected half-width with n = 12 subjects versus half the
    first-order predicted contrast |a + B/N| at the subject's geometry (subject mean), for every ref-arm cell."""
    n = int(proto['power_rule']['n_confirmation'])
    alpha = float(proto['inference']['alpha'])
    rows = []
    for (arm, g, i), c in sorted(cells.items()):
        if arm != 'ref' or c['sd'] is None:
            continue
        pc = []
        for s in subjects:
            geo = pred['subjects'][s]['geometries'][g]
            row = geo.get('row') or next(r for r in pred['subjects'][s]['rows'] if r['theta'] == 0.0)
            pc.append(row['a_best'] + row['B'] / c['N'][s])
        pred_c = float(np.mean(pc))
        half = float(student_t.ppf(1 - alpha / (2 * F), n - 1) * c['sd'] / np.sqrt(n))
        rows.append({'geometry': g, 'cell': i, 'N': c['N'], 'between_subject_sd': c['sd'], 'observed_mean': c['mean'],
                     'predicted_first_order': pred_c, 'projected_half_width_n12': half,
                     'passes_power_rule': bool(half < 0.5 * abs(pred_c))})
    return {'family_size': F, 'n_confirmation': n, 'cells': rows}


# ----------------------------------------------------------------------------------------------- preflight
def resolve_subjects(spec: str, proto: dict) -> list:
    if spec in ('dev', 'confirmation', 'analysed'):
        return list(proto['subjects'][spec])
    return spec.split(',')


def preflight(proto: dict, pred: dict, gates_: dict, derived: Path, subjects: list, confirmation: bool = False,
              protocol_sha256: str | None = None, smoke: bool = False) -> dict:
    """Everything that must hold before the first record is drawn.  Raises KitMismatch."""
    report = {}
    if not smoke:
        report['kit_identity'] = verify_kit()['kit_identity']
        bad = {v: os.environ.get(v) for v in THREAD_VARS if os.environ.get(v) != '1'}
        if bad:
            raise KitMismatch(f'thread variables must be 1: {bad}')
        if not sys.dont_write_bytecode:
            raise KitMismatch('set PYTHONDONTWRITEBYTECODE=1')
        report['dependencies'] = assert_dependencies(proto)
    unknown = [s for s in subjects if s not in proto['subjects']['analysed']]
    if unknown:
        raise KitMismatch(f'not analysed subjects: {unknown}')
    conf = sorted(set(subjects) & set(proto['subjects']['confirmation']))
    if conf:
        if not confirmation:
            raise KitMismatch(f'confirmation subjects {conf} requested without --confirmation')
        if not is_frozen(proto) or not PROTOCOL_FROZEN.is_file():
            raise KitMismatch('confirmation requires the frozen protocol.json in the kit')
        if protocol_sha256 is None or sha_file(PROTOCOL_FROZEN) != protocol_sha256:
            raise KitMismatch('frozen protocol SHA-256 does not match --protocol-sha256')
        if json.loads(PROTOCOL_FROZEN.read_text()) != proto:
            raise KitMismatch('loaded protocol is not the frozen protocol.json')
        if not proto.get('namespace_confirmation'):
            raise KitMismatch('frozen protocol has no confirmation namespace')
        reg = proto.get('registration', {})
        if reg.get('predictions_sha256') != sha_json(pred):
            raise KitMismatch('predictions differ from the registered predictions (protocol registration block)')
        if set(subjects) & set(proto['subjects']['dev']):
            raise KitMismatch('dev subjects are never part of a confirmation run')
    elif confirmation:
        raise KitMismatch('--confirmation given but no confirmation subject requested')
    if pred.get('protocol_sha256') != sha_json(proto):
        raise KitMismatch('predictions were computed under a different protocol')
    man = json.loads((Path(derived) / 'DERIVED_MANIFEST.json').read_text())
    ok_cfg = {sha_json(preprocess_config(proto))} if not smoke else {'fixture'}
    for s in subjects:
        ent = man['subjects'].get(s)
        if ent is None:
            raise KitMismatch(f'{s} missing from DERIVED_MANIFEST.json')
        if sha_file(Path(derived) / ent['file']) != ent['file_sha256']:
            raise KitMismatch(f'{s}: derived file SHA-256 differs from DERIVED_MANIFEST.json')
        if ent['config_sha256'] not in ok_cfg:
            raise KitMismatch(f'{s}: derived with a different preprocessing configuration')
        g = gates_['subjects'].get(s)
        if g is None or g.get('derived_file_sha256') != ent['file_sha256']:
            raise KitMismatch(f'{s}: gates.json missing or computed on a different derived file')
    mode = 'confirmation' if conf else 'dev'
    # budget / population rule (F5, review c): ineligible confirmation subjects are excluded (E1); in development
    # an ineligible dev subject is a design failure and stops the run
    extra = {}
    for s in subjects:
        if s not in pred['subjects']:
            if not gates_['subjects'][s]['excluded']:
                raise KitMismatch(f'{s}: no predictions')
            continue
        ps = pred['subjects'][s]
        reasons = list(ps.get('exclusion_reasons', []))
        grids = [ps['geometries'][g]['grid'] for g in GEOMS if ps['geometries'][g].get('grid')]
        if len(grids) < len(GEOMS):
            reasons.append('missing geometry grid')
        need = max(max(x) for x in grids) if grids else 0
        if ps['pools']['training'] < need or ps['pools']['calibration'] < int(pred['m_grid']['m_hi']) or \
                ps['pools']['evaluation'] < int(proto['n_eval']):
            reasons.append(f'window budget too small for the grid ({ps["pools"]})')
        if reasons:
            extra[s] = sorted(set(reasons))
    if mode == 'dev' and extra:
        raise KitMismatch(f'dev subjects fail the budget/population rule: {extra}')
    from .gates import exclusion
    e1 = exclusion({s: gates_['subjects'][s] for s in subjects}, proto, mode, extra)
    report.update(mode=mode, E1=e1, subjects=subjects)
    return report


# ----------------------------------------------------------------------------------------------- commands
def cmd_run(out: Path, derived: Path, pred_path: Path, gates_path: Path, subjects: list, workers: int,
            resamples: int | None = None, confirmation: bool = False, protocol_sha256: str | None = None,
            proto: dict | None = None, smoke: bool = False) -> dict:
    out = Path(out).resolve()
    if out == PACKAGE or PACKAGE in out.parents:
        raise KitMismatch('OUT must lie outside the kit directory (the kit hashes every file in it)')
    proto = proto or load_protocol()
    pred = json.loads(Path(pred_path).read_text())
    gates_ = json.loads(Path(gates_path).read_text())
    pf = preflight(proto, pred, gates_, derived, subjects, confirmation, protocol_sha256, smoke)
    R = int(resamples or proto['resamples'])
    if pf['mode'] == 'confirmation' and R != int(proto['resamples']):
        raise KitMismatch('the number of resamples is fixed by the frozen protocol')
    ns = proto['namespace_confirmation'] if pf['mode'] == 'confirmation' else proto['namespace']
    if pf['E1']['stop']:
        atomic_json(out / 'STOPPED.json', pf['E1'])
        raise KitMismatch(f'E1 stop: {pf["E1"]["reason"]}')
    incl = pf['E1']['included']
    tl = tasks(proto, pred, incl, derived, out, R, ns)
    expected = sum(len(t_['Ns']) for t_ in tl)
    out.mkdir(parents=True, exist_ok=True)
    start = out / 'START.json'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ident = {'protocol_sha256': sha_json(proto), 'predictions_sha256': sha_json(pred), 'gates_sha256': sha_json(gates_),
             'subjects': subjects, 'resamples': R, 'namespace': ns, 'mode': pf['mode']}
    if start.is_file():
        old = json.loads(start.read_text())
        if any(old[k] != v for k, v in ident.items()) or old.get('kit_identity') != pf.get('kit_identity'):
            raise KitMismatch('OUT holds a run of a different kit/protocol/predictions/subjects; use a fresh directory')
    else:
        atomic_json(start, {**ident, 'kit_identity': pf.get('kit_identity'), 'E1': pf['E1'], 'expected_records': expected,
                            'started_utc': now, 'environment': environment(), 'preflight': pf, 'workers': workers})
        for name, obj in (('protocol_used.json', proto), ('predictions_used.json', pred), ('gates_used.json', gates_)):
            atomic_json(out / name, obj)
    with (out / 'ATTEMPTS.jsonl').open('a') as f:
        f.write(json.dumps({'utc': now, 'host': socket.gethostname(), 'workers': workers,
                            'existing_records': len(list((out / 'records').rglob('*.json'))) if (out / 'records').is_dir() else 0}) + '\n')
    t0 = time.time()
    execute(tl, workers)
    summ = summarize(out)
    summ['wall_seconds_this_attempt'] = time.time() - t0
    atomic_json(out / 'summary.json', summ)
    return summ


def smoke_predictions(proto: dict, derived: Path, subjects: list) -> dict:
    targets, m_grid = population.study_constants(proto, derived)
    pr = {'protocol_sha256': sha_json(proto), 'reference_kind': proto['reference_metric']['kind'], 'theta_rule': targets,
          'm_grid': m_grid, 'subjects': {}}
    for s in subjects:
        pr['subjects'][s] = population.subject_predictions(load_subject(derived / f'{s}.h5'), proto, proto['namespace'],
                                                           targets, m_grid)
    return pr


def cmd_smoke(out: Path, workers: int = 1, overrides: dict | None = None) -> dict:
    """Synthetic end to end: fixture subjects (C = 4, T = 6) -> gates -> predictions (theta_s rule) -> run -> summary.
    ``overrides`` (tests only) updates top-level protocol blocks, e.g. {'resamples': 1}."""
    from . import fixture, gates
    from .preprocess import update_derived_manifest
    out = Path(out)
    proto = fixture.smoke_protocol(json.loads(PROTOCOL_DEV.read_text()))
    for k, v in (overrides or {}).items():
        proto[k] = dict(proto[k], **v) if isinstance(v, dict) else v
    derived = out / 'derived'
    sums = []
    for i, s in enumerate(fixture.SMOKE_SUBJECTS[:4]):
        if not (derived / f'{s}.h5').is_file():
            sums.append(fixture.make_derived_subject(derived, s, seed=i, kind='kron' if i % 2 == 0 else 'nonkron'))
    if sums:
        update_derived_manifest(derived, sums)
    man = json.loads((derived / 'DERIVED_MANIFEST.json').read_text())
    subs = proto['subjects']['dev']
    g = {'protocol_sha256': sha_json(proto), 'subjects': {}}
    for s in subs:
        g['subjects'][s] = dict(gates.subject_gates(load_subject(derived / f'{s}.h5'), proto),
                                derived_file_sha256=man['subjects'][s]['file_sha256'])
    atomic_json(out / 'gates.json', g)
    if not (out / 'predictions.json').is_file():
        atomic_json(out / 'predictions.json', smoke_predictions(proto, derived, subs))
    return cmd_run(out / 'run', derived, out / 'predictions.json', out / 'gates.json', subs, workers, proto=proto, smoke=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['verify-kit', 'write-dependencies', 'write-kit-hashes', 'smoke', 'preflight', 'run', 'summarize'])
    ap.add_argument('--out')
    ap.add_argument('--derived')
    ap.add_argument('--predictions')
    ap.add_argument('--gates')
    ap.add_argument('--subjects', default='dev')
    ap.add_argument('--resamples', type=int)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--confirmation', action='store_true')
    ap.add_argument('--protocol-sha256')
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--protocol', help='DEV ONLY: a development protocol variant (copy of protocol_dev.json) outside the kit')
    a = ap.parse_args(argv)
    if a.protocol and a.confirmation:
        raise KitMismatch('--protocol variants are for development runs; a confirmation run uses the frozen kit protocol.json')
    if a.command == 'verify-kit':
        k = verify_kit()
        assert_dependencies(load_protocol())
        print(json.dumps({'kit_identity': k['kit_identity'], 'files': len(k['files']), 'dependencies': 'ok',
                          'protocol': 'protocol.json (frozen)' if PROTOCOL_FROZEN.is_file() else 'protocol_dev.json (NOT FROZEN)'}, indent=1))
    elif a.command == 'write-dependencies':
        print(json.dumps(write_dependencies(), indent=1))
    elif a.command == 'write-kit-hashes':
        print(json.dumps(write_kit_hashes(), indent=1))
    elif a.command == 'smoke':
        s = cmd_smoke(Path(a.out), a.workers)
        print(json.dumps({'status': s['status'], 'readings': {k: v['reading'] for k, v in (s['readings'] or {}).items() if isinstance(v, dict) and 'reading' in v},
                          'timing': s['timing']}, indent=1))
    elif a.command == 'preflight':
        proto = load_protocol(a.protocol)
        pf = preflight(proto, json.loads(Path(a.predictions).read_text()), json.loads(Path(a.gates).read_text()),
                       Path(a.derived), resolve_subjects(a.subjects, proto), a.confirmation, a.protocol_sha256)
        print(json.dumps(pf, indent=1, default=common._default))
    elif a.command == 'run':
        proto = load_protocol(a.protocol)
        s = cmd_run(Path(a.out), Path(a.derived), Path(a.predictions), Path(a.gates), resolve_subjects(a.subjects, proto),
                    a.workers, a.resamples, a.confirmation, a.protocol_sha256, proto)
        print(json.dumps({'status': s['status'], 'readings': {k: v['reading'] for k, v in (s['readings'] or {}).items() if isinstance(v, dict) and 'reading' in v}}, indent=1))
    elif a.command == 'summarize':
        s = summarize(Path(a.out), a.partial)
        print(json.dumps({'status': s['status'], 'readings': {k: v['reading'] for k, v in (s['readings'] or {}).items() if isinstance(v, dict) and 'reading' in v}}, indent=1))


if __name__ == '__main__':
    try:
        main()
    except KitMismatch as e:
        print('KitMismatch: ' + str(e), file=sys.stderr)
        sys.exit(3)
