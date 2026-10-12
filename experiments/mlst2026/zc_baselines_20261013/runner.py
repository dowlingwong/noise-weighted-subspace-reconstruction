"""ZC confirmation runner (baselines on the T25 generator: CV-select, N*-rule, SHR).

Commands (EXP root; PYTHONPATH=src:. ; PYTHONDONTWRITEBYTECODE=1; OPENBLAS/OMP/MKL_NUM_THREADS=1):
    python -m experiments.mlst2026.zc_baselines_20261013.runner verify-kit
    python -m experiments.mlst2026.zc_baselines_20261013.runner run --out OUT [--workers 16]
    python -m experiments.mlst2026.zc_baselines_20261013.runner summarize --out OUT [--partial]
    python -m experiments.mlst2026.zc_baselines_20261013.runner smoke --out OUT        # T25 FIXTURE only (tests)
    python -m experiments.mlst2026.zc_baselines_20261013.runner write-protocol         # kit preparation only
    python -m experiments.mlst2026.zc_baselines_20261013.runner write-kit-hashes       # kit preparation only

`run` is fail-closed (preflight): kit SHA-256 (KIT_SHA256.json), pinned dependency bytes and the accepted T25
closure (protocol.json -> dependencies), thread variables = 1, PYTHONDONTWRITEBYTECODE, protocol.json equal to
design.build_protocol, seed derivation, the N* source SHA-256 and table, and the generator fields equal to the
T25 BASE must all agree before the first record is drawn.  Resumable: an existing record is never recomputed
or overwritten.  `summarize` (without --partial) re-verifies the kit and refuses unless OUT/START.json was
written by this kit identity and protocol.  A task exception stops the pool (fail-closed; see README).  Records: OUT/records/theta<th>/N<N>/<metric>/seed<ii>.json; at the end OUT/summary.json
(readings applied) and OUT/results.h5.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

from experiments.mlst2026.y19_20261012 import runner as y19                # Y19 helpers (T25 byte guard on import)
from experiments.mlst2026.t25_production_20260930 import science as t25   # noqa: E402
from experiments.mlst2026.x3_nonlinear_metric_20261005 import guard        # noqa: E402
from scipy.stats import t as student_t                                     # noqa: E402
from . import design, methods, nstar                                       # noqa: E402

PACKAGE = Path(__file__).resolve().parent
EXP_ROOT = PACKAGE.parents[2]
KIT_FILE = PACKAGE / 'KIT_SHA256.json'
THREAD_VARS = y19.THREAD_VARS
sha_file, sha_json, atomic_json, decide, environment = y19.sha_file, y19.sha_json, y19.atomic_json, y19.decide, y19.environment
CLASSES = ('U', 'F')
METHODS = ('U', 'F', 'CV', 'RULE', 'SHR')


class KitMismatch(y19.KitMismatch):
    pass


# ------------------------------------------------------------------------------------ kit identity
def kit_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob('*') if p.is_file() and p != KIT_FILE
                  and '__pycache__' not in p.parts and p.suffix != '.pyc' and '.pytest_cache' not in p.parts)


def kit_hashes() -> dict:
    files = {str(p.relative_to(PACKAGE)): sha_file(p) for p in kit_files()}
    return {'package': str(PACKAGE.relative_to(EXP_ROOT)), 'files': files, 'kit_identity': sha_json(files)}


def write_kit_hashes() -> dict:
    h = kit_hashes()
    KIT_FILE.write_text(json.dumps(h, indent=1) + '\n')
    return h


def verify_kit() -> dict:
    if not KIT_FILE.is_file():
        raise KitMismatch('KIT_SHA256.json missing')
    want = json.loads(KIT_FILE.read_text())
    got = kit_hashes()
    if want['files'] != got['files']:
        diff = sorted(set(want['files'].items()) ^ set(got['files'].items()))
        raise KitMismatch(f'kit files differ from KIT_SHA256.json: {diff[:6]}')
    if want['kit_identity'] != got['kit_identity']:
        raise KitMismatch('kit identity differs')
    return got


def loaded_repo_modules() -> list[str]:
    own = str(PACKAGE.relative_to(EXP_ROOT)) + '/'
    return [r for r in guard.loaded_repo_modules() if not r.startswith(own) and Path(r).name != 'conftest.py']


def dependency_snapshot() -> dict:
    """SHA-256 of every in-repository module outside this package loaded by the runner (kit preparation)."""
    closure = guard.load_closure()
    files = {r: sha_file(EXP_ROOT / r) for r in loaded_repo_modules()}
    in_closure = sorted(r for r in files if r in closure['EXP_files'])
    guard.assert_files(in_closure, closure)
    return {'accepted_t25_identity': guard.ACCEPTED_T25_IDENTITY, 'files': files,
            'in_accepted_t25_closure': in_closure, 'outside_closure': sorted(set(files) - set(in_closure))}


def assert_dependencies(proto: dict) -> dict:
    want = proto['dependencies']['files']
    for rel, h in want.items():
        got = sha_file(EXP_ROOT / rel)
        if got != h:
            raise KitMismatch(f'dependency {rel} differs ({got[:12]} != {h[:12]})')
    for rel in loaded_repo_modules():
        if rel not in want:
            raise KitMismatch(f'unpinned in-repository module loaded: {rel}')
    guard.assert_repo_origin()
    guard.assert_files(proto['dependencies']['in_accepted_t25_closure'])
    return {'checked': len(want)}


def load_protocol() -> dict:
    return json.loads((PACKAGE / 'protocol.json').read_text())


def preflight(proto: dict) -> dict:
    """Everything that must hold before the first record is drawn (confirmation mode)."""
    kit = verify_kit()
    bad = {v: os.environ.get(v) for v in THREAD_VARS if os.environ.get(v) != '1'}
    if bad:
        raise KitMismatch(f'thread variables must be 1: {bad}')
    if not sys.dont_write_bytecode:
        raise KitMismatch('set PYTHONDONTWRITEBYTECODE=1')
    deps = assert_dependencies(proto)
    if design.build_protocol(proto['dependencies']) != proto:
        raise KitMismatch('protocol.json is not the protocol built by design.build_protocol')
    if design.seeds() != proto['seeds'] or proto['namespace'] != design.NAMESPACE:
        raise KitMismatch('seed list / namespace differ from the namespace derivation')
    if nstar.source_sha() != proto['nstar']['source']['sha256'] or nstar.table() != proto['nstar']['table']:
        raise KitMismatch('N* source or table differs from the pinned Y15 RESULT.json')
    if proto['generator'] != design.generator(design.t25p.BASE):
        raise KitMismatch('generator differs from the T25 BASE')
    return {'kit_identity': kit['kit_identity'], 'dependencies': deps, 'nstar_source_sha256': nstar.source_sha()}


# ------------------------------------------------------------------------------------ science
_DATA = {}


def make_cfg(proto: dict) -> dict:
    return dict(proto['generator'], namespace=proto['namespace'], N=list(proto['N']), m=list(proto['m']))


def data(proto: dict, seed: int, theta: float):
    """t25.science.make_data, cached for the last (protocol, seed, theta) in this process."""
    key = (sha_json(make_cfg(proto)), seed, float(theta))
    if key not in _DATA:
        _DATA.clear()
        _DATA[key] = t25.make_data(make_cfg(proto), seed, theta)
    return _DATA[key]


def metric_covariance(metric: str, sigma: np.ndarray, pool: np.ndarray, C: int, T: int) -> np.ndarray:
    if metric == 'oracle':
        return sigma
    for arm in ('kron', 'full'):
        if metric.startswith(arm):
            m = int(metric[len(arm):])
            return t25.estimate(pool[:m], C, T)[0][arm]
    raise ValueError(f'unknown metric arm {metric}')


def _risk(fit, Cs, sigma, W0) -> dict:
    return t25.risks(fit['A'], fit['b'], Cs, sigma, W0)


def fit_task(proto: dict, si: int, theta: float, N: int, metric: str, keep_arrays: bool = False):
    seed = proto['seeds'][si]
    ns = proto['namespace']
    g, sigma, L, Li, pool, training = data(proto, seed, theta)
    x, _, _ = t25.moments(training, N)
    K = g.K(theta)
    Cs = L @ K @ L
    W0 = Li @ Li
    cov = metric_covariance(metric, sigma, pool, g.C, g.T)
    W = np.linalg.inv(cov)
    dims = (g.C, g.T, g.k_c, g.k_t)
    recipe = {'max_iter': proto['learners']['max_iter'], 'gradient_tolerance': proto['learners']['gradient_tolerance']}
    start_key = (ns, seed, theta, N)
    fold_idx = methods.folds(N, (ns, seed, theta, N, 'cv-folds'), proto['methods']['folds'])
    all_methods = metric in proto['metrics']['all_methods']

    fits = {'U': methods.fit_U(x, W, dims), 'F': methods.fit_F(x, W, dims, start_key, recipe)}
    risks = {c: _risk(fits[c], Cs, sigma, W0) for c in CLASSES}
    rr = []
    for uc, ut in fits['F']['factors']:
        A, b = methods.affine(np.kron(ut, uc), W, fits['F']['mean'])
        rr.append(t25.risks(A, b, Cs, sigma, W0)['clean'])
    D = risks['F']['clean'] - risks['U']['clean']
    spread = float(max(rr) - min(rr))
    sel = fits['F']['selected']
    limited = bool(fits['F']['restarts'][sel]['status'] != 'stationary' or spread >= abs(D))

    cv = methods.cv_select(x, W, dims, fold_idx, recipe, start_key, full=fits)
    risks['CV'] = risks[cv['choice']]
    rec_cv = {k: cv[k] for k in ('choice', 'fold_scores', 'mean_scores', 'fold_F', 'fold_F_nonstationary')}
    rec_rule = rec_shr = None
    shr = None
    if all_methods:
        rule = methods.nstar_rule(theta, N, proto['nstar']['table'])
        risks['RULE'] = risks[rule['choice']]
        rec_rule = rule
        shr = methods.fit_shr(x, W, dims, fold_idx, proto['methods']['lambda_grid'], return_path=True)
        risks['SHR'] = _risk(shr['fit'], Cs, sigma, W0)
        path_clean = [_risk(shr['path'][i], Cs, sigma, W0)['clean'] for i in range(len(shr['lambdas']))]
        rec_shr = {'lambda_hat': shr['lambda_hat'], 'lambdas': shr['lambdas'], 'fold_scores': shr['fold_scores'],
                   'mean_scores': shr['mean_scores'], 'path_clean': path_clean, 'nkp_full': shr['nkp_full'],
                   'nkp_folds_max_rank1_residual_rel': max(f['rank1_residual_rel'] for f in shr['nkp_folds']),
                   'nkp_folds_max_psd_clip': max(max(f['psd_clip_channel'], f['psd_clip_time']) for f in shr['nkp_folds'])}
    clean = {mth: (risks[mth]['clean'] if mth in risks else None) for mth in METHODS}
    best = min(clean['U'], clean['F'])
    regret = {mth: (clean[mth] - best if clean[mth] is not None else None) for mth in METHODS}
    rec = {'study': proto['study'], 'evidence_role': proto['evidence_role'], 'namespace': ns, 'seed_index': si,
           'seed': seed, 'theta': theta, 'N': N, 'metric': metric,
           'metric_role': 'all_methods' if all_methods else 'cv_select_only',
           'clean': clean, 'regret_vs_seed_min': regret,
           'better_class_this_seed': 'U' if clean['U'] <= clean['F'] else 'F',
           'D_F_minus_U': D, 'risks': risks, 'cv': rec_cv, 'rule': rec_rule, 'shr': rec_shr,
           'optimization': {'selected': sel, 'restarts': fits['F']['restarts'], 'restart_clean_risks': rr,
                            'restart_spread': spread, 'optimization_limited': limited,
                            'cv_fold_F_nonstationary': cv['fold_F_nonstationary']},
           'training_sha': t25.digest(x), 'calibration_sha': t25.digest(pool), 'metric_cov_sha': t25.digest(cov),
           'folds_sha': sha_json([f.tolist() for f in fold_idx])}
    arrays = None
    if keep_arrays:
        arrays = {'mean': fits['U']['mean'], 'B_U': fits['U']['B'], 'B_F': fits['F']['B']}
        if shr is not None:
            arrays['B_SHR'] = shr['fit']['B']
    return rec, arrays


def record_path(out, theta: float, N: int, metric: str, si: int) -> Path:
    return Path(out) / 'records' / f'theta{theta:.2f}' / f'N{N}' / metric / f'seed{si:02d}.json'


def run_task(task):
    proto, si, theta, N, metric, out, save_arrays = task
    out = Path(out)
    p = record_path(out, theta, N, metric, si)
    if p.is_file():
        return {'written': 0}
    t0 = time.time()
    rec, arrays = fit_task(proto, si, theta, N, metric, keep_arrays=save_arrays)
    rec['seconds'] = time.time() - t0
    if save_arrays:
        ap = out / 'arrays' / p.relative_to(out / 'records').with_suffix('.npz')
        ap.parent.mkdir(parents=True, exist_ok=True)
        tmp = ap.with_name(ap.stem + '.tmp.npz')
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, ap)
    atomic_json(p, rec)
    with (out / 'RUN_LOG.txt').open('a') as f:
        f.write(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} theta={theta} N={N} {metric} '
                f'seed{si:02d} {rec["seconds"]:.1f}s limited={rec["optimization"]["optimization_limited"]}\n')
    return {'written': 1}


def all_metrics(proto: dict) -> list[str]:
    return list(proto['metrics']['all_methods']) + list(proto['metrics']['cv_select_only'])


def tasks(proto: dict, out, save_arrays: bool = False) -> list:
    """Largest N first (load balance); within N seed-major, so a worker often reuses its cached draw."""
    return [(proto, si, th, N, m, str(out), save_arrays)
            for N in sorted(proto['N'], reverse=True) for si in range(len(proto['seeds']))
            for th in proto['theta'] for m in all_metrics(proto)]


def execute(proto: dict, out, workers: int, save_arrays: bool = False) -> list:
    todo = [t_ for t_ in tasks(proto, out, save_arrays) if not record_path(out, t_[2], t_[3], t_[4], t_[1]).is_file()]
    if workers <= 1:
        return [run_task(t_) for t_ in todo]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=50) as p:
        return list(p.imap_unordered(run_task, todo, chunksize=1))


# ------------------------------------------------------------------------------------ reduction
def load_records(proto: dict, out, partial: bool = False) -> dict:
    recs, missing = {}, []
    for th in proto['theta']:
        for N in proto['N']:
            for m in all_metrics(proto):
                for si in range(len(proto['seeds'])):
                    p = record_path(out, th, N, m, si)
                    if p.is_file():
                        recs[(th, N, m, si)] = json.loads(p.read_text())
                    else:
                        missing.append((th, N, m, si))
    if missing and not partial:
        raise ValueError(f'{len(missing)} records missing, e.g. {missing[:3]}')
    return recs


def interval(values, family_size: int, alpha: float) -> dict:
    """Seed-mean and approximate simultaneous Bonferroni Student-t interval (df = n - 1)."""
    v = np.asarray(values, float)
    n = len(v)
    mean = float(v.mean())
    if n < 2:
        return {'mean': mean, 'sd': None, 'ci': None, 'zero_SE': False, 'decision': 'undecided', 'seeds': n}
    sd = float(v.std(ddof=1))
    half = float(student_t.ppf(1 - alpha / (2 * family_size), n - 1) * sd / np.sqrt(n))
    zero = bool(sd == 0)
    return {'mean': mean, 'sd': sd, 'ci': [mean - half, mean + half], 'zero_SE': zero,
            'decision': decide(mean - half, mean + half, zero), 'seeds': n}


def cell_records(recs: dict, proto: dict, th: float, N: int, m: str) -> list:
    return [recs[(th, N, m, si)] for si in range(len(proto['seeds'])) if (th, N, m, si) in recs]


def better_class(rs: list) -> str:
    """Per-cell better hard class: smaller seed-mean clean risk (ties -> U)."""
    mu_u = float(np.mean([r['clean']['U'] for r in rs]))
    mu_f = float(np.mean([r['clean']['F'] for r in rs]))
    return 'U' if mu_u <= mu_f else 'F'


def family_rows(proto: dict, recs: dict, name: str) -> list:
    fam = proto['families'][name]
    rows = []
    for th, N, m in fam['cells']:
        rs = cell_records(recs, proto, th, N, m)
        if not rs:
            rows.append({'theta': th, 'N': N, 'metric': m, 'decision': 'missing', 'seeds': 0})
            continue
        if name == 'R-C1':
            vals = [r['clean']['RULE'] - r['clean']['CV'] for r in rs]
            extra = {'rule_choice': rs[0]['rule']['choice'], 'N_star_pred': rs[0]['rule']['N_star_pred'],
                     'cv_choice_F': int(sum(r['cv']['choice'] == 'F' for r in rs))}
        else:
            bc = better_class(rs)
            vals = [r['clean']['SHR'] - r['clean'][bc] for r in rs]
            lam = [r['shr']['lambda_hat'] for r in rs]
            extra = {'better_class': bc, 'lambda_hat_counts': {repr(l): lam.count(l) for l in sorted(set(lam))}}
        lim = int(sum(r['optimization']['optimization_limited'] for r in rs))
        cvn = int(sum(r['optimization']['cv_fold_F_nonstationary'] > 0 for r in rs))
        marg = interval(vals, 1, fam['alpha'])
        rows.append({'theta': th, 'N': N, 'metric': m, **interval(vals, fam['family_size'], fam['alpha']), **extra,
                     'marginal_descriptive': {'ci': marg['ci'], 'decision': marg['decision'],
                                              'note': 'unadjusted 95% Student-t; descriptive, not a registered reading'},
                     'optimization_limited_records': lim, 'cv_fold_nonstationary_records': cvn,
                     'flag': bool(lim or cvn)})
    return rows


def reading(rows: list) -> dict:
    """MET iff no cell decided positive; INCOMPLETE if a cell has no records."""
    cell = lambda r: [r['theta'], r['N'], r['metric']]
    pos = [cell(r) for r in rows if r['decision'] == 'positive']
    missing = [cell(r) for r in rows if r['decision'] == 'missing']
    flagged = [cell(r) for r in rows if r.get('flag')]
    return {'reading': 'INCOMPLETE' if missing else ('NOT MET' if pos else 'MET'),
            'decided_positive_cells': pos, 'missing_cells': missing,
            'decisions': {'negative': sum(r['decision'] == 'negative' for r in rows),
                          'undecided': sum(r['decision'] == 'undecided' for r in rows), 'positive': len(pos)},
            'cells_with_optimization_flags': flagged,
            'descriptive_marginal_positive_cells': [cell(r) for r in rows
                                                    if r.get('marginal_descriptive', {}).get('decision') == 'positive'],
            'largest_mean_contrast': max((r['mean'] for r in rows if 'mean' in r), default=None),
            'flag': 'reading involves cells with optimization-limited records' if flagged else None}


def descriptive(proto: dict, recs: dict) -> list:
    cells = []
    for th in proto['theta']:
        for N in proto['N']:
            for m in all_metrics(proto):
                rs = cell_records(recs, proto, th, N, m)
                if not rs:
                    continue
                bc = better_class(rs)
                row = {'theta': th, 'N': N, 'metric': m, 'seeds': len(rs), 'better_class': bc,
                       'cv_choice_F': int(sum(r['cv']['choice'] == 'F' for r in rs)),
                       'optimization_limited_records': int(sum(r['optimization']['optimization_limited'] for r in rs)),
                       'cv_fold_nonstationary_records': int(sum(r['optimization']['cv_fold_F_nonstationary'] > 0 for r in rs)),
                       'mean_seconds': float(np.mean([r.get('seconds', 0.0) for r in rs]))}
                for mth in METHODS:
                    if rs[0]['clean'][mth] is None:
                        continue
                    c = np.array([r['clean'][mth] for r in rs])
                    rg = np.array([r['regret_vs_seed_min'][mth] for r in rs])
                    rb = c - np.array([r['clean'][bc] for r in rs])
                    row[mth] = {'mean_clean': float(c.mean()), 'mean_regret_vs_seed_min': float(rg.mean()),
                                'sd_regret_vs_seed_min': float(rg.std(ddof=1)) if len(rs) > 1 else None,
                                'mean_regret_vs_cell_better': float(rb.mean()),
                                'sd_regret_vs_cell_better': float(rb.std(ddof=1)) if len(rs) > 1 else None}
                if rs[0]['shr'] is not None:
                    lam = [r['shr']['lambda_hat'] for r in rs]
                    row['lambda_hat_counts'] = {repr(l): lam.count(l) for l in sorted(set(lam))}
                    row['shr_path_mean_clean'] = [float(np.mean([r['shr']['path_clean'][i] for r in rs]))
                                                  for i in range(len(rs[0]['shr']['lambdas']))]
                if rs[0]['rule'] is not None:
                    row['rule_choice'] = rs[0]['rule']['choice']
                cells.append(row)
    return cells


def reduce(proto: dict, recs: dict, partial: bool = False) -> dict:
    fams = {name: family_rows(proto, recs, name) for name in ('R-C1', 'R-C2')}
    return {'status': 'COMPLETE' if not partial else 'INCOMPLETE - not a registered reading',
            'interval': 'approximate simultaneous Bonferroni Student-t over each registered family, df = n - 1, '
                        f'alpha = {proto["inference"]["alpha"]}',
            'family_sizes': {n: proto['families'][n]['family_size'] for n in fams},
            'families': fams, 'readings': {n: reading(rows) for n, rows in fams.items()},
            'descriptive_cells': descriptive(proto, recs), 'records': len(recs),
            'optimization_limited_records_total': int(sum(r['optimization']['optimization_limited'] for r in recs.values()))}


def consolidate(proto: dict, recs: dict, out) -> str:
    S, Th, Ns, Ms = len(proto['seeds']), list(proto['theta']), list(proto['N']), all_metrics(proto)
    shape = (len(Th), len(Ns), len(Ms), S)
    tables = {f'clean/{mth}': np.full(shape, np.nan) for mth in METHODS}
    tables.update({k: np.full(shape, np.nan) for k in ('lambda_hat', 'cv_choice_is_F', 'optimization_limited')})
    for (th, N, m, si), r in recs.items():
        ix = (Th.index(th), Ns.index(N), Ms.index(m), si)
        for mth in METHODS:
            if r['clean'][mth] is not None:
                tables[f'clean/{mth}'][ix] = r['clean'][mth]
        tables['cv_choice_is_F'][ix] = float(r['cv']['choice'] == 'F')
        tables['optimization_limited'][ix] = float(r['optimization']['optimization_limited'])
        if r['shr'] is not None:
            tables['lambda_hat'][ix] = r['shr']['lambda_hat']
    out = Path(out)
    try:
        import h5py
        path = out / 'results.h5'
        with h5py.File(path, 'w') as f:
            for k, v in tables.items():
                f.create_dataset(k, data=v)
            f.attrs['axes'] = json.dumps({'theta': Th, 'N': Ns, 'metric': Ms, 'seed_index': list(range(S))})
            f.attrs['protocol_sha256'] = sha_json(proto)
    except ImportError:
        path = out / 'results.npz'
        np.savez_compressed(path, **{k.replace('/', '__'): v for k, v in tables.items()})
    return path.name


def check_start(proto: dict, out) -> dict:
    """Registered summary: the kit must verify and OUT/START.json must bind OUT to this kit and protocol."""
    kit = verify_kit()
    start = Path(out) / 'START.json'
    if not start.is_file():
        raise KitMismatch('OUT/START.json missing: not a run of this kit')
    old = json.loads(start.read_text())
    if old.get('kit_identity') != kit['kit_identity'] or old.get('protocol_sha256') != sha_json(proto):
        raise KitMismatch('OUT/START.json was written by a different kit or protocol')
    return {'kit_identity': kit['kit_identity']}


def summarize(proto: dict, out, partial: bool = False, check_kit: bool = False) -> dict:
    """check_kit=True (the CLI `summarize` without --partial, and `run`) refuses unless verify_kit() passes and
    START.json matches the current kit identity and protocol."""
    out = Path(out)
    if check_kit:
        check_start(proto, out)
    recs = load_records(proto, out, partial)
    summ = reduce(proto, recs, partial)
    summ['results_file'] = consolidate(proto, recs, out)
    summ['protocol_sha256'] = sha_json(proto)
    atomic_json(out / ('summary.json' if not partial else 'summary_partial.json'), summ)
    return summ


# ------------------------------------------------------------------------------------ commands
def cmd_run(out, workers: int, save_arrays: bool = False) -> dict:
    out = Path(out).resolve()
    if out == PACKAGE or PACKAGE in out.parents:
        raise KitMismatch('OUT must lie outside the kit directory (the kit hashes every file in it)')
    proto = load_protocol()
    pf = preflight(proto)
    out.mkdir(parents=True, exist_ok=True)
    start = out / 'START.json'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if start.is_file():
        old = json.loads(start.read_text())
        if old['kit_identity'] != pf['kit_identity'] or old['protocol_sha256'] != sha_json(proto):
            raise KitMismatch('OUT holds a run of a different kit; use a fresh directory')
    else:
        atomic_json(start, {'kit_identity': pf['kit_identity'], 'protocol_sha256': sha_json(proto), 'started_utc': now,
                            'environment': environment(), 'preflight': pf, 'workers': workers,
                            'save_arrays': save_arrays})
    with (out / 'ATTEMPTS.jsonl').open('a') as f:
        f.write(json.dumps({'utc': now, 'host': socket.gethostname(), 'workers': workers,
                            'existing_records': len(load_records(proto, out, partial=True))}) + '\n')
    t0 = time.time()
    done = execute(proto, out, workers, save_arrays)
    summ = summarize(proto, out, check_kit=True)
    summ['wall_seconds_this_attempt'] = time.time() - t0
    summ['tasks_written_this_attempt'] = int(sum(d['written'] for d in done))
    atomic_json(out / 'summary.json', summ)
    return summ


def cmd_smoke(out, workers: int = 1) -> dict:
    """T25 FIXTURE geometry, smoke namespace, fabricated N* table (tests only; never the confirmation namespace)."""
    proto = design.smoke_protocol()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    execute(proto, out, workers)
    return summarize(proto, out)


def write_protocol() -> dict:
    proto = design.build_protocol(dependency_snapshot())
    (PACKAGE / 'protocol.json').write_text(json.dumps(proto, indent=1, allow_nan=False) + '\n')
    return proto


def _short(s: dict) -> dict:
    return {'status': s['status'], 'records': s['records'],
            'readings': {k: {kk: v[kk] for kk in ('reading', 'decisions', 'flag')} for k, v in s['readings'].items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['verify-kit', 'run', 'summarize', 'smoke', 'write-protocol', 'write-kit-hashes'])
    ap.add_argument('--out')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--partial', action='store_true', help='summarize an incomplete run (diagnostic only)')
    ap.add_argument('--save-arrays', action='store_true', help='also write fitted bases and mean per task (NPZ)')
    a = ap.parse_args(argv)
    if a.command == 'verify-kit':
        k = verify_kit()
        proto = load_protocol()
        assert_dependencies(proto)
        print(json.dumps({'kit_identity': k['kit_identity'], 'files': len(k['files']), 'dependencies': 'ok'}, indent=1))
    elif a.command == 'write-kit-hashes':
        print(json.dumps(write_kit_hashes(), indent=1))
    elif a.command == 'write-protocol':
        p = write_protocol()
        print(json.dumps({'dependencies': len(p['dependencies']['files']), 'seeds': len(p['seeds'])}, indent=1))
    elif a.command == 'run':
        print(json.dumps(_short(cmd_run(a.out, a.workers, a.save_arrays)), indent=1))
    elif a.command == 'summarize':
        print(json.dumps(_short(summarize(load_protocol(), a.out, a.partial, check_kit=not a.partial)), indent=1))
    elif a.command == 'smoke':
        print(json.dumps(_short(cmd_smoke(a.out, a.workers)), indent=1))


if __name__ == '__main__':
    try:
        main()
    except KitMismatch as e:
        print('KitMismatch: ' + str(e), file=sys.stderr)
        sys.exit(3)
