"""E2X+ runner: verify-kit, bench, smoke, run, summarize.

Run from the EXP root with PYTHONPATH=. (and OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1):
    python -m experiments.mlst2026.e2xplus_plugin_20261023.runner verify-kit
    python -m experiments.mlst2026.e2xplus_plugin_20261023.runner bench
    python -m experiments.mlst2026.e2xplus_plugin_20261023.runner smoke --out /tmp/e2x_smoke
    python -m experiments.mlst2026.e2xplus_plugin_20261023.runner run --out RUN_DIR --workers 2 --n-q2 100
    python -m experiments.mlst2026.e2xplus_plugin_20261023.runner summarize --out RUN_DIR

One atomic JSON per (generator, geometry, seed index): records/{G}/{geometry}/seed{iii}.json.  A record that
exists is never recomputed (resume only); `run` refuses an output directory frozen for another n_Q2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.mlst2026.t25_production_20260930 import science as t25
from . import data as D
from . import plugin as PL

PACKAGE = Path(__file__).resolve().parent
EXP_ROOT = PACKAGE.parents[2]
N_Q1_SEEDS = 200
FIXTURE_NAMESPACE = 'e2xplus-fixture'


# ------------------------------------------------------------------------------------------------ kit
def kit_files() -> list[Path]:
    own = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.name not in ('KIT_SHA256.json',) and 'pinned' not in p.parts or (p.is_file() and 'pinned' in p.parts))
    return [p for p in own if p.is_file() and '__pycache__' not in p.parts and p.name != 'KIT_SHA256.json']


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def protocol() -> dict:
    return json.loads((PACKAGE / 'protocol.json').read_text())


THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


def import_closure() -> list[str]:
    """Every loaded module file under the EXP root (experiments/, src/, package __init__ files), EXP-relative."""
    files = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if not f:
            continue
        fp = Path(f).resolve()
        try:
            rel = fp.relative_to(EXP_ROOT)
        except ValueError:
            continue
        if rel.parts[0] in ('experiments', 'src') and fp.suffix == '.py' and PACKAGE not in fp.parents:
            files.add(rel.as_posix())
    return sorted(files)


def write_manifest() -> dict:
    """Write KIT_SHA256.json over the kit's own files (excluding the manifest itself) - the freeze step."""
    files = {p.relative_to(PACKAGE).as_posix(): sha(p) for p in sorted(PACKAGE.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'KIT_SHA256.json' and p.suffix != '.pyc'}
    out = {'kit': PACKAGE.name, 'files': files}
    (PACKAGE / 'KIT_SHA256.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    return {'files': len(files)}


def verify_kit(require_manifest: bool = False) -> dict:
    proto = protocol()
    bad = {}
    for rel, want in proto['imports_sha256'].items():
        got = sha(EXP_ROOT / rel)
        if got != want:
            bad[rel] = {'want': want, 'got': got}
    for rel, want in proto['pinned_sha256'].items():
        got = sha(PACKAGE / rel)
        if got != want:
            bad[rel] = {'want': want, 'got': got}
    kit_path = PACKAGE / 'KIT_SHA256.json'
    if require_manifest and not kit_path.exists():
        raise SystemExit('verify-kit FAILED: KIT_SHA256.json missing (freeze not done)')
    if kit_path.exists():
        kit = json.loads(kit_path.read_text())
        for rel, want in kit['files'].items():
            got = sha(PACKAGE / rel)
            if got != want:
                bad[f'kit:{rel}'] = {'want': want, 'got': got}
    if bad:
        raise SystemExit('verify-kit FAILED: ' + json.dumps(bad, indent=1))
    return {'status': 'ok', 'imports': len(proto['imports_sha256']), 'pinned': len(proto['pinned_sha256']),
            'kit_manifest': kit_path.exists()}


# ------------------------------------------------------------------------------------------- one unit
def restart_rng(namespace: str, generator: str, geometry: str, N: int, arm: str, seed: int):
    return t25.rng(namespace, generator, geometry, N, arm, seed, 'restarts')


def run_unit(args) -> dict:
    generator, geometry, idx, n_q2, namespace, out = args
    path = Path(out) / 'records' / generator / geometry / f'seed{idx:03d}.json'
    if path.exists():
        return {'skipped': str(path)}
    t0 = time.time()
    seed = D.seed_of(idx, namespace)
    dd = D.draw(generator, geometry, seed, namespace)
    C, T, kc, kt = dd['C'], dd['T'], dd['kc'], dd['kt']
    Ws = PL.whiteners(dd['sigma'], dd['calibration'], C, T)
    q1 = {}
    for N in D.Q1_GRID:
        for arm in PL.ARMS:
            q1[f'{N}/{arm}'] = PL.certificate_only(PL.k_hat(dd['training'], N, Ws[arm]), C, T, kc, kt)
    q2 = {}
    if idx < n_q2:
        for N in D.q2_grid(generator, geometry):
            for arm in PL.ARMS:
                t1 = time.time()
                c = PL.plugin_constants(PL.k_hat(dd['training'], N, Ws[arm]), C, T, kc, kt,
                                        restart_rng(namespace, generator, geometry, N, arm, seed))
                c['rule_choice'] = PL.rule_choice(N, c)
                c.update(PL.debias(N, c))
                c['rule_choice_db'] = PL.rule_choice_db(N, c)
                c['seconds'] = time.time() - t1
                q2[f'{N}/{arm}'] = c
    rec = {'generator': generator, 'geometry': geometry, 'seed_index': idx, 'seed': seed, 'namespace': namespace,
           'in_q2': idx < n_q2, 'q1': q1, 'q2': q2, 'seconds': time.time() - t0,
           'training_sha256': t25.digest(dd['training']), 'calibration_sha256': t25.digest(dd['calibration'])}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec, sort_keys=True, default=_default) + '\n')
    os.replace(tmp, path)
    return {'written': str(path), 'seconds': rec['seconds']}


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------- commands
def units(n_seeds: int, n_q2: int, namespace: str, out: str) -> list[tuple]:
    us = []
    # Q2 seeds first (slow units), G2 before G1 within them, so a time overrun is visible early
    order = sorted(D.geometries(), key=lambda g: (g[0] != 'G2', g))
    for idx in range(n_seeds):
        for generator, geometry in order:
            us.append((generator, geometry, idx, n_q2, namespace, out))
    return sorted(us, key=lambda u: (u[2] >= n_q2, u[2], u[0] != 'G2'))


def cmd_bench(n: int = 1) -> dict:
    """Whole-unit timing, single-threaded (protocol v2 section 8): one Q2 unit and one Q1-only unit per generator
    on the fixture namespace, scaled by grid sizes to the full run at 2 workers."""
    import tempfile
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for G, geo in (('G1', '0.25'), ('G2', 'A')):
            t0 = time.time(); run_unit((G, geo, 0, 1, FIXTURE_NAMESPACE, tmp)); tq2 = time.time() - t0
            t0 = time.time(); run_unit((G, geo, 1, 1, FIXTURE_NAMESPACE, tmp)); tq1 = time.time() - t0
            res[G] = {'q2_unit_seconds': tq2, 'q1_unit_seconds': tq1, 'q2_grid_size': len(D.q2_grid(G, geo))}
    geos = {'G1': [g for G, g in D.geometries() if G == 'G1'], 'G2': [g for G, g in D.geometries() if G == 'G2']}
    for n_q2 in (100, 50):
        sec = 0.0
        for G in ('G1', 'G2'):
            r = res[G]
            per_n = (r['q2_unit_seconds'] - r['q1_unit_seconds']) / r['q2_grid_size']
            for geo in geos[G]:
                sec += N_Q1_SEEDS * r['q1_unit_seconds'] + n_q2 * per_n * len(D.q2_grid(G, geo))
        res[f'projected_hours_2_workers_nq2_{n_q2}'] = sec / 2 / 3600
    res['rule'] = 'n_Q2 = 100 if projected <= 6 h, else 50; STOP if 50 projects > 10 h (protocol v2 section 8)'
    return res


def cmd_run(out: str, workers: int, n_q2: int, namespace: str = D.NAMESPACE, n_seeds: int = N_Q1_SEEDS) -> dict:
    bad = {v: os.environ.get(v) for v in THREAD_VARS if os.environ.get(v) != '1'}
    if bad:
        raise SystemExit(f'refusing to run: single-threaded BLAS required (protocol v2.1 section 8); got {bad}')
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    start = outp / 'START.json'
    meta = {'n_q2': n_q2, 'n_seeds': n_seeds, 'namespace': namespace, 'workers': workers,
            'python': platform.python_version(), 'numpy': np.__version__, 'host': platform.node(),
            'threads': {v: os.environ.get(v) for v in THREAD_VARS},
            'kit': verify_kit(require_manifest=True) if namespace == D.NAMESPACE else 'fixture'}
    if start.exists():
        old = json.loads(start.read_text())
        if (old['n_q2'], old['n_seeds'], old['namespace']) != (n_q2, n_seeds, namespace):
            raise SystemExit(f'output directory frozen for {old}; refusing {meta}')
    else:
        meta['started'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        start.write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    us = units(n_seeds, n_q2, namespace, out)
    done = 0
    with Pool(workers) as pool:
        for r in pool.imap_unordered(run_unit, us):
            done += 1
            if done % 25 == 0 or done == len(us):
                print(f'[{time.strftime("%H:%M:%S")}] {done}/{len(us)} {r}', flush=True)
    return {'units': len(us)}


# ------------------------------------------------------------------------------------------ summarize
def load_records(out: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(out, 'records').rglob('seed*.json'))]


def population() -> dict:
    """Frozen population references: N*_pred, a, B_cert from the pinned Y15 RESULT (G1) and Y19 predictions.json (G2);
    c_U, c_F from pinned/population_constants.json (checked by T1)."""
    proto = protocol()
    y15 = json.loads((PACKAGE / 'pinned' / 'y15_RESULT_20261012.json').read_text())
    pc = json.loads((PACKAGE / 'pinned' / 'population_constants.json').read_text())
    pop = {}
    for r in y15['locations']:
        if r['study'] == 'T20-S':
            g = f"{r['theta']:g}"
            pop[('G1', g)] = {'N_star_pred': r['N_star_pred'], 'B_cert': r['B_cert'], 'a_best': r['a'],
                              'cU': pc['G1'][g]['cU'], 'cF': pc['G1'][g]['cF']}
    pj = json.loads((EXP_ROOT / proto['y19_predictions']).read_text())
    for lab in ('control', 'A', 'B'):
        g = pj['geometries'][lab]
        pop[('G2', lab)] = {'N_star_pred': g['N_star_pred'], 'B_cert': g['B_cert'], 'a_best': g['a_best'],
                            'cU': g['cU'], 'cF': g['cF']}
    for v in pop.values():
        v['N_star_cell'] = int(np.floor(v['N_star_pred'] + 0.5)) if v['N_star_pred'] else None
    return pop


def references() -> dict:
    """Decided signs of the fitted class contrast D = F - U for the scored cells (protocol O4) and CV counts (ZC)."""
    import csv
    ref, cv, optlim = {}, {}, {}
    arm_of = {'oracle': 'oracle', 'kron_256': 'kron256', 'kron_1024': 'kron1024'}          # full arms are not scored
    with open(PACKAGE / 'pinned' / 't25_contrasts.csv', newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        kind, _, rest = r['contrast'].partition('/')
        if kind != 'class' or rest not in arm_of:
            continue
        lo, hi = float(r['ci95_lo']), float(r['ci95_hi'])
        ref[('G1', f"{float(r['theta']):g}", int(r['N']), arm_of[rest])] = \
            'negative' if hi < 0 else 'positive' if lo > 0 else 'undecided'
    y = json.loads((PACKAGE / 'pinned' / 'y19_summary.json').read_text())
    for c in y['cells']:
        ref[('G2', c['label'], int(c['N']), 'oracle')] = c['decision']
        optlim[('G2', c['label'], int(c['N']), 'oracle')] = int(c.get('optimization_limited_records', 0))
    zc = json.loads((PACKAGE / 'pinned' / 'Y19_ZC_EVIDENCE.json').read_text())['zc']['descriptive_cells']
    for c in zc:
        cv[('G1', f"{c['theta']:g}", int(c['N']), c['metric'])] = (int(c['cv_choice_F']), int(c['seeds']))
    return {'decided': ref, 'cv': cv, 'reference_optimization_limited_records': optlim}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def monotone_smallest(grid, ok: dict):
    """Smallest grid N such that ok holds at N and at every larger grid N (None if none); plus a non-monotonicity flag."""
    best = None
    for N in sorted(grid, reverse=True):
        if ok[N]:
            best = N
        else:
            break
    first = next((N for N in sorted(grid) if ok[N]), None)
    return best, (first is not None and first != best)


def _med(v):
    return float(np.median(v)) if len(v) else None


def _finite_json(o):
    """+inf / -inf / nan as strings so SUMMARY.json is strict JSON (export scripts map '+inf' back)."""
    if isinstance(o, float) and not np.isfinite(o):
        return '+inf' if o > 0 else ('-inf' if o < 0 else 'nan')
    if isinstance(o, dict):
        return {k: _finite_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite_json(v) for v in o]
    return o


ARMS_PLUGIN = {'naive': ('N_star_plug', 'rule_choice'), 'debiased': ('N_star_db', 'rule_choice_db')}
SCORED_ARMS = {'G1': ('oracle', 'kron256', 'kron1024'), 'G2': ('oracle',)}


def cmd_summarize(out: str) -> dict:
    recs = load_records(out)
    pop, refs = population(), references()
    res = {'n_records': len(recs), 'O1': {}, 'O2': {}, 'O3': {}, 'O4': {}, 'O5': {}}
    by = {}
    for r in recs:
        by.setdefault((r['generator'], r['geometry']), []).append(r)
    null_geom = {'G1': '0', 'G2': 'control'}
    for (G, geo), rs in sorted(by.items()):
        for N in D.Q1_GRID:
            for arm in PL.ARMS:
                b = np.array([x['q1'][f'{N}/{arm}']['B_cert'] for x in rs])
                diff = b - pop[(G, geo)]['B_cert']
                res['O1'][f'{G}/{geo}/{N}/{arm}'] = {'n': len(b), 'mean_bias': float(diff.mean()), 'sd': float(diff.std(ddof=1)) if len(b) > 1 else None,
                                                    'q05_50_95': [float(q) for q in np.quantile(diff, [0.05, 0.5, 0.95])],
                                                    'mean_B_cert_hat': float(b.mean())}
    for (G, geo), rs in sorted(by.items()):
        if geo == null_geom[G]:
            continue
        null = by[(G, null_geom[G])]
        for arm in PL.ARMS:
            power = {}
            for N in D.Q1_GRID:
                q95 = float(np.quantile([x['q1'][f'{N}/{arm}']['B_cert'] for x in null], 0.95))
                power[N] = {'q95_null': q95, 'power': float(np.mean([x['q1'][f'{N}/{arm}']['B_cert'] > q95 for x in rs]))}
            ndet, nonmono = monotone_smallest(D.Q1_GRID, {N: power[N]['power'] >= 0.9 for N in D.Q1_GRID})
            nst = pop[(G, geo)]['N_star_pred']
            res['O2'][f'{G}/{geo}/{arm}'] = {'power': {str(k): v for k, v in power.items()}, 'N_det_oracle_calibrated': ndet,
                                             'non_monotone': nonmono, 'N_star_pred': nst,
                                             'N_det_over_N_star': (ndet / nst) if (ndet and nst) else None}
    for (G, geo), rs in sorted(by.items()):
        q2 = [x for x in rs if x['in_q2']]
        P_ = pop[(G, geo)]
        nst = P_['N_star_pred']
        separable = geo == null_geom[G]
        for N in D.q2_grid(G, geo):
            for arm in PL.ARMS:
                cs = [x['q2'][f'{N}/{arm}'] for x in q2]
                key = f'{G}/{geo}/{N}/{arm}'
                res['O5'][key] = {
                    'n': len(cs),
                    'median_seconds': float(np.median([c.get('seconds', np.nan) for c in cs])) if cs else None,
                    'median_clipped': float(np.median([c.get('n_clipped', 0) for c in cs])) if cs else None,
                    'frac_all_starts_reach_best': float(np.mean([bool(c.get('all_starts_reach_best')) for c in cs])) if cs else None,
                    'min_H_min_eig': float(np.nanmin([c.get('H_min_eig', np.nan) for c in cs])) if cs else None,
                    'median_cU_ratio': _med([c['cU'] / P_['cU'] for c in cs if c.get('cU') is not None]),
                    'median_cF_ratio': _med([c['cF'] / P_['cF'] for c in cs if c.get('cF') is not None]),
                    'median_a_hat': _med([c['a_best'] for c in cs if c.get('a_best') is not None]),
                    'causes': {k: sum(c.get('cause') == k for c in cs) for k in ('exception', 'nonfinite', 'nonpositive_gap', 'H_not_pos_def')}}
                for p, (fld, choice) in ARMS_PLUGIN.items():
                    vals = [(float('inf') if (p == 'debiased' and c.get('db_cause') == 'a_db_nonpositive') else c.get(fld)) for c in cs]
                    av = [v for v in vals if v is not None]
                    causes = {}
                    for c, v in zip(cs, vals):
                        if v is None:
                            cause = c.get('cause') if p == 'naive' else c.get('db_cause')
                            causes[str(cause)] = causes.get(str(cause), 0) + 1
                    o3 = {'n': len(cs), 'n_available': len(av), 'unavailable_by_cause': causes,
                          'n_a_db_nonpositive': int(sum(c.get('db_cause') == 'a_db_nonpositive' for c in cs)) if p == 'debiased' else None,
                          'median_N_star': float(np.median(av)) if av else None}
                    if nst and cs:
                        ratio = np.array(av) / nst if av else np.array([])
                        o3.update({'median_ratio': float(np.median(ratio)) if av else None,
                                   'iqr_ratio': [float(q) for q in np.quantile(ratio, [0.25, 0.75], method='nearest')] if av else None,
                                   'frac_within_factor2_of_all': float(np.sum((ratio >= 0.5) & (ratio <= 2)) / len(cs)),
                                   'frac_unavailable': 1 - len(av) / len(cs)})
                    if p == 'naive' and cs:
                        iv = [c['N_star_plug_interval'] for c in cs if c.get('N_star_plug_interval')]
                        o3['median_interval'] = [float(np.median([i[0] for i in iv])), float(np.median([i[1] for i in iv]))] if iv else None
                    if nst and p == 'naive':
                        fo = N * nst / (N + nst)
                        o3['first_order_prediction'] = fo
                        o3['median_over_first_order'] = float(np.median(np.array(av) / fo)) if av else None
                    res['O3'][f'{key}/{p}'] = o3
                    if arm in SCORED_ARMS[G]:
                        dec = refs['decided'].get((G, geo, N, arm))
                        want = {'negative': 'F', 'positive': 'U'}.get(dec)
                        ch = [c.get(choice) for c in cs]
                        o4 = {'reference_decision': dec, 'scored': want is not None, 'separable': separable,
                              'rule_F': int(sum(x == 'F' for x in ch)), 'no_decision': int(sum(x is None for x in ch)), 'n': len(ch)}
                        if G == 'G2':
                            o4['reference_optimization_limited_records'] = refs['reference_optimization_limited_records'].get((G, geo, N, arm))
                        if want is not None and ch:
                            k = int(sum(x == want for x in ch))
                            o4.update({'agree': k, 'agree_wilson95': wilson(k, len(ch)),
                                       'population_rule_agrees': ('F' if (nst is None or N < nst) else 'U') == want})
                            cvk = refs['cv'].get((G, geo, N, arm))
                            if cvk is not None:
                                f, n = cvk
                                o4['cv_agree_of_24'] = f if want == 'F' else n - f
                        res['O4'][f'{key}/{p}'] = o4
    res['readings'] = readings(res, pop)
    (Path(out) / 'SUMMARY.json').write_text(json.dumps(_finite_json(json.loads(json.dumps(res, default=_default))), indent=1, sort_keys=True, allow_nan=False) + '\n')
    return res['readings']


TEMPLATES = {
    'A': ('Computed from the training records against an oracle-calibrated null, the certificate flagged nonseparability '
          'from N = {N_det} records, before the crossing, and the bias-corrected plug-in N\u22c6 was within a factor of two '
          'from N = {N_loc}.'),
    'B': ('The certificate flagged nonseparability from N = {N_det} records (oracle-calibrated), but the bias-corrected '
          'plug-in N\u22c6 did not locate the crossing reliably: at N = {N_last}, {P}% of seeds were within a factor of two '
          '(median factor {F}), so N\u22c6 is a population characterization, not yet a selector.'),
    'D': ('The bias-corrected plug-in N\u22c6 was within a factor of two from N = {N_loc}, but the certificate did not '
          'separate the geometry from the separable control before the crossing.'),
    'C': 'Neither plug-in quantity was reliable at the record counts studied.',
}
HEADLINE = (('G1', '0.25'), ('G2', 'A'))
ORDER = {'A': 0, 'B': 1, 'D': 2, 'C': 3}


def fmt_factor(f) -> str:
    if f is None:
        return 'unavailable'
    return 'more than 10' if (not np.isfinite(f) or f > 10) else f'{f:.2f}'


def later(x, y) -> bool:
    """x later than y on the grid; None = later than any value; None vs None = not later (protocol v2.1 section 7)."""
    if x is None and y is None:
        return False
    if x is None:
        return True
    if y is None:
        return False
    return x > y


def readings(res: dict, pop: dict) -> dict:
    """Protocol v2.1 section 7, per geometry and plug-in arm (the oracle arm decides the outcome)."""
    nonsep = [(G, s_) for G, s_ in D.geometries() if pop[(G, s_)]['N_star_pred']]
    rq1, rq2, outcome, text = {}, {}, {}, {}
    for G, geo in nonsep:
        cell = pop[(G, geo)]['N_star_cell']
        for arm in PL.ARMS:
            nd = res['O2'][f'{G}/{geo}/{arm}']['N_det_oracle_calibrated']
            rq1[f'{G}/{geo}/{arm}'] = {'N_det': nd, 'bound': cell, 'holds': nd is not None and nd <= cell}
            for p in ARMS_PLUGIN:
                grid = D.q2_grid(G, geo)
                ok = {N: (res['O3'][f'{G}/{geo}/{N}/{arm}/{p}']['frac_within_factor2_of_all'] >= 0.8
                          and res['O3'][f'{G}/{geo}/{N}/{arm}/{p}']['frac_unavailable'] <= 0.1) for N in grid}
                nloc, nonmono = monotone_smallest(grid, ok)
                below = [N for N in grid if N <= cell]
                nlast = max(below) if below else None
                fac = pct = None
                if nlast is not None:
                    o3 = res['O3'][f'{G}/{geo}/{nlast}/{arm}/{p}']
                    mr = o3.get('median_ratio')
                    fac = None if mr is None else (mr if mr >= 1 else (1 / mr if mr > 0 else float('inf')))
                    pct = round(100 * o3['frac_within_factor2_of_all'])
                rq2[f'{G}/{geo}/{arm}/{p}'] = {'N_loc': nloc, 'non_monotone': nonmono, 'holds': nloc is not None and nloc <= cell,
                                               'N_last': nlast, 'factor_at_N_last': fac, 'pct_within_at_N_last': pct}
        for p in ARMS_PLUGIN:
            r1, r2 = rq1[f'{G}/{geo}/oracle'], rq2[f'{G}/{geo}/oracle/{p}']
            o = 'A' if (r1['holds'] and r2['holds']) else 'B' if r1['holds'] else 'D' if r2['holds'] else 'C'
            outcome[f'{G}/{geo}/{p}'] = o
            text[f'{G}/{geo}/{p}'] = TEMPLATES[o].format(N_det=r1['N_det'], N_loc=r2['N_loc'], N_last=r2['N_last'],
                                                         P=r2['pct_within_at_N_last'], F=fmt_factor(r2['factor_at_N_last']))
    head = {f'{G}/{g}': outcome[f'{G}/{g}/debiased'] for G, g in HEADLINE}
    weakest = max(head.values(), key=lambda v: ORDER[v])
    if head['G1/0.25'] == head['G2/A']:
        main_geo, supp_geo = 'G1/0.25', 'G2/A'
    else:
        main_geo = max(head, key=lambda k: ORDER[head[k]])
        supp_geo = [k for k in head if k != main_geo][0]
    degraded = False
    detail = {}
    for G, geo in HEADLINE:
        for arm in ('kron256', 'kron1024'):
            d = (later(rq1[f'{G}/{geo}/{arm}']['N_det'], rq1[f'{G}/{geo}/oracle']['N_det'])
                 or later(rq2[f'{G}/{geo}/{arm}/debiased']['N_loc'], rq2[f'{G}/{geo}/oracle/debiased']['N_loc']))
            detail[f'{G}/{geo}/{arm}'] = d
            degraded = degraded or d
    ratios = [res['O3'][f'{G}/{geo}/{N}/oracle/naive'].get('median_over_first_order')
              for G, geo in nonsep for N in D.q2_grid(G, geo)]
    inside = [r is not None and 2 / 3 <= r <= 1.5 for r in ratios]
    naive = {'cells': len(ratios), 'cells_within_2/3_3/2': int(sum(inside)),
             'confirmed': bool(sum(inside) >= 0.8 * len(ratios))}
    return {'R-Q1': rq1, 'R-Q2': rq2, 'outcome': outcome, 'text': text, 'headline_debiased': head,
            'abstract_outcome_weakest': weakest, 'main_text_geometry': main_geo, 'supplement_geometry': supp_geo,
            'main_text_sentence': text[f'{main_geo}/debiased'], 'supplement_sentence': text[f'{supp_geo}/debiased'],
            'estimated_arms': 'degraded' if degraded else 'held', 'estimated_arms_detail': detail,
            'naive_first_order_prediction': naive}


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('verify-kit')
    sub.add_parser('write-manifest')
    sub.add_parser('closure')
    b = sub.add_parser('bench'); b.add_argument('--n', type=int, default=4)
    s = sub.add_parser('smoke'); s.add_argument('--out', required=True)
    r = sub.add_parser('run'); r.add_argument('--out', required=True); r.add_argument('--workers', type=int, default=2)
    r.add_argument('--n-q2', type=int, required=True)
    m = sub.add_parser('summarize'); m.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'verify-kit':
        print(json.dumps(verify_kit(), indent=1))
    elif a.cmd == 'write-manifest':
        print(json.dumps(write_manifest(), indent=1))
    elif a.cmd == 'closure':
        dd = D.draw('G2', 'A', D.seed_of(0, FIXTURE_NAMESPACE), FIXTURE_NAMESPACE)          # touch every import path
        D.draw('G1', '0.25', D.seed_of(0, FIXTURE_NAMESPACE), FIXTURE_NAMESPACE)
        from experiments.mlst2026.p1nl_dev_20261006.generator import SparseLatentModel   # noqa: F401
        print(json.dumps(import_closure(), indent=1))
    elif a.cmd == 'bench':
        print(json.dumps(cmd_bench(a.n), indent=1))
    elif a.cmd == 'smoke':
        cmd_run(a.out, workers=2, n_q2=1, namespace=FIXTURE_NAMESPACE, n_seeds=3)
        print(json.dumps(cmd_summarize(a.out), indent=1, default=_default))
    elif a.cmd == 'run':
        if a.n_q2 not in (50, 100):
            raise SystemExit('n_Q2 must be 100 or 50 (protocol section 8)')
        print(json.dumps(cmd_run(a.out, a.workers, a.n_q2), indent=1))
    elif a.cmd == 'summarize':
        print(json.dumps(cmd_summarize(a.out), indent=1, default=_default))
    return 0


if __name__ == '__main__':
    sys.exit(main())
