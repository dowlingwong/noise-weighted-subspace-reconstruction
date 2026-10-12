"""Y19 confirmation runner (oracle-metric linear classes on G2; T25 oracle-arm recipe).

Commands (EXP root; PYTHONPATH=src; PYTHONDONTWRITEBYTECODE=1; OPENBLAS/OMP/MKL_NUM_THREADS=1):
    python -m experiments.mlst2026.y19_20261012.runner verify-kit
    python -m experiments.mlst2026.y19_20261012.runner run --out OUT [--workers 4]
    python -m experiments.mlst2026.y19_20261012.runner summarize --out OUT
    python -m experiments.mlst2026.y19_20261012.runner smoke --out OUT          # T25 fixture only (tests)
    python -m experiments.mlst2026.y19_20261012.runner write-kit-hashes         # kit preparation only

`run` is fail-closed: kit SHA-256 (KIT_SHA256.json), pinned dependency bytes (protocol.json), the
accepted T25 closure, thread variables, and a population recomputation of predictions.json must
all agree before the first record is drawn.  It is resumable: an existing record is never
recomputed or overwritten, so an interrupted run is continued by repeating the same command.
Records: OUT/records/<label>/seed<ii>_N<N>.json (+ OUT/arrays/... .npz); at the end
OUT/summary.json (reading rule applied) and OUT/results.h5 (or results.npz without h5py).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

from . import generator as gen                                                    # T25 byte guard runs here
from experiments.mlst2026.t25_production_20260930 import science as t25         # noqa: E402
from experiments.mlst2026.x3_nonlinear_metric_20261005 import guard              # noqa: E402
from scipy.stats import t as student_t                                           # noqa: E402

PACKAGE = Path(__file__).resolve().parent
EXP_ROOT = PACKAGE.parents[2]
KIT_FILE = PACKAGE / 'KIT_SHA256.json'
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
LABEL_ORDER = ('control', 'A', 'B')


class KitMismatch(RuntimeError):
    pass


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


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
    return [r for r in guard.loaded_repo_modules() if not r.startswith(str(PACKAGE.relative_to(EXP_ROOT)) + '/')
            and Path(r).name != 'conftest.py']


def dependency_snapshot() -> dict:
    """SHA-256 of every in-repository module outside this package that is loaded (measured after importing
    runner and predictions).  T25-closure members are additionally checked against the accepted closure."""
    from . import predictions  # noqa: F401
    closure = guard.load_closure()
    files = {r: sha_file(EXP_ROOT / r) for r in loaded_repo_modules()}
    in_closure = sorted(r for r in files if r in closure['EXP_files'])
    guard.assert_files(in_closure, closure)
    return {'accepted_t25_identity': guard.ACCEPTED_T25_IDENTITY, 'files': files,
            'in_accepted_t25_closure': in_closure,
            'outside_closure': sorted(set(files) - set(in_closure))}


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


def environment() -> dict:
    import scipy
    try:
        h5 = __import__('h5py').__version__
    except Exception:          # noqa: BLE001
        h5 = None
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'h5py': h5,
            'platform': platform.platform(), 'host': socket.gethostname(),
            'threads': {v: os.environ.get(v) for v in THREAD_VARS}}


def compare_predictions(saved: dict, new: dict, rtol: float = 1e-6) -> dict:
    rows, worst = [], 0.0
    for lab in LABEL_ORDER:
        for q in ('a_best', 'cU', 'cF', 'B_cert', 'N_star_pred'):
            s, n = saved['geometries'][lab][q], new['geometries'][lab][q]
            if s is None or n is None:
                ok = s is None and n is None
                rel = 0.0 if ok else float('inf')
            elif lab == 'control' and q in ('a_best', 'B_cert'):
                rel = abs(s - n)                              # both are zero: absolute comparison
                ok = rel <= 1e-9
            else:
                rel = abs(s - n) / max(abs(s), 1e-300)
                ok = rel <= rtol
            worst = max(worst, rel)
            rows.append({'label': lab, 'quantity': q, 'saved': s, 'recomputed': n, 'rel_or_abs_diff': rel, 'ok': bool(ok)})
    grids_ok = saved['grids'] == new['grids']
    return {'ok': bool(all(r['ok'] for r in rows) and grids_ok), 'grids_equal': bool(grids_ok), 'max_diff': worst,
            'rtol': rtol, 'rows': rows}


# ------------------------------------------------------------------------------------ science
_GEOMETRY_CACHE = {}


def geometry(cfg: dict):
    key = sha_json(cfg)
    if key not in _GEOMETRY_CACHE:
        _GEOMETRY_CACHE[key] = gen.build(cfg)
    return _GEOMETRY_CACHE[key]


def fit_record(g, proto: dict, phi: float, seed: int, N: int, pool: np.ndarray):
    """One (geometry, seed, N) record: the T25 oracle arm (t25.science.fit_cell, 'oracle' estimate) on G2."""
    lr = proto['learners']
    x, mean, S = t25.moments(pool, N)
    K = g.K(phi)
    Cs = g.L @ K @ g.L
    W0 = g.Li @ g.Li
    W = np.linalg.inv(g.sigma)                                  # T25: W = inv(cov) with cov = Sigma (oracle)
    truth = t25.top(K, g.k)
    initial = t25.starts(S, g.C, g.T, g.k_c, g.k_t, (proto['namespace'], seed, phi, N))
    B, diag, factors = t25.structured(W, S, initial, lr['max_iter'], lr['gradient_tolerance'])
    Bu = t25.unrestricted(W, S, g.k)
    out, arrays = {}, {'mean': mean, 'B_U': Bu, 'B_F': B}
    for cls, basis in (('U', Bu), ('F', B)):
        A = t25.map_for(basis, W)
        b = (np.eye(g.d) - A) @ mean
        out[cls] = t25.risks(A, b, Cs, g.sigma, W0) | t25.representation(basis, g.Li, truth)
    rr = []
    for i, (uc, ut) in enumerate(factors):
        arrays[f'uc_{i}'], arrays[f'ut_{i}'] = uc, ut
        A = t25.map_for(np.kron(ut, uc), W)
        rr.append(t25.risks(A, (np.eye(g.d) - A) @ mean, Cs, g.sigma, W0)['clean'])
    D = out['F']['clean'] - out['U']['clean']
    spread = float(max(rr) - min(rr))
    restarts = [{'objective': r['objective'], 'gradient_norm': r['gradient_norm'], 'status': r['status'],
                 'iterations': len(r['trajectory']) - 1, 'accepted_steps': len(r['accepted_steps'])}
                for r in diag['restarts']]
    sel = diag['selected']
    rec = {'seed': seed, 'phi': phi, 'N': N, 'R_U': out['U']['clean'], 'R_F': out['F']['clean'], 'D': D,
           'signal_U': out['U']['signal'], 'signal_F': out['F']['signal'], 'risks': out,
           'optimization': {'selected': sel, 'restarts': restarts}, 'restart_clean_risks': rr, 'restart_spread': spread,
           'optimization_limited': bool(restarts[sel]['status'] != 'stationary' or spread >= abs(D)),
           'selected_is_hosvd_start': bool(sel == 0), 'training_sha': t25.digest(x)}
    return rec, arrays


def record_path(out: Path, label: str, si: int, N: int) -> Path:
    return out / 'records' / label / f'seed{si:02d}_N{N}.json'


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False))
    os.replace(tmp, path)


def run_group(task):
    """All N of one (geometry, seed): one pool of max(grid) records, N ascending (prefixes)."""
    proto, label, si, out = task
    out = Path(out)
    gm = next(x for x in proto['geometries'] if x['label'] == label)
    seed, phi, grid_ = proto['seeds'][si], float(gm['phi']), sorted(gm['grid'])
    todo = [N for N in grid_ if not record_path(out, label, si, N).is_file()]
    if not todo:
        return {'label': label, 'seed_index': si, 'written': 0}
    g = geometry(proto['generator'])
    pool = gen.draw(g, proto['namespace'], seed, phi, max(grid_))
    pool_sha = t25.digest(pool)
    for N in todo:
        t0 = time.time()
        rec, arrays = fit_record(g, proto, phi, seed, N, pool)
        rec.update(label=label, seed_index=si, pool_sha=pool_sha, pool_size=max(grid_), seconds=time.time() - t0)
        ap = out / 'arrays' / label / f'seed{si:02d}_N{N}.npz'
        ap.parent.mkdir(parents=True, exist_ok=True)
        tmp = ap.with_name(ap.stem + '.tmp.npz')
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, ap)
        atomic_json(record_path(out, label, si, N), rec)
        with (out / 'RUN_LOG.txt').open('a') as f:
            f.write(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {label} seed{si:02d} N={N} '
                    f'{rec["seconds"]:.1f}s limited={rec["optimization_limited"]}\n')
    return {'label': label, 'seed_index': si, 'written': len(todo)}


def tasks(proto: dict, out: Path) -> list:
    # largest pools first for load balance
    order = sorted(proto['geometries'], key=lambda gm: -max(gm['grid']))
    return [(proto, gm['label'], si, str(out)) for gm in order for si in range(len(proto['seeds']))]


def execute(proto: dict, out: Path, workers: int) -> list:
    todo = tasks(proto, out)
    if workers <= 1:
        return [run_group(t_) for t_ in todo]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=4) as p:
        return list(p.imap_unordered(run_group, todo))


# ------------------------------------------------------------------------------------ reduction
def load_records(proto: dict, out: Path, partial: bool = False) -> dict:
    recs, missing = {}, []
    for gm in proto['geometries']:
        for si in range(len(proto['seeds'])):
            for N in gm['grid']:
                p = record_path(out, gm['label'], si, N)
                if p.is_file():
                    recs[(gm['label'], si, N)] = json.loads(p.read_text())
                else:
                    missing.append((gm['label'], si, N))
    if missing and not partial:
        raise ValueError(f'{len(missing)} records missing, e.g. {missing[:3]}')
    return recs


def decide(lo: float, hi: float, zero_se: bool) -> str:
    if zero_se:
        return 'undecided'
    return 'negative' if hi < 0 else 'positive' if lo > 0 else 'undecided'


def ols(x, y):
    X = np.stack([np.ones_like(x), x], 1)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def reduce(proto: dict, pred: dict, recs: dict, partial: bool = False) -> dict:
    inf = proto['inference']
    F, alpha = inf['family_size'], inf['alpha']
    seeds = range(len(proto['seeds']))
    RK_PU = {lab: float(np.sum(np.asarray(geometry(proto['generator']).mu)[geometry(proto['generator']).k:]))
             for lab in LABEL_ORDER}
    cells = []
    for gm in proto['geometries']:
        lab = gm['label']
        a_best = pred['geometries'][lab]['a_best'] if pred else 0.0
        for N in gm['grid']:
            rs = [recs[(lab, si, N)] for si in seeds if (lab, si, N) in recs]
            n = len(rs)
            if n == 0:
                continue
            D = np.array([r['D'] for r in rs])
            mean, sd = float(D.mean()), float(D.std(ddof=1)) if n > 1 else float('nan')
            half = float(student_t.ppf(1 - alpha / (2 * F), n - 1) * sd / np.sqrt(n)) if n > 1 else float('nan')
            zero_se = bool(n > 1 and sd == 0)
            lim = int(sum(r['optimization_limited'] for r in rs))
            dU = np.array([r['signal_U'] for r in rs]) - RK_PU[lab]
            dF = np.array([r['signal_F'] for r in rs]) - (RK_PU[lab] + a_best)
            cells.append({'label': lab, 'phi': gm['phi'], 'N': N, 'seeds': n, 'mean_D': mean, 'sd_D': sd,
                          'ci': [mean - half, mean + half], 'zero_SE': zero_se,
                          'decision': decide(mean - half, mean + half, zero_se) if n > 1 else 'undecided',
                          'optimization_limited_records': lim, 'optimization_limited': lim > 0,
                          'mean_R_U': float(np.mean([r['R_U'] for r in rs])), 'mean_R_F': float(np.mean([r['R_F'] for r in rs])),
                          'mean_Delta_U': float(dU.mean()), 'mean_Delta_F': float(dF.mean()),
                          'selected_hosvd_records': int(sum(r['selected_is_hosvd_start'] for r in rs)),
                          'mean_seconds': float(np.mean([r['seconds'] for r in rs]))})
    by = {(c['label'], c['N']): c for c in cells}
    readings = {}
    for lab in ('A', 'B'):
        gm = next(x for x in proto['geometries'] if x['label'] == lab)
        g5 = gm['grid']
        dec = [by[(lab, N)]['decision'] if (lab, N) in by else 'missing' for N in g5]
        met = dec[0] == 'negative' and dec[1] == 'negative' and dec[3] == 'positive' and dec[4] == 'positive'
        deciding = [g5[i] for i in (0, 1, 3, 4)]
        flagged = [N for N in deciding if (lab, N) in by and by[(lab, N)]['optimization_limited']]
        sym = {'negative': '-', 'positive': '+', 'undecided': '?', 'missing': 'x'}
        readings[lab] = {'phi': gm['phi'], 'N_star_pred': gm['N_star_pred'], 'grid': g5, 'decisions': dec,
                         'pattern': ' '.join(sym[d] for d in dec), 'reading': 'MET' if met else 'NOT MET',
                         'deciding_cells_optimization_limited': flagged,
                         'flag': 'reading involves optimization-limited deciding cells' if flagged else None}
    ctrl = next(x for x in proto['geometries'] if x['label'] == 'control')
    cdec = [by[('control', N)]['decision'] if ('control', N) in by else 'missing' for N in ctrl['grid']]
    readings['control'] = {'grid': ctrl['grid'], 'decisions': cdec,
                           'reading': 'NOT MET' if 'positive' in cdec else 'MET',
                           'optimization_limited_cells': [N for N in ctrl['grid'] if ('control', N) in by and by[('control', N)]['optimization_limited']]}
    overall = 'MET' if readings['A']['reading'] == 'MET' and readings['B']['reading'] == 'MET' else 'NOT MET'
    # secondary descriptives
    sec = {}
    for lab in ('A', 'B'):
        gm = next(x for x in proto['geometries'] if x['label'] == lab)
        Ns = np.array([N for N in gm['grid'] if (lab, N) in by], float)
        y = np.array([by[(lab, int(N))]['mean_D'] for N in Ns])
        alpha_, beta_ = ols(1 / Ns, y) if len(Ns) >= 2 else (float('nan'), float('nan'))
        mid = gm['grid'][2]
        sec[lab] = {'N_star_cell': mid, 'mean_D_at_N_star_cell': by[(lab, mid)]['mean_D'] if (lab, mid) in by else None,
                    'ci_at_N_star_cell': by[(lab, mid)]['ci'] if (lab, mid) in by else None,
                    'fit_alpha': alpha_, 'fit_beta': beta_,
                    'N_star_fit': (-beta_ / alpha_) if (alpha_ > 0 and beta_ < 0) else None,
                    'predicted_alpha_a_best': pred['geometries'][lab]['a_best'] if pred else None,
                    'predicted_beta_B': pred['geometries'][lab]['B'] if pred else None}
    Ns = np.array([c['N'] for c in cells], float)
    yU = np.array([c['N'] * c['mean_Delta_U'] for c in cells])
    cfit, efit = ols(1 / Ns, yU) if len(cells) >= 2 else (float('nan'), float('nan'))
    sec['unrestricted_excess'] = {'model': 'N * mean Delta_U = c + e/N (OLS over all registered cells)',
                                  'c_fit': cfit, 'e_fit': efit,
                                  'c_U_predicted': pred['geometries']['A']['cU'] if pred else None,
                                  'ratio_c_fit_over_c_U': (cfit / pred['geometries']['A']['cU']) if pred else None}
    status = 'COMPLETE' if not partial else 'INCOMPLETE - not a registered reading'
    return {'status': status, 'family_size': F, 'interval': 'approximate Bonferroni Student-t over independent seeds, '
            f'df = n - 1, alpha = {alpha}', 'cells': cells, 'readings': readings, 'overall': overall,
            'secondary_descriptive': sec, 'records': len(recs),
            'optimization_limited_records_total': int(sum(r['optimization_limited'] for r in recs.values()))}


def consolidate(proto: dict, recs: dict, out: Path) -> str:
    keys = ('D', 'R_U', 'R_F', 'signal_U', 'signal_F', 'restart_spread', 'seconds')
    tables = {}
    for gm in proto['geometries']:
        lab, Ns = gm['label'], gm['grid']
        for k in keys + ('optimization_limited',):
            M = np.full((len(proto['seeds']), len(Ns)), np.nan)
            for si in range(len(proto['seeds'])):
                for j, N in enumerate(Ns):
                    r = recs.get((lab, si, N))
                    if r is not None:
                        M[si, j] = float(r[k])
            tables[f'{lab}/{k}'] = M
        tables[f'{lab}/N'] = np.asarray(Ns, float)
    try:
        import h5py
        path = out / 'results.h5'
        with h5py.File(path, 'w') as f:
            for k, v in tables.items():
                f.create_dataset(k, data=v)
            f.attrs['seeds'] = json.dumps(proto['seeds'])
            f.attrs['protocol_sha256'] = sha_json(proto)
            f.attrs['layout'] = 'rows = seed index, columns = grid N (ascending as in protocol)'
    except ImportError:
        path = out / 'results.npz'
        np.savez_compressed(path, **{k.replace('/', '__'): v for k, v in tables.items()})
    return str(path.name)


def summarize(proto: dict, pred: dict, out: Path, partial: bool = False) -> dict:
    recs = load_records(proto, out, partial)
    summ = reduce(proto, pred, recs, partial)
    summ['results_file'] = consolidate(proto, recs, out)
    summ['protocol_sha256'] = sha_json(proto)
    atomic_json(out / ('summary.json' if not partial else 'summary_partial.json'), summ)
    return summ


def load_kit() -> tuple[dict, dict]:
    return json.loads((PACKAGE / 'protocol.json').read_text()), json.loads((PACKAGE / 'predictions.json').read_text())


def preflight(proto: dict, pred: dict) -> dict:
    """Everything that must hold before the first record is drawn (confirmation mode)."""
    kit = verify_kit()
    bad = {v: os.environ.get(v) for v in THREAD_VARS if os.environ.get(v) != '1'}
    if bad:
        raise KitMismatch(f'thread variables must be 1: {bad}')
    if not sys.dont_write_bytecode:
        raise KitMismatch('set PYTHONDONTWRITEBYTECODE=1')
    from . import predictions as P
    deps = assert_dependencies(proto)
    if P.build_protocol(pred, proto['dependencies']) != proto:
        raise KitMismatch('protocol.json is not the protocol built from predictions.json')
    if P.seeds() != proto['seeds']:
        raise KitMismatch('seed list differs from the namespace derivation')
    new = P.compute(scan=False)                       # population quantities only; no record is drawn
    cmp_ = compare_predictions(pred, new)
    if not cmp_['ok']:
        raise KitMismatch(f'population recomputation differs from predictions.json: {cmp_}')
    return {'kit_identity': kit['kit_identity'], 'dependencies': deps, 'prediction_recomputation': cmp_}


def cmd_run(out: Path, workers: int) -> dict:
    out = out.resolve()
    if out == PACKAGE or PACKAGE in out.parents:
        raise KitMismatch('OUT must lie outside the kit directory (the kit hashes every file in it)')
    proto, pred = load_kit()
    pf = preflight(proto, pred)
    out.mkdir(parents=True, exist_ok=True)
    start = out / 'START.json'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if start.is_file():
        old = json.loads(start.read_text())
        if old['kit_identity'] != pf['kit_identity'] or old['protocol_sha256'] != sha_json(proto):
            raise KitMismatch('OUT holds a run of a different kit; use a fresh directory')
    else:
        atomic_json(start, {'kit_identity': pf['kit_identity'], 'protocol_sha256': sha_json(proto),
                            'predictions_sha256': sha_file(PACKAGE / 'predictions.json'), 'started_utc': now,
                            'environment': environment(), 'preflight': pf, 'workers': workers})
    with (out / 'ATTEMPTS.jsonl').open('a') as f:
        f.write(json.dumps({'utc': now, 'host': socket.gethostname(), 'workers': workers,
                            'existing_records': len(load_records(proto, out, partial=True))}) + '\n')
    t0 = time.time()
    done = execute(proto, out, workers)
    summ = summarize(proto, pred, out)
    summ['wall_seconds_this_attempt'] = time.time() - t0
    summ['groups'] = len(done)
    atomic_json(out / 'summary.json', summ)
    return summ


# ------------------------------------------------------------------------------------ smoke (T25 fixture, never G2)
SMOKE_GENERATOR = {'family': 't25', 'C': 3, 'T': 4, 'kc': 2, 'kt': 2,
                   'spectrum': {'top': [6.0, 5.0, 4.0, 3.0], 'gap': 1.5, 'tail_min': 0.05},
                   'geometry_seed': 20260913, 'rho_c': 0.4, 'rho_t': 0.7}


def smoke_protocol(n_seeds: int = 3) -> dict:
    from . import predictions as P
    fake = {'generator': SMOKE_GENERATOR,
            'geometries': {'control': {'phi': 0.0, 'N_star_pred': None, 'N_star_interval': None},
                           'A': {'phi': 0.3, 'N_star_pred': 40.0, 'N_star_interval': None},
                           'B': {'phi': 0.6, 'N_star_pred': 24.0, 'N_star_interval': None}},
            'grids': {'A': P.grid(40.0), 'B': P.grid(24.0)}}
    fake['grids']['control'] = sorted(set(fake['grids']['A']) | set(fake['grids']['B']))
    proto = P.build_protocol(fake, {'files': {}, 'in_accepted_t25_closure': []})
    proto.update(namespace='y19-smoke-fixture', evidence_role='SMOKE (T25 fixture geometry; not G2)',
                 seeds=[gen.seed_int('y19-smoke-fixture', 'seed', i) % 10 ** 12 for i in range(n_seeds)])
    proto['learners'] = dict(proto['learners'], max_iter=50)
    proto['inference'] = dict(proto['inference'], df=n_seeds - 1)
    return proto


def cmd_smoke(out: Path, workers: int = 1) -> dict:
    proto = smoke_protocol()
    out.mkdir(parents=True, exist_ok=True)
    execute(proto, out, workers)
    return summarize(proto, None, out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['verify-kit', 'run', 'summarize', 'smoke', 'write-kit-hashes'])
    ap.add_argument('--out')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--partial', action='store_true', help='summarize an incomplete run (diagnostic only)')
    a = ap.parse_args(argv)
    if a.command == 'verify-kit':
        k = verify_kit()
        proto, _ = load_kit()
        assert_dependencies(proto)
        print(json.dumps({'kit_identity': k['kit_identity'], 'files': len(k['files']), 'dependencies': 'ok'}, indent=1))
    elif a.command == 'write-kit-hashes':
        print(json.dumps(write_kit_hashes(), indent=1))
    elif a.command == 'run':
        s = cmd_run(Path(a.out), a.workers)
        print(json.dumps({'overall': s['overall'], 'readings': s['readings']}, indent=1))
    elif a.command == 'summarize':
        proto, pred = load_kit()
        s = summarize(proto, pred, Path(a.out), a.partial)
        print(json.dumps({'status': s['status'], 'overall': s['overall'], 'readings': s['readings']}, indent=1))
    elif a.command == 'smoke':
        s = cmd_smoke(Path(a.out), a.workers)
        print(json.dumps({'overall': s['overall'], 'readings': s['readings']}, indent=1))


if __name__ == '__main__':
    try:
        main()
    except KitMismatch as e:
        print('KitMismatch: ' + str(e), file=sys.stderr)
        sys.exit(3)
