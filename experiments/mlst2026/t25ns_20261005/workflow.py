"""X2 (T25-NS) deployment: one bounded attempt, in-campaign replay, reduction, export.

Process design follows the accepted T25 deployment (t25_production_20260930/
workflow.py), whose generic helpers are imported read-only after guard.py checks
their bytes: frozen configuration and source identity, an independent review
naming the exact code identity, host qualification (fixtures, benchmark,
immutable snapshot), atomic START/END ledgers, no resume and no retry, one
supervised process group with RSS / wall / scratch caps, a per-cell wall cap,
and a full numerical replay (optimizer included) in the campaign-worker process,
separate from the cell-fitting processes, before the reduction.  Deviation from T25, stated in the protocol: the export copies the
verified run and writes SHA256SUMS instead of replaying a second time; a portable
saved-map audit then runs on the export.
"""
from __future__ import annotations

from . import guard

guard.assert_direct()

import copy                                                           # noqa: E402
import csv                                                            # noqa: E402
import io                                                             # noqa: E402
import json                                                           # noqa: E402
import os                                                             # noqa: E402
import re                                                             # noqa: E402
import secrets                                                        # noqa: E402
import shutil                                                         # noqa: E402
import socket                                                         # noqa: E402
import subprocess                                                     # noqa: E402
import tempfile                                                       # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402
from experiments.mlst2026.t25_production_20260930 import workflow as t25w  # noqa: E402
from experiments.mlst2026.t25_production_20260930.workflow import (   # noqa: E402,F401
    sha, identity, read, write_new, tree_bytes, kill_group, parent_watch, supervise, inventory,
    finite_between, thread_gate, preflight_space, _peak_children_rss as peak_children_rss)
from .protocol import BASE, FIXTURE, BENCHMARK, BENCHMARK_CELLS, CELL_CHARGES, config, enumeration  # noqa: E402
from . import science                                                 # noqa: E402

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
MODULE = 'experiments.mlst2026.t25ns_20261005'
VERSION = 't25ns-deployment/20261005-v1'
PROTOCOL_MS = 'plan/T25NS_PROTOCOL_2026-10-05.md'
VERDICT = 'ACCEPTED_T25NS_IMPLEMENTATION'
EXPECTED_TESTS = 39          # the exact size of tests/ (a vanished test module fails qualification)


def msroot():
    return t25w.msroot()


def external(p):
    return t25w.external(p)


def environment():
    env = t25w.environment()
    env['package_origin'] = str(PACKAGE)
    return env


def git(root, args):
    return subprocess.check_output(['git', '-c', 'core.fsmonitor=false'] + args, cwd=root, text=True,
                                   env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).strip()


def package_files():
    return sorted(p for p in PACKAGE.rglob('*') if p.is_file() and p.suffix in ('.py', '.json')
                  and '__pycache__' not in p.parts)


def source_files():
    """X2 code identity closure: this package (code, configs, tests, the accepted T25
    closure copy), every accepted T25 file a run loads (guard.T25_REUSED, byte-checked),
    pyproject.toml, and the MS protocol.  Unrelated later EXP/MS files do not enter it."""
    reused = guard.assert_files(guard.T25_REUSED)
    exp = {str(p.relative_to(ROOT)): sha(p) for p in package_files()}
    exp.update(reused)
    exp['pyproject.toml'] = sha(ROOT / 'pyproject.toml')
    return {'EXP_files': dict(sorted(exp.items())), 'MS_files': {PROTOCOL_MS: sha(msroot() / PROTOCOL_MS)},
            't25_accepted_identity': guard.ACCEPTED_T25_IDENTITY}


def code_identity(s=None):
    s = s or source_files()
    return identity({k: s[k] for k in ('EXP_files', 'MS_files', 't25_accepted_identity')})


def sources(include_git=True):
    out = source_files()
    if include_git:
        out.update({'EXP_head': git(ROOT, ['rev-parse', 'HEAD']), 'MS_head': git(msroot(), ['rev-parse', 'HEAD']),
                    'EXP_status': git(ROOT, ['status', '--short']), 'MS_status': git(msroot(), ['status', '--short'])})
    return out


def verify_source(f):
    guard.assert_loaded_modules()
    if code_identity(source_files()) != code_identity(f['sources']) or environment() != f['environment']:
        raise ValueError('source/environment changed since freeze')


# --------------------------------------------------------------------------------------
# freeze and gates
# --------------------------------------------------------------------------------------
def evidence(record, name, json_body=True):
    path, digest = record.get(name + '_path'), record.get(name + '_sha256')
    if not isinstance(path, str) or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('evidence identity absent: ' + name)
    p = Path(path)
    if not p.is_file() or sha(p) != digest:
        raise ValueError('evidence missing/changed: ' + name)
    if json_body:
        if p.stat().st_size > 4 * 1024 ** 2:
            raise ValueError('oversized evidence')
        return read(p)
    return p


def production_gate(acceptance, qualification, s, env):
    cid = code_identity(s)
    a, q = acceptance, qualification
    if a.get('verdict') != VERDICT or a.get('deployment') != VERSION or a.get('code_identity') != cid:
        raise ValueError('independent acceptance missing or stale')
    review = evidence(a, 'independent_review', False).read_text(errors='strict')
    if cid not in review or VERDICT not in review or 'NOT ACCEPTED' in review:
        raise ValueError('independent review does not accept this exact code identity')
    if env.get('dont_write_bytecode') is not True:
        raise ValueError('PYTHONDONTWRITEBYTECODE=1 required for a bound source tree')
    if (q.get('host') != socket.gethostname() or q.get('execution_mode') != 'direct' or q.get('code_identity') != cid
            or q.get('environment') != env):
        raise ValueError('host/source/environment qualification missing or stale')
    fixture = evidence(q, 'fixture_report')
    evidence(q, 'fixture_log', False)
    bench = evidence(q, 'benchmark_report')
    snap = evidence(q, 'snapshot_record')
    if (fixture.get('exit') != 0 or fixture.get('tests_passed') != EXPECTED_TESTS or fixture.get('tests_failed', 1) != 0
            or fixture.get('code_identity') != cid or fixture.get('environment') != env
            or fixture.get('log_sha256') != q['fixture_log_sha256']):
        raise ValueError('fixture qualification failed')
    if fixture.get('peak_rss_bytes', float('inf')) > BASE['budget']['rss_bytes']:
        raise ValueError('fixture RSS cap')
    if (snap.get('code_identity') != cid or snap.get('EXP_root') != str(ROOT) or snap.get('MS_root') != str(msroot())
            or snap.get('writable_source_files') != [] or snap.get('stale_bytecode') != []):
        raise ValueError('isolated snapshot identity absent or not immutable')
    for rel in s['EXP_files']:
        if (ROOT / rel).stat().st_mode & 0o222:
            raise ValueError('source file is writable: ' + rel)
    for rel in s['MS_files']:
        if (msroot() / rel).stat().st_mode & 0o222:
            raise ValueError('source file is writable: ' + rel)
    if not env['threadpools'] or any(p['num_threads'] != 1 for p in env['threadpools']):
        raise ValueError('actual BLAS thread count unqualified')
    import psutil
    if psutil.virtual_memory().total < 8 * 1024 ** 3:
        raise ValueError('hardware RAM below requirement')
    scratch = q.get('scratch_root')
    if not scratch or shutil.disk_usage(scratch).free < BASE['budget']['scratch_bytes']:
        raise ValueError('scratch qualification failed')
    if (bench.get('code_identity') != cid or bench.get('environment') != env
            or bench.get('scientific_endpoints_viewed') is not False
            or not 0 < bench.get('projected_lifecycle_seconds_with_margin', 0) <= BASE['budget']['campaign_seconds']
            or bench.get('worst_cell_within_half_cap') is not True
            or not 0 < bench.get('peak_rss_bytes', 0) <= BASE['budget']['rss_bytes']):
        raise ValueError('benchmark absent/stale/too slow')


def frozen_evidence_identities(f):
    """Recorded acceptance/qualification identities, checkable without the host (as T25)."""
    a, q, cid = f['acceptance'], f['qualification'], code_identity(f['sources'])
    if a.get('verdict') != VERDICT or a.get('deployment') != VERSION or a.get('code_identity') != cid:
        raise ValueError('frozen acceptance identity')
    if (q.get('code_identity') != cid or q.get('execution_mode') != 'direct' or q.get('environment') != f['environment']
            or not q.get('host')):
        raise ValueError('frozen qualification identity')
    for record, names in ((a, ('independent_review',)),
                          (q, ('fixture_report', 'fixture_log', 'benchmark_report', 'snapshot_record'))):
        for n in names:
            if not re.fullmatch('[0-9a-f]{64}', str(record.get(n + '_sha256', ''))):
                raise ValueError('frozen evidence hash absent: ' + n)


FREEZE_KEYS = {'deployment', 'status', 'config', 'config_identity', 'enumeration', 'population_references',
               'sources', 'environment', 'acceptance', 'qualification', 't25_guard'}


def validate_freeze(f, portable=False):
    if set(f) != FREEZE_KEYS:
        raise ValueError('freeze schema')
    c = f['config']
    if (f['deployment'] != VERSION or c not in (BASE, FIXTURE) or f['config_identity'] != identity(c)
            or f['enumeration'] != enumeration(c)):
        raise ValueError('freeze identity/config')
    if f['status'] != ('fixture_only' if c['development_only'] else 'confirmation'):
        raise ValueError('freeze status')
    if f['t25_guard'].get('accepted_t25_identity') != guard.ACCEPTED_T25_IDENTITY:
        raise ValueError('freeze lacks the accepted T25 identity')
    if portable:
        guard.assert_loaded_modules()
        if code_identity(source_files()) != code_identity(f['sources']):
            raise ValueError('portable audit must run from the frozen source bytes')
        if not c['development_only']:
            frozen_evidence_identities(f)
        return
    if not c['development_only']:
        if f['qualification'].get('host') != socket.gethostname():
            raise ValueError('host-scoped operation: run on the qualified host; use the portable audit elsewhere')
        production_gate(f['acceptance'], f['qualification'], f['sources'], f['environment'])
    verify_source(f)


def freeze(c, out, acceptance=None, qualification=None):
    thread_gate()
    out = external(out)
    guard_report = guard.report()
    guard.assert_loaded_modules()
    s, env = sources(), environment()
    a = read(acceptance) if acceptance else {}
    q = read(qualification) if qualification else {}
    if not c['development_only']:
        production_gate(a, q, s, env)
    f = {'deployment': VERSION, 'status': 'fixture_only' if c['development_only'] else 'confirmation',
         'config': c, 'config_identity': identity(c), 'enumeration': enumeration(c),
         'population_references': science.population_references(c), 'sources': s, 'environment': env,
         'acceptance': a, 'qualification': q, 't25_guard': guard_report}
    validate_freeze(f)
    write_new(out, f)
    return f


# --------------------------------------------------------------------------------------
# one bounded attempt
# --------------------------------------------------------------------------------------
NONCE_ENV = 'PAPER1_T25NS_WORKER_NONCE'


def cell_dir(root, i):
    return Path(root) / f'cell-{i:04d}'


def worker_cell(fpath, index, out):
    parent_watch()
    thread_gate()
    f = read(fpath)
    validate_freeze(f)
    row = f['enumeration']['cells'][index]
    root = Path(fpath).parent
    reservation = read(root / f'{index:04d}.START.json')
    nonce = os.environ.get(NONCE_ENV, '')
    if (not nonce or identity(nonce) != read(root / 'ATTEMPT.START.json')['worker_nonce_identity']
            or os.getppid() != read(root / 'WORKER.START.json')['pid'] or reservation['cell'] != row
            or Path(out).resolve() != cell_dir(root, index).resolve()):
        raise ValueError('worker lacks matching parent/reservation')
    if (root / f'{index:04d}.END.json').exists() or Path(out).exists():
        raise ValueError('duplicate worker claim')
    record, arrays = science.fit_cell(f['config'], row['seed'], row['rho'], row['theta'], row['N'])
    out = Path(out)
    out.mkdir(exist_ok=False)
    with (out / 'arrays.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write_new(out / 'record.json', record)
    verify_source(f)
    write_new(out / 'COMPLETE.json', {'arrays_sha': sha(out / 'arrays.npz'), 'record_sha': sha(out / 'record.json')})


def worker_campaign(root):
    parent_watch(group=True)
    thread_gate()
    root = Path(root)
    f = read(root / 'freeze.json')
    validate_freeze(f)
    c, b = f['config'], f['config']['budget']
    reservation = read(root / 'ATTEMPT.START.json')
    nonce = os.environ.get(NONCE_ENV, '')
    if not nonce or identity(nonce) != reservation['worker_nonce_identity'] or os.getppid() != reservation['supervisor_pid']:
        raise ValueError('campaign worker lacks parent reservation')
    write_new(root / 'WORKER.START.json', {'pid': os.getpid(), 'supervisor_pid': os.getppid(), 'nonce_identity': identity(nonce)})
    began = time.monotonic()
    for i, row in enumerate(f['enumeration']['cells']):
        verify_source(f)
        write_new(root / f'{i:04d}.START.json', {'cell': row, 'reserved_charges': CELL_CHARGES, 'timestamp': time.time()})
        p, start = None, time.monotonic()
        try:
            with (root / f'{i:04d}.log').open('x') as stream:
                p = subprocess.Popen([sys.executable, '-m', MODULE, '_cell', '--freeze', str(root / 'freeze.json'),
                                      '--index', str(i), '--out', str(cell_dir(root, i))],
                                     cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                while p.poll() is None:
                    if time.monotonic() - start > b['cell_seconds']:
                        p.kill()
                        p.wait()
                        raise ValueError('paired-cell wall cap')
                    time.sleep(.05)
                if p.returncode:
                    raise ValueError('paired-cell failed ' + str(i))
        finally:
            if p is not None and p.poll() is None:
                p.kill()
                p.wait()
        write_new(root / f'{i:04d}.END.json', {'status': 'COMPLETE', 'elapsed': time.monotonic() - start, 'charges': CELL_CHARGES})
    verify_source(f)
    write_new(root / 'FIT_COMPLETE.json', {'cells': len(f['enumeration']['cells']), 'elapsed': time.monotonic() - began})
    write_new(root / 'VERIFY.START.json', {'reserved_representation_replay_charges': f['enumeration']['representation_charges'], 'retries': 0})
    records = verify(root, require_complete=False, replay_optimizer=True)
    write_new(root / 'VERIFY.END.json', {'cells_replayed': len(records), 'all_restarts_replayed': True})
    write_new(root / 'REDUCE.START.json', {'cell_count': len(records)})
    write_new(root / 'reduction.json', {'development_only': c['development_only'], **science.reduce_records(c, records)})
    write_new(root / 'REDUCE.END.json', {'reduction_sha256': sha(root / 'reduction.json')})
    verify_source(f)
    write_new(root / 'WORKER_COMPLETE.json', {'development_only': c['development_only'], 'cells': len(records),
                                              'elapsed': time.monotonic() - began})


def execute(fpath, root):
    thread_gate()
    f = read(fpath)
    validate_freeze(f)
    root = external(root)
    root.mkdir(parents=True, exist_ok=False)
    nonce = secrets.token_hex(32)
    write_new(root / 'freeze.json', f)
    write_new(root / 'ATTEMPT.START.json', {'deployment': VERSION, 'confirmation': not f['config']['development_only'],
                                            'reserved_fit_charges': f['enumeration']['total_charges'],
                                            'reserved_replay_charges': f['enumeration']['total_charges'], 'retries': 0,
                                            'timestamp': time.time(), 'supervisor_pid': os.getpid(),
                                            'worker_nonce_identity': identity(nonce)})
    b = f['config']['budget']
    budget = {'seconds': b['campaign_seconds'], 'rss_bytes': b['rss_bytes'], 'scratch_bytes': b['scratch_bytes']}
    start = time.monotonic()
    try:
        report = supervise([sys.executable, '-m', MODULE, '_campaign', '--root', str(root)], root / 'campaign.log',
                           root, budget, extra_env={NONCE_ENV: nonce})
        if not (root / 'WORKER_COMPLETE.json').is_file():
            raise ValueError('missing worker completion')
        verify_source(f)
        write_new(root / 'COMPLETE.json', {'development_only': f['config']['development_only'],
                                           'cells': f['enumeration']['paired_cells'], 'resources': report,
                                           'deployment': VERSION})
        write_new(root / 'ATTEMPT.END.json', {'elapsed_seconds': time.monotonic() - start, 'status': 'COMPLETE'})
    except BaseException as e:
        write_new(root / 'FAILED.json', {'error': str(e), 'elapsed_seconds': time.monotonic() - start, 'retry_allowed': False})
        raise


# --------------------------------------------------------------------------------------
# verification by recomputation (host replay or portable saved-map audit)
# --------------------------------------------------------------------------------------
from experiments.mlst2026.t25_production_20260930.reconstruction import assert_close, check_optimizer_record  # noqa: E402


def verify(root, require_complete=True, replay_optimizer=True, notes=None):
    """Recompute every record from the saved maps and factors and the regenerated data.
    replay_optimizer=True (qualified host only) also re-runs every structured restart and
    requires bitwise-identical training moments; the portable audit skips both."""
    root = Path(root)
    if (root / 'FAILED.json').exists() or (require_complete and not (root / 'COMPLETE.json').is_file()):
        raise ValueError('failed/incomplete attempt')
    f = read(root / 'freeze.json')
    validate_freeze(f, portable=not replay_optimizer)
    c = f['config']
    check_population_references(f['population_references'], c, host=replay_optimizer, notes=notes)
    cells = f['enumeration']['cells']
    if {p.name for p in root.glob('cell-*')} != {cell_dir(root, i).name for i in range(len(cells))}:
        raise ValueError('incomplete cell inventory')
    records = []
    for i, cell in enumerate(cells):
        if not (root / f'{i:04d}.START.json').is_file() or not (root / f'{i:04d}.END.json').is_file():
            raise ValueError('interrupted attempt')
        assert_close(read(root / f'{i:04d}.START.json')['cell'], cell)
        dest = cell_dir(root, i)
        if read(dest / 'COMPLETE.json') != {'arrays_sha': sha(dest / 'arrays.npz'), 'record_sha': sha(dest / 'record.json')}:
            raise ValueError('tampered record/arrays')
        records.append(verify_cell(c, cell, read(dest / 'record.json'),
                                   dict(np.load(dest / 'arrays.npz', allow_pickle=False)), replay_optimizer, notes))
    return records


#: Portable audit: restart objectives recomputed from saved factors may reorder near-ties under another
#: BLAS; the saved selection is accepted if its objective is within this relative distance of the maximum.
#: The host replay keeps the exact first-maximum rule.
SELECTION_RTOL = 1e-12

#: Portable audit: the population restricted fits tie across their four starts to ~1e-14 relative in
#: objective while their clean risks differ by up to ~1e-6 (round-2 review), so another BLAS may select
#: another tied start.  Off the host the restricted population entries are compared to this absolute
#: tolerance (risks ~150, penalties ~1) and the largest difference is recorded; the host keeps the exact
#: comparison.
POPULATION_RESTRICTED_ATOL = 1e-5

#: Calibration-pool digests inside the covariance diagnostics (T25 estimate()).
CALIBRATION_DIGEST_KEYS = ('prefix_sha', 'mean_sha')


def _split_restricted(ref):
    ref = copy.deepcopy(ref)
    restricted = []
    for row in ref['population_penalties']:
        item = {'rho': row['rho'], 'theta': row['theta'], 'oracle': row['oracle_risk'].pop('restricted'),
                'status': row.pop('restricted_fit_status')}
        for k in sorted(k for k in row if k.startswith('penalty_')):
            item[k] = row[k].pop('restricted')
        restricted.append(item)
    return ref, restricted


def check_population_references(saved, c, host, notes=None):
    """Recompute the population references.  On the host: exact (assert_close).  Off the host: everything
    except the restricted population entries as on the host; those within POPULATION_RESTRICTED_ATOL, with
    the largest difference and any changed fit status recorded in the notes."""
    now = science.population_references(c)
    if host:
        assert_close(saved, now)
        return
    a, ra = _split_restricted(saved)
    b, rb = _split_restricted(now)
    assert_close(a, b)
    if [(x['rho'], x['theta'], sorted(x)) for x in ra] != [(y['rho'], y['theta'], sorted(y)) for y in rb]:
        raise ValueError('population reference layout differs')
    worst, statuses = 0.0, []
    for x, y in zip(ra, rb):
        for k in x:
            if k in ('rho', 'theta', 'status'):
                continue
            d = abs(float(x[k]) - float(y[k]))
            if not np.isfinite(d) or d > POPULATION_RESTRICTED_ATOL:
                raise ValueError('population restricted reference differs beyond tolerance: ' + k)
            worst = max(worst, d)
        if x['status'] != y['status']:
            statuses.append({'rho': x['rho'], 'theta': x['theta'], 'saved': x['status'], 'recomputed': y['status']})
    if notes is not None:
        notes['population_restricted_max_abs_difference'] = worst
        notes['population_restricted_status_changes'] = statuses


def verify_cell(c, cell, r, a, replay_optimizer, notes=None):
    seed, rho, th, N = cell['seed'], cell['rho'], cell['theta'], cell['N']
    g, sigma, L, Li, pool, training = science.make_data(c, seed, rho, th)
    x, mu, S = science.moments(training, N)
    K = g.K(th)
    Cs, W0, truth = L @ K @ L, Li @ Li, science.top(K, g.k)
    digests_match = (r['training_sha'], r['calibration_sha'], r['sigma_sha']) == (science.digest(x), science.digest(pool), science.digest(sigma))
    if replay_optimizer:
        if not np.array_equal(a['S'], S):
            raise ValueError('host replay requires bitwise-identical training moments')
        if not digests_match:
            raise ValueError('seed/data identities')
    elif not digests_match and notes is not None:
        notes.setdefault('data_digest_mismatches', []).append(dict(cell))
    if not np.allclose(a['S'], S, rtol=1e-10, atol=1e-10) or not np.allclose(a['mean'], mu, rtol=1e-10, atol=1e-10):
        raise ValueError('data/moment mismatch')
    if (r['seed'], r['rho'], r['theta'], r['N']) != (seed, rho, th, N):
        raise ValueError('cell identity')
    estimated, cd = science.covariance_estimates(c, pool, g)
    if replay_optimizer:
        assert_close(r['covariance'], cd)
    else:                                   # digests of the regenerated pool may differ under another BLAS
        saved_cd = {m: {k: v for k, v in d.items() if k not in CALIBRATION_DIGEST_KEYS} for m, d in r['covariance'].items()}
        now_cd = {m: {k: v for k, v in d.items() if k not in CALIBRATION_DIGEST_KEYS} for m, d in cd.items()}
        assert_close(saved_cd, now_cd)
        if any(r['covariance'][m].get(k) != cd[m].get(k) for m in cd for k in CALIBRATION_DIGEST_KEYS) and notes is not None:
            notes.setdefault('calibration_digest_mismatches', []).append(dict(cell))
    covs = {'oracle': sigma, **estimated}
    methods, expected = {}, {'S', 'mean'}
    initial = science.starts(S, g.C, g.T, g.k_c, g.k_t, (c['namespace'], seed, th, N))
    for name in science.variants(c):
        cov = covs[name]
        if name != 'oracle':
            expected.add(name + '_cov')
            if not np.allclose(a[name + '_cov'], cov, rtol=1e-10, atol=1e-12):
                raise ValueError('covariance differs ' + name)
        W = np.linalg.inv(cov)
        for cls in science.CLASSES:
            tag = name + '__' + cls
            expected.add(tag + '_B')
            B = a[tag + '_B']
            if np.linalg.matrix_rank(B) != g.k:
                raise ValueError('rank mismatch')
            A = science.map_for(B, W)
            b = (np.eye(g.d) - A) @ mu
            row = science.risks(A, b, Cs, sigma, W0) | science.representation(B, Li, truth)
            row['training_loss'] = float(np.trace(W @ S) - np.trace(np.linalg.solve(B.T @ W @ B, B.T @ W @ S @ W @ B)))
            if cls == 'restricted':
                opt = r['methods'][tag]['optimization']
                if len(opt['restarts']) != 4 or len(initial) != 4:
                    raise ValueError('restart inventory')
                if replay_optimizer:
                    rB, rOpt, rFac = science.structured(W, S, initial, c['max_iter'], c['gradient_tolerance'])
                    assert_close(opt, rOpt)
                    if not np.allclose(B, rB, atol=1e-9):
                        raise ValueError('optimizer replay mismatch')
                rr, objectives = [], []
                for j in range(4):
                    uc, ut = a[tag + f'_uc_{j}'], a[tag + f'_ut_{j}']
                    expected.update([tag + f'_uc_{j}', tag + f'_ut_{j}'])
                    if replay_optimizer and (not np.allclose(uc, rFac[j][0], rtol=1e-8, atol=1e-9)
                                             or not np.allclose(ut, rFac[j][1], rtol=1e-8, atol=1e-9)):
                        raise ValueError('restart factor replay mismatch')
                    if not np.allclose(uc.T @ uc, np.eye(g.k_c)) or not np.allclose(ut.T @ ut, np.eye(g.k_t)):
                        raise ValueError('factor orthogonality')
                    val, gc, gt = science.objective_gradient(uc, ut, W, S)
                    objectives.append(val)
                    scale = np.trace(W @ S)
                    gn = float(np.sqrt(np.sum(((gc - uc @ (uc.T @ gc)) / scale) ** 2) + np.sum(((gt - ut @ (ut.T @ gt)) / scale) ** 2)))
                    assert_close(opt['restarts'][j]['objective'], val)
                    assert_close(opt['restarts'][j]['gradient_norm'], gn)
                    traj = opt['restarts'][j]['trajectory']
                    if not np.isfinite(traj).all() or np.min(np.diff(traj)) < -1e-8:
                        raise ValueError('invalid optimization trajectory')
                    check_optimizer_record(opt['restarts'][j], val, gn, c)
                    ar = science.map_for(np.kron(ut, uc), W)
                    rr.append(science.risks(ar, (np.eye(g.d) - ar) @ mu, Cs, sigma, W0)['clean'])
                sel, saved = int(np.argmax(objectives)), opt['selected']
                if replay_optimizer:
                    ok = saved == sel
                else:
                    ok = (type(saved) is int and 0 <= saved < 4
                          and objectives[saved] >= max(objectives) - SELECTION_RTOL * abs(max(objectives)))
                if not ok or not np.allclose(B, np.kron(a[tag + f'_ut_{saved}'], a[tag + f'_uc_{saved}'])):
                    raise ValueError('class/selection mismatch')
                row.update(optimization=opt, restart_clean_risks=rr, restart_risk_spread=float(max(rr) - min(rr)))
            else:
                ref = science.unrestricted(W, S, g.k)
                if not np.allclose(science.qr(B) @ science.qr(B).T, ref @ ref.T, atol=1e-8):
                    raise ValueError('unrestricted fit mismatch')
            methods[tag] = row
        if name != 'oracle':
            methods[name + '__domain'] = science.response(Li @ S @ Li, Li @ cov @ Li - np.eye(g.d), g.k)[0]
    if set(a) != expected:
        raise ValueError('unexpected/missing arrays')
    assert_close(r['methods'], methods)
    return r


# --------------------------------------------------------------------------------------
# ledgers, export, portable audit
# --------------------------------------------------------------------------------------
RUN_FIXED = ('freeze.json', 'campaign.log', 'ATTEMPT.START.json', 'WORKER.START.json', 'FIT_COMPLETE.json',
             'VERIFY.START.json', 'VERIFY.END.json', 'REDUCE.START.json', 'REDUCE.END.json',
             'WORKER_COMPLETE.json', 'COMPLETE.json', 'ATTEMPT.END.json', 'reduction.json')


def check_ledgers(root, f):
    """Exact receipts, charges, cell inventory and finite resource ledgers."""
    root = Path(root)
    b, e, dev = f['config']['budget'], f['enumeration'], f['config']['development_only']
    if (root / 'FAILED.json').exists():
        raise ValueError('failed attempt')
    for n in RUN_FIXED:
        if not (root / n).is_file():
            raise ValueError('attempt not complete: ' + n)
    start, end = read(root / 'ATTEMPT.START.json'), read(root / 'ATTEMPT.END.json')
    worker, fit, complete = read(root / 'WORKER_COMPLETE.json'), read(root / 'FIT_COMPLETE.json'), read(root / 'COMPLETE.json')
    if (start.get('deployment') != VERSION or start.get('retries') != 0 or start.get('confirmation') != (not dev)
            or start.get('reserved_fit_charges') != e['total_charges']):
        raise ValueError('attempt reservation invalid')
    if end.get('status') != 'COMPLETE' or not finite_between(end.get('elapsed_seconds'), 0, b['campaign_seconds']):
        raise ValueError('attempt completion invalid')
    if worker.get('cells') != e['paired_cells'] or fit.get('cells') != e['paired_cells'] or complete.get('cells') != e['paired_cells']:
        raise ValueError('completion census')
    if not finite_between(fit.get('elapsed'), 0, worker.get('elapsed', -1)) or not finite_between(worker['elapsed'], 0, end['elapsed_seconds']):
        raise ValueError('inconsistent elapsed ledger')
    r = complete['resources']
    if (r.get('exit') != 0 or not finite_between(r.get('peak_tree_rss_bytes'), 0, b['rss_bytes'])
            or not finite_between(r.get('scratch_bytes'), 0, b['scratch_bytes'])
            or not finite_between(r.get('elapsed_seconds'), 0, b['campaign_seconds'])):
        raise ValueError('recorded resource failure')
    if read(root / 'VERIFY.END.json') != {'cells_replayed': e['paired_cells'], 'all_restarts_replayed': True}:
        raise ValueError('replay receipt')
    if read(root / 'REDUCE.END.json') != {'reduction_sha256': sha(root / 'reduction.json')}:
        raise ValueError('reduction receipt mismatch')
    n = e['paired_cells']
    allowed = set(RUN_FIXED) | {f'{i:04d}.{s}' for i in range(n) for s in ('START.json', 'END.json', 'log')} \
        | {cell_dir(root, i).name for i in range(n)} | {'EXPORT.json', 'SHA256SUMS', 'tables'}
    unknown = sorted({p.name for p in root.iterdir()} - allowed)
    if unknown:
        raise ValueError('unexpected entries in run tree: ' + ', '.join(unknown[:5]))
    starts = {p.name for p in root.glob('[0-9][0-9][0-9][0-9].START.json')}
    ends = {p.name for p in root.glob('[0-9][0-9][0-9][0-9].END.json')}
    if starts != {f'{i:04d}.START.json' for i in range(n)} or ends != {f'{i:04d}.END.json' for i in range(n)}:
        raise ValueError('cell ledger inventory (interrupted or incomplete attempt)')
    for i, cell in enumerate(e['cells']):
        begin, stop = read(root / f'{i:04d}.START.json'), read(root / f'{i:04d}.END.json')
        if begin.get('cell') != cell or begin.get('reserved_charges') != CELL_CHARGES:
            raise ValueError('cell reservation mismatch')
        if stop.get('status') != 'COMPLETE' or stop.get('charges') != CELL_CHARGES or not finite_between(stop.get('elapsed'), 0, b['cell_seconds']):
            raise ValueError('cell ledger/resource mismatch')
        if {p.name for p in cell_dir(root, i).iterdir()} != {'arrays.npz', 'record.json', 'COMPLETE.json'}:
            raise ValueError('cell directory inventory')


def write_tables(root, reduction):
    """E1/E2 tables and the full contrast table as CSV beside reduction.json."""
    tdir = Path(root) / 'tables'
    tdir.mkdir()
    rows = reduction['contrasts']
    fields = ['rho', 'theta', 'N', 'contrast', 'mean', 'ci_low', 'ci_high', 'seeds', 'zero_standard_error',
              'optimization_limited', 'optimization_limited_seeds']

    def dump(name, selected):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in selected:
            w.writerow({**{k: r[k] for k in fields if k in r}, 'ci_low': repr(r['ci'][0]), 'ci_high': repr(r['ci'][1]),
                        'mean': repr(r['mean'])})
        with (tdir / name).open('x') as fh:
            fh.write(buf.getvalue())
    dump('contrasts.csv', rows)
    dump('E1_kronecker_penalty.csv', [r for r in rows if r['contrast'].startswith(('metric/kron_', 'metric_calibration/kron/',
                                                                                    'rho_change/', 'floor_extrapolated/'))])
    dump('E2_full_minus_kronecker.csv', [r for r in rows if r['contrast'].startswith('estimator/')])
    dump('E3_interaction.csv', [r for r in rows if r['contrast'].startswith('interaction/')])
    write_new(tdir / 'predictions.json', reduction['predictions'])


def sha256sums(root):
    root = Path(root)
    lines = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('symlink in export')
        if p.is_file() and p.name != 'SHA256SUMS':
            lines.append(f'{sha(p)}  {p.relative_to(root).as_posix()}')
    return '\n'.join(lines) + '\n'


def export(root, out, allow_development=False):
    """Copy the complete, replay-verified run, add tables, EXPORT.json and SHA256SUMS.
    allow_development exists for the package tests only (fixture runs); a production export
    is a confirmation export."""
    thread_gate()
    root, out = external(root), external(out)
    f = read(root / 'freeze.json')
    validate_freeze(f)
    dev = f['config']['development_only']
    if dev and not allow_development:
        raise ValueError('fixture scientific export forbidden')
    if out.exists() or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('export requires a separate fresh output')
    check_ledgers(root, f)
    preflight_space(out.parent if out.parent.exists() else root.parent, tree_bytes(root) + 1024 ** 3)
    before = inventory(root)
    shutil.copytree(root, out)
    if inventory(root) != before or inventory(out) != before:
        raise ValueError('run changed while exporting')
    write_tables(out, read(out / 'reduction.json'))
    write_new(out / 'EXPORT.json', {'deployment': VERSION, 'status': 'DEVELOPMENT_EXPORT' if dev else 'CONFIRMATION_EXPORT',
                                    'source_run': str(root), 'code_identity': code_identity(f['sources']),
                                    'files_before_tables': before, 'reduction_sha256': sha(out / 'reduction.json')})
    with (out / 'SHA256SUMS').open('x') as fh:
        fh.write(sha256sums(out))


def check_sha256sums(root):
    root = Path(root)
    if (root / 'SHA256SUMS').read_text() != sha256sums(root):
        raise ValueError('SHA256SUMS mismatch')


def check_export(root, f):
    """A confirmation export must carry SHA256SUMS, EXPORT.json and tables/, all consistent: checksums,
    the export record (deployment, status, frozen code identity, reduction hash, pre-table inventory),
    and tables regenerated byte for byte from the verified reduction."""
    root = Path(root)
    parts = [(root / n).exists() for n in ('SHA256SUMS', 'EXPORT.json', 'tables')]
    if not any(parts):
        if not f['config']['development_only']:
            raise ValueError('confirmation export lacks SHA256SUMS/EXPORT.json/tables')
        return False
    if not all(parts):
        raise ValueError('incomplete export: SHA256SUMS, EXPORT.json and tables are all required')
    check_sha256sums(root)
    e = read(root / 'EXPORT.json')
    want_status = 'DEVELOPMENT_EXPORT' if f['config']['development_only'] else 'CONFIRMATION_EXPORT'
    if (set(e) != {'deployment', 'status', 'source_run', 'code_identity', 'files_before_tables', 'reduction_sha256'}
            or e['deployment'] != VERSION or e['status'] != want_status or e['code_identity'] != code_identity(f['sources'])
            or e['reduction_sha256'] != sha(root / 'reduction.json')):
        raise ValueError('export record invalid')
    inv = inventory(root)
    for name in [k for k in inv if k == 'EXPORT.json' or k == 'SHA256SUMS' or k.startswith('tables/')]:
        inv.pop(name)
    if inv != e['files_before_tables']:
        raise ValueError('export inventory differs from the recorded run inventory')
    with tempfile.TemporaryDirectory() as tmp:
        write_tables(tmp, read(root / 'reduction.json'))
        want = {str(p.relative_to(tmp)): p.read_bytes() for p in Path(tmp, 'tables').rglob('*') if p.is_file()}
        got = {str(p.relative_to(root)): p.read_bytes() for p in (root / 'tables').rglob('*') if p.is_file()}
        if want != got:
            raise ValueError('exported tables differ from the verified reduction')
    return True


def audit(root, out):
    """Portable receiver audit (any machine with the frozen source bytes): export record, checksums and
    tables (confirmation exports), ledgers, all records recomputed from saved maps/factors and regenerated
    data, and the full reduction.  The cell optimizer is not replayed here (that was the host's job); the
    population references are recomputed (their restricted entries to POPULATION_RESTRICTED_ATOL); data and
    calibration digest mismatches, possible across BLAS builds, are reported rather than ignored."""
    root, out = Path(root).resolve(), external(out)
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('audit output overlaps input')
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'START.json', {'operation': 'portable_audit', 'source_run': str(root), 'retries': 0})
    try:
        f = read(root / 'freeze.json')
        exported = check_export(root, f)
        check_ledgers(root, f)
        notes = {}
        records = verify(root, require_complete=True, replay_optimizer=False, notes=notes)
        want = {'development_only': f['config']['development_only'], **science.reduce_records(f['config'], records)}
        assert_close(read(root / 'reduction.json'), want)
        result = {'verified': True, 'mode': 'portable_saved_map_audit', 'optimizer_replayed': False,
                  'export_checked': exported, 'cells': len(records), 'source_identity': code_identity(f['sources']),
                  'data_digest_mismatches': notes.get('data_digest_mismatches', []),
                  'calibration_digest_mismatches': notes.get('calibration_digest_mismatches', []),
                  'population_restricted_max_abs_difference': notes.get('population_restricted_max_abs_difference'),
                  'population_restricted_status_changes': notes.get('population_restricted_status_changes', []),
                  'predictions': {k: want['predictions'][k].get('verdict') for k in ('P1', 'P2')}}
        write_new(out / 'audit.json', result)
        return result
    except BaseException as e:
        write_new(out / 'FAILED.json', {'error': str(e), 'retry_allowed': False})
        raise


# --------------------------------------------------------------------------------------
# host qualification
# --------------------------------------------------------------------------------------
def qualify_fixtures(out):
    """Run the package's complete test suite in a child with the frozen interpreter."""
    thread_gate()
    out = external(out)
    out.mkdir(parents=True, exist_ok=False)
    s, env, log = sources(), environment(), out / 'fixture.log'
    with log.open('x') as stream:
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp', str(out / 'pytest'),
                            str(PACKAGE / 'tests')], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                           env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    text = log.read_text()
    passed, failed = re.search(r'(\d+) passed', text), re.search(r'(\d+) (failed|error)', text)
    report = {'exit': r.returncode, 'tests_passed': int(passed.group(1)) if passed else 0,
              'tests_failed': int(failed.group(1)) if failed else 0, 'code_identity': code_identity(s),
              'environment': env, 'log_sha256': sha(log), 'peak_rss_bytes': peak_children_rss(),
              'host': socket.gethostname()}
    write_new(out / 'fixture_report.json', report)
    if r.returncode != 0 or report['tests_failed']:
        raise ValueError('fixture qualification failed; see ' + str(log))
    return report


def benchmark(out, cells=None):
    """Production-shaped timing in the X2 benchmark namespace (development-only, disjoint
    seeds); fitted outputs are discarded unviewed.  Projects fit + in-campaign replay with
    margin 2 (export is a copy and the audit skips the optimizer)."""
    import psutil
    thread_gate()
    out = external(out)
    out.mkdir(parents=True, exist_ok=False)
    c = BENCHMARK
    if set(c['seeds']) & (set(BASE['seeds']) | set(FIXTURE['seeds'])):
        raise ValueError('benchmark seed overlaps a registered namespace')
    rows, proc = [], psutil.Process()
    for rho, theta, N in (cells or BENCHMARK_CELLS):
        t0 = time.monotonic()
        record, arrays = science.fit_cell(c, c['seeds'][0], rho, theta, N)
        fit = time.monotonic() - t0
        statuses = [x['status'] for k, v in record['methods'].items() if k.endswith('__restricted')
                    for x in v['optimization']['restarts']]
        iters = [len(x['accepted_steps']) for k, v in record['methods'].items() if k.endswith('__restricted')
                 for x in v['optimization']['restarts']]
        del record, arrays
        rows.append({'rho': rho, 'theta': theta, 'N': N, 'fit_seconds': fit, 'rss_bytes': proc.memory_info().rss,
                     'restart_statuses': {k: statuses.count(k) for k in sorted(set(statuses))},
                     'max_restart_iterations': max(iters)})
    e = enumeration(BASE)
    mean = float(np.mean([r['fit_seconds'] for r in rows]))
    worst = float(max(r['fit_seconds'] for r in rows))
    projected = mean * e['paired_cells'] * 2 * 2
    report = {'development_only': True, 'scientific_endpoints_viewed': False, 'namespace': c['namespace'],
              'note': 'confirmation endpoints never computed; benchmark-namespace fits (risks included) are computed '
                      'in memory and discarded unviewed',
              'code_identity': code_identity(sources()), 'environment': environment(), 'host': socket.gethostname(),
              'cells': rows, 'mean_cell_fit_seconds': mean, 'worst_cell_fit_seconds': worst,
              'projected_fit_seconds': mean * e['paired_cells'],
              'projected_lifecycle_seconds_with_margin': projected,
              'margin': 'x2 over fit + in-campaign optimizer replay', 'cell_seconds_cap': BASE['budget']['cell_seconds'],
              'worst_cell_within_half_cap': worst * 2 <= BASE['budget']['cell_seconds'],
              'peak_rss_bytes': max(r['rss_bytes'] for r in rows)}
    write_new(out / 'benchmark_report.json', report)
    if (not report['worst_cell_within_half_cap'] or projected > BASE['budget']['campaign_seconds']
            or report['peak_rss_bytes'] > BASE['budget']['rss_bytes']):
        raise ValueError('benchmark does not fit registered caps with margin')
    return report


def snapshot_record(out):
    out = external(out)
    s = sources()
    writable = [r for r in s['EXP_files'] if (ROOT / r).stat().st_mode & 0o222] + \
               [r for r in s['MS_files'] if (msroot() / r).stat().st_mode & 0o222]
    stale = [str(p.relative_to(ROOT)) for p in (ROOT / 'experiments').rglob('*.pyc')] + \
            [str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.pyc')]
    record = {'code_identity': code_identity(s), 'EXP_root': str(ROOT), 'MS_root': str(msroot()),
              'EXP_head': s['EXP_head'], 'MS_head': s['MS_head'], 'EXP_status': s['EXP_status'],
              'MS_status': s['MS_status'], 'writable_source_files': writable, 'stale_bytecode': stale,
              'host': socket.gethostname()}
    write_new(out, record)
    if writable or stale:
        raise ValueError('snapshot not immutable/clean: make sources read-only and remove bytecode first')
    return record


# --------------------------------------------------------------------------------------
# the two modes
# --------------------------------------------------------------------------------------
def fixture_mode(work):
    """Complete development lifecycle in the fixture namespace: freeze, supervised run
    with in-campaign optimizer replay and reduction, then an independent portable audit."""
    work = external(work)
    work.mkdir(parents=True, exist_ok=False)
    freeze(FIXTURE, work / 'freeze.json')
    execute(work / 'freeze.json', work / 'run')
    result = audit(work / 'run', work / 'audit')
    return {'fixture': True, 'audit': result, 'reduction': str(work / 'run' / 'reduction.json')}


def production_mode(work, acceptance, qualification):
    """The single confirmation attempt: freeze (production gate), run, export, audit."""
    work = external(work)
    work.mkdir(parents=True, exist_ok=False)
    freeze(BASE, work / 'freeze.json', acceptance, qualification)
    execute(work / 'freeze.json', work / 'run')
    export(work / 'run', work / 'return')
    return audit(work / 'return', work / 'return_audit')
