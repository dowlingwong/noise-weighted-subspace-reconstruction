"""X3 deployment: one bounded attempt, in-campaign verification, reduction, export.

The same process design as X2 (t25ns_20261005/workflow.py) and the accepted T25
deployment, whose generic helpers are imported read-only after guard.py checks their
bytes.  In-campaign verification re-evaluates every saved model from regenerated data in
a fresh process and replays every training of the first seed's cells (registered replay
subset); the export copies the verified run and writes SHA256SUMS; a portable audit
re-evaluates everything again from the export.  Runs on Deepthought or on a Mac with
the pinned torch; the qualified host is recorded and enforced.
"""
from __future__ import annotations

from . import guard

guard.assert_direct()

import csv                                                            # noqa: E402
import io                                                             # noqa: E402
import json                                                           # noqa: E402
import os                                                             # noqa: E402
import re                                                             # noqa: E402
import secrets                                                        # noqa: E402
import shutil                                                         # noqa: E402
import socket                                                         # noqa: E402
import subprocess                                                     # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402
import torch                                                         # noqa: E402,F401  (imported before any threadpool census)
from experiments.mlst2026.t25_production_20260930 import workflow as t25w  # noqa: E402
from experiments.mlst2026.t25_production_20260930.workflow import (   # noqa: E402,F401
    sha, identity, read, write_new, tree_bytes, kill_group, parent_watch, supervise, inventory,
    finite_between, thread_gate, preflight_space, _peak_children_rss as peak_children_rss)
from .protocol import BASE, FIXTURE, BENCHMARK, DEEP, CELL_CHARGES, config, enumeration  # noqa: E402
from . import science                                                 # noqa: E402

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
MODULE = 'experiments.mlst2026.x3_nonlinear_metric_20261005'
VERSION = 'x3-deployment/20261005-v1'
PROTOCOL_MS = 'plan/X3_NONLINEAR_METRIC_PROTOCOL_2026-10-05.md'
VERDICT = 'ACCEPTED_X3_IMPLEMENTATION'
MIN_TESTS = 15


def msroot():
    return t25w.msroot()


def external(p):
    return t25w.external(p)


def environment():
    env = t25w.environment()           # torch is already imported (module top), so its pools are always listed
    env['package_origin'] = str(PACKAGE)
    env['torch'] = torch.__version__
    env['torch_config'] = torch.__config__.show()
    return env


def git(root, args):
    return subprocess.check_output(['git', '-c', 'core.fsmonitor=false'] + args, cwd=root, text=True,
                                   env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).strip()


def package_files():
    return sorted(p for p in PACKAGE.rglob('*') if p.is_file() and p.suffix in ('.py', '.json')
                  and '__pycache__' not in p.parts)


def source_files():
    """X3 code identity closure: this package (code, configs, tests, the accepted T25
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
    if q.get('host') != socket.gethostname() or q.get('code_identity') != cid or q.get('environment') != env:
        raise ValueError('host/source/environment qualification missing or stale')
    fixture = evidence(q, 'fixture_report')
    evidence(q, 'fixture_log', False)
    bench = evidence(q, 'benchmark_report')
    snap = evidence(q, 'snapshot_record')
    if (fixture.get('exit') != 0 or fixture.get('tests_passed', 0) < MIN_TESTS or fixture.get('tests_failed', 1) != 0
            or fixture.get('code_identity') != cid or fixture.get('environment') != env
            or fixture.get('log_sha256') != q['fixture_log_sha256']):
        raise ValueError('fixture qualification failed')
    if snap.get('code_identity') != cid or snap.get('writable_source_files') != [] or snap.get('stale_bytecode') != []:
        raise ValueError('isolated snapshot identity absent or not immutable')
    for rel in s['EXP_files']:
        if (ROOT / rel).stat().st_mode & 0o222:
            raise ValueError('source file is writable: ' + rel)
    for rel in s['MS_files']:
        if (msroot() / rel).stat().st_mode & 0o222:
            raise ValueError('source file is writable: ' + rel)
    if not env['threadpools'] or any(p['num_threads'] != 1 for p in env['threadpools']):
        raise ValueError('actual BLAS thread count unqualified')
    scratch = q.get('scratch_root')
    if not scratch or shutil.disk_usage(scratch).free < BASE['budget']['scratch_bytes']:
        raise ValueError('scratch qualification failed')
    if (bench.get('code_identity') != cid or bench.get('environment') != env
            or bench.get('scientific_endpoints_evaluated') is not False
            or not 0 < bench.get('projected_lifecycle_seconds_with_margin', 0) <= BASE['budget']['campaign_seconds']
            or bench.get('worst_cell_within_half_cap') is not True
            or not 0 < bench.get('peak_rss_bytes', 0) <= BASE['budget']['rss_bytes']):
        raise ValueError('benchmark absent/stale/too slow')


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
            a = f['acceptance']
            if a.get('verdict') != VERDICT or a.get('code_identity') != code_identity(f['sources']):
                raise ValueError('frozen acceptance identity')
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
    record, arrays = science.fit_cell(f['config'], row['seed'], row['N'])
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
    write_new(root / 'VERIFY.START.json', {'reserved_reevaluations': f['enumeration']['evaluated_models'], 'retries': 0})
    records = verify(root, require_complete=False, replay=True)
    write_new(root / 'VERIFY.END.json', {'cells_reevaluated': len(records), 'training_replay_cells': len(replay_cells(c))})
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
# verification by re-evaluation (and training replay on the registered subset)
# --------------------------------------------------------------------------------------
from experiments.mlst2026.t25_production_20260930.reconstruction import assert_close  # noqa: E402


def replay_cells(c):
    """Registered training-replay subset: every cell of the first seed."""
    return [{'seed': c['seeds'][0], 'N': n} for n in c['N']]


def verify(root, require_complete=True, replay=True):
    """Re-evaluate every saved model from regenerated data; on the qualified host (replay=True)
    also require identical data digests and replay every training of the registered subset."""
    root = Path(root)
    if (root / 'FAILED.json').exists() or (require_complete and not (root / 'COMPLETE.json').is_file()):
        raise ValueError('failed/incomplete attempt')
    f = read(root / 'freeze.json')
    validate_freeze(f, portable=not replay)
    c = f['config']
    assert_close(f['population_references'], science.population_references(c))
    cells = f['enumeration']['cells']
    if {p.name for p in root.glob('cell-*')} != {cell_dir(root, i).name for i in range(len(cells))}:
        raise ValueError('incomplete cell inventory')
    torch = science.import_torch(c)
    subset = replay_cells(c)
    records = []
    for i, cell in enumerate(cells):
        if not (root / f'{i:04d}.START.json').is_file() or not (root / f'{i:04d}.END.json').is_file():
            raise ValueError('interrupted attempt')
        assert_close(read(root / f'{i:04d}.START.json')['cell'], cell)
        dest = cell_dir(root, i)
        if read(dest / 'COMPLETE.json') != {'arrays_sha': sha(dest / 'arrays.npz'), 'record_sha': sha(dest / 'record.json')}:
            raise ValueError('tampered record/arrays')
        records.append(verify_cell(c, torch, cell, read(dest / 'record.json'),
                                   dict(np.load(dest / 'arrays.npz', allow_pickle=False)),
                                   replay_training=replay and cell in subset, host=replay))
    return records


def check_training_record(c, run):
    rec = c['recipe']
    keys = {'best_validation_loss', 'best_epoch', 'epochs_run', 'stopped', 'final_train_loss_at_best', 'validation_history'}
    if set(run) != keys:
        raise ValueError('training record schema')
    h, best, e = run['validation_history'], run['best_epoch'], run['epochs_run']
    if len(h) != e or not 1 <= e <= rec['max_epochs'] or not np.isfinite(h).all():
        raise ValueError('training history length/finite')
    if int(np.argmin(h)) != best or h[best] != run['best_validation_loss']:
        raise ValueError('best epoch is not the first validation minimum')
    if run['stopped'] == 'patience':
        ok = e == best + rec['patience'] + 1
    elif run['stopped'] == 'max_epochs':
        ok = e == rec['max_epochs'] and e - 1 - best < rec['patience']
    else:
        ok = False
    if not ok:
        raise ValueError('early-stopping record inconsistent with the registered rule')


def verify_cell(c, torch, cell, r, a, replay_training, host):
    seed, N = cell['seed'], cell['N']
    g, sigma, L, Li, train, eclean, enoisy = science.make_data(c, seed)
    x, mu, S = science.moments(train, N)
    if host and (r['training_sha'], r['evaluation_sha']) != (science.digest(x), science.digest(enoisy)):
        raise ValueError('seed/data identities')
    n_tr, n_val = science.split(N, c['recipe']['validation_fraction'])
    if (r['seed'], r['N'], r['n_train'], r['n_validation']) != (seed, N, n_tr, n_val):
        raise ValueError('cell identity')
    Wmat = {'isotropic': np.eye(g.d), 'sigma_inverse': Li @ Li}
    whiten = {'isotropic': None, 'sigma_inverse': Li}
    Cs = L @ g.K(c['theta']) @ L
    models, training, expected = {}, {}, set()
    for loss in c['losses']:
        A, b = science.closed_form(c, S, mu, Wmat[loss], c['k'])
        tag = f'tied_linear__{loss}'
        expected.update([tag + '__A', tag + '__b'])
        if not np.allclose(a[tag + '__A'], A, rtol=1e-9, atol=1e-11) or not np.allclose(a[tag + '__b'], b, rtol=1e-9, atol=1e-11):
            raise ValueError('closed-form map differs ' + tag)
        At, bt = torch.from_numpy(a[tag + '__A']), torch.from_numpy(a[tag + '__b'])
        ev = science.evaluate(c, torch, seed, lambda X: X @ At.T + bt, eclean, enoisy, Li)
        exact = science.risks(a[tag + '__A'], a[tag + '__b'], Cs, sigma, Li @ Li)
        ev.update(exact_clean_whitened=exact['clean'], exact_clean_euclidean=exact['euclidean_clean'],
                  exact_divergence=float(np.trace(a[tag + '__A'])))
        models[tag] = ev
    for cls in DEEP:
        for loss in c['losses']:
            tag = f'{cls}__{loss}'
            state = science.state_of(a, tag)
            expected.update(f'{tag}__{k}' for k in state)
            tr = r['training'][tag]
            runs = tr['restarts']
            if len(runs) != c['recipe']['restarts']:
                raise ValueError('restart inventory')
            for run in runs:
                check_training_record(c, run)
            sel = int(np.argmin([d['best_validation_loss'] for d in runs]))
            if tr['selected'] != sel:
                raise ValueError('restart selection mismatch')
            model = science.build_model(c, torch, cls, state)
            Wt = None if whiten[loss] is None else torch.from_numpy(whiten[loss])
            with torch.no_grad():
                vloss = float(science.metric_loss(model, torch.from_numpy(x[n_tr:N]), Wt))
                tloss = float(science.metric_loss(model, torch.from_numpy(x[:n_tr]), Wt))
            assert_close(vloss, runs[sel]['best_validation_loss'])
            assert_close(tloss, runs[sel]['final_train_loss_at_best'])
            if replay_training:
                replayed = []
                for k in range(c['recipe']['restarts']):
                    init = science.key_int(c['namespace'], seed, N, 'init', k)
                    m2, d2 = science.train_deep(c, torch, x[:n_tr], x[n_tr:N], whiten[loss], science.ACTIVATION[cls], init)
                    replayed.append(d2)
                    if k == sel:
                        for name, t2 in m2.state_dict().items():
                            if not np.array_equal(t2.detach().numpy(), state[name]):
                                raise ValueError('training replay changed the selected model ' + tag)
                if replayed != runs:
                    raise ValueError('training replay changed a restart record ' + tag)
            training[tag] = tr
            models[tag] = science.evaluate(c, torch, seed, model, eclean, enoisy, Li)
    if set(a) != expected:
        raise ValueError('unexpected/missing arrays')
    assert_close(r['models'], models)
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
    if read(root / 'VERIFY.END.json') != {'cells_reevaluated': e['paired_cells'], 'training_replay_cells': len(replay_cells(f['config']))}:
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
    """Contrast table and the P1/P2 tables as CSV beside reduction.json."""
    tdir = Path(root) / 'tables'
    tdir.mkdir()
    fields = ['N', 'contrast', 'mean', 'ci_low', 'ci_high', 'seeds', 'zero_standard_error']

    def dump(name, selected):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in selected:
            w.writerow({'N': r['N'], 'contrast': r['contrast'], 'mean': repr(r['mean']), 'ci_low': repr(r['ci'][0]),
                        'ci_high': repr(r['ci'][1]), 'seeds': r['seeds'], 'zero_standard_error': r['zero_standard_error']})
        with (tdir / name).open('x') as fh:
            fh.write(buf.getvalue())
    rows = reduction['contrasts']
    dump('contrasts.csv', rows)
    dump('P1_metric_effect.csv', [r for r in rows if r['contrast'].startswith('metric/')])
    dump('P2_class_contrasts.csv', [r for r in rows if r['contrast'].startswith('class/')])
    dump('SURE_agreement.csv', [r for r in rows if r['contrast'].startswith('sure_gap/')])
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


def export(root, out):
    """Copy the complete, replay-verified run, add tables, EXPORT.json and SHA256SUMS."""
    thread_gate()
    root, out = external(root), external(out)
    f = read(root / 'freeze.json')
    validate_freeze(f)
    if f['config']['development_only']:
        raise ValueError('fixture scientific export forbidden')
    if out.exists() or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('export requires a separate fresh output')
    check_ledgers(root, f)
    preflight_space(out.parent if out.parent.exists() else root.parent, tree_bytes(root) + 1024 ** 3)
    before = inventory(root)
    shutil.copytree(root, out)
    if inventory(root) != before or inventory(out) != before:
        raise ValueError('run changed while exporting')
    reduction = read(out / 'reduction.json')
    write_tables(out, reduction)
    write_new(out / 'EXPORT.json', {'deployment': VERSION, 'status': 'CONFIRMATION_EXPORT', 'source_run': str(root),
                                    'code_identity': code_identity(f['sources']), 'files_before_tables': before,
                                    'reduction_sha256': sha(out / 'reduction.json')})
    with (out / 'SHA256SUMS').open('x') as fh:
        fh.write(sha256sums(out))


def check_sha256sums(root):
    root = Path(root)
    if (root / 'SHA256SUMS').read_text() != sha256sums(root):
        raise ValueError('SHA256SUMS mismatch')


def audit(root, out):
    """Portable receiver audit (any machine with the frozen source bytes and torch): SHA256SUMS,
    ledgers, every model re-evaluated from saved weights and regenerated data, training records
    checked against the registered stopping and selection rules, and the full reduction.
    Training is not replayed here (that was the host's job on the registered subset)."""
    root, out = Path(root).resolve(), external(out)
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('audit output overlaps input')
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'START.json', {'operation': 'portable_audit', 'source_run': str(root), 'retries': 0})
    try:
        f = read(root / 'freeze.json')
        if (root / 'SHA256SUMS').exists():
            check_sha256sums(root)
        check_ledgers(root, f)
        records = verify(root, require_complete=True, replay=False)
        want = {'development_only': f['config']['development_only'], **science.reduce_records(f['config'], records)}
        assert_close(read(root / 'reduction.json'), want)
        result = {'verified': True, 'mode': 'portable_reevaluation_audit', 'training_replayed': False,
                  'cells': len(records), 'source_identity': code_identity(f['sources']),
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
    """Production-shaped timing in the X3 benchmark namespace (development-only, disjoint
    seed); fitted models and endpoints are discarded unviewed.  Projects the campaign from
    one cell per N, with x2 margin for the in-campaign verification and replay subset."""
    import psutil
    thread_gate()
    out = external(out)
    out.mkdir(parents=True, exist_ok=False)
    c = BENCHMARK
    if set(c['seeds']) & (set(BASE['seeds']) | set(FIXTURE['seeds'])):
        raise ValueError('benchmark seed overlaps a registered namespace')
    torch = science.import_torch(c)
    rows, proc = [], psutil.Process()
    for N in (cells or BASE['N']):
        t0 = time.monotonic()
        record, arrays = science.fit_cell(c, c['seeds'][0], N, torch)
        fit = time.monotonic() - t0
        stops = [d['stopped'] for t in record['training'].values() for d in t['restarts']]
        epochs = [d['epochs_run'] for t in record['training'].values() for d in t['restarts']]
        del record, arrays
        rows.append({'N': N, 'fit_seconds': fit, 'rss_bytes': proc.memory_info().rss,
                     'stops': {k: stops.count(k) for k in sorted(set(stops))}, 'max_epochs_run': max(epochs)})
    per_seed = sum(r['fit_seconds'] for r in rows)
    projected_fit = per_seed * len(BASE['seeds'])
    projected = (projected_fit + per_seed) * 2
    worst = max(r['fit_seconds'] for r in rows)
    report = {'development_only': True, 'scientific_endpoints_evaluated': False, 'namespace': c['namespace'],
              'note': 'confirmation endpoints never computed; benchmark-namespace models are discarded unviewed',
              'code_identity': code_identity(sources()), 'environment': environment(), 'host': socket.gethostname(),
              'cells': rows, 'seconds_per_seed': per_seed, 'projected_fit_seconds': projected_fit,
              'projected_lifecycle_seconds_with_margin': projected,
              'margin': 'x2 over fit + replay of one seed (re-evaluation is negligible)',
              'cell_seconds_cap': BASE['budget']['cell_seconds'],
              'worst_cell_fit_seconds': worst, 'worst_cell_within_half_cap': worst * 2 <= BASE['budget']['cell_seconds'],
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
