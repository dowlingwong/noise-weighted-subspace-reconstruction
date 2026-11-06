"""X-OPT production runner: one post-hoc strength pass over the frozen HCP Rnoise fits.

Modes
  --dry-run   verify freeze/inputs, print the plan, touch no result file.
  --worker ID internal: process one shard (launched by the driver).
  default     driver: verify, write STARTED.json, spawn workers, monitor, merge shards,
              write COMPLETED.json (only if complete and clean) or PARTIAL.json / STOP.json.

Selection is by training objective only; no clean-risk quantity is an input to selection.
No silent retries: a failing row is written as failed with the hash of its traceback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from . import kernel, select
from experiments.mlst2026.hcp_rnoise_20261019.signal import build

W_ROOT = '/ceph/dwong/hcp_xopt_20261106'
IN = W_ROOT + '/inputs/hcp_confirmation_20261020'
FITS = IN + '/FITS.jsonl'
FITS_SHA256 = 'c83198ea7f2b51fabb992ad5030997796b5abb9fb9a3663d88b785f7e645aa69'
SEAL_FILES = W_ROOT + '/inputs/SEAL_FILES.sha256'
FREEZE_RECORD = W_ROOT + '/MS/plan/onthefly/XOPT_FREEZE_2026-11-06.md'
FREEZE_JSON = W_ROOT + '/out-freeze/FREEZE.json'
PACKAGE = 'experiments/mlst2026/hcp_xopt_20261106'
KIT = 'experiments/mlst2026/hcp_rnoise_ad9_20261019'
KIT_IDENTITY = 'f01ed14caf682a3391edf5fd6e93566bd234e42b527d3077c4d79c684fe0596f'
EXPECTED = {'full': 135000, 'primary': 54000}
SCOPE_FRACTION = 0.10
SCOPE_SEED = 20261106
THREAD_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
              'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'BLIS_NUM_THREADS': '1',
              'PYTHONDONTWRITEBYTECODE': '1'}
REPLAY_OBJ_REL = 1e-9
REPLAY_D_FACTOR = 5.0


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def package_sha256():
    root = os.path.join(W_ROOT, 'EXP', PACKAGE)
    parts = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p) and name.endswith('.py'):
            parts.append(f'{name}:{sha256_file(p)}')
    tests = os.path.join(root, 'tests')
    if os.path.isdir(tests):
        for name in sorted(os.listdir(tests)):
            if name.endswith('.py'):
                parts.append(f'tests/{name}:{sha256_file(os.path.join(tests, name))}')
    return sha256_bytes('\n'.join(parts).encode())


def openblas_build():
    try:
        cfg = np.show_config(mode='dicts')
        blas = cfg.get('Build Dependencies', {}).get('blas', {})
        return f"{blas.get('name')} {blas.get('version')} {blas.get('openblas configuration')}"
    except Exception as exc:  # pragma: no cover
        return f'UNREADABLE:{exc}'


def stop(outdir, reason, detail=None):
    record = dict(stage='STOP', reason=reason, detail=detail,
                  time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'STOP.json'), 'w') as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print('STOP:', reason, json.dumps(detail)[:400], file=sys.stderr)
    return 8


def git(*args):
    return subprocess.run(['git', '-C', *args], capture_output=True, text=True)


def verify_freeze(args):
    """Verify the frozen identity and a clean tree.  Returns (ok, detail)."""
    detail = {}
    if not os.path.exists(FREEZE_JSON):
        return False, {'error': 'FREEZE.json missing: the freeze must precede the first fit'}
    with open(FREEZE_JSON) as fh:
        freeze = json.load(fh)
    detail['freeze'] = freeze
    ms, exp = os.path.join(W_ROOT, 'MS'), os.path.join(W_ROOT, 'EXP')
    r = git(exp, 'rev-parse', 'HEAD')
    if r.returncode or r.stdout.strip() != freeze['exp_commit']:
        return False, {'error': 'EXP HEAD is not the frozen code commit',
                       'head': r.stdout.strip(), 'frozen': freeze['exp_commit']}
    r = git(exp, 'describe', '--tags', '--exact-match')
    if r.returncode or r.stdout.strip() != freeze['tag']:
        return False, {'error': 'EXP HEAD is not exactly at the freeze tag', 'describe': r.stdout.strip()}
    r = git(ms, 'merge-base', '--is-ancestor', freeze['ms_commit'], 'HEAD')
    if r.returncode:
        return False, {'error': 'frozen MS commit is not an ancestor of MS HEAD'}
    proto = os.path.join(ms, 'plan/onthefly/XOPT_PROTOCOL_2026-11-06.md')
    detail['protocol_sha256'] = sha256_file(proto)
    if detail['protocol_sha256'] != freeze['protocol_sha256']:
        return False, {'error': 'protocol bytes differ from the freeze record',
                       'got': detail['protocol_sha256'], 'frozen': freeze['protocol_sha256']}
    detail['package_sha256'] = package_sha256()
    if detail['package_sha256'] != freeze['package_sha256']:
        return False, {'error': 'code bytes differ from the freeze record',
                       'got': detail['package_sha256'], 'frozen': freeze['package_sha256']}
    for name, path in (('EXP', exp), ('MS', ms)):
        dirty = git(path, 'status', '--porcelain').stdout.strip()
        if dirty:
            return False, {'error': f'{name} working tree is dirty', 'lines': dirty.splitlines()[:10]}
    detail['kit_identity'] = kit_identity()
    if detail['kit_identity'] != KIT_IDENTITY:
        return False, {'error': 'frozen kit identity mismatch', 'got': detail['kit_identity']}
    detail['openblas'] = openblas_build()
    detail['tag_object'] = git(exp, 'rev-parse', freeze['tag']).stdout.strip()
    return True, detail


def kit_identity():
    import pathlib
    root = pathlib.Path(W_ROOT, 'EXP', KIT)
    mani = json.load(open(root / 'KIT_SHA256.json'))
    for rel, want in sorted(mani['files'].items()):
        got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if got != want:
            return f'MISMATCH:{rel}'
    return mani['kit_identity']


def verify_inputs():
    """Verify the sealed, already-derived inputs before any fit."""
    if sha256_file(FITS) != FITS_SHA256:
        return False, {'error': 'FITS.jsonl sha256 mismatch', 'got': sha256_file(FITS)}
    n_dates = 0
    for name in sorted(os.listdir(IN + '/arrays')):
        d = os.path.join(IN, 'arrays', name)
        if not os.path.isdir(d):
            continue
        n_dates += 1
        for role in ('ref', 'train', 'cal', 'eval', 'Sref'):
            if not os.path.exists(os.path.join(d, role + ('.json' if role == 'Sref' else '.npz'))):
                return False, {'error': f'missing array file {name}/{role}'}
    if n_dates != 30:
        return False, {'error': f'expected 30 date array directories, found {n_dates}'}
    if os.path.exists(SEAL_FILES):
        r = subprocess.run(['sha256sum', '-c', SEAL_FILES], cwd=IN, capture_output=True, text=True)
        if r.returncode != 0:
            return False, {'error': 'SEAL_FILES.sha256 check failed', 'tail': r.stdout.strip().splitlines()[-3:]}
        ok_lines = sum(1 for ln in r.stdout.splitlines() if ln.endswith(': OK'))
        return True, {'fits_sha256': FITS_SHA256, 'arrays_dates': n_dates, 'seal_files_ok': ok_lines}
    return True, {'fits_sha256': FITS_SHA256, 'arrays_dates': n_dates, 'seal_files_ok': 'not available'}


def replicate_stratum(meta):
    key = f"{SCOPE_SEED}/{meta['date']}/{meta['geometry']}/{meta['N']}/{meta['arm']}/{meta['m']}"
    gen = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
    return set(gen.choice(50, int(round(SCOPE_FRACTION * 50)), replace=False).tolist())


def in_scope(meta, scope):
    if scope == 'full':
        return True
    if meta['arm'] in ('reference', 'isotropic'):
        return True
    return meta['replicate'] in replicate_stratum(meta)


def build_index(scope, path):
    n = 0
    with open(path, 'w') as fh:
        for meta in select.scan(FITS, wanted=lambda r: in_scope(r, scope)):
            fh.write(json.dumps(meta, sort_keys=True) + '\n')
            n += 1
    return n


def row_record(meta, original, result):
    return dict(
        date=meta['date'], geometry=meta['geometry'], theta=meta['theta'], N=meta['N'],
        arm=meta['arm'], m=meta['m'], replicate=meta['replicate'], date_index=meta['date_index'],
        original=dict(D=float(original['D']),
                      objective=float(original['kernel']['restarts'][original['kernel']['selected']]['objective']),
                      selected=int(original['kernel']['selected']),
                      status=original['kernel']['selected_status'],
                      optimization_limited=bool(original['kernel']['optimization_limited']),
                      optimization_limited_t25_rule=bool(original['kernel']['optimization_limited_t25_rule']),
                      restart_spread=float(original['kernel']['restart_spread'])),
        **result)


def process_row(meta, original, roles, Sref, model):
    t0 = time.time()
    frozen, cap_s, cap_m = kernel.frozen_route(roles, model, Sref, meta['N'], meta['m'], meta['arm'],
                                               int(meta['date_index']), meta['replicate'])
    t_frozen = time.time() - t0
    t0 = time.time()
    key = ('hcp-rnoise', int(meta['date_index']), meta['replicate'], float(meta['theta']), meta['N'])
    out, uwin, vwin, tied = kernel.strengthened(cap_s, cap_m, model.C, model.T, key)
    t_routes = time.time() - t0
    t0 = time.time()
    ev = kernel.evaluation_for(roles, model, Sref, int(meta['date_index']), meta['replicate'])
    _, _, win_scores = kernel.clean_scores(ev, uwin, vwin, cap_s['W'], cap_m['mu'])
    tied_risks = [float(kernel.clean_scores(ev, c['u'], c['v'], cap_s['W'], cap_m['mu'])[2]['clean_ref'])
                  for c in tied]
    t_scores = time.time() - t0
    spread = float(max(tied_risks) - min(tied_risks)) if len(tied_risks) > 1 else 0.0
    D = float(win_scores['clean_ref'] - frozen['U']['clean_ref'])
    flags = kernel.flags_for(out['selected_status'], out['selected_gradient_norm'], tied_risks, D, spread)
    obj_stored = float(original['kernel']['restarts'][original['kernel']['selected']]['objective'])
    obj_replay = float(frozen['restarts'][frozen['selected']]['objective'])
    replay_obj_rel = abs(obj_replay - obj_stored) / abs(obj_stored) if obj_stored else abs(obj_replay - obj_stored)
    replay_D_abs = abs(float(frozen['D']) - original['D'])
    spread_stored = float(original['kernel']['restart_spread'])
    return dict(
        frozen=dict(objective=obj_replay, D=float(frozen['D']), selected=int(frozen['selected']),
                    status=frozen['selected_status'],
                    gradient_norm=float(frozen['selected_gradient_norm']),
                    training_sha=frozen['training_sha'],
                    replay_obj_rel=replay_obj_rel, replay_D_abs=replay_D_abs,
                    replay_D_over_spread=replay_D_abs / spread_stored if spread_stored else None),
        strengthened=dict(routes=out['candidates'], route_seconds=out['route_seconds'],
                          selected_route=out['selected_route'], objective=out['objective'],
                          training_risk_at_best=out['training_risk_at_best'], n_tied=out['n_tied'],
                          tied_clean_risks=tied_risks, tied_spread=spread,
                          s2_seed_key=out.get('s2_seed_key'), s3=out.get('S3')),
        replay=dict(obj_rel=replay_obj_rel, D_abs=replay_D_abs,
                    D_over_spread=replay_D_abs / spread_stored if spread_stored else None,
                    obj_ok=replay_obj_rel <= REPLAY_OBJ_REL,
                    D_ok=(spread_stored > 0 and replay_D_abs <= REPLAY_D_FACTOR * spread_stored)),
        D=D, D_scores={k: float(v) for k, v in win_scores.items()},
        U=frozen['U'], F={k: float(v) for k, v in win_scores.items()},
        delta=dict(D=float(D - frozen['D']), objective=float(out['objective'] - obj_replay)),
        flags_after=flags,
        seconds=dict(frozen=t_frozen, routes=t_routes, scores=t_scores))


def worker_main(worker, n_workers, scope, outdir, estimate_seconds, index_path, deadline):
    shard = os.path.join(outdir, 'shards', f'shard_{worker:02d}.jsonl')
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    n_done = n_fail = 0
    date_roles, date_model = {}, {}
    with open(shard, 'w') as fh:
        with open(index_path) as idx:
            for line in idx:
                meta = json.loads(line)
                if int(hashlib.sha256(meta['date'].encode()).hexdigest()[:8], 16) % n_workers != worker:
                    continue
                if time.time() > deadline:
                    fh.write(json.dumps(dict(status='stopped', worker=worker, rows=n_done), sort_keys=True) + '\n')
                    return 3
                try:
                    original = select.read_row(FITS, meta['offset'])
                    if meta['date'] not in date_roles:
                        date_roles.clear()
                        date_model.clear()
                        date_roles[meta['date']] = select.load_inputs(meta['date'])
                    roles, Sref = date_roles[meta['date']]
                    key = float(meta['theta'])
                    if key not in date_model:
                        date_model[key] = build(Sref, key)
                    res = process_row(meta, original, roles, Sref, date_model[key])
                    rec = row_record(meta, original, res)
                    rec['status'] = 'ok'
                    n_done += 1
                    if not res['replay']['obj_ok'] or (res['replay']['D_over_spread'] or 0) > REPLAY_D_FACTOR:
                        return 6      # replication guard -> driver writes STOP.json
                except Exception:
                    tb = traceback.format_exc()
                    rec = row_record(meta, original, dict(
                        error=dict(type=tb.splitlines()[-1][:300], traceback_sha256=sha256_bytes(tb.encode())),
                        D=float('nan')))
                    rec['status'] = 'failed'
                    n_fail += 1
                fh.write(json.dumps(rec, sort_keys=True, allow_nan=True) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
    return 0 if n_fail == 0 else 4


def driver(args):
    outdir = args.out
    os.makedirs(os.path.join(outdir, 'shards'), exist_ok=True)
    if os.path.exists(os.path.join(outdir, 'STARTED.json')):
        return stop(outdir, 'output directory already started; no restart permitted')
    ok, detail = verify_freeze(args)
    if not ok:
        return stop(outdir, 'frozen identity check failed', detail)
    ok, inputs = verify_inputs()
    if not ok:
        return stop(outdir, 'sealed input verification failed', inputs)
    print(f'frozen identity OK: {json.dumps(detail)[:600]}', flush=True)
    print(f'inputs OK: {json.dumps(inputs)}', flush=True)
    index_path = os.path.join(outdir, 'scope_index.jsonl')
    temp_index = index_path + '.tmp'
    n_plan = build_index(args.scope, temp_index)
    expected = EXPECTED[args.scope]
    if args.limit:
        keep = []
        with open(temp_index) as fh:
            for i, line in enumerate(fh):
                if i >= args.limit:
                    break
                keep.append(line)
        with open(temp_index, 'w') as fh:
            fh.writelines(keep)
        n_plan = len(keep)
        expected = n_plan
    if n_plan != expected:
        os.unlink(temp_index)
        return stop(outdir, f'scope census {n_plan} != expected {expected}', {'scope': args.scope})
    os.replace(temp_index, index_path)
    if args.dry_run:
        os.unlink(index_path)
        print(f'DRY RUN OK: scope={args.scope} planned_rows={n_plan}; no result file touched.')
        return 0
    estimate = args.estimate_seconds
    started = dict(stage='STARTED', scope=args.scope, planned_rows=n_plan, workers=args.workers,
                   limit=args.limit, estimate_seconds=estimate,
                   graceful_stop_seconds=1.5 * estimate,
                   start_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   freeze=detail, inputs=inputs, kernel=args.kernel,
                   thread_env={k: os.environ.get(k) for k in THREAD_ENV},
                   openblas=detail.get('openblas'), host=os.uname().nodename,
                   python=sys.version.split()[0], numpy=np.__version__,
                   scope_index_sha256=sha256_file(index_path))
    with open(os.path.join(outdir, 'STARTED.json'), 'w') as fh:
        json.dump(started, fh, indent=2, sort_keys=True)
        fh.write('\n')
    deadline = time.time() + 1.5 * estimate
    env = dict(os.environ, OPENBLAS_CORETYPE=args.kernel, **THREAD_ENV)
    procs = []
    for w in range(args.workers):
        cmd = [sys.executable, '-m', 'experiments.mlst2026.hcp_xopt_20261106.run',
               '--worker', str(w), '--workers', str(args.workers), '--scope', args.scope,
               '--out', outdir, '--estimate-seconds', str(estimate), '--kernel', args.kernel]
        log = open(os.path.join(W_ROOT, 'logs', f'xopt_worker_{w:02d}.log'), 'ab')
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       preexec_fn=lambda: os.nice(10)), log))
    rcs = []
    try:
        for p, log in procs:
            p.wait()
            log.close()
            rcs.append(p.returncode)
    except KeyboardInterrupt:
        for p, _ in procs:
            p.send_signal(signal.SIGTERM)
        rcs = [130]
    shards = sorted(os.path.join(outdir, 'shards', f) for f in os.listdir(os.path.join(outdir, 'shards')))
    merged = os.path.join(outdir, 'XOPT_FITS.jsonl')
    n_rows = n_ok = n_fail = n_stop = 0
    with open(merged, 'wb') as out:
        for s in shards:
            with open(s, 'rb') as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get('status') == 'stopped':
                        n_stop += 1
                        continue
                    n_rows += 1
                    n_ok += rec.get('status') == 'ok'
                    n_fail += rec.get('status') == 'failed'
                    out.write(line)
    base = dict(scope=args.scope, planned_rows=n_plan, rows=n_rows, ok=n_ok, failed=n_fail,
                stopped_workers=n_stop, workers=args.workers, kernel=args.kernel,
                end_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                xopt_fits_sha256=sha256_file(merged))
    if 6 in rcs:
        return stop(outdir, 'replication guard triggered: a scoped row was not reproduced', base)
    if rcs and all(rc == 0 for rc in rcs) and n_rows == n_plan and n_fail == 0:
        base['stage'] = 'COMPLETED'
        name = 'COMPLETED.json'
    else:
        base['stage'] = 'PARTIAL'
        base['partial_reason'] = ('time_stop' if n_stop or any(rc == 3 for rc in rcs)
                                  else 'row_failures' if n_fail else f'worker_rc={rcs}')
        name = 'PARTIAL.json'
    with open(os.path.join(outdir, name), 'w') as fh:
        json.dump(base, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(json.dumps(base, indent=2, sort_keys=True))
    return 0 if name == 'COMPLETED.json' else 5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--scope', default='full', choices=['full', 'primary'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=W_ROOT + '/out')
    ap.add_argument('--estimate-seconds', type=float, required=True)
    ap.add_argument('--kernel', default='Haswell')
    ap.add_argument('--worker', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args(argv)
    if args.estimate_seconds <= 0:
        raise SystemExit('--estimate-seconds must be positive (the graceful stop depends on it)')
    if args.worker is not None:
        deadline = time.time() + 1.5 * args.estimate_seconds
        return worker_main(args.worker, args.workers, args.scope, args.out, args.estimate_seconds,
                           os.path.join(args.out, 'scope_index.jsonl'), deadline)
    return driver(args)


if __name__ == '__main__':
    sys.exit(main())
