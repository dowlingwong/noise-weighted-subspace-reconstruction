"""X-OPT resume-4 production runner: two hosts, one shared claim queue on /ceph.

Modes
  prepare     verify freeze + sealed inputs, build the scoped row list, split it into fixed
              chunks (200 rows by default), write the queue and STARTED.json (before the
              first fit), exit.
  worker      on one host: spawn N claim-workers (subprocesses), wait, report.
  worker-child  internal: loop over the queue claiming chunks, process, publish, exit.
  merge       validate exactly-once across chunk files, merge the ledger, write
              COMPLETED.json / PARTIAL.json / STOP.json.
  seal        archive results/claims/logs into sealed/ per addendum A2.d, verify, delete the
              loose copies, make sealed read-only.
  supervise   autonomous, agent-free supervisor for one host (run under tmux): hourly progress
              snapshots, worker-pool relaunch, quota guard, graceful-stop, and merge+seal when
              no chunk remains.
  status      append a progress snapshot (entries, bytes, quota) and print queue status.

Selection is by training objective only; no clean-risk quantity is an input to selection.
No silent retries: a failing row is written as failed with the hash of its traceback.
Invariant: a non-finite, missing, duplicated or failed value produces only a counted
failure / PARTIAL / STOP -- never an 'ok' row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

from . import claim, kernel, scopes, seal, select
from experiments.mlst2026.hcp_rnoise_20261019.signal import build

from .config import ROOT
W_ROOT = ROOT
IN = W_ROOT + '/inputs/hcp_confirmation_20261020'
FITS = IN + '/FITS.jsonl'
FITS_SHA256 = 'c83198ea7f2b51fabb992ad5030997796b5abb9fb9a3663d88b785f7e645aa69'
SEAL_FILES = W_ROOT + '/inputs/SEAL_FILES.sha256'
FREEZE_JSON = W_ROOT + '/out-freeze/FREEZE.json'
PACKAGE = 'experiments/mlst2026/hcp_xopt_20261106'
KIT = 'experiments/mlst2026/hcp_rnoise_ad9_20261019'
KIT_IDENTITY = 'f01ed14caf682a3391edf5fd6e93566bd234e42b527d3077c4d79c684fe0596f'
PLANNED = {'full': 135000, 'primary': 54000, 'core': 31500}
QDIR = W_ROOT + '/run_20261108'
PROGRESS = W_ROOT + '/records/resume4/progress.log'
CEPH_ROOT = '/ceph/dwong'
CEPH_MAX_FILES = 500000
CHUNK = 200
JOB_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'BLIS_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPYCACHEPREFIX': f'/tmp/xopt_pycache_{os.environ.get("USER", "dwong")}',
           'MPLCONFIGDIR': f'/tmp/xopt_mpl_{os.environ.get("USER", "dwong")}',
           'TMPDIR': '/tmp'}
REPLAY_OBJ_REL = 1e-9
REPLAY_D_FACTOR = 5.0
# Absolute floor added by the resume-5 reviewer (p2, recorded deviation): the 5x-restart_spread
# factor was calibrated on 60 replay rows; over 135,000 rows a row whose stored spread is
# ultratiny (e.g. 3.4e-10) trips it on a last-bit host difference while reproducing to ~1e-9
# absolute.  A breach now needs BOTH the relative and this absolute exceedance.  1e-5 is 8e-5 of
# the smallest registered half-width 0.1206 and 1e-4 of the closest decision margin 0.1024.
REPLAY_D_ABS_FLOOR = 1e-5
REQUIRED_FREEZE_KEYS = ('exp_commit', 'tag', 'ms_commit', 'protocol_sha256', 'package_sha256')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _source_names(directory):
    out = []
    for name in sorted(os.listdir(directory)):
        p = os.path.join(directory, name)
        if name.startswith('.') or not os.path.isfile(p) or not name.endswith('.py'):
            continue
        out.append(name)
    return out


def package_sha256():
    root = os.path.join(W_ROOT, 'EXP', PACKAGE)
    parts = [f'{name}:{sha256_file(os.path.join(root, name))}' for name in _source_names(root)]
    tests = os.path.join(root, 'tests')
    if os.path.isdir(tests):
        parts += [f'tests/{name}:{sha256_file(os.path.join(tests, name))}'
                  for name in _source_names(tests)]
    return sha256_bytes('\n'.join(parts).encode())


def openblas_build():
    try:
        cfg = np.show_config(mode='dicts')
        blas = cfg.get('Build Dependencies', {}).get('blas', {})
        return f"{blas.get('name')} {blas.get('version')} {blas.get('openblas configuration')}"
    except Exception as exc:  # pragma: no cover
        return f'UNREADABLE:{exc}'


def _freeze_path(args):
    return getattr(args, 'freeze_json', None) or FREEZE_JSON


def _queue(args):
    return getattr(args, 'queue', None) or QDIR


def stop(qdir, reason, detail=None, status='STOP'):
    os.makedirs(qdir, exist_ok=True)
    record = dict(stage=status, reason=reason, detail=detail,
                  time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    name = 'STOP.json' if status == 'STOP' else 'PARTIAL.json'
    with open(os.path.join(qdir, name), 'w') as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(status + ':', reason, json.dumps(detail, default=str)[:400], file=sys.stderr)
    return 8 if status == 'STOP' else 5


def git(*args):
    return subprocess.run(['git', '-C', *args], capture_output=True, text=True)


def kit_identity():
    import pathlib
    root = pathlib.Path(W_ROOT, 'EXP', KIT)
    with open(root / 'KIT_SHA256.json') as fh:
        mani = json.load(fh)
    for rel in sorted(mani['files']):
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != mani['files'][rel]:
            return f'MISMATCH:{rel}'
    return mani['kit_identity']


def verify_freeze(args):
    detail = {}
    path = _freeze_path(args)
    if not os.path.exists(path):
        return False, {'error': 'FREEZE.json missing: the freeze must precede the first fit',
                       'path': path}
    try:
        with open(path) as fh:
            freeze = json.load(fh)
    except (OSError, ValueError) as exc:
        return False, {'error': 'FREEZE.json is not valid JSON', 'path': path, 'detail': str(exc)}
    if not isinstance(freeze, dict):
        return False, {'error': 'FREEZE.json schema invalid: not an object', 'path': path}
    missing = [k for k in REQUIRED_FREEZE_KEYS if k not in freeze]
    if missing:
        return False, {'error': 'FREEZE.json schema invalid', 'missing': missing, 'path': path}
    detail['freeze'] = freeze
    ms, exp = os.path.join(W_ROOT, 'MS'), os.path.join(W_ROOT, 'EXP')
    r = git(exp, 'rev-parse', 'HEAD')
    if r.returncode or r.stdout.strip() != freeze['exp_commit']:
        return False, {'error': 'EXP HEAD is not the frozen code commit',
                       'head': r.stdout.strip(), 'frozen': freeze['exp_commit']}
    r = git(exp, 'describe', '--tags', '--exact-match')
    if r.returncode or r.stdout.strip() != freeze['tag']:
        return False, {'error': 'EXP HEAD is not exactly at the freeze tag', 'describe': r.stdout.strip()}
    tag_type = git(exp, 'cat-file', '-t', freeze['tag']).stdout.strip()
    detail['tag_object_type'] = tag_type
    if tag_type != 'tag':
        return False, {'error': 'freeze tag is not an annotated tag object', 'type': tag_type}
    detail['tag_object'] = git(exp, 'rev-parse', freeze['tag']).stdout.strip()
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
    for name, path_ in (('EXP', exp), ('MS', ms)):
        dirty = git(path_, 'status', '--porcelain').stdout.strip()
        if dirty:
            return False, {'error': f'{name} working tree is dirty', 'lines': dirty.splitlines()[:10]}
    detail['kit_identity'] = kit_identity()
    if detail['kit_identity'] != KIT_IDENTITY:
        return False, {'error': 'frozen kit identity mismatch', 'got': detail['kit_identity']}
    detail['openblas'] = openblas_build()
    detail['ots_status'] = freeze.get('ots_status', 'not-recorded')
    return True, detail


def verify_inputs():
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
    manifest = SEAL_FILES
    if not os.path.exists(manifest):
        manifest = os.path.join(W_ROOT, 'MS/reviews/hcp_rnoise_confirmation_seal_20261027/SEAL_FILES.sha256')
    if not os.path.exists(manifest):
        return False, {'error': 'no SEAL_FILES.sha256 manifest available'}
    r = subprocess.run(['sha256sum', '-c', manifest], cwd=IN, capture_output=True, text=True)
    if r.returncode != 0:
        return False, {'error': 'SEAL_FILES.sha256 check failed',
                       'tail': r.stdout.strip().splitlines()[-3:]}
    ok_lines = sum(1 for ln in r.stdout.splitlines() if ln.endswith(': OK'))
    if ok_lines != 248:
        return False, {'error': f'seal file count {ok_lines} != 248'}
    return True, {'fits_sha256': FITS_SHA256, 'arrays_dates': n_dates, 'seal_files_ok': ok_lines,
                  'seal_manifest': manifest}


def getfattr_value(attr, path):
    r = subprocess.run(['getfattr', '--only-values', '-n', attr, path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def ceph_usage():
    ent = getfattr_value('ceph.dir.rentries', CEPH_ROOT)
    bts = getfattr_value('ceph.dir.rbytes', CEPH_ROOT)
    maxf = getfattr_value('ceph.quota.max_files', CEPH_ROOT)
    maxb = getfattr_value('ceph.quota.max_bytes', CEPH_ROOT)
    w2 = getfattr_value('ceph.dir.rentries', W_ROOT)
    try:
        ent_i = int(ent) if ent is not None else None
        maxf_i = int(maxf) if maxf else None
    except (TypeError, ValueError):
        ent_i = maxf_i = None
    free = None if ent_i is None or maxf_i is None else maxf_i - ent_i
    return dict(rentries=ent_i, max_files=maxf_i, free_entries=free,
                rbytes=int(bts) if bts else None, max_bytes=int(maxb) if maxb else None,
                w2_rentries=int(w2) if w2 else None)


def build_scope_rows(scope):
    if scope == 'full':
        rows = [{k: meta[k] for k in ('offset', 'date', 'geometry', 'N', 'arm', 'm',
                                      'replicate', 'date_index', 'theta')}
                for meta in select.scan(FITS)]
        rows.sort(key=lambda r: r['offset'])
        return rows
    mapping = {'primary': 'S_primary', 'core': 'S_core'}
    return scopes.rows_for_scope(mapping[scope], FITS)


def row_base(meta):
    return dict(date=meta['date'], geometry=meta['geometry'], theta=meta['theta'], N=meta['N'],
                arm=meta['arm'], m=meta['m'], replicate=meta['replicate'],
                date_index=meta['date_index'], offset=meta['offset'])


def row_record(meta, original, result):
    rec = row_base(meta)
    rec.update(original=dict(D=float(original['D']),
                             objective=float(original['kernel']['restarts'][original['kernel']['selected']]['objective']),
                             selected=int(original['kernel']['selected']),
                             status=original['kernel']['selected_status'],
                             optimization_limited=bool(original['kernel']['optimization_limited']),
                             optimization_limited_t25_rule=bool(original['kernel']['optimization_limited_t25_rule']),
                             restart_spread=float(original['kernel']['restart_spread'])))
    rec.update(result)
    return rec


def _all_finite(values):
    return all(isinstance(v, (int, float)) and np.isfinite(float(v)) for v in values)


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
    obj_replay = float(frozen['restarts'][frozen['selected']]['objective'])
    if not _all_finite([D, float(frozen['D']), obj_replay, spread]):
        raise ValueError('non-finite value in the strengthened result')
    if not _all_finite(list(win_scores.values())) or not _all_finite(tied_risks):
        raise ValueError('non-finite clean-risk score')
    flags = kernel.flags_for(out['selected_status'], out['selected_gradient_norm'], tied_risks, D, spread)
    obj_stored = float(original['kernel']['restarts'][original['kernel']['selected']]['objective'])
    replay_obj_rel = abs(obj_replay - obj_stored) / abs(obj_stored) if obj_stored else abs(obj_replay - obj_stored)
    replay_D_abs = abs(float(frozen['D']) - original['D'])
    spread_stored = float(original['kernel']['restart_spread'])
    if not _all_finite([replay_obj_rel, replay_D_abs, spread_stored]):
        raise ValueError('non-finite replication diagnostic')
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
                    D_over_spread=replay_D_abs / spread_stored if spread_stored else None),
        D=D, D_scores={k: float(v) for k, v in win_scores.items()},
        U=frozen['U'], F={k: float(v) for k, v in win_scores.items()},
        delta=dict(D=float(D - frozen['D']), objective=float(out['objective'] - obj_replay)),
        flags_after=flags,
        seconds=dict(frozen=t_frozen, routes=t_routes, scores=t_scores))


def _replay_breach(res):
    rep = res['replay']
    obj = rep['obj_rel']
    if not isinstance(obj, (int, float)) or not np.isfinite(obj):
        return 'non-finite replay objective'
    if obj > REPLAY_OBJ_REL:
        return f'replay objective rel {obj:.3e} > {REPLAY_OBJ_REL:g}'
    d_abs = rep['D_abs']
    if not isinstance(d_abs, (int, float)) or not np.isfinite(d_abs):
        return 'non-finite replay D'
    d = rep['D_over_spread']
    if d is None:
        if d_abs > REPLAY_D_ABS_FLOOR:
            return (f'replay |D| {d_abs:.3e} > floor {REPLAY_D_ABS_FLOOR:g} '
                    f'with zero stored spread')
        return None
    if not np.isfinite(d):
        return 'non-finite replay D/spread'
    if d > REPLAY_D_FACTOR and d_abs > REPLAY_D_ABS_FLOOR:
        return (f'replay D/spread {d:.3f} > {REPLAY_D_FACTOR:g} AND '
                f'|D| {d_abs:.3e} > floor {REPLAY_D_ABS_FLOOR:g}')
    return None


# --------------------------------------------------------------------------- prepare
def prepare(args):
    qdir = _queue(args)
    os.makedirs(qdir, exist_ok=True)
    if os.path.exists(os.path.join(qdir, 'STARTED.json')):
        return stop(qdir, 'output directory already started; no restart permitted')
    scratch = bool(getattr(args, 'scratch', False))
    if scratch:
        detail = dict(scratch=True, note='scratch run: freeze/input checks bypassed by '
                                        'explicit --scratch; output is discarded')
        inputs = dict(scratch=True)
        print('SCRATCH MODE: freeze/input checks bypassed (discarded run)', file=sys.stderr)
    else:
        ok, detail = verify_freeze(args)
        if not ok:
            return stop(qdir, 'frozen identity check failed', detail)
        ok, inputs = verify_inputs()
        if not ok:
            return stop(qdir, 'sealed input verification failed', inputs)
    final_scope = {'S_full': 'full', 'S_primary': 'primary', 'S_core': 'core'}.get(
        getattr(args, 'scope', 'full'), getattr(args, 'scope', 'full'))
    rows = build_scope_rows(final_scope)
    if args.limit:
        rows = rows[:args.limit]
    expected = PLANNED[final_scope]
    if args.limit is None and len(rows) != expected:
        return stop(qdir, f'scope census {len(rows)} != expected {expected}', {'scope': final_scope})
    chunk_size = getattr(args, 'chunk_size', None) or CHUNK
    manifest = claim.prepare(qdir, final_scope, rows, chunk_size=chunk_size)
    projected = manifest['n_chunks'] * 3 + 60 + manifest['n_chunks'] * 0  # claim dir+claim.json+result
    estimate = args.estimate_seconds
    usage = ceph_usage()
    started = dict(stage='STARTED', scope=final_scope, planned_rows=manifest['planned_rows'],
                   chunks=manifest['n_chunks'], chunk_size=manifest['chunk_size'],
                   row_list_sha256=manifest['row_list_sha256'],
                   limit=args.limit, estimate_seconds=estimate,
                   graceful_stop_seconds=1.5 * estimate,
                   deadline_epoch=time.time() + 1.5 * estimate,
                   start_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   freeze=detail, inputs=inputs, kernel=args.kernel,
                   thread_env=dict(JOB_ENV), openblas=(detail or {}).get('openblas'),
                   host=os.uname().nodename, python=sys.version.split()[0],
                   numpy=np.__version__,
                   p_dt=getattr(args, 'p_dt', None), p_dt2=getattr(args, 'p_dt2', None),
                   p_total=None if getattr(args, 'p_dt', None) is None else
                   (args.p_dt + (args.p_dt2 or 0)),
                   projected_peak_entries=projected, quota_at_start=usage,
                   scratch=scratch, scope_index_sha256=manifest['row_list_sha256'])
    if args.dry_run:
        print(json.dumps({k: started[k] for k in ('scope', 'planned_rows', 'chunks', 'chunk_size',
                                                  'row_list_sha256', 'projected_peak_entries')},
                         indent=2))
        print('DRY RUN OK: no STARTED.json written, no result file touched.')
        return 0
    with open(os.path.join(qdir, 'STARTED.json'), 'w') as fh:
        json.dump(started, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(json.dumps({k: started[k] for k in ('scope', 'planned_rows', 'chunks', 'chunk_size',
                                              'row_list_sha256', 'graceful_stop_seconds',
                                              'kernel', 'projected_peak_entries')}, indent=2))
    return 0


# --------------------------------------------------------------------------- workers
def _load_started(qdir):
    with open(os.path.join(qdir, 'STARTED.json')) as fh:
        return json.load(fh)


def _claim_flags(qdir):
    out = {}
    for name in ('STOP_FLAG.json', 'TIME_STOP.json', 'QUOTA_STOP.json'):
        p = os.path.join(qdir, name)
        if os.path.exists(p):
            with open(p) as fh:
                out[name] = json.load(fh)
    return out


def _write_flag(qdir, name, payload):
    p = os.path.join(qdir, name)
    tmp = p + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write('\n')
    os.replace(tmp, p)


def _claim_next(qdir, chunks, host, pid):
    for cid in chunks:
        if claim.is_done(qdir, cid):
            continue
        res = claim.claim(qdir, cid, host, pid)
        if res['action'] in ('claimed', 'reclaimed'):
            return cid
    return None


def _quota_guard(qdir, manifest, host, pid):
    """A2.e: stop claiming when free entries < 1.5 x projected remaining entries."""
    usage = ceph_usage()
    st = claim.status(qdir, manifest)
    remaining = st['remaining']
    projected = remaining * 3 + 60
    free = usage.get('free_entries')
    if free is not None and free < 1.5 * projected:
        _write_flag(qdir, 'QUOTA_STOP.json',
                    dict(host=host, pid=pid, free_entries=free, projected=projected,
                         usage=usage, time=time.time(),
                         reason='free entries < 1.5 x projected remaining entries'))
        return False, usage
    return True, usage


def record_done(qdir, cid, host, n_failed):
    """Append one line to the single done_index.jsonl (used by the supervisor snapshots)."""
    line = json.dumps(dict(chunk=cid, host=host, failed=int(n_failed), time=time.time()),
                      sort_keys=True) + '\n'
    os.makedirs(os.path.join(qdir, 'logs'), exist_ok=True)
    fd = os.open(os.path.join(qdir, 'logs', 'done_index.jsonl'), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def run_chunk(args, qdir, cid, host, pid, deadline, manifest):
    rows = claim.load_chunk_rows(qdir, cid, manifest)
    date_roles, date_model = {}, {}
    records = []
    last_hb = time.time()
    for meta in rows:
        now = time.time()
        if now > deadline:
            _write_flag(qdir, 'TIME_STOP.json', dict(chunk=cid, host=host, pid=pid,
                                                     rows=len(records), time=now))
            return 'time_stop'
        if now - last_hb > claim.HEARTBEAT_SECONDS:
            if not claim.still_owner(qdir, cid, host, pid):
                return 'done'
            claim.heartbeat(qdir, cid, host, pid)
            last_hb = now
        try:
            original = select.read_row(FITS, meta['offset'])
            if meta['date'] not in date_roles:
                date_roles.clear()
                date_model.clear()
                date_roles[meta['date']] = select.load_inputs(meta['date'])
            roles, Sref = date_roles[meta['date']]
            model_key = float(meta['theta'])
            if model_key not in date_model:
                date_model[model_key] = build(Sref, model_key)
            res = process_row(meta, original, roles, Sref, date_model[model_key])
            rec = row_record(meta, original, res)
            rec['status'] = 'ok'
            if not _all_finite([rec['D'], rec['delta']['D'], rec['delta']['objective']]):
                raise ValueError('non-finite D/delta in a row that would otherwise be ok')
            breach = _replay_breach(res)
            if breach:
                _write_flag(qdir, 'STOP_FLAG.json', dict(chunk=cid, host=host, pid=pid,
                                                         reason='replication guard: ' + breach,
                                                         meta=meta))
                return 'guard'
        except Exception:
            tb = traceback.format_exc()
            rec = row_base(meta)
            rec['original'] = None
            rec['status'] = 'failed'
            rec['D'] = None
            rec['error'] = dict(type=tb.splitlines()[-1][:300],
                                traceback_sha256=sha256_bytes(tb.encode()))
        rec['host'] = host
        rec['chunk_id'] = cid
        rec['worker_pid'] = pid
        rec['openblas_coretype'] = os.environ.get('OPENBLAS_CORETYPE', args.kernel)
        records.append(rec)
    if not claim.still_owner(qdir, cid, host, pid):
        return 'done'
    claim.publish(qdir, cid, records, manifest)
    record_done(qdir, cid, host, sum(1 for r in records if r.get('status') != 'ok'))
    return 'done'


def worker_child(args):
    qdir = _queue(args)
    manifest = claim.load_manifest(qdir)
    started = _load_started(qdir)
    deadline = started['deadline_epoch']
    host = socket.gethostname()
    pid = os.getpid()
    last_quota = 0.0
    while True:
        flags = _claim_flags(qdir)
        if 'STOP_FLAG.json' in flags:
            return 6
        if 'QUOTA_STOP.json' in flags:
            return 5
        if time.time() > deadline:
            _write_flag(qdir, 'TIME_STOP.json', dict(host=host, pid=pid, time=time.time(),
                                                     reason='graceful stop threshold reached'))
            return 3
        if time.time() - last_quota > 60:
            ok, usage = _quota_guard(qdir, manifest, host, pid)
            last_quota = time.time()
            if not ok:
                return 5
        cid = _claim_next(qdir, claim.chunk_ids(manifest), host, pid)
        if cid is None:
            if claim.status(qdir, manifest)['remaining'] == 0:
                return 0
            time.sleep(20)
            continue
        rc = run_chunk(args, qdir, cid, host, pid, deadline, manifest)
        if rc == 'guard':
            return 6
        if rc == 'time_stop':
            return 3


def worker(args):
    qdir = _queue(args)
    if not os.path.exists(os.path.join(qdir, 'STARTED.json')):
        return stop(qdir, 'worker started without STARTED.json')
    host = socket.gethostname()
    env = dict(os.environ, OPENBLAS_CORETYPE=args.kernel, **JOB_ENV)
    logs = os.path.join(qdir, 'logs')
    os.makedirs(logs, exist_ok=True)
    procs = []
    for w in range(args.host_workers):
        cmd = [sys.executable, '-m', 'experiments.mlst2026.hcp_xopt_20261106.run',
               '--mode', 'worker-child', '--queue', qdir, '--kernel', args.kernel,
               '--host-workers', str(args.host_workers)]
        log = open(os.path.join(logs, f'worker_{host}_{w:03d}.log'), 'ab')
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       preexec_fn=lambda: os.nice(10)), log))
    rcs = []
    for p, log in procs:
        p.wait()
        log.close()
        rcs.append(p.returncode)
    print(json.dumps(dict(host=host, workers=args.host_workers, rcs=rcs), indent=2))
    return 0



# --------------------------------------------------------------------------- supervisor
def _done_index(qdir):
    out = {}
    p = os.path.join(qdir, 'logs', 'done_index.jsonl')
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                d = out.setdefault(r.get('host', '?'), dict(chunks=0, failed_rows=0))
                d['chunks'] += 1
                d['failed_rows'] += int(r.get('failed', 0))
    return out


def _reclaim_lines(qdir):
    p = os.path.join(qdir, 'reclaims.log')
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def progress_snapshot(qdir, started, manifest, tag='supervisor'):
    """Append one progress snapshot to PROGRESS (records/resume4/progress.log)."""
    usage = ceph_usage()
    st = claim.status(qdir, manifest)
    w2_bytes = getfattr_value('ceph.dir.rbytes', W_ROOT)
    maxb = usage.get('max_bytes') or 0
    rbytes = usage.get('rbytes')
    snap = dict(tag=tag, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                host=socket.gethostname(), scope=started.get('scope'),
                chunks_planned=manifest['n_chunks'], done=st['done'], claimed=st['claimed'],
                remaining=st['remaining'], done_per_host=_done_index(qdir),
                reclaims=_reclaim_lines(qdir),
                loadavg=open('/proc/loadavg').read().split()[:3],
                w2_entries=usage.get('w2_rentries'),
                w2_bytes=int(w2_bytes) if w2_bytes else None,
                ceph_rentries=usage.get('rentries'), ceph_max_files=usage.get('max_files'),
                free_entries=usage.get('free_entries'), ceph_rbytes=rbytes,
                ceph_max_bytes=maxb,
                free_bytes=(int(maxb) - int(rbytes)) if maxb and rbytes else None,
                result_bytes=st.get('result_bytes'))
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, 'a') as fh:
        fh.write(json.dumps(snap, sort_keys=True) + '\n')
    return snap


def _spawn_local_pool(args):
    """Spawn this host's worker pool (N claim-workers) as a child process."""
    qdir = _queue(args)
    logs = os.path.join(qdir, 'logs')
    os.makedirs(logs, exist_ok=True)
    cmd = [sys.executable, '-m', 'experiments.mlst2026.hcp_xopt_20261106.run',
           '--mode', 'worker', '--queue', qdir, '--host-workers', str(args.host_workers),
           '--kernel', args.kernel]
    log = open(os.path.join(logs, f'supervisor_pool_{socket.gethostname()}.log'), 'ab')
    env = dict(os.environ, OPENBLAS_CORETYPE=args.kernel, **JOB_ENV)
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=lambda: os.nice(10))


def _acquire_merge_lock(qdir):
    try:
        os.mkdir(os.path.join(qdir, '.merge.lock'))
        return True
    except FileExistsError:
        return False


def finalize(args):
    """Merge, then seal, exactly once (guarded by an atomic mkdir lock)."""
    qdir = _queue(args)
    if not _acquire_merge_lock(qdir):
        return None
    rc = merge(args)
    try:
        seal.archive_run(qdir)
        seal.freeze_sealed(qdir)
    except seal.SealError as exc:
        print('SEAL ERROR:', exc, file=sys.stderr)
        return rc
    print('SEALED:', os.path.join(qdir, seal.SEALED), file=sys.stderr)
    return rc


def supervise(args):
    """Autonomous, agent-free supervisor for one host (run under tmux).

    (i) appends a progress snapshot every --interval seconds (default 900 s, <= 1 h) to
        W2/records/resume4/progress.log;
    (ii) keeps this host's worker pool alive, relaunching it if it dies before the deadline
        (the workers themselves apply the 2 h stale-claim rule);
    (iii) applies the A2.e quota guard before relaunching;
    (iv) stops claiming at the graceful-stop threshold and lets the current chunk finish;
    (v) once no chunk remains, merges and seals exactly once via an atomic lock.
    """
    qdir = _queue(args)
    if not os.path.exists(os.path.join(qdir, 'STARTED.json')):
        return stop(qdir, 'supervisor started without STARTED.json')
    started = _load_started(qdir)
    manifest = claim.load_manifest(qdir)
    deadline = started['deadline_epoch']
    interval = max(30.0, float(args.interval))
    pool = None
    last = None
    while True:
        now = time.time()
        if last is None or now - last >= interval:
            progress_snapshot(qdir, started, manifest)
            last = now
        flags = _claim_flags(qdir)
        if 'STOP_FLAG.json' in flags or 'QUOTA_STOP.json' in flags:
            break
        st = claim.status(qdir, manifest)
        if st['remaining'] == 0:
            break
        if now > deadline:
            _write_flag(qdir, 'TIME_STOP.json', dict(host=socket.gethostname(), time=now,
                                                     reason='graceful stop threshold reached'))
            break
        ok, _usage = _quota_guard(qdir, manifest, socket.gethostname(), os.getpid())
        if not ok:
            break
        if pool is None or pool.poll() is not None:
            pool = _spawn_local_pool(args)
        time.sleep(min(60.0, interval))
    if pool is not None and pool.poll() is None:
        t0 = time.time()
        while pool.poll() is None and time.time() - t0 < 1800:
            time.sleep(10)
        if pool.poll() is None:
            pool.terminate()
    progress_snapshot(qdir, started, manifest, tag='final')
    rc = finalize(args)
    return 0 if rc in (None, 0) else rc


# --------------------------------------------------------------------------- merge / seal
def merge(args):
    qdir = _queue(args)
    if not os.path.exists(os.path.join(qdir, 'STARTED.json')):
        return stop(qdir, 'merge started without STARTED.json')
    started = _load_started(qdir)
    summary = claim.merge(qdir)
    stopped = _claim_flags(qdir)
    base = dict(scope=started['scope'], planned_rows=started['planned_rows'],
                chunks=started['chunks'], chunk_size=started['chunk_size'],
                row_list_sha256=started['row_list_sha256'],
                rows=summary['rows'], ok=summary['ok'], failed=summary['failed'],
                done_chunks=summary['done_chunks'], kernel=started.get('kernel'),
                end_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                xopt_fits_path=summary['merged_path'],
                xopt_fits_sha256=summary['merged_sha256'], problems=summary['problems'],
                stop_flags=stopped)
    if 'STOP_FLAG.json' in stopped:
        return stop(qdir, 'replication guard triggered: a scoped row was not reproduced',
                    base, status='STOP')
    if summary['complete'] and summary['failed'] == 0:
        base['stage'] = 'COMPLETED'
        name, rc = 'COMPLETED.json', 0
    else:
        base['stage'] = 'PARTIAL'
        base['partial_reason'] = ('time_stop' if 'TIME_STOP.json' in stopped else
                                  'quota_stop' if 'QUOTA_STOP.json' in stopped else
                                  'row_failures' if summary['failed'] else 'incomplete_chunks')
        name, rc = 'PARTIAL.json', 5
    with open(os.path.join(qdir, name), 'w') as fh:
        json.dump(base, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(json.dumps(base, indent=2, sort_keys=True))
    return rc


def seal_run(args):
    qdir = _queue(args)
    rec = seal.archive_run(qdir)
    seal.freeze_sealed(qdir)
    print(json.dumps({k: rec[k] for k in ('scope', 'merged_rows', 'chunk_count', 'n_members')},
                     indent=2))
    return 0


def status(args):
    qdir = _queue(args)
    manifest = claim.load_manifest(qdir)
    st = claim.status(qdir, manifest)
    usage = ceph_usage()
    snap = dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                host=socket.gethostname(), **st, quota=usage,
                entries_under_run=claim.count_entries(qdir))
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, 'a') as fh:
        fh.write(json.dumps(snap, sort_keys=True) + '\n')
    print(json.dumps(snap, indent=2, sort_keys=True))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', default='prepare',
                    choices=['prepare', 'worker', 'worker-child', 'merge', 'seal', 'supervise',
                             'status'])
    ap.add_argument('--scope', default='full', choices=['full', 'primary', 'core',
                                                        'S_full', 'S_primary', 'S_core'])
    ap.add_argument('--queue', default=None)
    ap.add_argument('--freeze-json', default=None)
    ap.add_argument('--host-workers', type=int, default=8)
    ap.add_argument('--p-dt', type=int, default=None)
    ap.add_argument('--p-dt2', type=int, default=None)
    ap.add_argument('--estimate-seconds', type=float, default=None)
    ap.add_argument('--chunk-size', type=int, default=None)
    ap.add_argument('--interval', type=float, default=900.0,
                    help='supervisor progress-snapshot interval seconds (<= 3600)')
    ap.add_argument('--kernel', default='Haswell')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--scratch', action='store_true')
    args = ap.parse_args(argv)
    if args.mode == 'prepare':
        if args.estimate_seconds is None or args.estimate_seconds <= 0:
            raise SystemExit('--estimate-seconds must be positive (the graceful stop depends on it)')
        return prepare(args)
    return {'worker': worker, 'worker-child': worker_child, 'merge': merge,
            'seal': seal_run, 'supervise': supervise, 'status': status}[args.mode](args)


if __name__ == '__main__':
    sys.exit(main())
