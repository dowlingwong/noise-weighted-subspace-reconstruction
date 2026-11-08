"""Shared, crash-safe claim queue for the two-host X-OPT resume-4 run (addendum A2 layout).

File-count-first layout (the /ceph quota binds on entries):

  run_20261108/
    queue/manifest.json           scope, chunk_size, planned_rows, chunk list (id, offset, nrows)
    queue/rows.jsonl              the deterministic scoped row list, one row per line
    claims/<chunk_id>/claim.json  the atomic claim (mkdir) + the in-place-rewritten heartbeat
    results/<chunk_id>.jsonl.gz   one gzipped result file per chunk (done == this file exists)
    logs/                         one log file per worker process / host
    reclaims.log                  every stale takeover, one line
    STARTED.json / COMPLETED.json / PARTIAL.json / STOP.json / sealed/

A chunk is claimed by an **atomic mkdir** of `claims/<chunk_id>/`; only one process can
create it.  `claim.json` holds host, pid, start time and heartbeat time; the heartbeat is
**rewritten in place** (write `claim.json.tmp`, then `os.replace`), not a separate file.
A chunk is done when its result file exists -- there is no DONE marker.

Stale-claim rule: a claim with heartbeat older than 2 h and no result file may be reclaimed
**once**; the old claim directory is renamed aside by an atomic `os.rename`, the takeover is
appended to `reclaims.log`, and the old directory removed.  A second takeover is refused.

`publish` validates that the result rows are exactly the chunk's scoped rows, in order, and
writes `<id>.tmp` then renames to `<id>.jsonl.gz`.  `merge` re-checks exactly-once across all
chunk files.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import socket
import time

CHUNK = 200
HEARTBEAT_SECONDS = 300
STALE_SECONDS = 2 * 3600
MAX_RECLAIMS = 1

IDENTITY = ('date', 'replicate', 'geometry', 'N', 'arm', 'm')


def _p(qdir, *parts):
    return os.path.join(qdir, *parts)


def identity(row):
    return tuple(row[k] for k in IDENTITY)


def chunk_id(i):
    return f'chunk_{i:05d}'


def prepare(qdir, scope, rows, chunk_size=CHUNK):
    """Write queue/rows.jsonl + queue/manifest.json.  Returns the manifest."""
    rows = list(rows)
    if len({identity(r) for r in rows}) != len(rows):
        raise ValueError('scoped row list contains duplicate identities')
    for d in ('queue', 'claims', 'results', 'logs'):
        os.makedirs(_p(qdir, d), exist_ok=True)
    chunks = []
    blob = bytearray()
    for i in range(0, len(rows), chunk_size):
        block = rows[i:i + chunk_size]
        cid = chunk_id(i // chunk_size)
        off = len(blob)
        for r in block:
            blob += (json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n').encode()
        chunks.append(dict(id=cid, offset=off, nrows=len(block)))
    with open(_p(qdir, 'queue', 'rows.jsonl'), 'wb') as fh:
        fh.write(bytes(blob))
    manifest = dict(scope=scope, chunk_size=chunk_size, planned_rows=len(rows),
                    n_chunks=len(chunks), chunks=chunks,
                    row_list_sha256=hashlib.sha256(bytes(blob)).hexdigest(),
                    created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with open(_p(qdir, 'queue', 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return manifest


def load_manifest(qdir):
    with open(_p(qdir, 'queue', 'manifest.json')) as fh:
        return json.load(fh)


def chunk_ids(manifest):
    return [c['id'] for c in manifest['chunks']]


def load_chunk_rows(qdir, cid, manifest=None):
    manifest = manifest or load_manifest(qdir)
    ent = next(c for c in manifest['chunks'] if c['id'] == cid)
    with open(_p(qdir, 'queue', 'rows.jsonl'), 'rb') as fh:
        fh.seek(ent['offset'])
        return [json.loads(fh.readline()) for _ in range(ent['nrows'])]


def result_path(qdir, cid):
    return _p(qdir, 'results', cid + '.jsonl.gz')


def is_done(qdir, cid):
    return os.path.exists(result_path(qdir, cid))


def _claim_dir(qdir, cid):
    return _p(qdir, 'claims', cid)


def _claim_file(qdir, cid):
    return _p(qdir, 'claims', cid, 'claim.json')


def _read_claim(qdir, cid):
    try:
        with open(_claim_file(qdir, cid)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_claim(qdir, cid, record):
    tmp = _claim_file(qdir, cid) + '.tmp.' + str(os.getpid())
    with open(tmp, 'w') as fh:
        json.dump(record, fh, sort_keys=True)
        fh.write('\n')
    os.replace(tmp, _claim_file(qdir, cid))


def _reclaim_count(qdir, cid):
    p = _p(qdir, 'reclaims.log')
    if not os.path.exists(p):
        return 0
    n = 0
    with open(p) as fh:
        for line in fh:
            if f' chunk={cid} ' in line:
                n += 1
    return n


def _log(qdir, msg):
    line = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg + '\n'
    fd = os.open(_p(qdir, 'reclaims.log'), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def claim(qdir, cid, host=None, pid=None, now=None):
    host = host or socket.gethostname()
    pid = pid or os.getpid()
    now = time.time() if now is None else now
    if is_done(qdir, cid):
        return dict(action='done', owner=_read_claim(qdir, cid))
    cdir = _claim_dir(qdir, cid)
    rec = dict(host=host, pid=pid, start=now, heartbeat=now)
    try:
        os.mkdir(cdir)
        _write_claim(qdir, cid, rec)
        return dict(action='claimed', owner=rec)
    except FileExistsError:
        pass
    if is_done(qdir, cid):
        return dict(action='done', owner=_read_claim(qdir, cid))
    old = _read_claim(qdir, cid)
    age = None if not old else now - float(old.get('heartbeat', 0))
    if age is None or age < STALE_SECONDS:
        return dict(action='busy', owner=old, age_seconds=age)
    if _reclaim_count(qdir, cid) >= MAX_RECLAIMS:
        return dict(action='exhausted', owner=old, age_seconds=age)
    trash = _p(qdir, 'claims', f'.stale-{cid}-{int(now)}-{pid}')
    try:
        os.rename(cdir, trash)
    except OSError:
        return dict(action='busy', owner=old, age_seconds=age)
    try:
        os.mkdir(cdir)
    except FileExistsError:
        return dict(action='busy', owner=_read_claim(qdir, cid))
    _write_claim(qdir, cid, rec)
    _log(qdir, f'RECLAIM chunk={cid} old_host={old.get("host")} old_pid={old.get("pid")} '
               f'new_host={host} new_pid={pid} age_seconds={round(age, 1)} '
               f'reason=no_heartbeat_gt_2h_and_no_result')
    shutil.rmtree(trash, ignore_errors=True)
    return dict(action='reclaimed', owner=old, reclaimed_from=old)


def heartbeat(qdir, cid, host=None, pid=None):
    """Rewrite claim.json in place with a fresh heartbeat.  False if we no longer own it."""
    host = host or socket.gethostname()
    pid = pid or os.getpid()
    rec = _read_claim(qdir, cid)
    if not rec or rec.get('host') != host or rec.get('pid') != pid:
        return False
    rec['heartbeat'] = time.time()
    _write_claim(qdir, cid, rec)
    return True


def still_owner(qdir, cid, host=None, pid=None):
    host = host or socket.gethostname()
    pid = pid or os.getpid()
    rec = _read_claim(qdir, cid)
    return bool(rec) and rec.get('host') == host and rec.get('pid') == pid


def publish(qdir, cid, records, manifest=None):
    """Atomically publish one chunk's gzipped result file and validate its identities."""
    want = [identity(r) for r in load_chunk_rows(qdir, cid, manifest)]
    got = [identity(r) for r in records]
    if got != want:
        raise ValueError(f'chunk {cid}: result identities do not match the scoped rows '
                         f'(want {len(want)}, got {len(got)}; '
                         f'missing={len(set(want) - set(got))} extra={len(set(got) - set(want))})')
    tmp = _p(qdir, 'results', cid + '.tmp')
    with gzip.open(tmp, 'wt') as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, result_path(qdir, cid))


def read_results(qdir, cid):
    out = []
    with gzip.open(result_path(qdir, cid), 'rt') as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def merge(qdir, manifest=None):
    """Validate completed chunks and write the merged gzipped ledger.

    Exactly-once: each result file must equal its chunk's row list in order; each chunk once;
    the global identity set must equal the scoped row list exactly.  `complete` is False if
    any chunk is missing.
    """
    manifest = manifest or load_manifest(qdir)
    problems = []
    merged_rel = _p(qdir, 'ledger_merged.jsonl.gz')
    tmp = merged_rel + '.tmp'
    seen = {}
    done_chunks = 0
    n_rows = n_ok = n_fail = 0
    with gzip.open(tmp, 'wt') as out:
        for ent in manifest['chunks']:
            cid = ent['id']
            if not is_done(qdir, cid):
                problems.append(dict(chunk=cid, problem='missing'))
                continue
            done_chunks += 1
            try:
                recs = read_results(qdir, cid)
            except (OSError, EOFError, ValueError) as exc:
                problems.append(dict(chunk=cid, problem=f'unreadable: {exc}'))
                continue
            want = [identity(r) for r in load_chunk_rows(qdir, cid, manifest)]
            got = [identity(r) for r in recs]
            if len(got) != len(set(got)):
                problems.append(dict(chunk=cid, problem='duplicate identity inside chunk'))
            if set(got) != set(want) or len(got) != len(want):
                problems.append(dict(chunk=cid, problem='result rows do not match chunk rows',
                                     want=len(want), got=len(got),
                                     missing=len(set(want) - set(got)),
                                     extra=len(set(got) - set(want))))
            for r in recs:
                k = identity(r)
                if k in seen:
                    problems.append(dict(chunk=cid, problem='identity seen twice across chunks',
                                         other=seen[k]))
                    continue
                seen[k] = cid
                out.write(json.dumps(r, sort_keys=True) + '\n')
                n_rows += 1
                if r.get('status') == 'ok':
                    n_ok += 1
                elif r.get('status') == 'failed':
                    n_fail += 1
    os.replace(tmp, merged_rel)
    sha = hashlib.sha256()
    with open(merged_rel, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            sha.update(chunk)
    return dict(scope=manifest['scope'], planned_rows=manifest['planned_rows'],
                n_chunks=manifest['n_chunks'], done_chunks=done_chunks,
                rows=n_rows, ok=n_ok, failed=n_fail,
                complete=(done_chunks == manifest['n_chunks'] and n_rows == manifest['planned_rows']
                          and not problems),
                problems=problems, merged_path=merged_rel, merged_sha256=sha.hexdigest(),
                row_list_sha256=manifest['row_list_sha256'])


def status(qdir, manifest=None):
    manifest = manifest or load_manifest(qdir)
    ids = chunk_ids(manifest)
    done = sum(1 for c in ids if is_done(qdir, c))
    claimed = sum(1 for c in ids if os.path.isdir(_claim_dir(qdir, c)) and not is_done(qdir, c))
    return dict(planned=manifest['n_chunks'], done=done, claimed=claimed,
                remaining=manifest['n_chunks'] - done,
                result_bytes=sum(os.path.getsize(result_path(qdir, c)) for c in ids
                                 if is_done(qdir, c)))


def count_entries(qdir):
    """Files + directories under qdir, recursively (the quota unit)."""
    n = 0
    for root, dirs, files in os.walk(qdir):
        n += len(dirs) + len(files)
    return n
