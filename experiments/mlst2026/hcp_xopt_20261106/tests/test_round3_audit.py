"""Round-3 (resume 5) independent audit tests (Parts 1c/1d).

Written by the round-3 reviewer (self-review, author-authorized).  They exercise the
invariant, the scope guard, the claim queue, the A2 entry budget and the archive verifier
independently of the resume-4 test suite.
"""
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.mlst2026.hcp_xopt_20261106 import claim, family, report, run, scopes, seal

FIELDS = ('date', 'replicate', 'geometry', 'N', 'arm', 'm', 'D')


@pytest.fixture(scope='session')
def base():
    rows = []
    with open(run.FITS) as f:
        for line in f:
            r = json.loads(line)
            rows.append({k: r[k] for k in FIELDS})
    return rows


@pytest.fixture(scope='session')
def scoped_primary():
    return scopes.rows_for_scope('S_primary', run.FITS)


def _write_ledger(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _reduce(monkeypatch, tmp_path, rows, scope):
    for name in ('reproducibility', 'route_statistics', 'per_cell_arm_delta'):
        monkeypatch.setattr(report, name, lambda rows: {})
    p = tmp_path / 'ledger.jsonl'
    _write_ledger(p, rows)
    return report.main(str(p), str(tmp_path / 'ev.json'), scope=scope)['outcome']


# ---------------------------------------------------------------- Part 1c adversarial
def test_nonfinite_objective_ok_row_is_incomplete(monkeypatch, tmp_path, base):
    rows = [dict(x) for x in base]
    rows[0] = dict(rows[0], status='ok', delta={'D': 0.0, 'objective': float('inf')},
                   original={'D': rows[0]['D']})
    out = _reduce(monkeypatch, tmp_path, rows, 'full')
    assert out['family'] == 'INCOMPLETE'


def test_failed_row_with_full_census_is_incomplete(monkeypatch, tmp_path, base):
    rows = [dict(x) for x in base]
    for r in rows:
        r['status'] = 'ok'
        r['delta'] = {'D': 0.0, 'objective': 0.0}
        r['original'] = {'D': r['D']}
    rows[0]['status'] = 'failed'
    out = _reduce(monkeypatch, tmp_path, rows, 'full')
    assert out['family'] == 'INCOMPLETE'


def test_malformed_ok_row_missing_delta_is_incomplete(monkeypatch, tmp_path, base):
    rows = [dict(x) for x in base]
    for r in rows:
        r['status'] = 'ok'
        r['delta'] = {'D': 0.0, 'objective': 0.0}
        r['original'] = {'D': r['D']}
    del rows[0]['delta']
    out = _reduce(monkeypatch, tmp_path, rows, 'full')
    assert out['family'] == 'INCOMPLETE'


def test_scope_guard_primary_pz4_perturbation_is_robust(monkeypatch, tmp_path, scoped_primary, base):
    dm = {(r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m']): r['D'] for r in base}
    rows = []
    for r in scoped_primary:
        rows.append(dict(date=r['date'], replicate=r['replicate'], geometry=r['geometry'],
                         N=r['N'], arm=r['arm'], m=r['m'],
                         D=dm[(r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m'])],
                         status='ok', delta={'D': 0.0, 'objective': 0.0},
                         original={'D': 0.0}))
    i = next(i for i, r in enumerate(rows) if r['arm'] == 'lw')
    rows[i]['D'] = 1e12          # PZ4 interactions are SAMPLE-BASED in S_primary
    out = _reduce(monkeypatch, tmp_path, rows, 'primary')
    assert out['family'] == 'ROBUST' and out['n'] == 0


def test_inf_replication_guard_never_ok(tmp_path, monkeypatch):
    qdir = str(tmp_path / 'q')
    manifest = claim.prepare(qdir, 'full',
                             [dict(offset=0, date='d', geometry='A', N=56, arm='reference',
                                   m=None, replicate=0, date_index=0, theta=0.0)], chunk_size=1)
    cid = claim.chunk_ids(manifest)[0]
    assert claim.claim(qdir, cid, host='h', pid=1)['action'] == 'claimed'
    Path(qdir, 'STARTED.json').write_text(json.dumps(dict(stage='STARTED', scope='full',
               planned_rows=1, chunks=1, chunk_size=1, row_list_sha256=manifest['row_list_sha256'],
               deadline_epoch=time.time() + 600, kernel='Haswell')))
    monkeypatch.setattr(run.select, 'read_row', lambda *a, **k: dict(
        D=0.0, kernel={'selected': 0, 'restarts': [{'objective': 1.0}], 'selected_status': 'ok',
                       'optimization_limited': False, 'optimization_limited_t25_rule': False,
                       'restart_spread': 1.0}))
    monkeypatch.setattr(run.select, 'load_inputs', lambda d: ({}, None))
    monkeypatch.setattr(run, 'build', lambda *a: None)
    monkeypatch.setattr(run, 'process_row', lambda *a, **k: dict(
        D=0.0, frozen={'D': 0.0}, delta={'D': 0.0, 'objective': 0.0},
        replay=dict(obj_rel=float('inf'), D_abs=0.0, D_over_spread=0.0),
        strengthened={}, flags_after={}, seconds={}, F={}, U={}, D_scores={}))
    rc = run.run_chunk(SimpleNamespace(kernel='Haswell'), qdir, cid, 'h', 1, time.time() + 600, manifest)
    assert rc == 'guard'
    assert Path(qdir, 'STOP_FLAG.json').exists()


# ---------------------------------------------------------------- Part 1d claim queue
def _claim_worker(qdir, ids, out, idx):
    claimed = []
    while True:
        cid = run._claim_next(qdir, ids, f'host{idx}', os.getpid())
        if cid is None:
            break
        rows = claim.load_chunk_rows(qdir, cid, claim.load_manifest(qdir))
        claim.publish(qdir, cid, [dict(r, status='ok', D=0.0, delta={'D': 0.0, 'objective': 0.0})
                                  for r in rows])
        claimed.append(cid)
    Path(out).write_text(json.dumps(claimed))


def test_two_process_claim_simulation_no_double_claim(tmp_path):
    qdir = str(tmp_path / 'q')
    rows = [dict(offset=i, date='d%02d' % (i % 3), geometry='A', N=56, arm='reference',
                 m=None, replicate=i, date_index=0, theta=0.0) for i in range(12)]
    manifest = claim.prepare(qdir, 'full', rows, chunk_size=1)
    ids = claim.chunk_ids(manifest)
    outs = []
    procs = []
    for k in range(2):
        out = str(tmp_path / f'claimed_{k}.json')
        outs.append(out)
        p = mp.get_context('fork').Process(target=_claim_worker, args=(qdir, ids, out, k))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    a = json.loads(Path(outs[0]).read_text())
    b = json.loads(Path(outs[1]).read_text())
    assert len(set(a) & set(b)) == 0, 'a chunk was claimed twice'
    assert sorted(a + b) == sorted(ids), 'not every chunk was claimed exactly once'
    assert all(claim.is_done(qdir, cid) for cid in ids)


def test_two_chunk_dry_run_entry_budget(tmp_path, monkeypatch):
    qdir = str(tmp_path / 'q')
    rows = [dict(offset=i, date='d%02d' % i, geometry='A', N=56, arm='reference', m=None,
                 replicate=i, date_index=0, theta=0.0) for i in range(2)]
    manifest = claim.prepare(qdir, 'full', rows, chunk_size=1)
    Path(qdir, 'STARTED.json').write_text(json.dumps(dict(
        stage='STARTED', scope='full', planned_rows=2, chunks=2, chunk_size=1,
        row_list_sha256=manifest['row_list_sha256'], deadline_epoch=time.time() + 600,
        kernel='Haswell')))
    monkeypatch.setattr(run.select, 'read_row', lambda *a, **k: dict(
        D=0.0, kernel={'selected': 0, 'restarts': [{'objective': 1.0}], 'selected_status': 'ok',
                       'optimization_limited': False, 'optimization_limited_t25_rule': False,
                       'restart_spread': 1.0}))
    monkeypatch.setattr(run.select, 'load_inputs', lambda d: ({}, None))
    monkeypatch.setattr(run, 'build', lambda *a: None)
    monkeypatch.setattr(run, 'process_row', lambda *a, **k: dict(
        D=0.0, frozen={'D': 0.0}, delta={'D': 0.0, 'objective': 0.0},
        replay=dict(obj_rel=0.0, D_abs=0.0, D_over_spread=0.0),
        strengthened={}, flags_after={}, seconds={}, F={}, U={}, D_scores={}))
    args = SimpleNamespace(queue=qdir, kernel='Haswell', host_workers=1)

    def entries():
        out = set()
        for root, dirs, files in os.walk(qdir):
            for f in files:
                out.add(os.path.relpath(os.path.join(root, f), qdir))
        return out

    before = entries()
    assert run.worker_child(args) == 0
    new = entries() - before
    assert all(p.startswith(('claims/', 'results/', 'logs/')) for p in new), new
    assert len(new) <= 3 * 2 + 1, new          # claim dir + claim.json + result per chunk + index


def test_archive_tamper_rejected(tmp_path):
    qdir = str(tmp_path / 'q')
    rows = [dict(offset=0, date='d', geometry='A', N=56, arm='reference', m=None,
                 replicate=0, date_index=0, theta=0.0)]
    manifest = claim.prepare(qdir, 'full', rows, chunk_size=1)
    cid = claim.chunk_ids(manifest)[0]
    claim.publish(qdir, cid, [dict(r, status='ok', D=0.0, delta={'D': 0.0, 'objective': 0.0})
                              for r in claim.load_chunk_rows(qdir, cid)])
    claim.merge(qdir)
    members = seal._loose_members(qdir)
    blobs = {m: Path(qdir, m).read_bytes() for m in members}
    seal.archive_run(qdir)
    seal.verify_archive(qdir)
    target = members[0]
    blobs[target] = blobs[target] + b'TAMPER'
    with tarfile.open(Path(qdir, 'sealed', 'run_archive.tar'), 'w') as tar:
        for m in members:
            ti = tarfile.TarInfo(m)
            ti.size = len(blobs[m])
            tar.addfile(ti, io.BytesIO(blobs[m]))
    with pytest.raises(seal.SealError):
        seal.verify_archive(qdir)
