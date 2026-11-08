"""Claim-queue and seal controls for the two-host resume-4 run (addendum A2 layout).

Spec: atomic mkdir claims; in-place heartbeat rewrite keeps a claim fresh; a stale claim
(heartbeat > 2 h, no result file) is reclaimed exactly once; a chunk with a result file is
never rerun; the merge rejects a missing or duplicated row; archive verification fails on a
tampered member; during a 2-chunk dry run no file is created outside claims/, results/,
logs/ and node-local /tmp.
"""
import json
import multiprocessing as mp
import os
import tarfile
import time
from pathlib import Path

import pytest

from experiments.mlst2026.hcp_xopt_20261106 import claim, run, seal


def _row(i):
    return dict(offset=i, date='d%02d' % (i % 3), geometry='A', N=56, arm='reference',
                m=None, replicate=i, date_index=0, theta=0.0)


def _mkqueue(tmp_path, n_rows=6, chunk_size=2):
    qdir = str(tmp_path / 'q')
    manifest = claim.prepare(qdir, 'full', [_row(i) for i in range(n_rows)],
                             chunk_size=chunk_size)
    return qdir, manifest


def _fake_result(meta):
    return dict(D=0.0, frozen={'D': 0.0}, delta={'D': 0.0, 'objective': 0.0},
                replay=dict(obj_rel=0.0, D_abs=0.0, D_over_spread=0.0),
                strengthened={}, flags_after={}, seconds={}, F={}, U={}, D_scores={})


def _publish_all(qdir, manifest):
    for cid in claim.chunk_ids(manifest):
        rows = claim.load_chunk_rows(qdir, cid, manifest)
        claim.publish(qdir, cid, [dict(_row(0), **r, status='ok')
                                  for r in rows], manifest)


def _race_claim(qdir, cid, outpath):
    res = claim.claim(qdir, cid, host='racehost', pid=os.getpid())
    Path(outpath).write_text(res['action'])


def test_two_processes_never_claim_the_same_chunk(tmp_path):
    qdir, manifest = _mkqueue(tmp_path, n_rows=8, chunk_size=1)
    for cid in claim.chunk_ids(manifest):
        outs = []
        procs = []
        for k in range(2):
            p = str(tmp_path / f'{cid}_{k}.txt')
            outs.append(p)
            pr = mp.get_context('fork').Process(target=_race_claim, args=(qdir, cid, p))
            pr.start()
            procs.append(pr)
        for pr in procs:
            pr.join()
        actions = sorted(Path(p).read_text() for p in outs)
        assert actions.count('claimed') == 1, (cid, actions)


def test_heartbeat_in_place_keeps_claim_fresh(tmp_path):
    qdir, manifest = _mkqueue(tmp_path)
    cid = claim.chunk_ids(manifest)[0]
    assert claim.claim(qdir, cid, host='h', pid=7)['action'] == 'claimed'
    p = Path(qdir, 'claims', cid, 'claim.json')
    rec = json.loads(p.read_text())
    rec['heartbeat'] = time.time() - 3 * 3600
    p.write_text(json.dumps(rec))
    assert claim.claim(qdir, cid, host='other', pid=8)['action'] == 'reclaimed'
    assert claim.heartbeat(qdir, cid, host='other', pid=8) is True
    fresh = json.loads(p.read_text())
    assert time.time() - fresh['heartbeat'] < 60
    assert claim.claim(qdir, cid, host='third', pid=9)['action'] == 'busy'


def test_stale_claim_reclaimed_exactly_once(tmp_path):
    qdir, manifest = _mkqueue(tmp_path)
    cid = claim.chunk_ids(manifest)[0]
    assert claim.claim(qdir, cid, host='dead', pid=1)['action'] == 'claimed'
    p = Path(qdir, 'claims', cid, 'claim.json')
    rec = json.loads(p.read_text())
    rec['heartbeat'] = time.time() - 3 * 3600
    p.write_text(json.dumps(rec))
    assert claim.claim(qdir, cid, host='new', pid=2)['action'] == 'reclaimed'
    rec = json.loads(p.read_text())
    rec['heartbeat'] = time.time() - 3 * 3600
    p.write_text(json.dumps(rec))
    assert claim.claim(qdir, cid, host='third', pid=3)['action'] == 'exhausted'
    log = Path(qdir, 'reclaims.log').read_text().splitlines()
    assert sum(1 for ln in log if f'chunk={cid}' in ln) == 1


def test_chunk_with_result_file_is_never_rerun(tmp_path):
    qdir, manifest = _mkqueue(tmp_path)
    cid = claim.chunk_ids(manifest)[0]
    rows = claim.load_chunk_rows(qdir, cid, manifest)
    claim.publish(qdir, cid, [dict(_row(0), **r, status='ok') for r in rows], manifest)
    before = os.path.getmtime(claim.result_path(qdir, cid))
    assert claim.claim(qdir, cid, host='x', pid=1)['action'] == 'done'
    assert os.path.getmtime(claim.result_path(qdir, cid)) == before


def test_merge_rejects_missing_row(tmp_path):
    qdir, manifest = _mkqueue(tmp_path, n_rows=4, chunk_size=2)
    _publish_all(qdir, manifest)
    ok = claim.merge(qdir, manifest)
    assert ok['complete'] and ok['rows'] == 4
    p = Path(claim.result_path(qdir, claim.chunk_ids(manifest)[0]))
    import gzip
    with gzip.open(p, 'rt') as fh:
        lines = fh.read().splitlines()
    with gzip.open(p, 'wt') as fh:
        fh.write('\n'.join(lines[:-1]) + '\n')
    bad = claim.merge(qdir, manifest)
    assert any('do not match' in pr['problem'] for pr in bad['problems'])


def test_merge_rejects_duplicated_row(tmp_path):
    qdir, manifest = _mkqueue(tmp_path, n_rows=4, chunk_size=2)
    _publish_all(qdir, manifest)
    p = Path(claim.result_path(qdir, claim.chunk_ids(manifest)[0]))
    import gzip
    with gzip.open(p, 'rt') as fh:
        lines = fh.read().splitlines()
    with gzip.open(p, 'wt') as fh:
        fh.write('\n'.join(lines + [lines[0]]) + '\n')
    bad = claim.merge(qdir, manifest)
    assert any('duplicate' in pr['problem'] for pr in bad['problems'])


def test_merge_requires_all_chunks(tmp_path):
    qdir, manifest = _mkqueue(tmp_path, n_rows=4, chunk_size=2)
    cid = claim.chunk_ids(manifest)[0]
    rows = claim.load_chunk_rows(qdir, cid, manifest)
    claim.publish(qdir, cid, [dict(_row(0), **r, status='ok') for r in rows], manifest)
    bad = claim.merge(qdir, manifest)
    assert not bad['complete']
    assert any(pr['problem'] == 'missing' for pr in bad['problems'])


def test_archive_verification_fails_on_tampered_member(tmp_path):
    qdir, manifest = _mkqueue(tmp_path, n_rows=2, chunk_size=1)
    _publish_all(qdir, manifest)
    claim.merge(qdir, manifest)
    # capture member bytes before sealing deletes the loose files
    members = seal._loose_members(qdir)
    blobs = {rel: open(os.path.join(qdir, rel), 'rb').read() for rel in members}
    rec = seal.archive_run(qdir)
    assert rec['n_members'] == len(members)
    seal.verify_archive(qdir)  # clean archive verifies
    tar_path = Path(qdir, 'sealed', 'run_archive.tar')
    target = members[0]
    blobs[target] = blobs[target] + b'TAMPER'
    with tarfile.open(tar_path, 'w') as tar:
        for rel in members:
            import io
            ti = tarfile.TarInfo(rel)
            ti.size = len(blobs[rel])
            tar.addfile(ti, io.BytesIO(blobs[rel]))
    with pytest.raises(seal.SealError):
        seal.verify_archive(qdir)


def test_no_files_outside_queue_dirs_in_2chunk_dry_run(tmp_path, monkeypatch):
    qdir, manifest = _mkqueue(tmp_path, n_rows=2, chunk_size=1)
    started = dict(stage='STARTED', scope='full', planned_rows=2, chunks=2,
                   chunk_size=1, row_list_sha256=manifest['row_list_sha256'],
                   deadline_epoch=time.time() + 600, kernel='Haswell')
    Path(qdir, 'STARTED.json').write_text(json.dumps(started))
    monkeypatch.setattr(run.select, 'read_row', lambda *a, **k: {})
    monkeypatch.setattr(run.select, 'load_inputs', lambda d: ({}, None))
    monkeypatch.setattr(run, 'build', lambda *a: None)
    monkeypatch.setattr(run, 'process_row', lambda *a, **k: _fake_result(None))
    before = set()
    for root, dirs, files in os.walk(qdir):
        for f in files:
            before.add(os.path.relpath(os.path.join(root, f), qdir))
    args = type('A', (), dict(queue=qdir, kernel='Haswell', host_workers=1))()
    rc = run.worker_child(args)
    assert rc == 0
    after = set()
    for root, dirs, files in os.walk(qdir):
        for f in files:
            after.add(os.path.relpath(os.path.join(root, f), qdir))
    new = after - before
    allowed = ('claims/', 'results/', 'logs/', 'queue/')
    outside = [p for p in new if not p.startswith(allowed)]
    assert not outside, outside
