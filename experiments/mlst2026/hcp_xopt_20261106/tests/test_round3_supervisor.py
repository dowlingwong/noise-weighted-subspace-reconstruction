"""Round-3 (resume 5) supervisor tests — Part 1e.

The frozen code must contain an agent-free supervisor that, under tmux:
 (i) appends a progress snapshot at least hourly,
 (ii) keeps the local worker pool alive,
 (iii) applies the A2.e quota guard,
 (iv) stops claiming at the graceful-stop threshold,
 (v) when no chunk remains, merges, writes COMPLETED/PARTIAL/STOP and seals.

These tests fail before the supervisor existed (import/attribute error) and pass after.
"""
import gzip
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mlst2026.hcp_xopt_20261106 import claim, run, seal


def _rows(n):
    return [dict(offset=i, date='d%02d' % (i % 3), geometry='A', N=56, arm='reference',
                 m=None, replicate=i, date_index=0, theta=0.0) for i in range(n)]


FAKE_ORIGINAL = {'D': 0.0, 'kernel': {'selected': 0, 'restarts': [{'objective': 1.0}],
                 'selected_status': 'ok', 'optimization_limited': False,
                 'optimization_limited_t25_rule': False, 'restart_spread': 1.0}}


def _fake_result():
    return dict(D=0.0, frozen={'D': 0.0}, delta={'D': 0.0, 'objective': 0.0},
                replay=dict(obj_rel=0.0, D_abs=0.0, D_over_spread=0.0),
                strengthened={}, flags_after={}, seconds={}, F={}, U={}, D_scores={})


def _start_queue(tmp_path, rows_n=2, chunk_size=1, deadline_delta=600):
    qdir = str(tmp_path / 'q')
    manifest = claim.prepare(qdir, 'full', _rows(rows_n), chunk_size=chunk_size)
    started = dict(stage='STARTED', scope='full', planned_rows=manifest['planned_rows'],
                   chunks=manifest['n_chunks'], chunk_size=manifest['chunk_size'],
                   row_list_sha256=manifest['row_list_sha256'],
                   deadline_epoch=time.time() + deadline_delta,
                   graceful_stop_seconds=1.5 * deadline_delta, kernel='Haswell')
    Path(qdir, 'STARTED.json').write_text(json.dumps(started))
    return qdir, manifest, started


class _DoneProc:
    def poll(self):
        return 0


def _patch_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(run.select, 'read_row', lambda *a, **k: dict(FAKE_ORIGINAL))
    monkeypatch.setattr(run.select, 'load_inputs', lambda d: ({}, None))
    monkeypatch.setattr(run, 'build', lambda *a: None)
    monkeypatch.setattr(run, 'process_row', lambda *a, **k: _fake_result())
    monkeypatch.setattr(run, 'PROGRESS', str(tmp_path / 'progress.log'))
    monkeypatch.setattr(run.time, 'sleep', lambda s: None)
    monkeypatch.setattr(run, 'ceph_usage', lambda: dict(rentries=459000, max_files=500000,
                                                        free_entries=41000, rbytes=10 ** 12,
                                                        max_bytes=0, w2_rentries=10300))
    monkeypatch.setattr(run, 'getfattr_value', lambda a, p: None)


def _process_all(qdir, manifest):
    args = SimpleNamespace(kernel='Haswell')
    for cid in claim.chunk_ids(manifest):
        if claim.is_done(qdir, cid):
            continue
        res = claim.claim(qdir, cid, host='testhost', pid=os.getpid())
        if res['action'] not in ('claimed', 'reclaimed'):
            continue
        run.run_chunk(args, qdir, cid, 'testhost', os.getpid(), time.time() + 600, manifest)


def test_supervisor_progress_then_merge_and_seal(tmp_path, monkeypatch):
    qdir, manifest, started = _start_queue(tmp_path, rows_n=2, chunk_size=1)
    _patch_fast(monkeypatch, tmp_path)
    monkeypatch.setattr(run, '_spawn_local_pool',
                        lambda args: (_process_all(qdir, manifest), _DoneProc())[-1])
    args = SimpleNamespace(queue=qdir, host_workers=1, kernel='Haswell', interval=30)
    rc = run.supervise(args)
    assert rc == 0
    assert Path(qdir, 'COMPLETED.json').exists()
    assert json.loads(Path(qdir, 'COMPLETED.json').read_text())['stage'] == 'COMPLETED'
    assert Path(qdir, 'sealed', 'MANIFEST.json').exists()
    seal.verify_archive(qdir)
    snaps = [json.loads(l) for l in Path(tmp_path / 'progress.log').read_text().splitlines()]
    assert snaps, 'no progress snapshot written'
    for key in ('chunks_planned', 'done', 'remaining', 'done_per_host', 'reclaims',
                'loadavg', 'w2_entries', 'free_entries', 'free_bytes'):
        assert key in snaps[0], key
    assert snaps[-1]['tag'] == 'final'


def test_supervisor_graceful_stop_writes_partial_and_seals(tmp_path, monkeypatch):
    qdir, manifest, started = _start_queue(tmp_path, rows_n=2, chunk_size=1, deadline_delta=-5)
    _patch_fast(monkeypatch, tmp_path)
    monkeypatch.setattr(run, '_spawn_local_pool', lambda args: _DoneProc())
    args = SimpleNamespace(queue=qdir, host_workers=1, kernel='Haswell', interval=30)
    rc = run.supervise(args)
    assert rc != 0
    part = json.loads(Path(qdir, 'PARTIAL.json').read_text())
    assert part['stage'] == 'PARTIAL' and part['partial_reason'] == 'time_stop'
    assert Path(qdir, 'TIME_STOP.json').exists()
    assert Path(qdir, 'sealed', 'MANIFEST.json').exists()


def test_supervisor_quota_guard_stops_claiming(tmp_path, monkeypatch):
    qdir, manifest, started = _start_queue(tmp_path, rows_n=2, chunk_size=1)
    _patch_fast(monkeypatch, tmp_path)
    monkeypatch.setattr(run, 'ceph_usage', lambda: dict(rentries=499999, max_files=500000,
                                                        free_entries=1, rbytes=10 ** 12,
                                                        max_bytes=0, w2_rentries=499999))
    monkeypatch.setattr(run, '_spawn_local_pool', lambda args: _DoneProc())
    args = SimpleNamespace(queue=qdir, host_workers=1, kernel='Haswell', interval=30)
    rc = run.supervise(args)
    assert Path(qdir, 'QUOTA_STOP.json').exists()
    part = json.loads(Path(qdir, 'PARTIAL.json').read_text())
    assert part['partial_reason'] == 'quota_stop'
