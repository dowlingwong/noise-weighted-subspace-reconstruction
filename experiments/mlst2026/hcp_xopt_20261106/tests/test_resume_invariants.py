"""Adversarial ingestion / worker / freeze controls for the resume-4 invariant.

Every test here is a *regression specification*: it failed before the Step D fixes and must
pass after them.  The invariant under test: a non-finite, missing, duplicated or failed value
can only produce a counted failure and the label INCOMPLETE -- never a decision, a verdict
label or an interval.
"""
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.mlst2026.hcp_xopt_20261106 import claim, family, report, run, scopes, select
from experiments.mlst2026.hcp_xopt_20261106.config import ROOT

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
def scoped_core():
    return scopes.rows_for_scope('S_core', run.FITS)


def _write_ledger(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(dict(r, status='ok', original={'D': r['D']},
                                    delta={'D': 0.0, 'objective': 0.0})) + '\n')


def _reduce(monkeypatch, tmp_path, rows, scope):
    for name in ('reproducibility', 'route_statistics', 'per_cell_arm_delta'):
        monkeypatch.setattr(report, name, lambda rows: {})
    p = tmp_path / 'ledger.jsonl'
    _write_ledger(p, rows)
    out = report.main(str(p), str(tmp_path / 'evidence.json'), scope=scope)
    return out['outcome']


# ---------------------------------------------------------------- ingestion invariant
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_ok_d_never_decision(monkeypatch, tmp_path, base, value):
    rows = [dict(x) for x in base]
    rows[0]['D'] = value
    out = _reduce(monkeypatch, tmp_path, rows, 'full')
    assert out['family'] == 'INCOMPLETE'
    assert 'decision' not in out and 'changes' not in out


def test_missing_row_is_incomplete(monkeypatch, tmp_path, base):
    rows = [dict(x) for x in base]
    rows.pop(0)
    assert _reduce(monkeypatch, tmp_path, rows, 'full')['family'] == 'INCOMPLETE'


def test_duplicate_row_is_incomplete(monkeypatch, tmp_path, base):
    rows = [dict(x) for x in base]
    rows.append(dict(rows[0]))
    assert _reduce(monkeypatch, tmp_path, rows, 'full')['family'] == 'INCOMPLETE'


def test_hole_in_pz4_cell_is_incomplete(monkeypatch, tmp_path, base):
    rows = [dict(x) for x in base]
    rows.pop(next(i for i, r in enumerate(rows) if r['geometry'] == 'A' and r['arm'] == 'lw'))
    assert _reduce(monkeypatch, tmp_path, rows, 'full')['family'] == 'INCOMPLETE'


def test_hole_outside_34_is_incomplete(monkeypatch, tmp_path, base):
    rows = [dict(x) for x in base]
    rows.pop(next(i for i, r in enumerate(rows) if r['geometry'] == 'control' and r['arm'] == 'lw'))
    assert _reduce(monkeypatch, tmp_path, rows, 'full')['family'] == 'INCOMPLETE'


@pytest.mark.parametrize('kind', ['pz4', 'nonfinite'])
def test_family_direct_guard(base, kind):
    ledger = family._ingest([(*[r[k] for k in ('date', 'replicate', 'geometry', 'N', 'arm', 'm')],
                             r['D']) for r in base])
    key = next(k for k in ledger if k[4] == 'lw' and k[2] == 'A')
    if kind == 'nonfinite':
        ledger[key] = float('nan')
    else:
        del ledger[key]
    dates = sorted({k[0] for k in ledger})
    with pytest.raises(family.IncompleteLedger):
        family.compute_cells(ledger, dates, family.REFIT_BY_SCOPE['full'])


# ---------------------------------------------------------------- verdicts
def test_end_to_end_dry_run_chosen_scope_robust(monkeypatch, tmp_path, base):
    out = _reduce(monkeypatch, tmp_path, base, 'full')
    assert out['family'] == 'ROBUST' and out['n'] == 0
    assert out['changes'] == []


def test_one_finite_d_across_boundary_sensitivity(monkeypatch, tmp_path, base):
    rows = [dict(x) for x in base]
    i = next(i for i, r in enumerate(rows) if r['geometry'] == 'control' and r['arm'] == 'reference')
    rows[i]['D'] += 1e8
    out = _reduce(monkeypatch, tmp_path, rows, 'full')
    assert out['family'] == 'SENSITIVE'
    assert any(c['cell'].startswith('control/') for c in out['changes'])


def test_scope_guard_sampled_cell_never_changes_verdict(monkeypatch, tmp_path, scoped_core, base):
    # S_core: only the 15 reference cells are refit; isotropic and estimated arms are sampled.
    dm = {(r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m']): r['D'] for r in base}
    rows = [dict(date=r['date'], replicate=r['replicate'], geometry=r['geometry'], N=r['N'],
                 arm=r['arm'], m=r['m'],
                 D=dm[(r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m'])])
            for r in scoped_core]
    iso = next(i for i, r in enumerate(rows) if r['arm'] == 'isotropic')
    rows[iso]['D'] = 1e12
    out = _reduce(monkeypatch, tmp_path, rows, 'core')
    assert out['family'] == 'ROBUST' and out['n'] == 0


# ---------------------------------------------------------------- freeze STOP path
@pytest.mark.parametrize('content', [None, '{}', 'not json', '[]'])
def test_invalid_or_missing_freeze_writes_stop(monkeypatch, tmp_path, content):
    p = tmp_path / 'FREEZE.json'
    if content is not None:
        p.write_text(content)
    args = SimpleNamespace(out=str(tmp_path / 'out'), queue=str(tmp_path / 'q'),
                           freeze_json=str(p), limit=None, estimate_seconds=1.0,
                           scope='full', scratch=False, kernel='Haswell', dry_run=False)
    assert run.prepare(args) != 0
    assert (Path(args.queue) / 'STOP.json').exists()
    assert not (Path(args.queue) / 'STARTED.json').exists()


def test_missing_freeze_writes_stop(monkeypatch, tmp_path):
    args = SimpleNamespace(out=str(tmp_path / 'out'), queue=str(tmp_path / 'q'),
                           freeze_json=str(tmp_path / 'nope.json'), limit=None,
                           estimate_seconds=1.0, scope='full', scratch=False,
                           kernel='Haswell', dry_run=False)
    assert run.prepare(args) != 0
    assert (Path(args.queue) / 'STOP.json').exists()


# ---------------------------------------------------------------- worker paths
def _scratch_queue(tmp_path, n_rows=2):
    rows = run.build_scope_rows('full')[:n_rows]
    qdir = str(tmp_path / 'q')
    manifest = claim.prepare(qdir, 'full', rows, chunk_size=n_rows)
    cid = claim.chunk_ids(manifest)[0]
    res = claim.claim(qdir, cid, host='testhost', pid=4242)
    assert res['action'] == 'claimed'
    started = dict(stage='STARTED', scope='full', planned_rows=manifest['planned_rows'],
                   chunks=manifest['n_chunks'], chunk_size=manifest['chunk_size'],
                   deadline_epoch=time.time() + 600, kernel='Haswell')
    with open(Path(qdir) / 'STARTED.json', 'w') as fh:
        json.dump(started, fh)
    return qdir, cid, manifest


def _wargs():
    return SimpleNamespace(kernel='Haswell', queue=None, host_workers=1)


def test_worker_read_row_failure_is_counted(monkeypatch, tmp_path):
    qdir, cid, manifest = _scratch_queue(tmp_path)
    monkeypatch.setattr(select, 'read_row', lambda *a, **k: (_ for _ in ()).throw(OSError('injected')))
    rc = run.run_chunk(_wargs(), qdir, cid, 'testhost', 4242, time.time() + 600, manifest)
    assert rc == 'done'
    recs = claim.read_results(qdir, cid)
    assert len(recs) == 2
    assert all(r['status'] == 'failed' for r in recs)
    assert all(r['error']['traceback_sha256'] for r in recs)
    assert claim.is_done(qdir, cid)


def test_replication_guard_nan_never_ok(monkeypatch, tmp_path):
    qdir, cid, manifest = _scratch_queue(tmp_path)
    monkeypatch.setattr(select, 'read_row', lambda *a, **k: {'D': 0.0,
                        'kernel': {'selected': 0, 'restarts': [{'objective': 1.0}],
                                   'selected_status': 'x', 'optimization_limited': False,
                                   'optimization_limited_t25_rule': False, 'restart_spread': 1.0}})
    monkeypatch.setattr(run, 'process_row', lambda *a, **k: dict(
        D=float('nan'), frozen={'D': 0.0}, delta={'D': float('nan'), 'objective': 0.0},
        replay=dict(obj_rel=float('nan'), D_abs=float('nan'), D_over_spread=float('nan')),
        strengthened={}, flags_after={}, seconds={}, F={}, U={}, D_scores={}))
    rc = run.run_chunk(_wargs(), qdir, cid, 'testhost', 4242, time.time() + 600, manifest)
    recs = claim.read_results(qdir, cid)
    assert rc == 'done'
    assert all(r['status'] == 'failed' for r in recs)


# ---------------------------------------------------------------- work root
def test_workroot_imports():
    import sys
    for name, mod in list(sys.modules.items()):
        if name.startswith('experiments.mlst2026.') and getattr(mod, '__file__', None):
            assert Path(mod.__file__).resolve().is_relative_to(Path(ROOT) / 'EXP')
