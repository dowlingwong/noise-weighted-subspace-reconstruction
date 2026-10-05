"""End-to-end fixture lifecycle and adversarial checks (development namespace only)."""
import json
import os
import shutil

import numpy as np
import pytest

from experiments.mlst2026.t25ns_20261005 import workflow as w
from experiments.mlst2026.t25ns_20261005.protocol import BASE, FIXTURE, config

THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


@pytest.fixture(scope='module')
def fixture_run(tmp_path_factory):
    old = {k: os.environ.get(k) for k in THREADS}
    os.environ.update(THREADS)
    try:
        work = tmp_path_factory.mktemp('x2') / 'work'
        result = w.fixture_mode(work)
        yield work, result
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_fixture_lifecycle_completes_and_audits(fixture_run):
    work, result = fixture_run
    assert result['audit']['verified'] and result['audit']['cells'] == 12
    red = json.loads((work / 'run' / 'reduction.json').read_text())
    assert red['development_only'] is True and red['family_size'] == 4 * 27 + 2 * 2
    assert sum(red['domain_census'].values()) == 12 * 4
    assert set(red['predictions']) >= {'P1', 'P2', 'P3', 'E2'}
    f = json.loads((work / 'freeze.json').read_text())
    assert f['status'] == 'fixture_only' and f['t25_guard']['accepted_t25_identity'].startswith('01f2f82f')
    assert w.PROTOCOL_MS in f['sources']['MS_files']


def copy_run(work, tmp_path, name):
    dest = tmp_path / name
    shutil.copytree(work / 'run', dest)
    return dest


def test_tampered_record_is_rejected(fixture_run, tmp_path):
    work, _ = fixture_run
    run = copy_run(work, tmp_path, 'record')
    p = run / 'cell-0003' / 'record.json'
    r = json.loads(p.read_text())
    r['methods']['oracle__restricted']['clean'] += 1.0
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='tampered'):
        w.audit(run, tmp_path / 'a1')


def test_consistently_forged_record_is_rejected(fixture_run, tmp_path):
    work, _ = fixture_run
    run = copy_run(work, tmp_path, 'forged')
    d = run / 'cell-0005'
    r = json.loads((d / 'record.json').read_text())
    r['methods']['kron_64__unrestricted']['clean'] *= 0.5
    (d / 'record.json').write_text(json.dumps(r, indent=2, sort_keys=True))
    (d / 'COMPLETE.json').write_text(json.dumps({'arrays_sha': w.sha(d / 'arrays.npz'), 'record_sha': w.sha(d / 'record.json')}))
    with pytest.raises(ValueError, match='numerical reconstruction mismatch'):
        w.audit(run, tmp_path / 'a2')


def test_forged_arrays_are_rejected(fixture_run, tmp_path):
    work, _ = fixture_run
    run = copy_run(work, tmp_path, 'arrays')
    d = run / 'cell-0001'
    a = dict(np.load(d / 'arrays.npz'))
    a['oracle__restricted_B'] = a['oracle__unrestricted_B']
    (d / 'arrays.npz').unlink()
    with (d / 'arrays.npz').open('xb') as fh:
        np.savez_compressed(fh, **a)
    (d / 'COMPLETE.json').write_text(json.dumps({'arrays_sha': w.sha(d / 'arrays.npz'), 'record_sha': w.sha(d / 'record.json')}))
    with pytest.raises(ValueError):
        w.audit(run, tmp_path / 'a3')


def test_missing_ledger_and_unexpected_entries_are_rejected(fixture_run, tmp_path):
    work, _ = fixture_run
    run = copy_run(work, tmp_path, 'ledger')
    (run / '0002.END.json').unlink()
    with pytest.raises(ValueError):
        w.audit(run, tmp_path / 'a4')
    run2 = copy_run(work, tmp_path, 'extra')
    (run2 / 'stray.txt').write_text('x')
    with pytest.raises(ValueError, match='unexpected entries'):
        w.audit(run2, tmp_path / 'a5')


def test_reduction_must_match_recomputation(fixture_run, tmp_path):
    work, _ = fixture_run
    run = copy_run(work, tmp_path, 'reduction')
    p = run / 'reduction.json'
    red = json.loads(p.read_text())
    red['predictions']['P1']['verdict'] = 'HOLDS'
    p.write_text(json.dumps(red))
    (run / 'REDUCE.END.json').write_text(json.dumps({'reduction_sha256': w.sha(p)}))
    with pytest.raises(ValueError):
        w.audit(run, tmp_path / 'a6')


def test_production_freeze_refuses_without_acceptance_and_qualification(tmp_path, monkeypatch):
    for k, v in THREADS.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match='acceptance'):
        w.freeze(BASE, tmp_path / 'freeze.json')
    assert not (tmp_path / 'freeze.json').exists()


def test_fixture_export_forbidden_and_config_closed(fixture_run, tmp_path):
    work, _ = fixture_run
    with pytest.raises(ValueError, match='fixture'):
        w.export(work / 'run', tmp_path / 'export')
    bad = dict(BASE, rho=[0.25, 0.75])
    p = tmp_path / 'bad.json'
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='registered'):
        config(p)
    p.write_text(json.dumps(FIXTURE))
    assert config(p) == FIXTURE


def test_outputs_inside_source_trees_are_refused():
    with pytest.raises(ValueError, match='inside source'):
        w.external(w.ROOT / 'tmp_x2_output')
