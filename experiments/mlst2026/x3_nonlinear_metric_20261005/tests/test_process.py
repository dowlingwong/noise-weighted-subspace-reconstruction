"""End-to-end fixture lifecycle and adversarial checks (development namespace only)."""
import json
import os
import shutil

import numpy as np
import pytest

from experiments.mlst2026.x3_nonlinear_metric_20261005 import workflow as w
from experiments.mlst2026.x3_nonlinear_metric_20261005.protocol import BASE, FIXTURE, config

THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


@pytest.fixture(scope='module')
def fixture_run(tmp_path_factory):
    old = {k: os.environ.get(k) for k in THREADS}
    os.environ.update(THREADS)
    try:
        work = tmp_path_factory.mktemp('x3') / 'work'
        yield work, w.fixture_mode(work)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_fixture_lifecycle_completes_and_audits(fixture_run):
    work, result = fixture_run
    assert result['audit']['verified'] and result['audit']['cells'] == 6
    red = json.loads((work / 'run' / 'reduction.json').read_text())
    assert red['family_size'] == 2 * 18 and red['development_only'] is True
    end = json.loads((work / 'run' / 'VERIFY.END.json').read_text())
    assert end == {'cells_reevaluated': 6, 'training_replay_cells': 2}
    f = json.loads((work / 'freeze.json').read_text())
    assert f['environment']['torch'].split('+')[0] == BASE['torch_version']


def copy_run(work, tmp_path, name):
    dest = tmp_path / name
    shutil.copytree(work / 'run', dest)
    return dest


def rehash(d):
    (d / 'COMPLETE.json').write_text(json.dumps({'arrays_sha': w.sha(d / 'arrays.npz'), 'record_sha': w.sha(d / 'record.json')}))


def test_tampered_record_is_rejected(fixture_run, tmp_path):
    run = copy_run(fixture_run[0], tmp_path, 'r')
    p = run / 'cell-0002' / 'record.json'
    r = json.loads(p.read_text())
    r['models']['deep_silu__sigma_inverse']['clean_whitened'] -= 1.0
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='tampered'):
        w.audit(run, tmp_path / 'a1')
    rehash(run / 'cell-0002')
    with pytest.raises(ValueError, match='numerical reconstruction mismatch'):
        w.audit(run, tmp_path / 'a2')


def test_swapped_weights_are_rejected(fixture_run, tmp_path):
    run = copy_run(fixture_run[0], tmp_path, 'w')
    d = run / 'cell-0004'
    a = dict(np.load(d / 'arrays.npz'))
    silu = sorted((k for k in a if k.startswith('deep_silu__sigma_inverse__')), key=lambda k: list(a).index(k))
    lin = sorted((k for k in a if k.startswith('deep_linear__sigma_inverse__')), key=lambda k: list(a).index(k))
    for ks, kl in zip(silu, lin):                      # same shapes in the same order
        a[ks] = a[kl]
    (d / 'arrays.npz').unlink()
    with (d / 'arrays.npz').open('xb') as fh:
        np.savez_compressed(fh, **a)
    rehash(d)
    with pytest.raises(ValueError):
        w.audit(run, tmp_path / 'a3')


def test_forged_selection_and_stopping_records_are_rejected(fixture_run, tmp_path):
    run = copy_run(fixture_run[0], tmp_path, 's')
    p = run / 'cell-0001' / 'record.json'
    r = json.loads(p.read_text())
    t = r['training']['deep_linear__isotropic']
    t['selected'] = (t['selected'] + 1) % 4
    p.write_text(json.dumps(r))
    rehash(run / 'cell-0001')
    with pytest.raises(ValueError, match='selection'):
        w.audit(run, tmp_path / 'a4')
    run2 = copy_run(fixture_run[0], tmp_path, 's2')
    p = run2 / 'cell-0001' / 'record.json'
    r = json.loads(p.read_text())
    run0 = r['training']['deep_linear__isotropic']['restarts'][0]
    run0['stopped'] = 'patience' if run0['stopped'] == 'max_epochs' else 'max_epochs'
    p.write_text(json.dumps(r))
    rehash(run2 / 'cell-0001')
    with pytest.raises(ValueError, match='early-stopping'):
        w.audit(run2, tmp_path / 'a5')


def test_missing_ledger_is_rejected(fixture_run, tmp_path):
    run = copy_run(fixture_run[0], tmp_path, 'l')
    (run / '0003.END.json').unlink()
    with pytest.raises(ValueError):
        w.audit(run, tmp_path / 'a6')


def test_production_freeze_refuses_without_acceptance(tmp_path, monkeypatch):
    for k, v in THREADS.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match='acceptance'):
        w.freeze(BASE, tmp_path / 'freeze.json')


def test_fixture_export_forbidden_and_config_closed(fixture_run, tmp_path):
    with pytest.raises(ValueError, match='fixture'):
        w.export(fixture_run[0] / 'run', tmp_path / 'export')
    p = tmp_path / 'bad.json'
    p.write_text(json.dumps(dict(BASE, recipe=dict(BASE['recipe'], lr=3e-3))))
    with pytest.raises(ValueError, match='registered'):
        config(p)
