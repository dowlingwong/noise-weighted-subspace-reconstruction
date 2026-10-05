"""End-to-end fixture lifecycle and adversarial checks (development namespace only)."""
import json
import os
import shutil

import numpy as np
import pytest
from numpy.testing import assert_allclose

from experiments.mlst2026.x3_nonlinear_metric_20261005 import science
from experiments.mlst2026.x3_nonlinear_metric_20261005 import workflow as w
from experiments.mlst2026.x3_nonlinear_metric_20261005.protocol import BASE, FIXTURE, config

THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                            'VECLIB_MAXIMUM_THREADS')}


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


def test_tied_linear_closed_form_uses_the_training_split(fixture_run):
    """Every class sees the same N - floor(0.2 N) training records; the closed form never sees validation."""
    work, _ = fixture_run
    f = json.loads((work / 'freeze.json').read_text())
    c = f['config']
    for i, cell in enumerate(f['enumeration']['cells'][:2]):
        d = work / 'run' / f'cell-{i:04d}'
        a = dict(np.load(d / 'arrays.npz'))
        r = json.loads((d / 'record.json').read_text())
        g, sigma, L, Li, train, ec, en = science.make_data(c, cell['seed'])
        n_tr, n_val = science.split(cell['N'], c['recipe']['validation_fraction'])
        assert (r['n_train'], r['n_validation']) == (n_tr, n_val) and n_val == cell['N'] // 5
        _, mu, S = science.moments(train, n_tr)
        _, muN, SN = science.moments(train, cell['N'])
        for loss, W in (('isotropic', np.eye(g.d)), ('sigma_inverse', Li @ Li)):
            A, b = science.closed_form(c, S, mu, W, c['k'])
            assert_allclose(a[f'tied_linear__{loss}__A'], A, rtol=0, atol=1e-12)
            assert_allclose(a[f'tied_linear__{loss}__b'], b, rtol=0, atol=1e-12)
            AN, _ = science.closed_form(c, SN, muN, W, c['k'])
            assert not np.allclose(a[f'tied_linear__{loss}__A'], AN, atol=1e-6)


def test_export_tables_and_records_are_bound_by_the_audit(fixture_run, tmp_path, monkeypatch):
    for k, v in THREADS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(w, 'preflight_space', lambda *a: None)   # disk-size gate is the host's concern
    work, _ = fixture_run
    out = tmp_path / 'export'
    w.export(work / 'run', out, allow_development=True)
    assert w.audit(out, tmp_path / 'ok')['export_checked'] is True
    forged = tmp_path / 'forged'                                  # verdict flipped, checksums regenerated
    shutil.copytree(out, forged)
    p = forged / 'tables' / 'predictions.json'
    pred = json.loads(p.read_text())
    pred['P1']['verdict'] = 'HOLDS' if pred['P1']['verdict'] != 'HOLDS' else 'FAILS'
    p.unlink()
    p.write_text(json.dumps(pred))
    (forged / 'SHA256SUMS').unlink()
    (forged / 'SHA256SUMS').write_text(w.sha256sums(forged))
    with pytest.raises(ValueError, match='tables differ'):
        w.audit(forged, tmp_path / 'bad1')
    csvf = tmp_path / 'csv'                                       # one table value changed
    shutil.copytree(out, csvf)
    q = csvf / 'tables' / 'P1_metric_effect.csv'
    lines = q.read_text().splitlines()
    cells = lines[1].split(',')
    cells[2] = repr(float(cells[2]) - 1.0)
    lines[1] = ','.join(cells)
    q.unlink()
    q.write_text('\n'.join(lines) + '\n')
    (csvf / 'SHA256SUMS').unlink()
    (csvf / 'SHA256SUMS').write_text(w.sha256sums(csvf))
    with pytest.raises(ValueError, match='tables differ'):
        w.audit(csvf, tmp_path / 'bad2')
    rec = tmp_path / 'record'                                     # export record identity zeroed
    shutil.copytree(out, rec)
    e = json.loads((rec / 'EXPORT.json').read_text())
    e['code_identity'] = '0' * 64
    (rec / 'EXPORT.json').unlink()
    (rec / 'EXPORT.json').write_text(json.dumps(e))
    (rec / 'SHA256SUMS').unlink()
    (rec / 'SHA256SUMS').write_text(w.sha256sums(rec))
    with pytest.raises(ValueError, match='export record'):
        w.audit(rec, tmp_path / 'bad3')
    gone = tmp_path / 'gone'                                      # checksums removed
    shutil.copytree(out, gone)
    (gone / 'SHA256SUMS').unlink()
    with pytest.raises(ValueError, match='incomplete export'):
        w.audit(gone, tmp_path / 'bad4')


def test_forged_data_digest_is_reported_by_the_portable_audit(fixture_run, tmp_path):
    run = copy_run(fixture_run[0], tmp_path, 'dg')
    d = run / 'cell-0000'
    r = json.loads((d / 'record.json').read_text())
    r['training_sha'] = '0' * 64
    (d / 'record.json').write_text(json.dumps(r))
    rehash(d)
    result = w.audit(run, tmp_path / 'a7')
    assert result['data_digest_mismatches'] == [json.loads((run / 'freeze.json').read_text())['enumeration']['cells'][0]]
    with pytest.raises(ValueError, match='seed/data'):
        w.verify(run, require_complete=True, replay=True)


def test_stale_acceptance_is_refused():
    s = w.source_files()
    with pytest.raises(ValueError, match='stale'):
        w.production_gate({'verdict': w.VERDICT, 'deployment': w.VERSION, 'code_identity': '0' * 64}, {}, s, w.environment())


def test_thread_gate_requires_veclib_single_thread(monkeypatch):
    for k, v in THREADS.items():
        monkeypatch.setenv(k, v)
    w.thread_gate()
    monkeypatch.delenv('VECLIB_MAXIMUM_THREADS')
    with pytest.raises(ValueError, match='VECLIB'):
        w.thread_gate()
