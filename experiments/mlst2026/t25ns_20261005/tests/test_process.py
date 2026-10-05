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
    assert red['development_only'] is True and red['family_size'] == 4 * 31 + 2 * 2
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


def test_export_tables_and_records_are_bound_by_the_audit(fixture_run, tmp_path, monkeypatch):
    for k, v in THREADS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(w, 'preflight_space', lambda *a: None)   # disk-size gate is the host's concern
    work, _ = fixture_run
    out = tmp_path / 'export'
    w.export(work / 'run', out, allow_development=True)
    assert w.audit(out, tmp_path / 'ok')['export_checked'] is True
    forged = tmp_path / 'forged'
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
    gone = tmp_path / 'gone'
    shutil.copytree(out, gone)
    (gone / 'SHA256SUMS').unlink()
    with pytest.raises(ValueError, match='incomplete export'):
        w.audit(gone, tmp_path / 'bad2')


def test_consistent_restart_swap_passes_portable_audit_but_not_host_replay(fixture_run, tmp_path, monkeypatch):
    for k, v in THREADS.items():
        monkeypatch.setenv(k, v)
    work, _ = fixture_run
    run = copy_run(work, tmp_path, 'swap')
    d = run / 'cell-0000'
    tag = 'oracle__restricted'
    a = dict(np.load(d / 'arrays.npz'))
    for part in ('uc', 'ut'):
        a[f'{tag}_{part}_0'], a[f'{tag}_{part}_1'] = a[f'{tag}_{part}_1'].copy(), a[f'{tag}_{part}_0'].copy()
    (d / 'arrays.npz').unlink()
    with (d / 'arrays.npz').open('xb') as fh:
        np.savez_compressed(fh, **a)
    r = json.loads((d / 'record.json').read_text())
    opt = r['methods'][tag]['optimization']
    opt['restarts'][0], opt['restarts'][1] = opt['restarts'][1], opt['restarts'][0]
    opt['selected'] = {0: 1, 1: 0}.get(opt['selected'], opt['selected'])
    rr = r['methods'][tag]['restart_clean_risks']
    rr[0], rr[1] = rr[1], rr[0]
    (d / 'record.json').write_text(json.dumps(r, indent=2, sort_keys=True))
    (d / 'COMPLETE.json').write_text(json.dumps({'arrays_sha': w.sha(d / 'arrays.npz'), 'record_sha': w.sha(d / 'record.json')}))
    assert w.audit(run, tmp_path / 'portable')['verified']
    with pytest.raises(ValueError):
        w.verify(run, require_complete=True, replay_optimizer=True)


def test_stale_acceptance_is_refused():
    s = w.source_files()
    with pytest.raises(ValueError, match='stale'):
        w.production_gate({'verdict': w.VERDICT, 'deployment': w.VERSION, 'code_identity': '0' * 64}, {}, s, w.environment())


def _shift_restricted(refs, delta, flip_status=False):
    import copy
    refs = copy.deepcopy(refs)
    for row in refs['population_penalties']:
        row['oracle_risk']['restricted'] += delta
        for k in row:
            if k.startswith('penalty_'):
                row[k]['restricted'] += delta
        if flip_status:
            for name in row['restricted_fit_status']:
                row['restricted_fit_status'][name] = 'other-start-' + str(row['restricted_fit_status'][name])
    return refs


def test_portable_audit_tolerates_tied_population_restarts(fixture_run, tmp_path, monkeypatch):
    """Another BLAS may select another tied start in a population restricted fit (risk shifts ~1e-6):
    the portable audit records the difference; the host keeps the exact comparison."""
    for k, v in THREADS.items():
        monkeypatch.setenv(k, v)
    work, _ = fixture_run
    real = w.science.population_references
    monkeypatch.setattr(w.science, 'population_references', lambda c: _shift_restricted(real(c), 5e-7, True))
    result = w.audit(work / 'run', tmp_path / 'tied')
    assert result['verified'] and 4e-7 < result['population_restricted_max_abs_difference'] < 6e-7
    assert result['population_restricted_status_changes']
    with pytest.raises(ValueError):
        w.verify(work / 'run', require_complete=True, replay_optimizer=True)
    monkeypatch.setattr(w.science, 'population_references', lambda c: _shift_restricted(real(c), 1e-3))
    with pytest.raises(ValueError, match='beyond tolerance'):
        w.audit(work / 'run', tmp_path / 'far')


def test_portable_audit_reports_calibration_digest_mismatch(fixture_run, tmp_path, monkeypatch):
    """A one-ulp change in the regenerated calibration pool (another BLAS) changes its digests: reported
    off the host, refused on the host."""
    for k, v in THREADS.items():
        monkeypatch.setenv(k, v)
    work, _ = fixture_run
    real = w.science.covariance_estimates
    monkeypatch.setattr(w.science, 'covariance_estimates', lambda c, pool, g: real(c, np.nextafter(pool, np.inf), g))
    result = w.audit(work / 'run', tmp_path / 'ulp')
    assert result['verified'] and len(result['calibration_digest_mismatches']) == 12
    with pytest.raises(ValueError):
        w.verify(work / 'run', require_complete=True, replay_optimizer=True)
