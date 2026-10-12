"""Runner: synthetic smoke end to end, resume determinism, kit identity, preflight refusals."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.mlst2026.meg_z_20261013 import common, fixture, runner as R, secondary_d8
from experiments.mlst2026.meg_z_20261013.preprocess import load_subject
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol

EXP = Path(__file__).resolve().parents[4]
OVR = {'resamples': 1, 'signal': {'theta': [0.0, 0.25]}, 'grid': {'control_absolute': [30, 60, 120]}}


@pytest.fixture(scope='module')
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp('megz_smoke')
    summ = R.cmd_smoke(out, 1, OVR)
    return out, summ


def smoke_proto():
    p = fixture.smoke_protocol(load_protocol())
    for k, v in OVR.items():
        p[k] = dict(p[k], **v) if isinstance(v, dict) else v
    return p


def test_smoke_complete(smoke):
    out, summ = smoke
    assert summ['status'] == 'COMPLETE'
    start = json.loads((out / 'run' / 'START.json').read_text())
    recs = list((out / 'run' / 'records').rglob('*.json'))
    assert len(recs) == start['expected_records'] == summ['records'] > 0
    assert set(summ['readings']) == {'PZ1', 'PZ2', 'PZ3', 'PZ4', 'PZ5', 'descriptive_iso_class_contrast'}
    assert summ['family_size'] == sum(len(g['grid']) for g in json.loads((out / 'predictions.json').read_text())['subjects']['sub-91']['geometries'].values()) + 6
    r = json.loads(recs[0].read_text())
    assert set(r['arms']) == {'ref', 'lw_lo', 'lw_hi', 'kron_lo', 'kron_hi', 'iso'} and r['geometry'] in ('control', 'A', 'B')
    assert r['metric_diagnostics']['lw_hi']['m'] == 256 and r['metric_diagnostics']['lw_lo']['m'] == 64
    for a in r['arms'].values():
        assert abs(a['D'] - (a['F']['clean_ref'] - a['U']['clean_ref'])) < 1e-12
    assert 'dev_power' in summ and summ['dev_power']['cells']
    opt = summ['optimization']
    assert set(opt['per_arm']) == set(r['arms']) and opt['max_iter_decision'].startswith(('raise', 'keep'))
    for a in opt['per_arm'].values():
        assert sum(a['selected_status_counts'].values()) == summ['records'] and a['flags_refined_rule'] <= summ['records']
    assert all('optimization_limited_t25_rule' in a for a in r['arms'].values())
    assert summ['readings']['PZ1']['reading'] == 'MET'


def test_resume_is_deterministic(smoke):
    out, _ = smoke
    run = out / 'run'
    recs = sorted((run / 'records').rglob('*.json'))
    p = recs[len(recs) // 2]
    before = json.loads(p.read_text())
    mt = {q: q.stat().st_mtime_ns for q in recs}
    os.remove(p)
    proto = smoke_proto()
    R.cmd_run(run, out / 'derived', out / 'predictions.json', out / 'gates.json', proto['subjects']['dev'], 1, proto=proto, smoke=True)
    after = json.loads(p.read_text())
    for q, tns in mt.items():
        if q != p:
            assert q.stat().st_mtime_ns == tns
    assert after['pool_sha'] == before['pool_sha'] and after['training_noise_index_sha'] == before['training_noise_index_sha']
    for arm in before['arms']:
        assert abs(after['arms'][arm]['D'] - before['arms'][arm]['D']) <= 1e-10
    with pytest.raises(R.KitMismatch):          # a different resample count in the same OUT is refused
        R.cmd_run(run, out / 'derived', out / 'predictions.json', out / 'gates.json', proto['subjects']['dev'], 1, resamples=2,
                  proto=proto, smoke=True)


def test_secondary_d8_on_fixture(smoke):
    out, _ = smoke
    sd = load_subject(out / 'derived' / 'sub-91.h5', cache=False)
    r = secondary_d8.subject_d8(sd, smoke_proto())
    for k in ('decoding_accuracy_U', 'decoding_accuracy_F', 'decoding_accuracy_raw_whitened_pca6'):
        assert 0.0 <= r[k] <= 1.0
    assert r['decoding_accuracy_F'] > 0.6                                     # face trials carry 2x the evoked field
    with pytest.raises(SystemExit, match='frozen protocol'):                  # F7
        secondary_d8.main(['--derived', str(out / 'derived'), '--subjects', 'sub-12', '--out', str(out / 'd8.json')])
    assert r['heldout_weighted_residual_U'] > 0 and r['heldout_weighted_residual_F'] > 0


def _args(out):
    proto = smoke_proto()
    return proto, json.loads((out / 'predictions.json').read_text()), json.loads((out / 'gates.json').read_text())


def test_preflight_refusals(smoke, tmp_path):
    out, _ = smoke
    proto, pred, g = _args(out)
    der = out / 'derived'
    dev, conf = proto['subjects']['dev'], proto['subjects']['confirmation']
    ok = R.preflight(proto, pred, g, der, dev, smoke=True)
    assert ok['mode'] == 'dev' and ok['E1']['included'] == dev
    with pytest.raises(R.KitMismatch, match='without --confirmation'):
        R.preflight(proto, pred, g, der, conf, smoke=True)
    with pytest.raises(R.KitMismatch, match='frozen protocol'):
        R.preflight(proto, pred, g, der, conf, confirmation=True, smoke=True)
    fake = dict(proto, frozen=True, status='FROZEN (fake)')
    with pytest.raises(R.KitMismatch, match='frozen protocol'):                # no protocol.json in the kit
        R.preflight(fake, pred, g, der, conf, confirmation=True, protocol_sha256='0' * 64, smoke=True)
    with pytest.raises(R.KitMismatch, match='no confirmation subject'):
        R.preflight(proto, pred, g, der, dev, confirmation=True, smoke=True)
    with pytest.raises(R.KitMismatch, match='not analysed'):
        R.preflight(proto, pred, g, der, ['sub-01'], smoke=True)
    other = dict(proto, version='changed')
    with pytest.raises(R.KitMismatch, match='different protocol'):
        R.preflight(other, pred, g, der, dev, smoke=True)
    # tampered derived file
    d2 = tmp_path / 'derived'
    shutil.copytree(der, d2)
    with open(d2 / 'sub-92.h5', 'ab') as f:
        f.write(b'\0')
    with pytest.raises(R.KitMismatch, match='SHA-256 differs'):
        R.preflight(proto, pred, g, d2, dev, smoke=True)
    # window budget too small for the grid
    small = json.loads(json.dumps(pred))
    small['subjects']['sub-91']['pools']['training'] = 10
    with pytest.raises(R.KitMismatch, match='budget'):
        R.preflight(proto, small, g, der, dev, smoke=True)
    inel = json.loads(json.dumps(pred))
    inel['subjects']['sub-92']['exclusion_reasons'] = ['A: no sign change']
    with pytest.raises(R.KitMismatch, match='population rule'):                # a dev subject without theta_s root
        R.preflight(proto, inel, g, der, dev, smoke=True)
    # gates computed on another derived file
    g2 = json.loads(json.dumps(g))
    g2['subjects']['sub-93']['derived_file_sha256'] = 'x'
    with pytest.raises(R.KitMismatch, match='gates.json'):
        R.preflight(proto, pred, g2, der, dev, smoke=True)
    with pytest.raises(R.KitMismatch, match='outside the kit'):
        R.cmd_run(R.PACKAGE / 'out', der, out / 'predictions.json', out / 'gates.json', dev, 1, proto=proto, smoke=True)


def test_preflight_threads_and_kit_subprocess(smoke):
    out, _ = smoke
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    code = ('import json; from experiments.mlst2026.meg_z_20261013 import runner as r; '
            'r.preflight(r.load_protocol(), {}, {}, None, [])')
    res = subprocess.run([sys.executable, '-c', code], cwd=EXP, env=env, capture_output=True, text=True)
    assert res.returncode != 0 and 'thread variables' in res.stderr
    env['OPENBLAS_NUM_THREADS'] = '1'
    r = subprocess.run([sys.executable, '-m', 'experiments.mlst2026.meg_z_20261013.runner', 'verify-kit'], cwd=EXP, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)['dependencies'] == 'ok'


def test_kit_tamper_detected(tmp_path):
    k = R.verify_kit()
    copy = tmp_path / R.PACKAGE.name
    shutil.copytree(R.PACKAGE, copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert R.verify_kit(copy)['kit_identity'] == k['kit_identity']
    p = copy / 'protocol_dev.json'
    p.write_text(p.read_text().replace('"resamples": 20', '"resamples": 21'))
    with pytest.raises(R.KitMismatch):
        R.verify_kit(copy)


def test_namespaces():
    proto = load_protocol()
    assert proto['frozen'] is False and proto['namespace_confirmation'] is None
    assert proto['namespace'] == common.NAMESPACE_DEV != common.NAMESPACE_SMOKE
    for prior in ('y19-confirm-20261012', 'zc-confirm-20261013'):
        assert proto['namespace'] != prior
