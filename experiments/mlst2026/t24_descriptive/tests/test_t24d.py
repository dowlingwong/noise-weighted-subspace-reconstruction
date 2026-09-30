"""T24-D fixtures: synthetic headerless release with the seven registered test keys."""
import json, os, shutil
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.t24_descriptive import workflow as w, science as sc
from experiments.mlst2026.t24_cumulant import core

HEAVY = '35/57'


@pytest.fixture(autouse=True)
def threads(monkeypatch):
    for k in w.THREADS:
        monkeypatch.setenv(k, '1')


def make_release(root, units=sc.CONFIG['units'], n_noise=420, n_pulse=240, seed=5):
    root.mkdir(parents=True)
    g = np.random.default_rng(seed); t = np.arange(512)
    pulse = np.exp(-np.clip(t - 160, 0, None) / 60.) * (t >= 160)
    kernel = 0.6 ** np.arange(12)
    X = []; rows = []
    for u in units:
        run, ch = u.split('/')
        hours = g.permutation(n_noise + n_pulse) * 0.05
        for i in range(n_noise + n_pulse):
            e = g.normal(size=512 + 11)
            if u == HEAVY:  # per-record scale mixture (multivariate t, 4 dof): survives projection
                e = e * np.sqrt(4 / g.chisquare(4))
            x = np.convolve(e, kernel, mode='valid') + g.normal(scale=0.3)
            noise = i < n_noise
            if not noise:
                x = x + g.uniform(4, 9) * pulse
            X.append(x.astype('<f4')); rows.append((run, ch, hours[i], noise, True))
    np.asarray(X, '<f4').tofile(root / 'X_test.npy')
    with (root / 'features_test.csv').open('w') as f:
        f.write('run,channel,hours,noise,clean,pulse_height\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]:.6f},{r[3]},{r[4]},1.0\n')
    return root


@pytest.fixture(scope='module')
def done(tmp_path_factory):
    old = {k: os.environ.get(k) for k in w.THREADS}
    for k in w.THREADS:
        os.environ[k] = '1'
    base = tmp_path_factory.mktemp('t24d'); data = make_release(base / 'raw'); out = base / 'run'
    w.run(data, out)
    for k, v in old.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return data, out


def test_run_and_both_verifications(done):
    data, out = done
    assert w.verify(out)['mode'] == 'saved_arrays'
    assert w.verify(out, data)['mode'] == 'raw_rebuild'
    s = w.read(out / 'summary.json')['units']
    assert set(s) == set(sc.CONFIG['units'])
    for u in s.values():
        assert u['retained']['p'] == 6 and u['residual']['p'] == 32
        assert u['noise_evaluation'] == 126 and u['noise_calibration'] == 294


def test_reference_scale_behaviour(done):
    """Gaussian synthetic noise sits near the iid scale reference; heavy-tailed does not."""
    s = w.read(done[1] / 'summary.json')['units']
    for u, v in s.items():
        ratio = v['residual']['b2_ratio_to_reference_mean']
        assert (ratio > 1.3) if u == HEAVY else (0.85 < ratio < 1.15), (u, ratio)


def test_no_inferential_output(done):
    keys = set()
    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                keys.add(k.lower()); walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
    walk(w.read(done[1] / 'summary.json'))
    header = (done[1] / 'table.csv').read_text().splitlines()[0].lower()
    for word in ('p_value', 'pvalue', 'reject', 'holm', 'class', 'z_score', 'significan'):
        assert not any(word in k for k in keys) and word not in header


def test_tampering_detected(done, tmp_path):
    out = tmp_path / 'copy'; shutil.copytree(done[1], out)
    a = dict(np.load(out / 'unit_33_36.npz')); a['residual'] = a['residual'] * 1.001
    (out / 'unit_33_36.npz').unlink(); np.savez_compressed(out / 'unit_33_36.npz', **a)
    c = w.read(out / 'COMPLETE.json'); c['files'] = w.inventory(out); (out / 'COMPLETE.json').write_text(json.dumps(c))
    with pytest.raises(ValueError, match='summary not reproduced'):
        w.verify(out)
    (out / 'extra.txt').write_text('x'); c['files'] = w.inventory(out); (out / 'COMPLETE.json').write_text(json.dumps(c))
    with pytest.raises(ValueError, match='unexpected'):
        w.verify(out)


def test_raw_rebuild_detects_changed_data(done, tmp_path):
    data = tmp_path / 'raw'; shutil.copytree(done[0], data)
    b = bytearray((data / 'X_test.npy').read_bytes()); b[100] ^= 1; (data / 'X_test.npy').write_bytes(bytes(b))
    with pytest.raises(ValueError, match='raw data differ'):
        w.verify(done[1], data)


def test_gates(tmp_path, monkeypatch):
    data = make_release(tmp_path / 'raw', n_noise=60, n_pulse=40)
    with pytest.raises(ValueError, match='outside the source'):
        w.run(data, w.EXP / 'should_not_exist')
    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    with pytest.raises(ValueError, match='one numerical thread'):
        w.run(data, tmp_path / 'o')


def test_unit_set_must_match(tmp_path):
    data = make_release(tmp_path / 'raw', units=sc.CONFIG['units'] + ['35/99'], n_noise=60, n_pulse=40)
    with pytest.raises(ValueError, match='unit set differs'):
        w.run(data, tmp_path / 'o')
    assert (tmp_path / 'o' / 'FAILED.json').is_file()


def test_affine_invariance_and_basis_geometry():
    g = np.random.default_rng(1); y = g.normal(size=(200, 6)); A = g.normal(size=(6, 6)) + 3 * np.eye(6)
    s1, s2 = core.statistics(y), core.statistics(y @ A + 5.)
    assert np.isclose(s1['b2'], s2['b2']) and np.isclose(s1['b1'], s2['b1'])
    x = g.normal(size=(400, 512)); m, T = sc.whitening(x[:300], 0.1)
    R, S = sc.bases(x[300:] + 3 * np.sin(np.arange(512) / 40.), m, T, 6, 32, 'u', 'ns')
    assert np.allclose(R @ R.T, np.eye(6)) and np.allclose(S.T @ S, np.eye(32)) and np.abs(R @ S).max() < 1e-10


def test_evaluation_rows_do_not_enter_the_fit(done):
    """Scaling/perturbing every evaluation noise row leaves metric and bases bit-identical."""
    data, _ = done
    x, f = w.load(data, sc.CONFIG)
    x = np.array(x, dtype=np.float64)
    unit = '33/36'; a = sc.unit_arrays(x, f, unit, sc.CONFIG)
    y = x.copy(); y[a['noise_evaluation_ids']] = 50 * y[a['noise_evaluation_ids']] + np.random.default_rng(0).normal(size=(len(a['noise_evaluation_ids']), 512))
    b = sc.unit_arrays(y, f, unit, sc.CONFIG)
    for k in ('transform', 'retained_basis', 'residual_basis', 'noise_calibration_ids', 'pulse_calibration_ids'):
        assert np.array_equal(a[k], b[k]), k


def test_residual_block_stable_under_roundoff():
    """Degenerate circulant eigenpairs must not rotate the residual block."""
    g = np.random.default_rng(2); x = g.normal(size=(600, 512)); p = g.normal(size=(200, 512)) + np.sin(np.arange(512) / 30.)
    m1, t1 = sc.whitening(x, 0.1)
    m2 = m1 * (1 + 1e-15 * g.normal(size=m1.shape)); m2 = (m2 + m2.T) / 2
    vals, vecs = np.linalg.eigh(m2); t2 = (vecs * np.sqrt(vals)) @ vecs.T
    r1, s1 = sc.bases(p, m1, t1, 6, 32, 'u', 'ns'); r2, s2 = sc.bases(p, m2, t2, 6, 32, 'u', 'ns')
    cos = np.linalg.svd(s1.T @ s2, compute_uv=False)
    assert cos.min() > 1 - 1e-8
