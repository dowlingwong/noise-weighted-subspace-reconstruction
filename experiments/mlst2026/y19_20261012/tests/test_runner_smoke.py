"""Runner smoke tests on the T25 FIXTURE geometry (C = 3, T = 4) and on fabricated numbers; never G2."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.mlst2026.y19_20261012 import runner

EXP = Path(__file__).resolve().parents[4]


@pytest.fixture(scope='module')
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp('y19smoke') / 'run'
    summ = runner.cmd_smoke(out)
    return out, summ


def test_smoke_never_uses_g2():
    proto = runner.smoke_protocol()
    assert proto['generator']['family'] == 't25' and proto['namespace'] != 'y19-confirm-20261012'


def test_smoke_records_and_summary(smoke):
    out, summ = smoke
    proto = runner.smoke_protocol()
    n = sum(len(g['grid']) for g in proto['geometries']) * len(proto['seeds'])
    assert len(list((out / 'records').rglob('*.json'))) == n == summ['records']
    assert summ['status'] == 'COMPLETE' and summ['family_size'] == 20
    assert set(summ['readings']) == {'A', 'B', 'control'}
    assert summ['overall'] in ('MET', 'NOT MET')
    assert (out / summ['results_file']).is_file()
    r = json.loads(next((out / 'records').rglob('*.json')).read_text())
    for k in ('D', 'R_U', 'R_F', 'signal_U', 'signal_F', 'optimization_limited', 'restart_spread', 'training_sha', 'pool_sha'):
        assert k in r
    assert abs(r['D'] - (r['R_F'] - r['R_U'])) < 1e-12


def test_resume_and_determinism(smoke):
    out, _ = smoke
    p = runner.record_path(out, 'A', 1, 20)
    before = json.loads(p.read_text())
    mtimes = {q: q.stat().st_mtime_ns for q in (out / 'records').rglob('*.json')}
    os.remove(p)
    runner.cmd_smoke(out)                              # resumes: recomputes only the missing record
    after = json.loads(p.read_text())
    for q, t in mtimes.items():
        if q != p:
            assert q.stat().st_mtime_ns == t
    assert after['training_sha'] == before['training_sha'] and after['pool_sha'] == before['pool_sha']
    assert abs(after['D'] - before['D']) <= 1e-10


def _fake(proto, dvals, limited=()):
    recs = {}
    for g in proto['geometries']:
        for si in range(len(proto['seeds'])):
            for j, N in enumerate(g['grid']):
                m = dvals[g['label']][j]
                D = m + 0.01 * (si - 1)
                recs[(g['label'], si, N)] = {'D': D, 'R_U': 1.0, 'R_F': 1.0 + D, 'signal_U': 1.0, 'signal_F': 1.0,
                                             'optimization_limited': (g['label'], N) in limited and si == 0,
                                             'selected_is_hosvd_start': True, 'seconds': 0.0, 'restart_spread': 0.0}
    return recs


def test_reading_rule_logic():
    proto = runner.smoke_protocol()
    nc = len(next(g for g in proto['geometries'] if g['label'] == 'control')['grid'])
    met = {'A': [-3, -1, 0.0, 0.5, 0.7], 'B': [-3, -1, 0.3, 0.5, 0.7], 'control': [-1] * nc}
    s = runner.reduce(proto, None, _fake(proto, met))
    assert s['readings']['A']['reading'] == 'MET' and s['readings']['B']['reading'] == 'MET' and s['overall'] == 'MET'
    assert s['readings']['A']['pattern'] == '- - ? + +'
    assert s['readings']['control']['reading'] == 'MET'
    bad = dict(met, B=[-3, -1, 0.0, 0.0, 0.7], control=[-1] * (nc - 1) + [0.5])
    s = runner.reduce(proto, None, _fake(proto, bad))
    assert s['readings']['B']['reading'] == 'NOT MET' and s['overall'] == 'NOT MET'
    assert s['readings']['control']['reading'] == 'NOT MET'
    gA = next(g for g in proto['geometries'] if g['label'] == 'A')['grid']
    s = runner.reduce(proto, None, _fake(proto, met, limited={('A', gA[3])}))
    assert s['readings']['A']['reading'] == 'MET' and s['readings']['A']['deciding_cells_optimization_limited'] == [gA[3]]
    assert s['readings']['A']['flag']
    # secondary: alpha + beta/N fit recovers an exact law
    lawA = [0.8 * (1 - 40 / N) for N in gA]
    s = runner.reduce(proto, None, _fake(proto, dict(met, A=lawA)))
    assert abs(s['secondary_descriptive']['A']['N_star_fit'] - 40) < 1e-6


def test_kit_identity_and_dependencies_subprocess():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-m', 'experiments.mlst2026.y19_20261012.runner', 'verify-kit'],
                       cwd=EXP, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)['dependencies'] == 'ok'


def test_confirmation_preflight_subprocess():
    """The fail-closed preflight of `run` (kit, threads, dependencies, protocol, seeds, population
    recomputation) passes; it draws no record."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    code = ('import json; from experiments.mlst2026.y19_20261012 import runner as r; '
            'pf = r.preflight(*r.load_kit()); print(json.dumps({"ok": pf["prediction_recomputation"]["ok"], '
            '"max_diff": pf["prediction_recomputation"]["max_diff"], "kit": pf["kit_identity"]}))')
    res = subprocess.run([sys.executable, '-c', code], cwd=EXP, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.strip().splitlines()[-1])['ok']
    bad = dict(env, OPENBLAS_NUM_THREADS='4')
    res = subprocess.run([sys.executable, '-c', code], cwd=EXP, env=bad, capture_output=True, text=True)
    assert res.returncode != 0 and 'thread variables' in res.stderr
