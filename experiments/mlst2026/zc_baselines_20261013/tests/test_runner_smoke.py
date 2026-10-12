"""Runner on the T25 FIXTURE geometry (C = 3, T = 4; smoke namespace) and on fabricated records."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.mlst2026.zc_baselines_20261013 import runner, design

EXP = Path(__file__).resolve().parents[4]
PKG = 'experiments.mlst2026.zc_baselines_20261013'


@pytest.fixture(scope='module')
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp('zcsmoke') / 'run'
    summ = runner.cmd_smoke(out)
    return out, summ


def test_smoke_never_uses_confirmation_namespace():
    p = design.smoke_protocol()
    assert p['namespace'] == design.SMOKE_NAMESPACE != design.NAMESPACE
    assert p['generator']['C'] == 3 and p['generator']['T'] == 4
    assert not set(p['seeds']) & set(design.seeds())


def test_smoke_records_and_summary(smoke):
    out, summ = smoke
    p = design.smoke_protocol()
    n = len(p['seeds']) * len(p['theta']) * len(p['N']) * 5
    assert len(list((out / 'records').rglob('*.json'))) == n == summ['records'] == 60
    assert summ['status'] == 'COMPLETE'
    assert summ['family_sizes'] == {'R-C1': 1, 'R-C2': 12}
    assert set(summ['readings']) == {'R-C1', 'R-C2'}
    assert all(v['reading'] in ('MET', 'NOT MET') for v in summ['readings'].values())
    assert (out / summ['results_file']).is_file()


def test_record_arithmetic(smoke):
    out, _ = smoke
    p = design.smoke_protocol()
    for f in (out / 'records').rglob('*.json'):
        r = json.loads(f.read_text())
        c, g = r['clean'], r['regret_vs_seed_min']
        best = min(c['U'], c['F'])
        assert abs(r['D_F_minus_U'] - (c['F'] - c['U'])) < 1e-12
        assert c['CV'] == c[r['cv']['choice']]
        for m, v in c.items():
            if v is None:
                assert g[m] is None
            else:
                assert g[m] == v - best
        assert min(g['U'], g['F']) == 0.0 and g['CV'] >= 0
        if r['metric'] in p['metrics']['all_methods']:
            assert c['RULE'] == c[r['rule']['choice']]
            assert r['rule']['choice'] == ('F' if r['theta'] == 0.0 or r['N'] < 60 else 'U')
            sh = r['shr']
            assert sh['lambda_hat'] in sh['lambdas'] and sh['path_clean'][sh['lambdas'].index(sh['lambda_hat'])] == c['SHR']
            assert abs(sh['path_clean'][0] - c['U']) <= 1e-12 * abs(c['U'])        # lambda = 0 is U
        else:
            assert r['metric'].startswith('full') and c['RULE'] is None and c['SHR'] is None and r['shr'] is None


def test_metric_arms_distinct_and_paired(smoke):
    out, _ = smoke
    rs = {m: json.loads(runner.record_path(out, 0.3, 40, m, 0).read_text()) for m in ('oracle', 'kron32', 'kron64', 'full32')}
    assert len({r['metric_cov_sha'] for r in rs.values()}) == 4
    assert len({r['training_sha'] for r in rs.values()}) == 1 and len({r['folds_sha'] for r in rs.values()}) == 1


def test_resume_and_determinism(smoke):
    out, _ = smoke
    p = runner.record_path(out, 0.3, 80, 'kron32', 1)
    before = json.loads(p.read_text())
    mtimes = {q: q.stat().st_mtime_ns for q in (out / 'records').rglob('*.json')}
    os.remove(p)
    runner.cmd_smoke(out)                              # resumes: recomputes only the missing record
    after = json.loads(p.read_text())
    for q, t in mtimes.items():
        if q != p:
            assert q.stat().st_mtime_ns == t
    for k in ('training_sha', 'calibration_sha', 'metric_cov_sha', 'folds_sha', 'clean', 'cv', 'shr', 'rule'):
        assert after[k] == before[k], k


def test_parallel_equals_serial(smoke, tmp_path):
    out, _ = smoke
    out2 = tmp_path / 'par'
    runner.cmd_smoke(out2, workers=2)
    for f in (out / 'records').rglob('*.json'):
        a, b = json.loads(f.read_text()), json.loads((out2 / f.relative_to(out)).read_text())
        assert a['clean'] == b['clean'] and a['cv'] == b['cv'] and a['shr'] == b['shr']


# ------------------------------------------------------------------------------ reading rules (fabricated)
def _fake(proto, rc1=0.0, rc2=0.0, rc1_cell=None, rc2_cell=None, spread=0.01, limited=()):
    """Records with prescribed seed-mean contrasts: RULE - CV = rc1 (+ noise) and SHR - better = rc2 (+ noise)."""
    recs = {}
    ms = proto['metrics']['all_methods'] + proto['metrics']['cv_select_only']
    for th in proto['theta']:
        for N in proto['N']:
            for m in ms:
                for si in range(len(proto['seeds'])):
                    e = spread * (si - (len(proto['seeds']) - 1) / 2)
                    U, F = 10.0, 10.5                        # U better in every cell
                    d1 = (rc1 if rc1_cell in (None, (th, N)) else 0.0) + e
                    d2 = (rc2 if rc2_cell in (None, (th, N, m)) else 0.0) + e
                    allm = m in proto['metrics']['all_methods']
                    clean = {'U': U, 'F': F, 'CV': U, 'RULE': (U + d1) if allm else None, 'SHR': (U + d2) if allm else None}
                    recs[(th, N, m, si)] = {
                        'clean': clean, 'regret_vs_seed_min': {k: (v - U if v is not None else None) for k, v in clean.items()},
                        'cv': {'choice': 'U'}, 'rule': {'choice': 'U', 'N_star_pred': 1.0} if allm else None,
                        'shr': {'lambda_hat': 0.3, 'lambdas': [0.0, 0.3], 'path_clean': [U, U + d2]} if allm else None,
                        'optimization': {'optimization_limited': (th, N, m) in limited and si == 0,
                                         'cv_fold_F_nonstationary': 0}, 'seconds': 0.0}
    return recs


def test_reading_rules_on_fabricated_inputs():
    p = design.smoke_protocol()
    s = runner.reduce(p, _fake(p, rc1=-1.0, rc2=-1.0))
    assert s['readings']['R-C1']['reading'] == 'MET' and s['readings']['R-C1']['decisions']['negative'] == 1
    assert s['readings']['R-C2']['reading'] == 'MET' and s['readings']['R-C2']['decisions']['negative'] == 12
    s = runner.reduce(p, _fake(p, rc1=0.0, rc2=0.0))           # undecided counts as "not larger"
    assert s['readings']['R-C1']['reading'] == 'MET' and s['readings']['R-C1']['decisions']['undecided'] == 1
    assert s['readings']['R-C2']['reading'] == 'MET'
    s = runner.reduce(p, _fake(p, rc1=+1.0))
    assert s['readings']['R-C1']['reading'] == 'NOT MET' and s['readings']['R-C1']['decided_positive_cells'] == [[0.3, 40, 'oracle']]
    s = runner.reduce(p, _fake(p, rc2=+1.0, rc2_cell=(0.0, 80, 'kron64')))
    assert s['readings']['R-C2']['reading'] == 'NOT MET'
    assert s['readings']['R-C2']['decided_positive_cells'] == [[0.0, 80, 'kron64']]
    s = runner.reduce(p, _fake(p, rc1=5.0, spread=0.0))        # zero SE -> undecided
    assert s['families']['R-C1'][0]['zero_SE'] and s['readings']['R-C1']['reading'] == 'MET'
    s = runner.reduce(p, _fake(p, rc1=-1.0, limited={(0.3, 40, 'oracle')}))
    assert s['readings']['R-C1']['reading'] == 'MET' and s['readings']['R-C1']['flag']
    recs = _fake(p)
    del recs[(0.3, 40, 'oracle', 0)], recs[(0.3, 40, 'oracle', 1)], recs[(0.3, 40, 'oracle', 2)]
    s = runner.reduce(p, recs, partial=True)
    assert s['readings']['R-C1']['reading'] == 'INCOMPLETE'


def test_bonferroni_interval_and_better_class():
    from scipy.stats import t
    v = [0.1, 0.4, -0.2, 0.3, 0.05]
    r = runner.interval(v, 4, 0.05)
    half = t.ppf(1 - 0.05 / 8, 4) * np.std(v, ddof=1) / np.sqrt(5)
    assert np.allclose(r['ci'], [np.mean(v) - half, np.mean(v) + half])
    rs = [{'clean': {'U': 1.0, 'F': 0.5}}, {'clean': {'U': 1.0, 'F': 2.0}}]
    assert runner.better_class(rs) == 'U'                       # means 1.0 vs 1.25
    p = design.smoke_protocol()
    s = runner.reduce(p, _fake(p, rc2=-1.0))
    assert all(r['better_class'] == 'U' for r in s['families']['R-C2'])


# ------------------------------------------------------------------------------ kit identity and preflight
def _env(**kw):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    e.update(kw)
    return e


def test_verify_kit_subprocess():
    r = subprocess.run([sys.executable, '-m', PKG + '.runner', 'verify-kit'], cwd=EXP, env=_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)['dependencies'] == 'ok'


PF = ('import json; from experiments.mlst2026.zc_baselines_20261013 import runner as r; '
      'pf = r.preflight(r.load_protocol()); print(json.dumps({"kit": pf["kit_identity"]}))')


def test_preflight_passes_subprocess():
    res = subprocess.run([sys.executable, '-c', PF], cwd=EXP, env=_env(), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    kit = json.loads((EXP / 'experiments/mlst2026/zc_baselines_20261013/KIT_SHA256.json').read_text())
    assert json.loads(res.stdout.strip().splitlines()[-1])['kit'] == kit['kit_identity']


@pytest.mark.parametrize('var', ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'])
def test_preflight_refuses_threads(var):
    res = subprocess.run([sys.executable, '-c', PF], cwd=EXP, env=_env(**{var: '4'}), capture_output=True, text=True)
    assert res.returncode != 0 and 'thread variables' in res.stderr


def test_preflight_refuses_bytecode():
    code = PF.replace('pf = r.preflight', 'import sys; sys.dont_write_bytecode = False; pf = r.preflight')
    res = subprocess.run([sys.executable, '-c', code], cwd=EXP, env=_env(), capture_output=True, text=True)
    assert res.returncode != 0 and 'PYTHONDONTWRITEBYTECODE' in res.stderr


def test_preflight_refuses_dependency_and_kit_mismatch(tmp_path, monkeypatch):
    proto = runner.load_protocol()
    bad = json.loads(json.dumps(proto))
    rel = 'experiments/mlst2026/t25_production_20260930/science.py'
    bad['dependencies']['files'][rel] = '0' * 64
    with pytest.raises(runner.KitMismatch, match='dependency'):
        runner.assert_dependencies(bad)
    fake = tmp_path / 'KIT_SHA256.json'
    k = json.loads(runner.KIT_FILE.read_text())
    k['files']['methods.py'] = '0' * 64
    fake.write_text(json.dumps(k))
    monkeypatch.setattr(runner, 'KIT_FILE', fake)
    with pytest.raises(runner.KitMismatch, match='kit files differ'):
        runner.verify_kit()
    monkeypatch.setattr(runner, 'KIT_FILE', tmp_path / 'missing.json')
    with pytest.raises(runner.KitMismatch, match='missing'):
        runner.verify_kit()


def test_protocol_json_is_built_protocol():
    proto = runner.load_protocol()
    assert design.build_protocol(proto['dependencies']) == proto
    assert proto['namespace'] == 'zc-confirm-20261013' and len(proto['seeds']) == 24
    assert proto['inference']['df'] == 23
    assert proto['families']['R-C1']['family_size'] == 4 and proto['families']['R-C2']['family_size'] == 72
    assert proto['metrics']['all_methods'] == ['oracle', 'kron256', 'kron1024']
    assert proto['metrics']['cv_select_only'] == ['full256', 'full1024']
    assert proto['methods']['lambda_grid'] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_run_refuses_out_inside_kit():
    with pytest.raises(runner.KitMismatch, match='outside the kit'):
        runner.cmd_run(runner.PACKAGE / 'out', 1)


def test_summarize_checks_kit_and_start(smoke, tmp_path):
    out, _ = smoke
    p = design.smoke_protocol()
    with pytest.raises(runner.KitMismatch, match='START.json missing'):
        runner.summarize(p, out, check_kit=True)
    kit = runner.verify_kit()['kit_identity']
    start = out / 'START.json'
    start.write_text(json.dumps({'kit_identity': '0' * 64, 'protocol_sha256': runner.sha_json(p)}))
    with pytest.raises(runner.KitMismatch, match='different kit'):
        runner.summarize(p, out, check_kit=True)
    start.write_text(json.dumps({'kit_identity': kit, 'protocol_sha256': '0' * 64}))
    with pytest.raises(runner.KitMismatch, match='different kit'):
        runner.summarize(p, out, check_kit=True)
    start.write_text(json.dumps({'kit_identity': kit, 'protocol_sha256': runner.sha_json(p)}))
    assert runner.summarize(p, out, check_kit=True)['status'] == 'COMPLETE'
    os.remove(start)


def test_marginal_intervals_descriptive():
    p = design.smoke_protocol()
    s = runner.reduce(p, _fake(p, rc2=0.05, rc2_cell=(0.0, 80, 'kron64'), spread=0.01))
    row = next(r for r in s['families']['R-C2'] if (r['theta'], r['N'], r['metric']) == (0.0, 80, 'kron64'))
    assert row['marginal_descriptive']['decision'] == 'positive' and row['decision'] == 'undecided'
    assert s['readings']['R-C2']['reading'] == 'MET'
    assert s['readings']['R-C2']['descriptive_marginal_positive_cells'] == [[0.0, 80, 'kron64']]
    assert abs(s['readings']['R-C2']['largest_mean_contrast'] - 0.05) < 1e-12
