"""Tests for the TR03 representation endpoints.

Unit tests use small hand-built references and the X3 FIXTURE configuration; the lifecycle tests build
one X3 fixture-namespace export (paper1-x3-v1/fixture, development only) with the frozen workflow and
then fabricate corrupted copies of it.  No confirmation namespace, seed or return is touched.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from experiments.mlst2026.x3_nonlinear_metric_20261005 import science as x3s
from experiments.mlst2026.x3_nonlinear_metric_20261005 import workflow as x3w
from experiments.mlst2026.x3_nonlinear_metric_20261005.protocol import FIXTURE
from experiments.mlst2026.x3_representation_addendum import representation_endpoints as rep

THREADS = {k: '1' for k in rep.THREAD_VARS}


def rot(D, i, j, phi):
    R = np.eye(D)
    R[[i, i, j, j], [i, j, i, j]] = [np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi)]
    return R


# --------------------------------------------------------------------------------------
# pure linear algebra
# --------------------------------------------------------------------------------------
def test_distance_is_mean_sin2_on_hand_built_case():
    D, phi = 5, 0.3
    U1 = np.eye(D)[:, :2]
    U2 = rot(D, 1, 2, phi) @ U1                    # rotate e2 towards e3 by phi
    d = rep.subspace_distance(rep.projector(U1), rep.projector(U2), 2)
    assert abs(d - np.sin(phi) ** 2 / 2) < 1e-14
    ang = rep.principal_angles_deg(U1, U2)
    assert np.allclose(ang, [np.degrees(phi), 0.0], atol=1e-6)
    assert abs(d - np.mean(np.sin(np.radians(ang)) ** 2)) < 1e-12


def test_eigh_desc_orders_descending():
    w, V = rep.eigh_desc(np.diag([1.0, 5.0, 3.0]))
    assert np.allclose(w, [5, 3, 1])
    assert np.allclose(np.abs(V[:, 0]), [0, 1, 0])


@pytest.fixture(scope='module')
def refs():
    return rep.references(FIXTURE)


def test_references_match_independent_construction(refs):
    c = FIXTURE
    g = x3s.geometry(c)
    sigma = x3s.t25.covariance(c)
    L, Li = x3s.powers(sigma)
    K = g.K(c['theta'])
    k = c['k']
    # P_Sigma = leading k columns of V(theta)
    Vt = g.V(c['theta'])[:, :k]
    assert np.allclose(Vt @ Vt.T, refs['P_sigma'], atol=1e-10)
    # P_iso by the addendum's formula Cov(x) = Sigma^{1/2}(K + I)Sigma^{1/2}, orthonormalised by SVD (not QR)
    w, V = np.linalg.eigh(L @ (K + np.eye(g.d)) @ L)
    M = Li @ V[:, np.argsort(w)[::-1][:k]]
    Uq = np.linalg.svd(M, full_matrices=False)[0]
    assert np.allclose(Uq @ Uq.T, refs['P_iso'], atol=1e-10)
    assert refs['p_sigma_vs_V_theta'] < 1e-10


def rand_inputs(c, n=300, seed=0):
    g = x3s.geometry(c)
    return np.random.default_rng(seed).normal(size=(n, g.d)) @ x3s.powers(x3s.t25.covariance(c))[0] * 2.0


@pytest.mark.parametrize('rank', [1, 3, 4])
def test_affine_rank_at_most_k_has_zero_leakage(refs, rank):
    rng = np.random.default_rng(rank)
    X = rand_inputs(FIXTURE)
    D = X.shape[1]
    A = rng.normal(size=(D, rank)) @ rng.normal(size=(rank, D))
    r = rep.endpoints(X @ A.T + rng.normal(size=D), refs)
    assert abs(r['leakage']) < 1e-12
    if rank == FIXTURE['k']:
        assert r['status'] == 'ok'
    else:                                       # rank < k: the top-k output subspace is not unique
        assert r['status'] == 'rank_deficient' and r['d_sigma'] is None and r['numerical_rank'] == rank


def test_nonlinear_output_has_positive_leakage(refs):
    X = rand_inputs(FIXTURE)
    r = rep.endpoints(np.tanh(X) + X ** 2 / 4, refs)
    assert r['status'] == 'ok' and r['leakage'] > 1e-3


def test_rows_are_records_and_centring(refs):
    X = rand_inputs(FIXTURE, n=80)
    W = refs['W']
    C = rep.output_covariance(X, W)
    Y = X @ W.T
    assert np.allclose(C, np.cov(Y, rowvar=False), atol=1e-12)
    a, b = rep.endpoints(X, refs), rep.endpoints(X + 7.5, refs)     # a constant offset changes nothing
    for key in ('d_sigma', 'd_iso', 'leakage', 'max_angle_sigma_deg'):
        assert abs(a[key] - b[key]) < 1e-10
    with pytest.raises(rep.Refusal):
        rep.output_covariance(X.T, W)                                 # column-per-record input rejected


def test_orthogonal_change_of_whitening_leaves_endpoints_unchanged(refs):
    rng = np.random.default_rng(3)
    X = rand_inputs(FIXTURE)
    F = np.tanh(X @ rng.normal(size=(X.shape[1], X.shape[1])) / 4)
    Q = np.linalg.qr(rng.normal(size=(X.shape[1], X.shape[1])))[0]
    refs_q = rep.references(FIXTURE, W=Q @ refs['Li'])
    a, b = rep.endpoints(F, refs), rep.endpoints(F, refs_q)
    for key in ('d_sigma', 'd_iso', 'leakage', 'max_angle_sigma_deg', 'max_angle_iso_deg'):
        assert abs(a[key] - b[key]) < 1e-9, key


def test_non_whitening_matrix_is_refused():
    with pytest.raises(rep.Refusal, match='differs from I'):
        rep.references(FIXTURE, W=np.eye(FIXTURE['C'] * FIXTURE['T']))


def test_zero_output_variance_is_undefined(refs):
    X = rand_inputs(FIXTURE)
    r = rep.endpoints(np.zeros_like(X) + 3.0, refs)
    assert r['status'] == 'zero_output_variance'
    assert all(r[m] is None for m in rep.METRICS)


def test_tied_linear_sigma_inverse_is_fitted_cw_pca_projector(refs):
    c = FIXTURE
    g, sigma, L, Li, train, eclean, enoisy = x3s.make_data(c, c['seeds'][0])
    _, mu, S = x3s.moments(train, 32)
    A, b = x3s.closed_form(c, S, mu, Li @ Li, c['k'])
    C = rep.output_covariance(enoisy @ A.T + b, refs['W'])
    U = rep.eigh_desc(C)[1][:, :c['k']]
    B = x3s.unrestricted(Li @ Li, S, c['k'])          # fitted CW-PCA basis, physical coordinates
    Uw = np.linalg.qr(Li @ B)[0]
    assert np.allclose(U @ U.T, Uw @ Uw.T, atol=1e-9)
    # isotropic loss: the fitted Euclidean PCA subspace, mapped by Li
    A2, b2 = x3s.closed_form(c, S, mu, np.eye(g.d), c['k'])
    U2 = rep.eigh_desc(rep.output_covariance(enoisy @ A2.T + b2, refs['W']))[1][:, :c['k']]
    B2 = x3s.unrestricted(np.eye(g.d), S, c['k'])
    Uw2 = np.linalg.qr(Li @ B2)[0]
    assert np.allclose(U2 @ U2.T, Uw2 @ Uw2.T, atol=1e-9)


# --------------------------------------------------------------------------------------
# reducer
# --------------------------------------------------------------------------------------
def fake_rows(c, value):
    rows = []
    for cl in c['classes']:
        for lo in c['losses']:
            for n in c['N']:
                for s in c['seeds']:
                    v = value(cl, lo, n, s)
                    rows.append({'class': cl, 'loss': lo, 'N': n, 'seed': s, 'd_sigma': v, 'd_iso': None if v is None else v + 1,
                                 'max_angle_sigma_deg': v, 'max_angle_iso_deg': v,
                                 'leakage': None if v is None else v / 10})
    return rows


def test_reducer_medians_iqr_and_pairing():
    c = {**FIXTURE, 'seeds': [11, 12, 13, 14, 15]}
    base = {11: 0.1, 12: 0.5, 13: 0.2, 14: 0.9, 15: 0.3}

    def value(cl, lo, n, s):
        return base[s] + (0.05 if lo == 'sigma_inverse' else 0.0) * s       # pairing-dependent difference
    rows = fake_rows(c, value)
    rows = rows[::-1]                                                           # order must not matter
    summary, paired, exp = rep.reduce_rows(c, rows)
    r = next(x for x in summary if x['class'] == 'deep_silu' and x['loss'] == 'isotropic' and x['metric'] == 'd_sigma')
    v = np.array(list(base.values()))
    assert r['median'] == pytest.approx(np.median(v)) and r['iqr'] == pytest.approx(np.subtract(*np.percentile(v, [75, 25])))
    p = next(x for x in paired if x['class'] == 'deep_silu')
    assert p['median'] == pytest.approx(np.median([0.05 * s for s in base]))
    assert all(set(x) == set(rep.SUMMARY_FIELDS) for x in summary)              # no intervals / tests
    e1 = [x for x in exp['E1_sigma_loss_nearer_P_sigma'] if x['class'] == 'deep_silu']
    assert all(x['status'] == 'evaluated_on_complete_summaries' and x['holds'] is False for x in e1)
    assert 'section5_silu_sentence_allowed' not in exp and 'withdrawn' in exp['section5_qualitative_sentence']


def test_reducer_refuses_duplicates_and_missing_seeds():
    c = {**FIXTURE, 'seeds': [1, 2, 3]}
    rows = fake_rows(c, lambda *a: 0.1)
    with pytest.raises(rep.Refusal, match='duplicate'):
        rep.reduce_rows(c, rows + rows[:1])
    with pytest.raises(rep.Refusal):
        rep.reduce_rows(c, rows[1:])


def test_reducer_flags_undefined_values():
    c = {**FIXTURE, 'seeds': [1, 2, 3]}
    rows = fake_rows(c, lambda cl, lo, n, s: None if s == 2 and cl == 'deep_silu' else 0.1 * s)
    summary, paired, _ = rep.reduce_rows(c, rows)
    r = next(x for x in summary if x['class'] == 'deep_silu' and x['metric'] == 'd_sigma')
    assert r['n_defined'] == 2 and r['complete'] is False and r['median'] == pytest.approx(0.2)
    assert r['summary_label'].startswith('INCOMPLETE available-case')


# --------------------------------------------------------------------------------------
# lifecycle on one fixture-namespace export, plus fabricated corruptions
# --------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def fixture_export(tmp_path_factory):
    # Built in a child process that imports only the frozen X3 package: the frozen guard
    # (assert_loaded_modules) refuses to freeze when any other in-repository module is loaded.
    work = tmp_path_factory.mktemp('tr03') / 'work'
    code = ('import sys\n'
            'from experiments.mlst2026.x3_nonlinear_metric_20261005 import workflow as w\n'
            'from pathlib import Path\n'
            'work = Path(sys.argv[1])\n'
            'w.fixture_mode(work)\n'
            'w.preflight_space = lambda *a: None   # disk-size gate is the host concern (as in the X3 tests)\n'
            "w.export(work / 'run', work / 'export', allow_development=True)\n")
    res = subprocess.run([sys.executable, '-c', code, str(work)], capture_output=True, text=True,
                         env={**os.environ, **THREADS})
    assert res.returncode == 0, res.stderr[-4000:]
    return work / 'export'


def verdicts_file(export, path, **override):
    red = json.loads((export / 'reduction.json').read_text())
    v = {k: red['predictions'][k]['verdict'] for k in ('P1', 'P2', 'SURE_agreement')}
    v.update(override)
    path.write_text(json.dumps(v))
    return path


def test_fixture_export_runs(fixture_export, tmp_path):
    out = tmp_path / 'out'
    res = rep.run('fixture', fixture_export, out, verdicts_file(fixture_export, tmp_path / 'v.json'))
    n = len(FIXTURE['seeds']) * len(FIXTURE['N']) * 6
    assert res['rows'] == n
    assert sorted(p.name for p in out.iterdir()) == ['RUN.json', 'checks.json', 'expectations.json',
                                                     'paired.csv', 'per_seed.csv', 'summary.csv']
    checks = json.loads((out / 'checks.json').read_text())
    lin = checks['linear_leakage_check']
    assert lin['rows'] == n * 2 // 3 and lin['passed'] == lin['rows'] and not lin['failed']
    run = json.loads((out / 'RUN.json').read_text())
    assert run['x3_code_identity'] == rep.REGISTERED_X3_IDENTITY and run['export']['export_record_status'] == 'DEVELOPMENT_EXPORT'
    per = (out / 'per_seed.csv').read_text().splitlines()
    assert len(per) == n + 1


def test_frozen_package_unchanged_after_run(fixture_export, tmp_path):
    before = x3w.code_identity()
    rep.run('fixture', fixture_export, tmp_path / 'o')
    assert x3w.code_identity() == before == rep.REGISTERED_X3_IDENTITY


def copy_export(src, tmp_path, name):
    dest = tmp_path / name
    shutil.copytree(src, dest)
    return dest


def assert_refused(mode, export, out, match, verdicts=None):
    with pytest.raises(rep.Refusal, match=match):
        rep.run(mode, export, out, verdicts)
    assert [p.name for p in out.iterdir()] == ['REFUSED.json']


def test_confirmation_mode_refuses_fixture_export(fixture_export, tmp_path):
    assert_refused('confirmation', fixture_export, tmp_path / 'o', 'registered confirmation configuration')


def test_confirmation_mode_requires_recorded_verdicts(fixture_export, tmp_path):
    with pytest.raises(rep.Refusal, match='record the X3 primary verdicts'):
        rep.check_verdicts(fixture_export, 'confirmation', None)
    v = rep.check_verdicts(fixture_export, 'confirmation', verdicts_file(fixture_export, tmp_path / 'v.json'))
    assert set(v) == {'P1', 'P2', 'SURE_agreement'}
    assert rep.check_verdicts(fixture_export, 'fixture', None) == v          # optional in fixture mode


def test_development_export_status_is_refused_in_confirmation_mode(fixture_export, tmp_path, monkeypatch):
    # let the fixture configuration pass as "confirmation" to reach the status gate (test only)
    monkeypatch.setattr(rep, 'BASE', FIXTURE)
    monkeypatch.setattr(rep, 'REGISTERED_K', FIXTURE['k'])
    monkeypatch.setattr(rep, 'REGISTERED_EVAL_RECORDS', FIXTURE['evaluation']['records'])
    assert_refused('confirmation', fixture_export, tmp_path / 'o', 'export status|export/ledger')


def test_mismatching_recorded_verdicts_are_refused(fixture_export, tmp_path):
    v = verdicts_file(fixture_export, tmp_path / 'v.json', P1='NOT-A-VERDICT')
    assert_refused('fixture', fixture_export, tmp_path / 'o', 'differ from the export reduction', v)


def test_corrupted_byte_is_refused(fixture_export, tmp_path):
    exp = copy_export(fixture_export, tmp_path, 'e')
    p = exp / 'cell-0001' / 'arrays.npz'
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))
    assert_refused('fixture', exp, tmp_path / 'o', 'SHA256SUMS')


def test_missing_checksums_are_refused(fixture_export, tmp_path):
    exp = copy_export(fixture_export, tmp_path, 'e')
    (exp / 'SHA256SUMS').unlink()
    assert_refused('fixture', exp, tmp_path / 'o', 'export/ledger check failed')


def test_changed_export_identity_is_refused_even_when_rehashed(fixture_export, tmp_path):
    exp = copy_export(fixture_export, tmp_path, 'e')
    e = json.loads((exp / 'EXPORT.json').read_text())
    e['code_identity'] = '0' * 64
    (exp / 'EXPORT.json').write_text(json.dumps(e))
    (exp / 'SHA256SUMS').write_text(x3w.sha256sums(exp))
    assert_refused('fixture', exp, tmp_path / 'o', 'export record invalid|code identity')


def test_unexported_run_is_refused(fixture_export, tmp_path):
    exp = copy_export(fixture_export, tmp_path, 'e')
    for n in ('SHA256SUMS', 'EXPORT.json'):
        (exp / n).unlink()
    shutil.rmtree(exp / 'tables')
    assert_refused('fixture', exp, tmp_path / 'o', 'absent')


def test_changed_addendum_bytes_are_refused(fixture_export, tmp_path, monkeypatch):
    monkeypatch.setattr(rep, 'ADDENDUM_SHA256', 'f' * 64)
    assert_refused('fixture', fixture_export, tmp_path / 'o', 'addendum bytes changed')


def test_existing_output_directory_is_refused(fixture_export, tmp_path):
    (tmp_path / 'o').mkdir()
    with pytest.raises(SystemExit):
        rep.run('fixture', fixture_export, tmp_path / 'o')


def edited_cell(fixture_export, tmp_path, edit_record=None, edit_arrays=None):
    exp = copy_export(fixture_export, tmp_path, 'e')
    d = exp / 'cell-0002'
    if edit_record:
        r = json.loads((d / 'record.json').read_text())
        edit_record(r)
        (d / 'record.json').write_text(json.dumps(r))
    if edit_arrays:
        a = dict(np.load(d / 'arrays.npz'))
        edit_arrays(a)
        (d / 'arrays.npz').unlink()
        with (d / 'arrays.npz').open('xb') as fh:
            np.savez_compressed(fh, **a)
    f = json.loads((exp / 'freeze.json').read_text())
    return exp, f


def test_data_digest_mismatch_is_refused(fixture_export, tmp_path):
    exp, f = edited_cell(fixture_export, tmp_path, edit_record=lambda r: r.update(evaluation_sha='0' * 64))
    with pytest.raises(rep.Refusal, match='data digest'):
        rep.check_data(f['config'], exp, f['enumeration']['cells'])


def test_changed_closed_form_is_refused(fixture_export, tmp_path):
    def bump(a):
        a['tied_linear__sigma_inverse__A'] = a['tied_linear__sigma_inverse__A'] * (1 + 1e-6)
    exp, f = edited_cell(fixture_export, tmp_path, edit_arrays=bump)
    with pytest.raises(rep.Refusal, match='closed-form'):
        rep.check_data(f['config'], exp, f['enumeration']['cells'])


def test_missing_weights_and_model_record_are_refused(fixture_export, tmp_path):
    def drop(a):
        for k in [k for k in a if k.startswith('deep_silu__isotropic__')]:
            del a[k]
    exp, f = edited_cell(fixture_export, tmp_path, edit_arrays=drop)
    with pytest.raises(rep.Refusal, match='weights'):
        rep.check_data(f['config'], exp, f['enumeration']['cells'])
    exp2, f2 = edited_cell(fixture_export, tmp_path / 'b', edit_record=lambda r: r['models'].pop('deep_linear__isotropic'))
    with pytest.raises(rep.Refusal, match='model record'):
        rep.check_data(f2['config'], exp2, f2['enumeration']['cells'])


def test_module_outside_frozen_closure_is_refused():
    code = ('import experiments.mlst2026.t24_descriptive\n'
            'from experiments.mlst2026.x3_representation_addendum import representation_endpoints as r\n'
            'try:\n    r.check_loaded_modules()\nexcept r.Refusal as e:\n    print("REFUSED", e)\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env={**os.environ, **THREADS})
    assert 'REFUSED' in out.stdout, out.stderr


# --------------------------------------------------------------------------------------
# review findings R2/R3 (6 Oct): defined small gaps are kept; ties, rank deficiency and zero variance are
# undefined with explicit reasons; incomplete summaries never drive E1/E2
# --------------------------------------------------------------------------------------
def diag_refs(D, k):
    U = np.eye(D)[:, :k]
    return {'k': k, 'W': np.eye(D), 'U_sigma': U, 'U_iso': U, 'P_sigma': U @ U.T, 'P_iso': U @ U.T}


def outputs_with_spectrum(eigs, n=4000, seed=0):
    """Rows whose sample covariance (divisor n - 1) has EXACTLY the given eigenvalues on the axes."""
    D = len(eigs)
    g = np.random.default_rng(seed).normal(size=(n, D))
    g -= g.mean(0)
    q, r = np.linalg.qr(g)                          # n x D with orthonormal columns
    q = q * np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * np.sqrt(np.asarray(eigs) * (n - 1))


def test_small_but_resolved_gap_keeps_the_registered_endpoint():
    eigs = [10, 9, 8, 7, 6, 1e-9, 0, 0, 0]          # reviewer reproducer: top six unique
    r = rep.endpoints(outputs_with_spectrum(eigs), diag_refs(9, 6))
    assert r['status'] == 'ok' and r['numerical_rank'] == 6
    assert r['d_sigma'] == pytest.approx(0.0, abs=1e-9) and r['relative_gap'] == pytest.approx(1e-10, rel=1e-3)


def test_exact_tie_rank_deficiency_and_zero_variance_are_undefined_with_reasons():
    refs = diag_refs(9, 6)
    tie = rep.endpoints(outputs_with_spectrum([10, 9, 8, 7, 6, 3, 3, 1, 0.5]), refs)
    assert tie['status'] == 'unresolved_tie' and tie['d_sigma'] is None and tie['leakage'] is not None
    rk = rep.endpoints(outputs_with_spectrum([10, 9, 8, 7, 6, 0, 0, 0, 0]), refs)
    assert rk['status'] == 'rank_deficient' and rk['d_sigma'] is None and abs(rk['leakage']) < 1e-12
    z = rep.endpoints(np.ones((50, 9)), refs)
    assert z['status'] == 'zero_output_variance' and z['leakage'] is None
    ok = rep.endpoints(outputs_with_spectrum([10, 9, 8, 7, 6, 5, 1, 0.5, 0.1]), refs)
    assert ok['status'] == 'ok' and ok['d_sigma'] == pytest.approx(0.0, abs=1e-9)


def rows_with_missing(c, missing):
    """missing(cl, lo, n, s) -> True makes that row undefined (status rank_deficient)."""
    rows = []
    for cl in c['classes']:
        for lo in c['losses']:
            for n in c['N']:
                for s in c['seeds']:
                    if missing(cl, lo, n, s):
                        rows.append({'class': cl, 'loss': lo, 'N': n, 'seed': s, 'status': 'rank_deficient',
                                     'd_sigma': None, 'd_iso': None, 'max_angle_sigma_deg': None,
                                     'max_angle_iso_deg': None, 'leakage': 0.0})
                    else:
                        v = 0.8 if lo == 'sigma_inverse' else 0.9
                        rows.append({'class': cl, 'loss': lo, 'N': n, 'seed': s, 'status': 'ok', 'd_sigma': v,
                                     'd_iso': v + 0.05, 'max_angle_sigma_deg': 40.0, 'max_angle_iso_deg': 50.0,
                                     'leakage': 0.01})
    return rows


C24 = {**FIXTURE, 'seeds': list(range(1, 25))}


def e1_for(exp, cl='deep_silu'):
    return [x for x in exp['E1_sigma_loss_nearer_P_sigma'] if x['class'] == cl]


def test_reviewer_r3_reproducer_one_defined_seed_does_not_evaluate_e1():
    summ, paired, exp = rep.reduce_rows(C24, rows_with_missing(C24, lambda cl, lo, n, s: s != 1))
    assert all(x['status'] == 'not_evaluable_incomplete' and x['holds'] is None for x in e1_for(exp))
    r = next(x for x in summ if x['class'] == 'deep_silu' and x['metric'] == 'd_sigma')
    assert r['n_defined'] == 1 and r['summary_label'].startswith('INCOMPLETE')
    assert json.loads(r['undefined_reasons']) == {'rank_deficient': 23}


@pytest.mark.parametrize('case', ['one_missing', 'all_missing', 'by_loss', 'all_complete'])
def test_missingness_patterns(case):
    miss = {'one_missing': lambda cl, lo, n, s: s == 7 and cl == 'deep_silu',
            'all_missing': lambda cl, lo, n, s: cl == 'deep_silu',
            'by_loss': lambda cl, lo, n, s: cl == 'deep_silu' and lo == 'isotropic' and s <= 3,
            'all_complete': lambda *a: False}[case]
    summ, paired, exp = rep.reduce_rows(C24, rows_with_missing(C24, miss))
    e1 = e1_for(exp)
    p = [x for x in paired if x['class'] == 'deep_silu']
    if case == 'all_complete':
        assert all(x['holds'] is True for x in e1) and all(x['complete'] for x in p)
    else:
        assert all(x['status'] == 'not_evaluable_incomplete' for x in e1)
        assert all(not x['complete'] and x['summary_label'].startswith('INCOMPLETE') for x in p)
    if case == 'all_missing':
        r = next(x for x in summ if x['class'] == 'deep_silu' and x['metric'] == 'd_sigma')
        assert r['median'] is None and r['n_defined'] == 0
    if case == 'by_loss':
        iso = next(x for x in summ if x['class'] == 'deep_silu' and x['loss'] == 'isotropic' and x['metric'] == 'd_sigma')
        sig = next(x for x in summ if x['class'] == 'deep_silu' and x['loss'] == 'sigma_inverse' and x['metric'] == 'd_sigma')
        assert iso['n_defined'] == 21 and sig['complete']
    # the linear classes are complete in every case and keep their evaluation
    assert all(x['status'] == 'evaluated_on_complete_summaries' for x in e1_for(exp, 'tied_linear'))
