"""Fabricated metadata/arrays only. No real recording or launcher invocation."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_ad9_20261019 import launcher, guard, protocol, prediction, readings
from experiments.mlst2026.hcp_rnoise_ad9_20261019.tests.test_ad9 import complete_prediction


@pytest.fixture
def registered(monkeypatch):
    """Mock git objects, never a real tag, receipt, registration or OSF action."""
    f = guard.FREEZE
    record = 'reviews/hcp_rnoise_registration_20261020/REGISTRATION.md'
    auth = str(Path(record).parent / 'CONFIRMATION_AUTHORIZATION.md')
    tag = 'hcp-rnoise-registration-20261020'
    commit = 'f' * 40
    names = ['PROTOCOL_FROZEN.md', 'protocol_frozen.json', 'FREEZE_KIT_SHA256.json',
             'POWER_AD9.json', 'DEV_PREDICTIONS.json', 'PILOT_SUMMARY.json',
             'PILOT_R50_REF_SUMMARY.json', 'CONFIRMATION_RUN_CLI.md', 'CONFIRMATION_AUTHORIZATION_TEMPLATE.md']
    blobs = {f + '/' + n: b'{}' for n in names}
    blobs.update({name: b'{}' for name in launcher.REQUIRED_BUNDLE})
    cfg = protocol.load()
    blobs[f + '/protocol_frozen.json'] = launcher.encoded(cfg)
    kit = {'kit_identity': 'fabricated-kit'}
    blobs[f + '/FREEZE_KIT_SHA256.json'] = launcher.encoded(kit)
    for name in launcher.SOURCES:
        blobs[f + '/' + name] = (protocol.PACKAGE / name).read_bytes()
    sums = b''.join(launcher.digest(raw).encode() + b'  ' + path.encode() + b'\n' for path, raw in blobs.items())
    bundle = launcher.digest(sums)
    blobs[f + '/REGISTRATION_SHA256SUMS'] = sums
    ots = launcher.digest(b'fabricated receipt, NOT an OTS timestamp')
    blobs[f + '/REGISTRATION_SHA256SUMS.ots'] = b'fabricated receipt, NOT an OTS timestamp'
    blobs[record] = (f'OSF URL: https://osf.io/abc12/\nGit tag: {tag}\nTagged commit: {commit}\n'
                     f'Bundle SHA-256: {bundle}\nOTS SHA-256: {ots}\n').encode()
    blobs[auth] = (f'OSF URL: https://osf.io/abc12/\nGit tag: {tag}\nOTS SHA-256: {ots}\n'
                  'Author signature: Dowling Wong; 2026-10-20 12:00 CEST\n').encode()
    calls = []
    state = {'current': 'a' * 40, 'tagged': commit, 'fail_ancestry': False}
    def git(ms, *args):
        calls.append(args)
        if args[0] == 'rev-parse':
            return (state['current'] if args[1].startswith('origin/') else state['tagged']).encode()
        if args[0] == 'merge-base':
            if state['fail_ancestry']: raise subprocess.CalledProcessError(1, args)
            return b''
        if args[0] == 'show':
            ref, path = args[1].split(':', 1)
            if (ref, path) in state: return state[ref, path]
            if path in blobs: return blobs[path]
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(guard, 'git', git)
    monkeypatch.setattr(launcher, 'git_run', lambda ms, *args, **kw: calls.append(args) or b'')
    def verify(expected=None):
        if state.get('bad_kit') or expected is not None and expected != kit:
            raise PermissionError('STOP local/new/inherited kit identity')
        return kit
    monkeypatch.setattr(protocol, 'verify', verify)
    monkeypatch.setattr(guard, 'verify', verify)
    return SimpleNamespace(blobs=blobs, record=record, auth=auth, bundle=bundle, tag=tag,
                           commit=commit, state=state, calls=calls)


@pytest.mark.parametrize('bad', ['missing', 'placeholder', 'url', 'tag', 'commit', 'bundle',
    'changed_bundle', 'kit', 'ots', 'empty_tagged_ots', 'missing_tagged_ots', 'unsigned',
    'signature_placeholder', 'pretag', 'ancestry', 'source', 'duplicate_field', 'receipt_auth'])
def test_preflight_refuses_before_output_or_recording(monkeypatch, tmp_path, registered, bad):
    r = registered; b = r.blobs; f = guard.FREEZE
    if bad == 'missing': del b[r.record]
    if bad == 'placeholder': b[r.record] = b[r.record].replace(b'abc12', b'registration')
    if bad == 'url': b[r.record] = b[r.record].replace(b'https://osf.io', b'http://osf.io')
    if bad == 'tag': b[r.record] = b[r.record].replace(r.tag.encode(), b'<hcp-rnoise-registration-YYYYMMDD>')
    if bad == 'commit': r.state['tagged'] = 'd' * 40
    if bad == 'bundle': r.bundle = '0' * 64
    if bad == 'changed_bundle': r.state[r.commit, f + '/REGISTRATION_SHA256SUMS'] = b'changed'
    if bad == 'kit': r.state['bad_kit'] = True
    if bad == 'ots': b[f + '/REGISTRATION_SHA256SUMS.ots'] = b'changed receipt'
    if bad == 'empty_tagged_ots': r.state[r.commit, f + '/REGISTRATION_SHA256SUMS.ots'] = b''
    if bad == 'missing_tagged_ots': del b[f + '/REGISTRATION_SHA256SUMS.ots']
    if bad == 'unsigned': b[r.auth] = b[r.auth].replace(b'Dowling Wong;', b'UNSIGNED;')
    if bad == 'signature_placeholder': b[r.auth] = b[r.auth].replace(b'2026-10-20 12:00 CEST', b'YYYY-MM-DD HH:MM timezone')
    if bad == 'pretag': r.state['current'] = r.commit
    if bad == 'ancestry': r.state['fail_ancestry'] = True
    if bad == 'source': b[f + '/launcher.py'] += b'changed'
    if bad == 'duplicate_field': b[r.record] += b'Bundle SHA-256: placeholder\n'
    if bad == 'receipt_auth': b[r.auth] = b[r.auth].replace(b'OTS SHA-256:', b'Wrong:')
    class Forbidden(launcher.Runtime):
        @staticmethod
        def units(): pytest.fail('unit metadata reached before preflight')
        @staticmethod
        def decode(unit): pytest.fail('recording access reached')
        @staticmethod
        def publish(*args): pytest.fail('output publication reached')
    out = tmp_path / 'new'
    with pytest.raises((PermissionError, subprocess.CalledProcessError)):
        launcher.run(tmp_path, r.record, r.bundle, out, backend=Forbidden())
    assert not out.exists()


def test_preflight_fetch_order_and_real_guard(registered):
    r = registered
    result = launcher.preflight('fake-ms', r.record, r.bundle)
    assert result['status'] == 'PASS'
    assert r.calls[0][:3] == ('fetch', '--no-tags', 'origin')
    tag_fetch = next(i for i, c in enumerate(r.calls) if c[0] == 'fetch' and 'refs/tags/' in c[-1])
    registration_rev = next(i for i, c in enumerate(r.calls) if c[0] == 'rev-parse')
    assert tag_fetch < registration_rev


@pytest.mark.parametrize('value', ['', '<SHA>', 'UNREGISTERED', '0' * 63])
def test_placeholder_bundle_refuses_without_fetch(tmp_path, monkeypatch, value):
    monkeypatch.setattr(launcher, 'fetch', lambda *a: pytest.fail('fetch reached'))
    with pytest.raises(PermissionError): launcher.preflight(tmp_path, 'bad', value)


def sample_prediction(S, date):
    # The inherited schema fixture is fabricated; retain real validation.
    p = complete_prediction('c59')
    p['date'] = date
    p['date_index'] = 12 + protocol.load()['access']['confirmation_analysis_dates'].index(date)
    p['Sref_sha256_f64le'] = prediction.sref_hash(S)
    return p


class Arrays:
    """Only the orchestration uses this backend; CLI cannot select it."""
    def __init__(self, tmp_path, n=18):
        self.tmp = tmp_path; self.n = n; self.calls = []; self.published = {}
        self.rows = launcher.units()
        self.S = np.eye(72)
        self.roles = {r: (np.zeros((2, 72)), {'block': np.array([0, 0])}) for r in ('ref', 'train', 'cal', 'eval')}
        self.fail_decode = False; self.fail_fit = False; self.fail_publish = False

    def preflight(self, *args):
        self.calls.append('preflight')
        return {'tag': 'hcp-rnoise-registration-20261020', 'bundle_sha256': 'a' * 64}

    def publish(self, ms, path, raw, message):
        if path in self.published or self.fail_publish: raise FileExistsError('STOP public START exists')
        p = self.tmp / path
        launcher.write_new(p, raw)
        self.published[path] = raw
        if '/predictions/' in path: self.calls.append(('published', json.loads(raw)['date']))
        return p, 'b' * 40

    def units(self): return self.rows

    def decode(self, unit):
        self.calls.append(('decode', unit['measurement_date']))
        if self.fail_decode: raise OSError('fabricated decode failure')
        return np.zeros((12, 2))

    def qualify(self, data, unit, index):
        d = unit['measurement_date']; self.calls.append(('qualified', d))
        passed = index - 12 < self.n
        return ({k: {'pass': passed if k == 'G1' else True} for k in launcher.GATES}, self.roles, self.S)

    def predict(self, S, date):
        self.calls.append(('predicted', date))
        return sample_prediction(S, date)

    def model(self, S, theta): return SimpleNamespace(theta=theta)

    def fit(self, roles, model, S, N, m, arm, index, replicate, **kw):
        d = kw['date']
        assert ('published', d) in self.calls
        assert index == 12 + [u['measurement_date'] for u in self.rows].index(d)
        assert 0 <= replicate < 50 and kw['geometry'] in prediction.GRIDS
        if self.fail_fit: raise RuntimeError('fabricated fit failure')
        return {'D': -.1 + index * .001, 'replicate': replicate, 'selected_status': 'flag retained',
                'restart_scores': [1., 2.], 'theta': model.theta}


@pytest.mark.parametrize('started', [False, True])
def test_output_claim_is_atomic_and_prevents_work(tmp_path, started):
    b = Arrays(tmp_path); out = tmp_path / 'out'; out.mkdir()
    if started: (out / 'STARTED.json').write_text('{}')
    with pytest.raises(FileExistsError): launcher.run(tmp_path, 'r', 'a' * 64, out, backend=b)
    assert b.calls == ['preflight']


def test_public_once_only_blocks_different_output(tmp_path):
    b = Arrays(tmp_path); b.fail_decode = True
    for out in (tmp_path / 'first', tmp_path / 'second'):
        with pytest.raises((OSError, FileExistsError)):
            launcher.run(tmp_path, 'r', 'a' * 64, out, backend=b)
    assert len([c for c in b.calls if isinstance(c, tuple) and c[0] == 'decode']) == 1
    assert json.loads((tmp_path / 'first' / 'STOP.json').read_bytes())['no_restart']
    assert (tmp_path / 'second' / 'STOP.json').is_file()
    start = json.loads((tmp_path / 'first' / 'STARTED.json').read_bytes())
    stop = json.loads((tmp_path / 'first' / 'STOP.json').read_bytes())
    assert stop['run_token'] == start['run_token']


def test_same_path_same_clock_start_ownership_is_unique(tmp_path):
    evidence = {'tag': 'hcp-rnoise-registration-20261020', 'bundle_sha256': 'a' * 64}
    # Independent hosts/clones can use the same absolute output name, Git user
    # and second. The payload must still differ before Git object construction.
    a = launcher.start_record(evidence, tmp_path / 'identical')
    b = launcher.start_record(evidence, tmp_path / 'identical')
    assert len(a['run_token']) == len(b['run_token']) == 64
    assert a['run_token'] != b['run_token']
    assert launcher.digest(launcher.encoded(a)) != launcher.digest(launcher.encoded(b))
    a.pop('run_token'); b.pop('run_token')
    assert a == b


@pytest.mark.parametrize('failure', ['decode', 'fit'])
def test_failure_persists_stop_without_retry(tmp_path, failure):
    b = Arrays(tmp_path)
    setattr(b, 'fail_' + failure, True)
    out = tmp_path / 'run'
    with pytest.raises((OSError, RuntimeError)): launcher.run(tmp_path, 'r', 'a' * 64, out, backend=b)
    stopped = json.loads((out / 'STOP.json').read_bytes())
    assert stopped['status'] == 'STOP' and stopped['confirmatory_claim_allowed'] is False
    assert not (out / 'COMPLETED.json').exists()
    count = len(b.calls)
    with pytest.raises(FileExistsError): launcher.run(tmp_path, 'r', 'a' * 64, out, backend=b)
    assert len(b.calls) == count + 1  # preflight only, no retry


def test_below18_reports_all60_and_no_fit(tmp_path):
    b = Arrays(tmp_path, n=17); out = tmp_path / 'below18'
    with pytest.raises(RuntimeError, match='below18'):
        launcher.run(tmp_path, 'r', 'a' * 64, out, backend=b)
    q = json.loads((out / 'QUALIFICATION.json').read_bytes())
    assert len(q['units']) == len(q['booleans']) == 60
    assert q['cohort']['n_included'] == 17 and not q['cohort']['replacement']
    assert len(q['cohort']['excluded']) == 43
    assert not any(c[0] == 'predicted' for c in b.calls if isinstance(c, tuple))
    s = json.loads((out / 'SUMMARY.json').read_bytes())
    assert set(s['outcomes']) == {f'PZ{i}' for i in range(1, 6)}
    assert not s['confirmatory_claim_allowed']


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'replacement', 'subject'])
def test_no_replacement_or_subset(tmp_path, bad):
    b = Arrays(tmp_path)
    if bad == 'missing': b.rows.pop()
    if bad == 'duplicate': b.rows[-1] = b.rows[0]
    if bad == 'replacement': b.rows[0] = {**b.rows[0], 'measurement_date': '2099-01-01'}
    if bad == 'subject': b.rows[0] = {**b.rows[0], 'subject_id': '999999'}
    with pytest.raises(PermissionError): launcher.run(tmp_path, 'r', 'a' * 64, tmp_path / 'out', backend=b)
    assert not any(c[0] == 'decode' for c in b.calls if isinstance(c, tuple))


def test_fabricated_happy_path_complete_r50_all_arms(tmp_path, monkeypatch):
    b = Arrays(tmp_path, 18); out = tmp_path / 'happy'
    # Keep full R50 orchestration + real readings, avoid per-row disk sync cost.
    monkeypatch.setattr(launcher.os, 'fsync', lambda fd: None)
    result = launcher.run(tmp_path, 'r', 'a' * 64, out, backend=b)
    assert result['status'] == 'COMPLETE' and result['cohort']['n_included'] == 18
    assert len(result['intervals']) == 90 and len(result['interactions']) == 4
    assert set(result['outcomes']) == {f'PZ{i}' for i in range(1, 6)}
    assert len([c for c in b.calls if isinstance(c, tuple) and c[0] == 'decode']) == 60
    assert len({c[1] for c in b.calls if isinstance(c, tuple) and c[0] == 'decode'}) == 60
    assert len([c for c in b.calls if isinstance(c, tuple) and c[0] == 'published']) == 18
    terminal = json.loads((out / 'COMPLETED.json').read_bytes())
    assert terminal['fits'] == 18 * 15 * 6 * 50
    row = json.loads((out / 'FITS.jsonl').open().readline())
    assert row['kernel']['selected_status'] == 'flag retained'
    last_qualification = max(i for i,c in enumerate(b.calls) if isinstance(c,tuple) and c[0] == 'qualified')
    first_prediction = min(i for i,c in enumerate(b.calls) if isinstance(c,tuple) and c[0] == 'predicted')
    assert last_qualification < first_prediction


def test_real_array_guard_precedes_learning_kernel(monkeypatch, tmp_path):
    from experiments.mlst2026.hcp_rnoise_20261019 import gates, signal, learning
    b = Arrays(tmp_path)
    d = b.rows[0]['measurement_date']
    gen = np.random.default_rng(714)
    roles = {}
    for role, count in [('ref', 600), ('train', 1111), ('cal', 606), ('eval', 606)]:
        X = gen.normal(size=(count, 72)) * 1e-12
        # Equal blocks for training/cal; reference metadata unused by fitting.
        roles[role] = (X, {'block': np.repeat(np.arange(count // 101), 101) if count % 101 == 0 else np.zeros(count, int)})
    S = gates.reference(roles['ref'][0]); p = sample_prediction(S, d)
    model = signal.build(S, 0.)
    calls = []
    def before(*args): calls.append('guard'); return {'prediction_record': p}
    monkeypatch.setattr(guard, 'before_fit', before)
    original = learning.fit_cell
    def fit(*args): calls.append('learning'); return original(*args)
    monkeypatch.setattr(learning, 'fit_cell', fit)
    row = guard.fit_confirmation(roles, model, S, 56, None, 'reference', 12, 0,
        ms=tmp_path, registration_record='fabricated', date=d, prediction_path='fixture', prediction_commit='fixture', geometry='control')
    assert calls == ['guard', 'learning'] and np.isfinite(row['D'])
    assert len(set(row['sampled_training_indices'])) == 56
    assert 'restarts' in row and 'selected_status' in row
    def denied(*a): raise PermissionError('STOP pre-tag prediction')
    monkeypatch.setattr(guard, 'before_fit', denied)
    calls.clear()
    with pytest.raises(PermissionError):
        guard.fit_confirmation(roles, model, S, 56, None, 'reference', 12, 0,
            ms=tmp_path, registration_record='fabricated', date=d, prediction_path='fixture', prediction_commit='fixture', geometry='control')
    assert not calls


def test_ad2_and_preprocessing_fabricated_arrays(monkeypatch):
    from experiments.mlst2026.hcp_rnoise_dev_20261019.artifact import screen
    from experiments.mlst2026.hcp_rnoise_20261019 import gates
    data = np.ones((12, 20000)) * 1e-8  # DC offset screened about median only
    assert screen(data)['pass']
    data[0, 0] = np.nan
    result, roles, S = launcher.qualify(data, {'n_times':20000, 'sfreq_hz':2034.5101318359375}, 12)
    assert result['artifact']['amplitude_count'] is None
    assert result['artifact']['channels']['MEG 001']['amplitude_count'] is None
    assert roles is S is None and not result['artifact']['pass']
    # Real pinned FIR/window/gate kernels on a fabricated complete native scan.
    gen = np.random.default_rng(319)
    data = gen.normal(size=(12, 610560)) * 1e-12
    u = {'n_times': data.shape[1], 'sfreq_hz':2034.5101318359375}
    result, roles, S = launcher.qualify(data, u, 12)
    assert result['artifact']['pass'] and S.shape == (72,72)
    assert np.array_equal(S, gates.reference(roles['ref'][0]))
    assert result['budget']['reference_strideT_count'] >= 576
    assert len(result['plan']['blocks']) == 29
    assert all(type(v) is bool for v in launcher.qualification_booleans(result).values())


@pytest.mark.parametrize('path', sorted(launcher.REQUIRED_BUNDLE))
def test_every_required_bundle_entry_is_mandatory(registered, path):
    r = registered; f = guard.FREEZE
    old = r.blobs[f + '/REGISTRATION_SHA256SUMS']
    new = b''.join(line for line in old.splitlines(keepends=True) if line.split(b'  ',1)[1].strip().decode() != path)
    r.blobs[f + '/REGISTRATION_SHA256SUMS'] = new
    r.bundle = launcher.digest(new)
    r.blobs[r.record] = r.blobs[r.record].replace(launcher.digest(old).encode(), r.bundle.encode())
    with pytest.raises(PermissionError): launcher.preflight('fake', r.record, r.bundle)


def test_publisher_uses_private_index_and_normal_push(tmp_path, monkeypatch):
    calls = []; raw = b'fabricated metadata\n'
    def fake(ms, *args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == 'rev-parse': return b'a' * 40
        if args[0] == 'show': return raw
        if args[0] in ('hash-object', 'write-tree', 'commit-tree'): return b'b' * 40
        return b''
    monkeypatch.setattr(guard, 'git', fake)
    monkeypatch.setattr(launcher, 'git_run', fake)
    path, commit = launcher.publish(tmp_path, 'reviews/fabricated/STARTED.json', raw, 'fixture')
    assert path.read_bytes() == raw and commit == 'b' * 40
    mutations = {a[0]: kw for a,kw in calls if a[0] in ('read-tree', 'update-index', 'write-tree')}
    assert all('GIT_INDEX_FILE' in kw['extra_env'] for kw in mutations.values())
    push = next(a for a,kw in calls if a[0] == 'push')
    assert push == ('push', 'origin', 'b' * 40 + ':refs/heads/main')
    assert not any('--force' in a or '--amend' in a for a,kw in calls)


def test_rejected_start_publication_never_decodes(tmp_path):
    b = Arrays(tmp_path); b.fail_publish = True
    with pytest.raises(FileExistsError): launcher.run(tmp_path, 'r', 'a' * 64, tmp_path / 'out', backend=b)
    assert b.calls == ['preflight']
    assert (tmp_path / 'out' / 'STOP.json').is_file()


def test_unavailable_predictions_keep_canonical_families(tmp_path, monkeypatch):
    b = Arrays(tmp_path); b.n = 18
    original = b.predict
    def predict(S, date):
        p = original(S, date)
        for g in ('A', 'B'):
            p['adaptive'][g] = {'root': False, 'available': False, 'reason': 'fabricated no root'}
            p['geometries'][g].update(available=False, theta=None, row=None, reason='fabricated no root')
        return p
    b.predict = predict
    monkeypatch.setattr(launcher.os, 'fsync', lambda fd: None)
    result = launcher.run(tmp_path, 'r', 'a' * 64, tmp_path / 'out', backend=b)
    assert len(result['intervals']) == 90
    assert sum(r['available'] for r in result['intervals']) == 30
    assert not result['outcomes']['PZ2']['met'] and not result['outcomes']['PZ5']['available']
    assert not result['outcomes']['PZ4']['available']
    assert len(list((tmp_path / 'out' / 'unavailable').iterdir())) == 36


def test_prediction_wrapper_calls_unchanged_kernel(monkeypatch):
    from experiments.mlst2026.hcp_rnoise_20261019 import population
    S = np.eye(72) * 1e-24
    date = protocol.load()['access']['confirmation_analysis_dates'][0]
    # Real population/64-start predictions and full fixed/adaptive diagnostics on
    # an invented isotropic covariance. No recording, pilot source or dev cache.
    result = prediction.prediction_record(S, date)
    assert result['pre_fit'] and result['date_index'] == 12
    assert len(result['fixed']) == 4 and set(result['adaptive']) == {'A', 'B'}
    assert all(r['n_starts'] == 64 for r in result['fixed'])
    assert result['Sref_sha256_f64le'] == prediction.sref_hash(S)


def test_concurrent_identical_path_clones_have_one_start_owner(tmp_path, monkeypatch):
    base = 'a' * 40
    evidence = {'tag': 'hcp-rnoise-registration-20261020', 'bundle_sha256': 'b' * 64}
    # Same absolute output string, base, author/committer second and message.
    payloads = [launcher.encoded(launcher.start_record(evidence, '/identical/output')) for _ in range(2)]
    relative = 'reviews/hcp_rnoise_confirmation_' + 'b' * 64 + '/STARTED.json'
    clones = {}; remote = {'tip': base, 'owners': []}
    def fake(ms, *args, **kw):
        state = clones.setdefault(str(ms), {})
        op = args[0]
        if op == 'rev-parse': return base.encode()  # intentionally stale same base
        if op == 'ls-tree': return b''
        if op == 'hash-object':
            state['raw'] = kw['data']
            state['object'] = hashlib.sha1(state['raw']).hexdigest()
            return state['object'].encode()
        if op == 'write-tree': return hashlib.sha1(state['object'].encode()).hexdigest().encode()
        if op == 'commit-tree':
            state['commit'] = hashlib.sha1((args[1] + base + 'fixed-author-time').encode() + kw['data']).hexdigest()
            return state['commit'].encode()
        if op == 'push':
            commit = args[-1].split(':', 1)[0]
            # Model normal Git: same-commit up-to-date pushes succeed too.
            if remote['tip'] not in (base, commit):
                raise subprocess.CalledProcessError(1, args, b'non-fast-forward')
            remote.update(tip=commit, raw=state['raw'])
            remote['owners'].append(str(ms))
            return b''
        if op == 'show': return remote['raw']
        return b''
    monkeypatch.setattr(guard, 'git', fake)
    monkeypatch.setattr(launcher, 'git_run', fake)
    launcher.publish(tmp_path / 'clone1', relative, payloads[0], 'HCP confirmation: once-only START')
    with pytest.raises(subprocess.CalledProcessError):
        launcher.publish(tmp_path / 'clone2', relative, payloads[1], 'HCP confirmation: once-only START')
    assert len(remote['owners']) == 1
    assert json.loads(remote['raw'])['run_token'] == json.loads(payloads[0])['run_token']


def test_start_publication_verifies_exact_owner_blob(tmp_path, monkeypatch):
    raw = launcher.encoded(launcher.start_record({'tag': 'hcp-rnoise-registration-20261020'}, '/same/output'))
    other = json.loads(raw); other['run_token'] = 'other-owner'
    def fake(ms, *args, **kw):
        if args[0] == 'rev-parse': return b'a' * 40
        if args[0] == 'ls-tree': return b''
        if args[0] in ('hash-object', 'write-tree', 'commit-tree'): return b'b' * 40
        if args[0] == 'show': return launcher.encoded(other)
        return b''
    monkeypatch.setattr(guard, 'git', fake)
    monkeypatch.setattr(launcher, 'git_run', fake)
    with pytest.raises(PermissionError, match='exact bytes'):
        launcher.publish(tmp_path, 'reviews/fabricated/STARTED.json', raw, 'fixture')
