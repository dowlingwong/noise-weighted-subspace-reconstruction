"""Future registered confirmation runner. Never invoke without real signed authority.

Import is metadata/array-only. Recording paths are constructed solely by decode(),
which run() reaches after fetching, exact registration validation and public START.
Fabricated tests inject array backends; the command line exposes no bypass flags.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
from urllib.parse import urlparse

import numpy as np
from . import guard, prediction, protocol, readings

FREEZE = guard.FREEZE
ARMS = (('reference', None), ('lw', 256), ('lw', 512),
        ('kron', 256), ('kron', 512), ('isotropic', None))
GATES = ('G1', 'G2', 'artifact', 'budget')
SOURCES = ('launcher.py', 'tests/test_launcher.py', 'confirmation-run')
REQUIRED_BUNDLE = {
    FREEZE + '/' + n for n in (
        'PROTOCOL_FROZEN.md', 'protocol_frozen.json', 'FREEZE_KIT_SHA256.json',
        'POWER_AD9.json', 'DEV_PREDICTIONS.json', 'PILOT_SUMMARY.json',
        'PILOT_R50_REF_SUMMARY.json', 'CONFIRMATION_RUN_CLI.md',
        'CONFIRMATION_AUTHORIZATION_TEMPLATE.md', 'PRESERVED_RECORDS_AD9.json')
} | {
    'reviews/hcp_rnoise_design_20261019/' + n
    for n in ('BUDGETS.json', 'CONFIG.json', 'DESIGN.md', 'GEOMETRY.json', 'SPLIT.json')
} | {
    'reviews/hcp_rnoise_author_decisions_20261019/' + n
    for n in ('AD8_POWER_CELLS.md', 'AD8a_AD9_AMENDMENT.md', 'DECISIONS.md')
} | {
    'reviews/hcp_rnoise_design_addendum_20261019/' + n
    for n in ('ADDENDUM.md', 'CONFIG_DELTA.json')
} | {FREEZE + '/' + n for n in SOURCES}


def encoded(value):
    """Keep nonfinite diagnostics explicit; never turn missing statistics into zero."""
    from experiments.mlst2026.hcp_rnoise_dev_20261019.pipeline import clean
    return (json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def git_run(ms, *args, data=None, extra_env=None):
    return subprocess.check_output(['git', '-C', str(ms), *args], input=data,
        env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0', **(extra_env or {})})


def fetch(ms):
    git_run(ms, 'fetch', '--no-tags', 'origin',
            'refs/heads/main:refs/remotes/origin/main')


def exact_fields(text, names):
    for name in names:
        if len(re.findall('^' + re.escape(name) + ':', text, re.M)) != 1:
            raise PermissionError('STOP missing or duplicate field: ' + name)
    return {name: guard.field(text, name) for name in names}


def preflight(ms, record, registered_bundle):
    """Metadata only; strengthen the frozen guard without changing its bytes."""
    if not re.fullmatch('[0-9a-f]{64}', registered_bundle):
        raise PermissionError('STOP exact registered bundle SHA-256 required')
    if not re.fullmatch(r'reviews/hcp_rnoise_registration_\d{8}/REGISTRATION.md', record):
        raise PermissionError('STOP registration record path')
    fetch(ms)
    text = guard.git(ms, 'show', 'origin/main:' + record).decode()
    fields = exact_fields(text, ('OSF URL', 'Git tag', 'Tagged commit',
                                 'Bundle SHA-256', 'OTS SHA-256'))
    url = fields['OSF URL']; parsed = urlparse(url)
    if (not re.fullmatch(r'https://osf\.io/[a-z0-9]+/', url)
            or parsed.path.strip('/') in {'registration', 'placeholder', 'example', 'todo', 'unsigned'}):
        raise PermissionError('STOP real OSF https registration URL required')
    tag = fields['Git tag']
    if not re.fullmatch(r'hcp-rnoise-registration-\d{8}', tag):
        raise PermissionError('STOP registration tag name')
    datetime.strptime(tag[-8:], '%Y%m%d')
    if not re.fullmatch('[0-9a-f]{40}', fields['Tagged commit']):
        raise PermissionError('STOP full recorded commit required')
    if fields['Bundle SHA-256'] != registered_bundle:
        raise PermissionError('STOP requested bundle differs from registration')
    # Fetch an existing tag only; never create, move, stamp or register a tag.
    git_run(ms, 'fetch', '--no-tags', 'origin', f'refs/tags/{tag}:refs/tags/{tag}')
    result = guard.registration(ms, record)
    current = guard.git(ms, 'rev-parse', 'origin/main^{commit}').decode().strip()
    if current == result['tagged_commit']:
        raise PermissionError('STOP registration/authorization must be published after tag')
    # Frozen guard requires these too; additionally reject empty tagged receipts.
    if not guard.git(ms, 'show', result['tagged_commit'] + ':' + FREEZE + '/REGISTRATION_SHA256SUMS.ots'):
        raise PermissionError('STOP empty tagged OTS')
    auth = guard.git(ms, 'show', 'origin/main:' + str(Path(record).parent / 'CONFIRMATION_AUTHORIZATION.md')).decode()
    auth_fields = exact_fields(auth, ('OSF URL', 'Git tag', 'OTS SHA-256', 'Author signature'))
    if not re.fullmatch(r'Dowling Wong; \d{4}-\d{2}-\d{2} \d{2}:\d{2} [A-Za-z][A-Za-z0-9_+/:-]*', auth_fields['Author signature']):
        raise PermissionError('STOP real dated author signature required')
    datetime.strptime(auth_fields['Author signature'].split('; ')[1].rsplit(' ', 1)[0], '%Y-%m-%d %H:%M')
    sums = guard.git(ms, 'show', 'origin/main:' + FREEZE + '/REGISTRATION_SHA256SUMS')
    if digest(sums) != registered_bundle:
        raise PermissionError('STOP bundle SHA mismatch')
    entries = dict(line.split('  ', 1)[::-1] for line in sums.decode().splitlines())
    if not REQUIRED_BUNDLE.issubset(entries):
        raise PermissionError('STOP incomplete registration bundle file set')
    kit = protocol.verify()
    # guard.registration already binds the manifest, protocol and inherited kits.
    # Bind the newly added executable bytes individually as well as via the kit.
    for name in SOURCES:
        public = FREEZE + '/' + name
        raw = (protocol.PACKAGE / name).read_bytes()
        if entries.get(public) != digest(raw):
            raise PermissionError('STOP launcher source absent or different in bundle: ' + name)
    return {**result, 'bundle_sha256': registered_bundle, 'kit_identity': kit['kit_identity']}


def publish(ms, relative, raw, message):
    """Create one new public blob with a private index; normal push is the CAS.

    A competing START/publication wins the fast-forward race: the loser stops.
    Never touches the operator's index/worktree history or retries a rejected push.
    """
    base = guard.git(ms, 'rev-parse', 'origin/main^{commit}').decode().strip()
    found = guard.git(ms, 'ls-tree', base, '--', relative)
    if found:
        raise FileExistsError('STOP public record already exists: ' + relative)
    path = Path(ms) / relative
    write_new(path, raw)
    with tempfile.TemporaryDirectory(prefix='hcp-confirmation-index-') as scratch:
        env = {'GIT_INDEX_FILE': str(Path(scratch) / 'index')}
        git_run(ms, 'read-tree', base, extra_env=env)
        blob = git_run(ms, 'hash-object', '-w', '--stdin', data=raw).decode().strip()
        git_run(ms, 'update-index', '--add', '--cacheinfo', '100644', blob, relative, extra_env=env)
        tree = git_run(ms, 'write-tree', extra_env=env).decode().strip()
        commit = git_run(ms, 'commit-tree', tree, '-p', base, data=(message + '\n').encode()).decode().strip()
    git_run(ms, 'push', 'origin', commit + ':refs/heads/main')
    fetch(ms)
    guard.git(ms, 'merge-base', '--is-ancestor', commit, 'origin/main')
    if guard.git(ms, 'show', 'origin/main:' + relative) != raw:
        raise PermissionError('STOP published record differs from exact bytes')
    return path, commit


def units():
    """Read only pinned design metadata; no filesystem discovery or replacements."""
    from experiments.mlst2026.hcp_rnoise_20261019.protocol import split
    rows = sorted((u for u in split()['units'] if u['split'] == 'confirmation'),
                  key=lambda u: u['measurement_date'])
    cfg = protocol.load()
    expected = cfg['confirmation_units']
    actual = [{'date': u['measurement_date'], 'subject_id': u['subject_id']} for u in rows]
    if (len(rows) != 60 or actual != expected
            or [u['measurement_date'] for u in rows] != sorted(cfg['access']['confirmation_analysis_dates'])):
        raise PermissionError('STOP exact all60 SPLIT representatives required')
    return rows


def decode(unit):
    """The only recording path/open site. Called after preflight and public START.

    Expected metadata/geometry failures are exclusions. Reader/get_data failures
    propagate to STOP; neither the date nor the run is retried.
    """
    import mne
    from experiments.mlst2026.hcp_rnoise_20261019.preprocess import selected_geometry, Excluded
    cfg = protocol.load()
    directory = Path(cfg['access']['data_root']) / unit['subject_id']
    raw = mne.io.read_raw_bti(str(directory / 'c,rfDC'), config_fname=str(directory / 'config'),
                            verbose='error', **cfg['preprocessing']['reader'])
    try:
        if (raw.preload or raw.n_times != unit['n_times']
                or not np.isclose(raw.info['sfreq'], unit['sfreq_hz'], rtol=0, atol=1e-9)):
            raise Excluded('native header differs from frozen SPLIT; no shortening')
        selected_geometry(raw.info)
        if raw.info.get('proc_history') or raw.info.get('projs'):
            raise Excluded('processed history or projections')
        data = raw.get_data(picks=cfg['geometry']['selected_channels'])
        if data.shape != (12, unit['n_times']):
            raise RuntimeError('STOP decoded shape differs from frozen scan')
        return data
    finally:
        raw.close()


def qualify(data, unit, index):
    """AD2 screen -> original native FIR -> fixed windows -> unchanged G1/G2."""
    from experiments.mlst2026.hcp_rnoise_dev_20261019.artifact import screen
    from experiments.mlst2026.hcp_rnoise_dev_20261019.pipeline import preprocess
    from experiments.mlst2026.hcp_rnoise_20261019 import gates, windows
    cfg = protocol.load()
    artifact = screen(data)
    outcome = {'artifact': artifact, 'G1': {'pass': False, 'evaluated': False},
               'G2': {'pass': False, 'evaluated': False}, 'budget': {'pass': False, 'evaluated': False}}
    if not artifact['pass']:
        outcome['reason'] = 'AD2 whole-scan exclusion; no filtering or repair'
        return outcome, None, None
    _, roles, plan = preprocess(data, unit)
    # Use the same divisor-n reference for qualification, prediction and learning.
    S = gates.reference(roles['ref'][0])
    outcome.update(gates.assess(roles, native_artifact=artifact, details=True, date_index=index))
    counts = {r: len(pair[0]) for r, pair in roles.items()}
    ref_strideT = sum(len(range(b['active_start'], b['end'] - 6 + 1, 6))
                      for b in plan['blocks'] if b['role'] == 'ref')
    block_counts = {r: sum(b['role'] == r for b in plan['blocks']) for r in roles}
    # Fixed absolute grids/budgets: never derive a smaller grid from a short scan.
    budget_pass = (ref_strideT >= 576 and counts['train'] >= 1.1 * 888
                   and counts['cal'] >= 1.1 * 512 and counts['eval'] >= 1.1 * 500
                   and all(n >= 6 for n in block_counts.values())
                   and windows.grids(counts['train'])['A_grid'] == prediction.GRIDS['A']
                   and windows.grids(counts['train'])['B_grid'] == prediction.GRIDS['B'])
    outcome['budget'] = {'pass': bool(budget_pass), 'counts': counts, 'blocks': block_counts,
                         'reference_strideT_count': ref_strideT, 'fixed_grids': prediction.GRIDS}
    outcome['plan'] = plan
    if outcome['G1']['pass']:
        outcome['Sref_sha256_f64le'] = prediction.sref_hash(S)
    return outcome, roles, S


def qualification_booleans(outcome):
    result = {}
    for gate in GATES:
        value = outcome[gate]['pass']
        if type(value) is not bool:
            raise ValueError('STOP explicit boolean qualification required: ' + gate)
        result[gate] = value
    return result


def unavailable(reason):
    return {'available': False, 'met': False, 'reason': reason, 'reading': 'UNAVAILABLE'}


def summarize(records, predictions, qualification):
    """All60 cohort first; canonical families remain intact if cells unavailable."""
    selection = protocol.cohort(qualification)
    if selection['status'] != 'PASS':
        return {'status': 'STOP', 'cohort': selection, 'confirmatory_claim_allowed': False,
                'outcomes': {f'PZ{i}': unavailable(selection['reason']) for i in range(1, 6)}}
    grouped = {}
    for row in records:
        key = (row['theta'], row['N'], row['arm'], row['m'])
        grouped.setdefault(key, []).append(row)
    intervals = []
    for g, grid in prediction.GRIDS.items():
        for N in grid:
            for arm, m in ARMS:
                key = (g, N, arm, m)
                cell = f'{g}/{grid.index(N)}/' + (arm if m is None else f'{arm}_m{m}')
                try:
                    ci = readings.date_interval(grouped.get(key, []), 34,
                        expected_dates=selection['included'], qualification=qualification)
                except ValueError as exc:
                    ci = {**unavailable(str(exc)), 'cell': cell}
                intervals.append(ci)
    reference = [r for r in intervals if r['cell'] in readings.REFERENCE_CELLS]
    isotropic = [r for r in intervals if r['cell'] in readings.PZ5_CELLS]
    pz1 = {d: readings.pz1(predictions[d]['fixed']) for d in selection['included']}
    paired = [r for r in records if r['theta'] in ('A', 'B') and r['arm'] != 'isotropic']
    try:
        interactions = readings.paired_interactions(paired, qualification=qualification)
        pz4 = readings.pz4({a: interactions[a + '_m256'] for a in ('lw', 'kron')},
                          {a: interactions[a + '_m512'] for a in ('lw', 'kron')})
    except ValueError as exc:
        interactions = {f'{a}_m{m}': unavailable(str(exc)) for a in ('lw', 'kron') for m in (256, 512)}
        pz4 = unavailable(str(exc))
    return {'status': 'COMPLETE', 'cohort': selection, 'family_size': 34,
            'resamples': 50, 'intervals': intervals, 'interactions': interactions,
            'outcomes': {'PZ1': {'available': all(r['available'] for r in pz1.values()),
                                'met': all(r['met'] for r in pz1.values()), 'per_date': pz1},
                         'PZ2': readings.pz2(reference, predictions=predictions, qualification=qualification),
                         'PZ3': readings.pz3([r for r in reference if r['cell'] in readings.PZ3_CELLS]),
                         'PZ4': pz4, 'PZ5': readings.pz5(isotropic)},
            'reporting': 'Every fit/optimizer/restart flag retained verbatim in FITS.jsonl; exclusions never replaced'}


class Runtime:
    """Production backend. Tests supply fabricated arrays, never real readers."""
    preflight = staticmethod(preflight)
    publish = staticmethod(publish)
    units = staticmethod(units)
    decode = staticmethod(decode)
    qualify = staticmethod(qualify)
    predict = staticmethod(prediction.prediction_record)
    fit = staticmethod(guard.fit_confirmation)

    @staticmethod
    def model(Sref, theta):
        from experiments.mlst2026.hcp_rnoise_20261019.signal import build
        return build(Sref, theta)


def start_record(evidence, out):
    """Distinct ownership even in same-path clones with identical Git clocks.

    This token affects only the access ledger, never study seeds or numerics.
    publish() verifies the complete token-bearing blob on origin/main. Without
    it, identical commits could make a second push idempotently succeed.
    """
    return {**evidence, 'status': 'STARTED', 'no_restart': True,
            'run_token': secrets.token_hex(32), 'output': str(Path(out).absolute()),
            'all60': True, 'resamples': 50, 'n_min': 18}


def run(ms, registration_record, registered_bundle, output, *, backend=None):
    """Once-only orchestration; there is deliberately no restart/resume API."""
    backend = Runtime() if backend is None else backend
    evidence = backend.preflight(ms, registration_record, registered_bundle)
    # No recording metadata/path/reader work takes place above this line.
    out = Path(output)
    out.mkdir(exist_ok=False)  # atomic claim, including empty/partially started outputs
    fsync_directory(out.parent)
    # The immutable checksum bundle, rather than an output name or tag date,
    # identifies the once-only run across different clones and authorizations.
    public = 'reviews/hcp_rnoise_confirmation_' + registered_bundle
    start = start_record(evidence, out)
    qualification, details, predictions, records = {}, {}, {}, []
    current_date = None
    started_public = False
    try:
        write_new(out / 'STARTED.json', encoded(start))
        # A registration-scoped, public once-only claim also prevents reruns into
        # another directory or checkout. A rejected concurrent push is terminal.
        backend.publish(ms, public + '/STARTED.json', encoded(start), 'HCP confirmation: once-only START')
        started_public = True
        selected = backend.units()
        expected_dates = protocol.load()['access']['confirmation_analysis_dates']
        if (len(selected) != 60 or [u['measurement_date'] for u in selected] != expected_dates
                or [{'date': u['measurement_date'], 'subject_id': u['subject_id']} for u in selected]
                != protocol.load()['confirmation_units']):
            raise PermissionError('STOP all60 fixed units required; no replacement')
        # Qualify every unit before inference/fitting. Passing arrays stay local.
        for i, unit in enumerate(selected):
            current_date = unit['measurement_date']
            try:
                data = backend.decode(unit)
            except Exception as exc:
                from experiments.mlst2026.hcp_rnoise_20261019.preprocess import Excluded
                if not isinstance(exc, Excluded):
                    raise
                outcome = {k: {'pass': False, 'evaluated': False, 'reason': str(exc)} for k in GATES}
                roles = S = None
            else:
                outcome, roles, S = backend.qualify(data, unit, 12 + i)
            flags = qualification_booleans(outcome)
            qualification[current_date] = flags
            details[current_date] = {'date_index': 12 + i, 'subject_id': unit['subject_id'], **outcome}
            write_new(out / 'qualification' / (current_date + '.json'), encoded(details[current_date]))
            if all(flags.values()):
                # Private local arrays avoid decoding twice or opening a replacement.
                for role, (X, meta) in roles.items():
                    path = out / 'arrays' / current_date / (role + '.npz')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as h:
                        np.savez(h, X=X, **meta)
                write_new(out / 'arrays' / current_date / 'Sref.json', encoded(S))
        selection = protocol.cohort(qualification)
        write_new(out / 'QUALIFICATION.json', encoded({'units': details, 'booleans': qualification, 'cohort': selection}))
        if selection['status'] != 'PASS':
            result = summarize([], {}, qualification)
            write_new(out / 'SUMMARY.json', encoded(result))
            raise RuntimeError('STOP below18 passing dates; no confirmatory claim')
        for current_date in selection['included']:
            roles = {}
            for role in ('ref', 'train', 'cal', 'eval'):
                with np.load(out / 'arrays' / current_date / (role + '.npz'), allow_pickle=False) as a:
                    roles[role] = (a['X'], {k: a[k] for k in a.files if k != 'X'})
            S = np.asarray(json.loads((out / 'arrays' / current_date / 'Sref.json').read_bytes()), float)
            p = backend.predict(S, current_date)
            prediction.validate_record(p, current_date)
            if prediction.sref_hash(S) != p['Sref_sha256_f64le']:
                raise PermissionError('STOP qualification/prediction Sref mismatch')
            predictions[current_date] = p
            prediction_path, prediction_commit = backend.publish(ms, public + '/predictions/' + current_date + '.json',
                encoded(p), 'HCP confirmation pre-fit predictions ' + current_date)
            # Publish/verify all fixed+adaptive predictions before the first fit.
            for g, grid in prediction.GRIDS.items():
                geo = p['geometries'][g]
                if g != 'control' and not geo['available']:
                    write_new(out / 'unavailable' / (current_date + '_' + g + '.json'),
                              encoded({'date': current_date, 'geometry': g, **unavailable(geo.get('reason') or 'prediction unavailable')}))
                    continue
                model = backend.model(S, geo['theta'])
                for N in grid:
                    for arm, m in ARMS:
                        for replicate in range(50):
                            row = backend.fit(roles, model, S, N, m, arm, p['date_index'], replicate,
                                ms=ms, registration_record=registration_record, date=current_date,
                                prediction_path=prediction_path, prediction_commit=prediction_commit, geometry=g)
                            if not np.isfinite(row['D']):
                                raise RuntimeError('STOP nonfinite learning contrast')
                            # Preserve exact full kernel result and every diagnostic.
                            record = {'date': current_date, 'geometry': g, 'theta': g, 'N': N, 'arm': arm, 'm': m,
                                      'replicate': replicate, 'kernel': row, 'D': row['D'],
                                      'prediction_commit': prediction_commit}
                            with (out / 'FITS.jsonl').open('ab') as h:
                                h.write(json.dumps(json.loads(encoded(record)), sort_keys=True, separators=(',', ':')).encode() + b'\n')
                                h.flush()
                                os.fsync(h.fileno())
                            records.append(record)
        result = summarize(records, predictions, qualification)
        write_new(out / 'SUMMARY.json', encoded(result))
        terminal = {'status': 'COMPLETE', 'cohort': selection, 'fits': len(records), 'no_restart': True,
                    'run_token': start['run_token'], 'bundle_sha256': registered_bundle}
        backend.publish(ms, public + '/COMPLETED.json', encoded(terminal), 'HCP confirmation completion')
        write_new(out / 'COMPLETED.json', encoded(terminal))
        return result
    except BaseException as exc:
        stop = {'status': 'STOP', 'no_restart': True, 'confirmatory_claim_allowed': False,
                'run_token': start['run_token'], 'bundle_sha256': registered_bundle,
                'date': current_date, 'exception': type(exc).__name__, 'reason': str(exc),
                'qualification': qualification, 'unprocessed_dates': [d for d in protocol.load()['access']['confirmation_analysis_dates'] if d not in qualification],
                'completed_fits': len(records), 'outcomes': {f'PZ{i}': unavailable('run STOP: ' + str(exc)) for i in range(1, 6)}}
        write_new(out / 'STOP.json', encoded(stop))
        if started_public:
            try:
                backend.publish(ms, public + '/STOP.json', encoded(stop), 'HCP confirmation STOP; no restart')
            except Exception as publication_error:
                write_new(out / 'STOP_PUBLICATION_FAILED.json', encoded({'reason': str(publication_error)}))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog='confirmation-run', description=__doc__)
    parser.add_argument('--ms', required=True)
    parser.add_argument('--registration-record', required=True)
    parser.add_argument('--registered-bundle', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.ms, args.registration_record, args.registered_bundle, args.output)
    except (Exception, KeyboardInterrupt) as exc:
        parser.exit(2, 'STOP ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
