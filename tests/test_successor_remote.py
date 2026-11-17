"""Repository handoff acceptance guards; never execute a production cell."""
import json
from pathlib import Path
import pytest
from experiments.mlst2026.successor_20260913.remote import prepare as R


def test_pilot_reduce_precedes_confirmation():
    for name in ('t19s','t20s','t21s','t22s'):
        _, phases = R.phase_plan(name)
        confirmation = next(i for i,(action,args) in enumerate(phases) if action=='run' and any('confirm' in x for x in args))
        assert any(action=='reduce' and '--allow-partial' in args for action,args in phases[:confirmation])
        assert phases[-1] == ('reduce',[])


def test_blocked_real_data_modes_are_not_launchable():
    for selection in ('t22-real','cresst-full'):
        with pytest.raises(ValueError,match='Unsupported'):
            R.phase_plan(selection)


def test_preflight_refuses_skipped_science_tests(tmp_path):
    p=tmp_path/'tests.xml'
    p.write_text('<testsuites><testsuite tests="129" failures="0" errors="0" skipped="1"/></testsuites>')
    with pytest.raises(ValueError,match='acceptance failed'):
        R.validate_test_report(p)
    p.write_text('<testsuites><testsuite tests="129" failures="0" errors="0" skipped="0"/></testsuites>')
    assert R.validate_test_report(p)['tests']==129


def test_manifest_detects_source_tampering(tmp_path,monkeypatch):
    exp=tmp_path/'exp';ms=tmp_path/'ms';exp.mkdir();ms.mkdir()
    p=exp/'source.py';p.write_text('original')
    m=tmp_path/'manifest.json'
    R.write(m,{'schema':'mlst-repository-handoff/1','files':{'experiment':{'source.py':R.digest(p)},'manuscript':{}},'matrices':{}})
    monkeypatch.setattr(R,'EXP',exp);monkeypatch.setattr(R,'matrix_records',lambda:{})
    R.verify(ms,m)
    p.write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        R.verify(ms,m)


def test_refuses_output_inside_source_repository(tmp_path):
    with pytest.raises(ValueError,match='outside'):
        R.require_external_out(tmp_path/'ms',R.EXP/'results/new')


def test_only_one_launcher_owns_run_root(tmp_path):
    with R.single_launcher(tmp_path):
        with pytest.raises(ValueError,match='Another launcher'):
            with R.single_launcher(tmp_path):
                pass


def test_launcher_refuses_a_third_retry(tmp_path):
    from experiments.mlst2026.successor_20260913.common import ledger as L, protocol as P
    protocol=P.load_protocol(R.ROUND/'t19s'/R.PROTOCOLS['t19s'])
    ledger=L.ExecutionLedger(tmp_path/'EXECUTION_LEDGER.jsonl',cap=4800,package='t19s',protocol_sha256=protocol['_sha256'])
    cell={'cell_id':'failed-cell','role':'pilot','n_fits':1}
    for _ in range(3):
        attempt=ledger.start(cell)
        ledger.fail(attempt,cell,reason='test failure',wall_s=0.0)
    with pytest.raises(ValueError,match='Two retries exhausted'):
        R.check_retry_limit(tmp_path,'t19s')


# ---------------------------------------------------------------------------
# Launcher phase-order regression coverage.
#
# The declarative guard above inspects only the ARGUMENTS a phase declares. It
# passes for a plan whose pre-freeze `run` phase is unrestricted, because an
# unrestricted phase names no role at all -- which is exactly the defect that
# stopped T19-S at 20/136 cells on 2026-09-13: `[('run',[]),('reduce',[])]`
# drew the whole matrix, reached the first confirmation cell at index 20 and
# was refused by the runner's frozen-selection guard.
#
# These tests resolve each phase against the package's real frozen matrix, so
# what the launcher would actually draw is what gets asserted.
# ---------------------------------------------------------------------------

PAIRED_PACKAGES = ('t19s', 't20s', 't21s', 't22s')


def _selector(extra, flag):
    """Values passed to `flag` in a phase's argument list, or None."""
    if flag not in extra:
        return None
    out = []
    for token in extra[extra.index(flag) + 1:]:
        if str(token).startswith('--'):
            break
        out.append(token)
    return out or None


def _frozen_matrix(pkg):
    """The package's production matrix, built in memory. Writes nothing."""
    import importlib
    from experiments.mlst2026.successor_20260913.common.protocol import load_protocol
    runner = importlib.import_module(
        f'experiments.mlst2026.successor_20260913.{pkg}.runner')
    protocol = load_protocol(R.ROUND / pkg / R.PROTOCOLS[pkg])
    return runner.build_matrix(protocol, False)


def _drawn_by(matrix, extra):
    roles = _selector(extra, '--roles')
    families = _selector(extra, '--families')
    if not roles and not families:
        return matrix          # an unrestricted phase draws the whole matrix
    return matrix.subset(roles=roles, families=families)


@pytest.mark.parametrize('selection', PAIRED_PACKAGES)
def test_no_run_phase_draws_a_confirmation_cell_before_the_pilot_freeze(selection):
    """The frozen selection must exist before any confirmation cell is drawn.

    Resolved against the real matrix, so an unrestricted pre-freeze run phase
    fails here even though it declares no confirmation argument.
    """
    pkg, phases = R.phase_plan(selection)
    matrix = _frozen_matrix(pkg)
    frozen = False
    for index, (action, extra) in enumerate(phases):
        if action == 'reduce':
            frozen = True
            continue
        confirmation = [c for c in _drawn_by(matrix, extra).cells
                        if c['role'] == 'confirmation']
        if confirmation and not frozen:
            raise AssertionError(
                f'{selection}: phase {index} {(action, extra)} would draw '
                f'{len(confirmation)} confirmation cell(s) before any reduce has '
                f'frozen PILOT_SELECTION.json; the runner refuses the first one '
                f'(e.g. {confirmation[0]["cell_id"]}) and production stops')


@pytest.mark.parametrize('selection', PAIRED_PACKAGES)
def test_production_phases_cover_every_frozen_cell(selection):
    """No frozen production cell may be silently dropped by the phase plan.

    Restricting phases by role/family is how the ordering defect is fixed;
    this keeps that fix from quietly narrowing the experiment.
    """
    pkg, phases = R.phase_plan(selection)
    matrix = _frozen_matrix(pkg)
    expected = {c['cell_id'] for c in matrix.cells if c['role'] != 'development'}
    covered = set()
    for action, extra in phases:
        if action == 'run':
            covered |= {c['cell_id'] for c in _drawn_by(matrix, extra).cells}
    missing = expected - covered
    assert not missing, (
        f'{selection}: {len(missing)} frozen cell(s) are never run by any phase, '
        f'e.g. {sorted(missing)[:3]}')


@pytest.mark.parametrize('selection', PAIRED_PACKAGES)
def test_no_unrestricted_run_phase_precedes_a_reduce(selection):
    """Regression guard for the 2026-09-13 T19-S stop.

    An unrestricted `run` before the pilot freeze is the defect itself; it can
    only be safe by accident of matrix ordering.
    """
    _, phases = R.phase_plan(selection)
    for index, (action, extra) in enumerate(phases):
        if action == 'reduce':
            break
        assert _selector(extra, '--roles') or _selector(extra, '--families'), (
            f'{selection}: phase {index} is an unrestricted run before the first '
            f'reduce; it draws the entire matrix including confirmation cells')


# ---------------------------------------------------------------------------
# Launcher FLOW coverage.
#
# The tests above check which cells a phase would draw. These drive the real
# `execute()` phase loop end to end against an emulated package, so the ordering
# contract is exercised as control flow: pilot completes, the partial reduction
# succeeds and writes the freeze, confirmation runs only then, and the final
# reduction closes the run. The refusal paths are covered too.
#
# Nothing production is touched: tmp_path roots, an emulated package, and the
# real frozen protocol is read but never modified.
# ---------------------------------------------------------------------------

FREEZE = 'PILOT_SELECTION.json'


def _receipt(tmp_path, manifest):
    r = tmp_path / 'PREFLIGHT.json'
    R.write(r, {'schema': 'mlst-preflight/1', 'manifest_sha256': R.digest(manifest),
                'environment': R.environment(), 'tests': {'tests': 136, 'failures': 0,
                                                          'errors': 0, 'skipped': 0}})
    return r


def _emulated_package(out, trace, *, reduce_fails_at=None, freeze_on_partial=True):
    """Stand-in for the package CLI, enforcing the same ordering contract."""
    def fake_command(argv, logfile, env=None):
        argv = [str(a) for a in argv]
        action = argv[3]
        extra = argv[8:]
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        Path(logfile).write_text(f'emulated {action} {extra}\n')
        trace.append((action, tuple(extra)))
        cells = out / 'cells'
        cells.mkdir(parents=True, exist_ok=True)
        if action == 'run':
            roles = _selector(extra, '--roles') or []
            if 'confirmation' in roles and not (out / FREEZE).exists():
                raise RuntimeError(
                    f'confirmation cell refused: {out / FREEZE} is absent; '
                    'reduce the pilot cells first')
            for role in roles:
                (cells / f'{role}.json').write_text('{}')
        else:
            partial = '--allow-partial' in extra
            if reduce_fails_at == ('partial' if partial else 'final'):
                raise RuntimeError(f'Command exited 1; emulated {"partial" if partial else "final"} reduction failure')
            if partial:
                if not (cells / 'pilot.json').exists():
                    raise RuntimeError('partial reduction has no pilot cells to freeze')
                if freeze_on_partial:
                    R.write(out / FREEZE, {'frozen': True})
            else:
                if not (out / FREEZE).exists():
                    raise RuntimeError('final reduction without a frozen selection')
            R.write(out / 'REDUCED.json', {'action': action, 'partial': partial})
    return fake_command


def _run_t19_flow(tmp_path, monkeypatch, **kw):
    trace = []
    out = tmp_path / 'root'
    manifest = R.ROUND / 'remote/MANIFEST.json'
    monkeypatch.setattr(R, 'verify', lambda ms, m: None)
    monkeypatch.setattr(R, 'command', _emulated_package(out, trace, **kw))
    R.execute(tmp_path / 'ms', manifest, _receipt(tmp_path, manifest), out, 't19s')
    return trace, out


def test_t19_launcher_flow_pilot_then_freeze_then_confirmation_then_final(tmp_path, monkeypatch):
    """pilot completion -> partial reduction + freeze -> confirmation -> final reduction."""
    trace, out = _run_t19_flow(tmp_path, monkeypatch)

    actions = [a for a, _ in trace]
    assert actions == ['run', 'reduce', 'run', 'reduce'], actions

    # the first run carries no confirmation; the second is the confirmation run
    assert 'confirmation' not in (_selector(trace[0][1], '--roles') or [])
    assert _selector(trace[1][1], '--allow-partial') is None and '--allow-partial' in trace[1][1]
    assert _selector(trace[2][1], '--roles') == ['confirmation']
    assert trace[3][1] == ()

    # the freeze existed before the confirmation run, and the run closed reduced
    assert (out / FREEZE).exists()
    assert (out / 'REDUCED.json').exists()

    # every phase reduction was retained in the invocation logs
    logs = sorted((out / 'logs').glob('*/*REDUCED.json'))
    assert len(logs) == 2, [p.name for p in logs]


def test_t19_launcher_stops_when_the_pilot_reduction_fails(tmp_path, monkeypatch):
    """A failed partial reduction must stop the run before any confirmation cell."""
    with pytest.raises(RuntimeError, match='reduction failure'):
        _run_t19_flow(tmp_path, monkeypatch, reduce_fails_at='partial')
    out = tmp_path / 'root'
    assert not (out / FREEZE).exists()
    assert not (out / 'cells' / 'confirmation.json').exists()


def test_t19_launcher_refuses_confirmation_when_the_freeze_is_absent(tmp_path, monkeypatch):
    """If the partial reduction leaves no freeze, confirmation must be refused."""
    with pytest.raises(RuntimeError, match='confirmation cell refused'):
        _run_t19_flow(tmp_path, monkeypatch, freeze_on_partial=False)
    out = tmp_path / 'root'
    assert not (out / 'cells' / 'confirmation.json').exists()


def test_real_t19_runner_refuses_a_confirmation_cell_without_the_freeze(tmp_path):
    """The production guard itself, on a real frozen confirmation cell.

    This is the consumer the launcher must satisfy; it refuses before computing.
    """
    from experiments.mlst2026.successor_20260913.t19s import runner as T19
    from experiments.mlst2026.successor_20260913.common.protocol import load_protocol
    protocol = load_protocol(R.ROUND / 't19s' / R.PROTOCOLS['t19s'])
    cell = next(c for c in T19.build_matrix(protocol, False).cells
                if c['role'] == 'confirmation')
    assert not (tmp_path / FREEZE).exists()
    with pytest.raises(RuntimeError, match='confirmation cell refused'):
        T19.run_cell(protocol, cell, False, {'out': tmp_path})
