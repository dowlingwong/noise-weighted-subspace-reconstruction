"""Repository handoff acceptance guards; never execute a production cell."""
import json
from pathlib import Path
import pytest
from experiments.mlst2026.successor_20260913.remote import prepare as R


def test_pilot_reduce_precedes_confirmation():
    for name in ('t20s','t21s','t22s'):
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
