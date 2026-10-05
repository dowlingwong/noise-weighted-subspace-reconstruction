import json
import sys
import types

import pytest
from experiments.mlst2026.t25ns_20261005 import guard


def test_shipped_closure_is_the_accepted_t25_identity():
    closure = guard.load_closure()
    assert guard.closure_identity(closure) == guard.ACCEPTED_T25_IDENTITY
    assert closure['EXP_head'] == '854a6216a1c20f31a70482086dae904b71816d03'
    assert closure['MS_head'] == 'f219c8399d5719060d130f9bddb6234bee0f2c23'
    tampered = json.loads(json.dumps(closure))
    key = 'experiments/mlst2026/t25_production_20260930/science.py'
    tampered['EXP_files'][key] = '0' * 64
    assert guard.closure_identity(tampered) != guard.ACCEPTED_T25_IDENTITY


def test_every_reused_file_is_accepted_byte_for_byte():
    got = guard.assert_files(guard.T25_REUSED)
    assert set(guard.T25_DIRECT) <= set(got) and len(got) == len(guard.T25_REUSED)
    guard.assert_direct()
    guard.assert_loaded_modules()


def test_mismatch_and_foreign_files_fail_closed():
    closure = guard.load_closure()
    bad = json.loads(json.dumps(closure))
    bad['EXP_files'][guard.T25_DIRECT[2]] = 'f' * 64
    with pytest.raises(guard.AcceptedCodeMismatch, match='differs from the accepted'):
        guard.assert_files([guard.T25_DIRECT[2]], bad)
    with pytest.raises(guard.AcceptedCodeMismatch, match='not part of the accepted'):
        guard.assert_files(['experiments/mlst2026/t25ns_20261005/science.py'], closure)


def test_repo_packages_must_come_from_this_repository(tmp_path):
    fake = types.ModuleType('experiments.fake_outside')
    fake.__file__ = str(tmp_path / 'fake.py')
    sys.modules['experiments.fake_outside'] = fake
    try:
        with pytest.raises(guard.AcceptedCodeMismatch, match='outside'):
            guard.assert_repo_origin()
    finally:
        del sys.modules['experiments.fake_outside']
