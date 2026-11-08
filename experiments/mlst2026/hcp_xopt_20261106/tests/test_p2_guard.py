"""p2 guard-calibration test (resume 5, post-freeze supervisor/safety fix).

The run tripped the replication guard on deepthought2 for a row that deepthought reproduced
bit-exactly: |delta D| = 1.85e-9 against a stored restart_spread of 3.4e-10 (ratio 5.45).
Fails before the fix (the tail case was flagged), passes after.
"""
from experiments.mlst2026.hcp_xopt_20261106 import run


def _res(obj_rel, d_abs, d_over):
    return dict(replay=dict(obj_rel=obj_rel, D_abs=d_abs, D_over_spread=d_over))


def test_relative_tail_with_small_absolute_is_not_a_breach():
    assert run._replay_breach(_res(1.49e-16, 1.846e-9, 5.454)) is None


def test_gross_absolute_breach_is_still_a_breach():
    assert run._replay_breach(_res(1e-16, 1e-3, 5.454)) is not None


def test_objective_breach_is_still_a_breach():
    assert run._replay_breach(_res(1e-6, 0.0, 0.0)) is not None


def test_zero_spread_with_gross_absolute_is_a_breach():
    assert run._replay_breach(_res(1e-16, 1e-3, None)) is not None


def test_zero_spread_with_tiny_absolute_is_not_a_breach():
    assert run._replay_breach(_res(1e-16, 1.846e-9, None)) is None


def test_nan_objective_is_still_a_breach():
    assert run._replay_breach(_res(float('nan'), 0.0, 0.0)) is not None
