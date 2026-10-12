"""Validation gate: the Y19 constants path reproduces the saved T25 weight table (T25 generator only)."""
from experiments.mlst2026.y19_20261012 import validation_gate


def test_t25_constants_reproduced():
    g = validation_gate.gate()
    for r in g['rows']:
        assert r['pass'], r
    assert g['pass']
