"""N* table: pinned Y15 RESULT.json (T20-S rows), SHA-256 check, and the constants-path recomputation."""
import json

import pytest

from experiments.mlst2026.zc_baselines_20261013 import nstar, runner


def test_pinned_source_and_table():
    assert nstar.source_sha() == nstar.SOURCE_SHA256
    t = nstar.table()
    assert t['0.0'] is None
    assert [round(t[k]) for k in ('0.12', '0.25', '0.4')] == [5241, 1221, 488]
    proto = runner.load_protocol()
    assert proto['nstar']['table'] == t and proto['nstar']['source']['sha256'] == nstar.SOURCE_SHA256
    rows = [r for r in json.loads(nstar.SOURCE.read_text())['locations'] if r['study'] == 'T25-oracle']
    assert {repr(float(r['theta'])): r['N_star_pred'] for r in rows} == t          # T25-oracle rows agree


def test_modified_source_refused(tmp_path):
    p = tmp_path / 'RESULT.json'
    p.write_bytes(nstar.SOURCE.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA-256'):
        nstar.table(p)


def test_constants_path_recomputation():
    """Recompute N*_pred via Y19 predictions.constants (TR09 core.py) on the T25/T20-S generator."""
    t = nstar.table()
    new = nstar.recompute()
    assert new['0.0'] is None
    for k in ('0.12', '0.25', '0.4'):
        assert abs(new[k] - t[k]) / t[k] <= 1e-6, (k, new[k], t[k])
