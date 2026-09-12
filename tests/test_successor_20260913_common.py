"""Behavioural tests of the shared local-package infrastructure, using a toy
package that fits nothing.  Every rule the amendment asks for is exercised
here once so that the four scientific packages inherit it."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.mlst2026.successor_20260913.common import cli, ledger as L, matrix as M, protocol as PR, results as R  # noqa: E402

TOY_PROTOCOL = {
    "schema": PR.SCHEMA, "package": "toy", "version": "2026-09-13", "title": "toy", "question": "q",
    "hypotheses": ["h"], "independent_unit": "seed", "estimands": {"m": "mean"}, "controls": ["zero"],
    "seed_namespaces": {"development": [900000, 900099], "pilot": [1000, 1019], "confirmation": [2000, 2099]},
    "inference": {"rule": "fixed"}, "stopping_rule": "fixed allocation: all matrix cells, no early stop",
    "failure_interpretation": "report", "budget": {"estimator_fits_cap": 6, "fixture_to_production_scale": {"default": 10.0}},
    "explicitly_not_claimed": ["anything"],
}


def _toy_matrix(P, fixture):
    cells = []
    seeds = range(900000, 900003) if fixture else range(1000, 1003)
    role = "development" if fixture else "pilot"
    for s in seeds:
        cells.append(M.make_cell(role=role, seed=s, params={"x": 1}, n_fits=2, family="A"))
    cells.append(M.make_cell(role="algebraic", seed=None, params={"check": "identity"}, n_fits=0, n_solves=1, family="alg"))
    return M.RunMatrix("toy", P["_sha256"], cells)


def _toy_run_cell(P, cell, fixture, ctx):
    if cell["role"] == "algebraic":
        return {"value": 0.0}, 0, 1
    rng = np.random.default_rng(cell["seed"])
    return {"value": float(rng.normal())}, 2, 0


def _toy_reduce(P, matrix, cells, out, partial):
    vals = [c["result"]["value"] for c in cells.values() if c["role"] != "algebraic"]
    return {"mean": float(np.mean(vals)) if vals else None, "n": len(vals)}


def _main(argv, tmp_protocol):
    return cli.main_template(package="toy", description="toy", build_matrix=_toy_matrix, run_cell=_toy_run_cell,
                             reduce_run=_toy_reduce, result_required={"value": float}, source_files=[Path(__file__)],
                             argv=[argv[0], "--protocol", str(tmp_protocol)] + argv[1:])


@pytest.fixture
def proto(tmp_path):
    p = tmp_path / "PROTOCOL_TOY.json"
    p.write_text(json.dumps(TOY_PROTOCOL))
    return p


def test_protocol_rejects_outcome_dependent_stopping(tmp_path):
    bad = dict(TOY_PROTOCOL, stopping_rule="budget spent, or acceptance decided, whichever first")
    with pytest.raises(PR.ProtocolError):
        PR.validate_protocol(bad)


def test_protocol_rejects_overlapping_namespaces():
    bad = dict(TOY_PROTOCOL, seed_namespaces={"development": [0, 10], "pilot": [5, 20], "confirmation": [30, 40]})
    with pytest.raises(PR.ProtocolError):
        PR.validate_protocol(bad)


def test_validate_and_enumerate_compute_nothing(proto, tmp_path, capsys):
    assert _main(["validate"], proto) == 0
    out = tmp_path / "enum"
    assert _main(["enumerate", "--out", str(out)], proto) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    summ = json.loads((out / "run_matrix_summary.json").read_text())
    assert summ["fits"] == 6 and summ["solves"] == 1 and summ["by_role"]["pilot"]["cells"] == 3


def test_run_refuses_archive_roots(proto):
    rc = _main(["run", "--out", str(REPO / "results" / "toy_should_never_exist")], proto)
    assert rc == 6
    assert not (REPO / "results" / "toy_should_never_exist").exists()


def test_fixture_uses_development_namespace_only(proto, tmp_path):
    out = tmp_path / "fx"
    assert _main(["run", "--out", str(out), "--fixture"], proto) == 0
    cells = R.read_cells(out)
    seeds = {js["seed"] for files in cells.values() for _, js in files if js["seed"] is not None}
    assert seeds and all(900000 <= s <= 900099 for s in seeds)
    assert all(js["development_only"] for files in cells.values() for _, js in files)
    assert _main(["reduce", "--out", str(out)], proto) == 0
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["development_only"] is True and red["reconciliation"]["complete"]


def test_full_run_reduce_is_deterministic_and_resumable(proto, tmp_path):
    out = tmp_path / "run"
    # partial run: stop after 2 cells, then resume
    assert _main(["run", "--out", str(out), "--max-cells", "2"], proto) == 0
    rc = _main(["reduce", "--out", str(out)], proto)
    assert rc == 3, "reduce must refuse a partial run without --allow-partial"
    assert _main(["reduce", "--out", str(out), "--allow-partial"], proto) == 0
    assert json.loads((out / "REDUCED.json").read_text())["partial"] is True
    assert _main(["run", "--out", str(out)], proto) == 0
    assert _main(["reduce", "--out", str(out)], proto) == 0
    a = json.loads((out / "REDUCED.json").read_text())
    assert _main(["reduce", "--out", str(out)], proto) == 0
    b = json.loads((out / "REDUCED.json").read_text())
    assert a["mean"] == b["mean"] and a["partial"] is False
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=6, package="toy", protocol_sha256="x")
    assert led.charged()["fits"] == 6 and led.charged()["by_state"]["completed"] == 4


def test_failure_is_charged_and_retry_accumulates(proto, tmp_path):
    out = tmp_path / "fail"
    P = PR.load_protocol(proto)
    m = _toy_matrix(P, False)
    victim = [c for c in m.cells if c["role"] == "pilot"][0]["cell_id"]
    rc = _main(["run", "--out", str(out), "--fail-cell", victim], proto)
    assert rc == 1
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=6, package="toy", protocol_sha256="x")
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1 and ch["fits"] == 2  # the failed attempt stays charged
    # the retry needs 6 more fits but only 4 remain under the cap -> the third pilot cell is refused
    rc = _main(["run", "--out", str(out)], proto)
    assert rc == 2
    ch = led.charged()
    assert ch["fits"] == 6 and ch["by_state"]["completed"] == 2 and ch["by_state"]["failed"] == 1


def test_duplicate_completion_refused_and_repeat_flagged(proto, tmp_path):
    out = tmp_path / "dup"
    assert _main(["run", "--out", str(out)], proto) == 0
    # re-running is a no-op skip; forcing a repeat needs room under the cap -> refused by budget
    assert _main(["run", "--out", str(out)], proto) == 0
    P = PR.load_protocol(proto)
    m = _toy_matrix(P, False)
    alg = [c for c in m.cells if c["role"] == "algebraic"][0]["cell_id"]
    rc = _main(["run", "--out", str(out), "--cells", alg, "--allow-repeat"], proto)
    assert rc == 0
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=6, package="toy", protocol_sha256="x")
    assert led.summary()["cells_with_repeat_completion"] == [alg]
    # a duplicated output file (copied by hand) is detected by reduce
    cells = list((out / "cells").glob("*.json"))
    src = cells[0]
    js = json.loads(src.read_text())
    js["cell_id"] = "deadbeefdeadbeef"
    (out / "cells" / "deadbeefdeadbeef.json").write_text(json.dumps(js))
    assert _main(["reduce", "--out", str(out)], proto) == 3


def test_tampered_output_is_rejected_at_reduce(proto, tmp_path):
    out = tmp_path / "tamper"
    assert _main(["run", "--out", str(out)], proto) == 0
    p = sorted((out / "cells").glob("*.json"))[0]
    js = json.loads(p.read_text())
    js["result"]["value"] = 42.0
    p.write_text(json.dumps(js))
    assert _main(["reduce", "--out", str(out)], proto) == 3


def test_interrupted_attempt_is_swept_and_charged(proto, tmp_path):
    out = tmp_path / "intr"
    P = PR.load_protocol(proto)
    m = _toy_matrix(P, False)
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=6, package="toy", protocol_sha256=P["_sha256"])
    cell = [c for c in m.cells if c["role"] == "pilot"][0]
    led.start(cell)
    n = led.sweep_interrupted(is_alive=lambda pid, host: False)
    assert n == 1
    ch = led.charged()
    assert ch["by_state"]["interrupted"] == 1 and ch["fits"] == 2


def test_help_never_computes(proto, tmp_path):
    with pytest.raises(SystemExit) as e:
        _main(["--help"], proto)
    assert e.value.code == 0
