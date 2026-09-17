"""R1 (T22 interval calibration) local acceptance tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402
from experiments.mlst2026.successor_20260913.common import protocol as PR  # noqa: E402
from experiments.mlst2026.successor_20260913.common.results import read_cells  # noqa: E402
from experiments.mlst2026.followup_20260917.r1_t22_calibration import candidates as C  # noqa: E402
from experiments.mlst2026.followup_20260917.r1_t22_calibration import runner as RN  # noqa: E402

PROTO = RN.PROTOCOL_DEFAULT


def main(*argv):
    return RN.main([argv[0], "--protocol", str(PROTO)] + list(argv[1:]))


@pytest.fixture(scope="module")
def pilot_root(tmp_path_factory):
    out = tmp_path_factory.mktemp("r1") / "pilot"
    assert main("run", "--out", str(out), "--fixture", "--families", "sim_pilot", "control") == 0
    assert main("reduce", "--out", str(out), "--allow-partial") == 0
    return out


@pytest.fixture(scope="module")
def full_root(tmp_path_factory, pilot_root):
    import shutil
    out = tmp_path_factory.mktemp("r1") / "full"
    shutil.copytree(pilot_root, out)
    assert main("run", "--out", str(out), "--fixture", "--families", "sim_confirm") == 0
    assert main("reduce", "--out", str(out)) == 0
    return out


def test_validate_and_enumerate_compute_nothing(tmp_path):
    assert main("validate") == 0
    out = tmp_path / "enum"
    assert main("enumerate", "--out", str(out)) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    s = json.loads((out / "run_matrix_summary.json").read_text())
    assert s["by_role"]["pilot"]["cells"] == 32 and s["by_role"]["confirmation"]["cells"] == 690
    assert s["fits"] == 0


def test_protocol_declares_new_version_and_frozen_candidates():
    P = PR.load_protocol(PROTO)
    assert P["schema"] == PR.SCHEMA and P["package"] == "r1_t22_calibration"
    assert P["supersedes"]["package"] == "t22s"
    assert "designed_after" in P["provenance"]
    assert tuple(P["candidate_family"]) == C.CANDIDATES
    assert C.all_targets_documented()
    assert P["calibration_tolerances"]["coverage"] == [0.925, 0.975]
    P0 = PR.load_protocol(REPO / "experiments/mlst2026/successor_20260913/t22s/PROTOCOL_T22S_v2.json")
    for name in ("development", "pilot", "confirmation"):
        a, b = P["seed_namespaces"][name], P0["seed_namespaces"][name]
        assert a[1] < b[0] or b[1] < a[0], f"R1 {name} namespace collides with the archived t22s namespace"


def test_candidate_math_ordering_pivot_and_coverage():
    pf = C.PerFile(y=np.array([1.0, 1.1, 0.9, 1.05]), s=np.array([0.2, 0.3, 0.25, 0.22]), n_bootstrap=200)
    ivs = C.candidate_intervals(pf, two_stage_point=1.0, two_stage_se=0.1)
    for name, v in ivs.items():
        assert v["lo"] <= v["point"] <= v["hi"], name
    # coherent p follows the same t pivot
    from scipy import stats as st
    p = C.coherent_pvalue("t_equal_file", ivs["t_equal_file"])
    expect = 2.0 * (1.0 - st.t.cdf(abs(ivs["t_equal_file"]["point"]) / ivs["t_equal_file"]["se"], ivs["t_equal_file"]["dof"]))
    assert abs(p - expect) < 1e-12
    cov = C.candidate_coverage(ivs, 1.0)
    assert set(cov) == set(C.CANDIDATES)


def test_namespace_disjoint_and_production_refuses_confirmation_without_freeze(tmp_path):
    out = tmp_path / "refuse"
    out.mkdir()
    (out / RN.FROZEN_NAME).write_text("{}")
    assert main("run", "--out", str(out), "--fixture", "--roles", "confirmation") == 6
    assert not (out / "cells").exists()


def test_fixture_end_to_end_freezes_a_candidate(full_root):
    red = json.loads((full_root / "REDUCED.json").read_text())
    assert red["development_only"] is True and red["reconciliation"]["complete"]
    fz = json.loads((full_root / RN.FROZEN_NAME).read_text())
    assert fz["chosen_candidate"] in set(C.CANDIDATES) | {"none"}
    assert fz["development_only"] is True
    assert "confirmation" in red and "frozen_candidate" in red
    for js in {k: v[0][1] for k, v in read_cells(full_root).items()}.values():
        assert js["development_only"]


def test_tampered_record_is_refused(pilot_root, tmp_path):
    import shutil
    dup = tmp_path / "tamper"
    shutil.copytree(pilot_root, dup)
    p = [q for q in sorted((dup / "cells").glob("*.json")) if json.loads(q.read_text())["family"] == "sim_pilot"][0]
    js = json.loads(p.read_text())
    js["result"]["records"][0]["candidates"]["raw-bin|empca"]["t_equal_file"]["hi"] += 1e-3
    p.write_text(json.dumps(js))
    assert main("reduce", "--out", str(dup), "--allow-partial") == 3


def test_selection_rule_is_frozen_and_not_reoptimised(full_root):
    fz = json.loads((full_root / RN.FROZEN_NAME).read_text())
    assert "all-scenario" in fz["adequacy_rule"]["rule"] or "ADEQUATE" in fz["adequacy_rule"]["rule"]
    assert fz["reason"]
    assert fz["frozen_from_pilot_cells"]


# --- A2: the R1 acceptance receipt consumed by the R4 gate ----------------------------
def test_acceptance_receipt_is_written_and_marks_development_only(full_root):
    red = json.loads((full_root / "REDUCED.json").read_text())
    acc = red["confirmation"]["acceptance"]
    assert acc["development_only"] is True
    assert set(acc["coverage_ok"]) <= {"4", "8", "16"}
    assert set(acc["type1_ok"]) <= {"4", "8", "16"}
    out_file = full_root / "R1_ACCEPTANCE.json"
    assert RN.main(["acceptance", "--run-root", str(full_root), "--out-file", str(out_file)]) == 0
    assert json.loads(out_file.read_text())["frozen_candidate"] == acc["frozen_candidate"]


def test_r4_gate_refuses_a_development_only_r1_receipt(full_root):
    from experiments.mlst2026.followup_20260917.r4_t22_feasibility import r1_gate
    acc = full_root / "R1_ACCEPTANCE.json"
    g = r1_gate.evaluate_r1_precondition(full_root / RN.FROZEN_NAME, acc)
    # a present-but-uncitable (development-only) freeze is BLOCKED, not a scientific cancel
    assert g["status"] == "blocked" and g["cancel"] is False and g["blocked"] is True
    assert r1_gate.ready_to_transfer(g) is False
    assert r1_gate.large_file_admissible(g) is False
