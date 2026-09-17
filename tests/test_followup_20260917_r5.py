"""R5 (optional T19 reachability) local acceptance tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026.successor_20260913.common import protocol as PR  # noqa: E402
from experiments.mlst2026.followup_20260917.r5_t19_reach import covariance as cov  # noqa: E402
from experiments.mlst2026.followup_20260917.r5_t19_reach import reachability as reach  # noqa: E402
from experiments.mlst2026.followup_20260917.r5_t19_reach import runner as RN  # noqa: E402

PROTO = RN.PROTOCOL_DEFAULT


def main(*argv):
    return RN.main([argv[0], "--protocol", str(PROTO)] + list(argv[1:]))


def test_validate_and_enumerate_compute_nothing(tmp_path):
    assert main("validate") == 0
    out = tmp_path / "enum"
    assert main("enumerate", "--out", str(out)) == 0
    assert not (out / "cells").exists()
    s = json.loads((out / "run_matrix_summary.json").read_text())
    # ONE cell per seed group over the full m ladder (nesting), not one per m
    assert s["by_role"]["confirmation"]["cells"] == 1 and s["fits"] == 80


def test_superseded_v1_protocol_is_refused():
    """R5-v1 is retained for the record but must never execute: validate/run
    refuse it the same way R1/R4 refuse their superseded v1 protocols."""
    v1 = PROTO.parent / "PROTOCOL_R5_T19_REACH_v1.json"
    assert v1.exists()
    rc = RN.main(["validate", "--protocol", str(v1)])
    assert rc != 0
    P = PR.load_protocol(v1)
    try:
        RN.build_matrix(P, False)
    except PR.ProtocolError as exc:
        assert "superseded" in str(exc)
    else:
        raise AssertionError("R5-v1 must be refused by build_matrix")


def test_runner_fixture_path_nests_across_m(tmp_path):
    """The runner's own (fixture-scale) path carries the full ladder in one cell,
    derives nested prefixes, and accounts one fit per (seed, m)."""
    P = PR.load_protocol(PROTO)
    m = RN.build_matrix(P, True)
    reach_cells = [c for c in m.cells if c["family"] == "reach"]
    assert len(reach_cells) == 1
    mg = reach_cells[0]["params"]["m_grid"]
    assert mg == P["matrix"]["fixture"]["m_grid"]
    assert reach_cells[0]["n_fits"] == P["matrix"]["fixture"]["n_seeds"] * len(mg)
    result, fits, _ = RN.run_cell(P, reach_cells[0], True, {"out": tmp_path})
    assert fits == P["matrix"]["fixture"]["n_seeds"] * len(mg)
    assert all("parent_seed" in r and "stream_seed" in r for r in result["records"])
    assert {r["m"] for r in result["records"]} == set(mg)


def test_streaming_covariance_matches_direct_construction():
    rng = np.random.default_rng(11)
    d, m, chunk = 7, 1500, 128
    draws = rng.standard_normal((m, d))
    direct = cov.direct_covariance(draws)
    it = iter(draws)
    streamed = cov.streaming_covariance(lambda n: np.array([next(it) for _ in range(n)]), m, d, chunk)
    assert np.max(np.abs(direct - streamed)) < 1e-10


def test_nested_arms_are_prefixes_of_one_stream():
    rng = np.random.default_rng(12)
    d, chunk = 5, 64
    draws = rng.standard_normal((900, d))
    it = iter(draws)
    covs = cov.nested_streaming_covariances(lambda n: np.array([next(it) for _ in range(n)]), [300, 900], d, chunk)
    assert np.max(np.abs(covs[300] - cov.direct_covariance(draws[:300]))) < 1e-10


def test_reachability_eta_declines_with_m_and_keeps_out_of_domain():
    from experiments.mlst2026.successor_20260913.t19s import synthetic as S
    g = S.make_generator(24)
    recs = reach.reachability_records(42, g, m_grid=[400, 3200], n_train=48, chunk=256, n_seeds=2)
    s = reach.summarise(recs, [400, 3200])
    assert s["by_m"]["400"]["eta_median"] > s["by_m"]["3200"]["eta_median"]
    assert s["by_m"]["400"]["n"] == 2 and s["by_m"]["3200"]["n"] == 2   # nothing discarded
    assert "never_discarded" in s


def test_fixture_end_to_end_and_controls(tmp_path):
    out = tmp_path / "fx"
    assert main("run", "--out", str(out), "--fixture") == 0
    assert main("reduce", "--out", str(out)) == 0
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["development_only"] is True
    assert all(v["passed"] for v in red["controls"].values())
    assert "reachability" in red and "eta_scaling" in red
    assert red["reachability"]["never_discarded"]


def test_namespaces_fresh_and_disjoint():
    P = PR.load_protocol(PROTO)
    for a in ("development", "pilot", "confirmation"):
        for b in ("development", "pilot", "confirmation"):
            if a < b:
                lo_a, hi_a = P["seed_namespaces"][a]
                lo_b, hi_b = P["seed_namespaces"][b]
                assert hi_a < lo_b or hi_b < lo_a
