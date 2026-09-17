"""R6 instrument-readiness acceptance tests.

Everything here runs against the R6 fail-closed interfaces and, for wiring only, a
small synthetic HDF5 / NPZ fixture.  **No instrument file exists in this
environment and none is opened.**  Every fixture output is labelled
``synthetic-fixture, not real data, not validated on real inputs`` and is never a
scientific result.

Covered, as the R6 package requires:

* ``validate`` / ``enumerate`` compute no science and open no data;
* the archived 350-cell grid enumerates with its original pairing and schema;
* a refusal when the raw CRESST pulse root is absent;
* a refusal on a provenance or split mismatch;
* the matched design uses identical evaluation windows and labels for both
  representations;
* a synthetic-fixture wiring test explicitly labelled non-real;
* missing files and an inconclusive development gate block confirmatory execution;
* no output claims real-data validation.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.mlst2026.followup_20260917.r6_instrument import cresst_readiness as CR  # noqa: E402
from experiments.mlst2026.followup_20260917.r6_instrument import runner as RN  # noqa: E402
from experiments.mlst2026.followup_20260917.r6_instrument import tidmad_readiness as TM  # noqa: E402
from experiments.mlst2026.successor_20260913.common.protocol import load_protocol  # noqa: E402
from experiments.mlst2026.successor_20260913.cresst_s.archive_recompute import InputNotAvailable  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import matched_stage as ms  # noqa: E402
from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402

PROTO = RN.PROTOCOL_DEFAULT


@pytest.fixture(scope="module")
def P():
    return load_protocol(PROTO)


# --------------------------------------------------------------------------------------------
# 1. validate / enumerate compute no science
# --------------------------------------------------------------------------------------------
def test_validate_computes_nothing_and_is_data_blocked(P, capsys):
    assert RN.main(["validate"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["computes"] is False and js["opens_real_data"] is False
    assert js["executes_production_cells"] is False and js["real_data_validated"] is False
    assert js["data_blocked"] is True and js["status"] == "blocked on data"
    assert js["cresst"]["cells_enumerated"] == 350 and js["cresst"]["fits_if_executed"] == 1050
    assert js["cresst"]["executable_here"] is False and js["cresst"]["grid_matches_archived"] is True
    assert js["tidmad"]["matched_design_declared"] is True and js["tidmad"]["executable_here"] is False


def test_enumerate_computes_nothing_and_writes_no_data(P, tmp_path):
    out = tmp_path / "enum"
    assert RN.main(["enumerate", "--out", str(out)]) == 0
    payload = json.loads((out / "R6_ENUMERATION.json").read_text())
    assert payload["computes"] is False and payload["real_data_validated"] is False
    assert not (out / "cells").exists()
    assert not any(p.suffix in (".h5", ".npz") for p in out.rglob("*"))
    cells = payload["cresst"]["cells"]
    assert len(cells) == 350
    assert all(c["executable_here"] is False for c in cells)
    assert all("CRESST_DATA_ROOT" in c["requires"] for c in cells)
    assert payload["tidmad"]["executable_here"] is False


# --------------------------------------------------------------------------------------------
# 2. the archived 350-cell grid enumerates with the correct pairing
# --------------------------------------------------------------------------------------------
def test_grid_enumerates_350_cells_with_archived_pairing(P):
    rows = CR.enumerate_grid(P)
    assert len(rows) == 350
    assert sum(r["arm"] == "pilot" for r in rows) == 35
    assert sum(r["arm"] == "confirmation" for r in rows) == 315

    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["contrast_group"])].append(r)
    assert len(groups) == 28  # 7 pilot + 21 confirmation (channel, time-cut) groups
    for key, g in groups.items():
        refs = [r for r in g if r["is_group_reference"]]
        assert len(refs) == 1, key
        ref = refs[0]
        for r in g:
            if r is ref:
                assert r["paired_against"] is None
            else:
                assert r["paired_against"] == (f"{r['contrast_group']}|rank{ref['rank']}"
                                               f"|shrink{ref['shrinkage']}")

    cost = CR.cost_grid(P, rows)
    assert cost["n_cells"] == 350 and cost["n_fits"] == 1050
    assert cost["n_contrast_groups"] == 28
    assert cost["n_paired_contrasts_total"] == 336
    assert cost["executable_here"] is False


def test_grid_rows_match_the_archived_design_field_by_field(P):
    archived = [r for arm in CR.ARMS for r in CR.archived_rows()[arm]]
    mine = CR.enumerate_grid(P)
    assert len(mine) == len(archived) == 350
    keys = ("arm", "run", "channel", "rank", "shrinkage", "time_cut", "contrast_group",
            "contrast_group_index", "channel_index", "time_cut_index", "is_group_reference",
            "is_primary_setting", "paired_against", "time_cut_paired_against")
    for m, a in zip(mine, archived):
        for k in keys:
            assert m[k] == a[k], (k, m[k], a[k])


# --------------------------------------------------------------------------------------------
# 3. refusal when raw pulses are absent
# --------------------------------------------------------------------------------------------
def test_refusal_when_raw_pulses_absent(P, capsys):
    with pytest.raises(InputNotAvailable) as e:
        CR.check_inputs(P, None)
    assert "raw CRESST pulse data root" in str(e.value)
    assert RN.main(["preflight"]) == RN.EXIT_DATA_BLOCKED
    out = capsys.readouterr().out
    assert "DATA-BLOCKED" in out and "raw CRESST pulse data root" in out


# --------------------------------------------------------------------------------------------
# 4. refusal on a provenance / split mismatch
# --------------------------------------------------------------------------------------------
def _fixture(tmp_path, P, name="cresst", seed=11):
    root = CR.write_fixture_root(tmp_path / name, seed=seed, P=P)
    return root


def _patch_index(root: Path, **changes):
    p = root / "index.json"
    js = json.loads(p.read_text())
    js.update(changes)
    p.write_text(json.dumps(js, indent=1, sort_keys=True))


def test_synthetic_fixture_passes_wiring_gate_then_provenance_refusal(P, tmp_path):
    root = _fixture(tmp_path, P)
    ok = CR.check_inputs(P, root)
    assert ok["synthetic"] is True and ok["real_data"] is False
    assert ok["n_channels"] == 7 and ok["label"] == CR.NON_REAL_LABEL

    js = json.loads((root / "index.json").read_text())
    js["provenance"]["archive_id"] = "forged-release"
    (root / "index.json").write_text(json.dumps(js, indent=1, sort_keys=True))
    with pytest.raises(InputNotAvailable) as e:
        CR.check_inputs(P, root)
    assert "provenance refused" in str(e.value)


def test_split_mismatch_is_refused(P, tmp_path):
    root = _fixture(tmp_path, P, name="cresst_split", seed=12)
    js = json.loads((root / "index.json").read_text())
    js["split"] = "training"
    (root / "index.json").write_text(json.dumps(js, indent=1, sort_keys=True))
    with pytest.raises(InputNotAvailable) as e:
        CR.check_inputs(P, root)
    assert "split refused" in str(e.value)


def test_non_chronological_times_are_refused(P, tmp_path):
    root = _fixture(tmp_path, P, name="cresst_time", seed=13)
    entry = json.loads((root / "index.json").read_text())["channels"][0]
    with np.load(root / entry["file"], allow_pickle=False) as z:
        pulses, times, noise = z["pulses"], z["times"], z["noise"]
    times = times[::-1].copy()
    np.savez_compressed(root / entry["file"], pulses=pulses, times=times, noise=noise)
    with pytest.raises(InputNotAvailable) as e:
        CR.check_inputs(P, root)
    assert "not chronological" in str(e.value)


# --------------------------------------------------------------------------------------------
# 5/6. matched design + synthetic-fixture wiring, explicitly labelled non-real
# --------------------------------------------------------------------------------------------
MATCHED = {"w": 1024, "n_cal": 64, "n_fit": 100, "n_eval": 100, "guard": 1024, "k_inj": 37, "B": 9}


@pytest.fixture(scope="module")
def matched_wiring(tmp_path_factory):
    """Build the synthetic TIDMAD fixture once and score the two named contrasts."""
    cfg = MATCHED
    tmp = tmp_path_factory.mktemp("r6_tidmad")
    path = tmp / "synthetic_fixture.h5"
    ms.write_tidmad_fixture(path, seed=101, w=cfg["w"], n_cal=cfg["n_cal"], n_fit=cfg["n_fit"],
                            n_eval=cfg["n_eval"], guard=cfg["guard"], k_inj=cfg["k_inj"])
    alloc = ms.fixture_allocation(w=cfg["w"], n_cal=cfg["n_cal"], n_fit=cfg["n_fit"],
                                  n_eval=cfg["n_eval"], guard=cfg["guard"])
    stretch = ms.load_fixture_stretch(path, 0, alloc)
    k = cfg["k_inj"]
    N = te.mean_bin_power(stretch["fit_read"])
    N[k] = te.offline_noise_floor(stretch["fit_read"], k)
    scored = TM.matched_contrasts(stretch, k_inj=k, B=cfg["B"], w=cfg["w"], N_bins=N,
                                  exposure_status="synthetic-fixture")
    b1 = TM.assert_b1_identity(stretch, k_inj=k, w=cfg["w"], N_bins=N)
    cov = TM.covariance_adequacy(stretch, k_inj=k, N_bins=N)
    return {"stretch": stretch, "N": N, "scored": scored, "b1": b1, "cov": cov, "cfg": cfg}


def test_matched_design_is_declared_and_same_units(P):
    d = TM.matched_design(P)
    assert d["executable_here"] is False
    assert d["representations"] == ["whole-window", "band"]
    assert d["rank"] == 2 and d["band_width"] == 9 and d["band_selected_here"] is False
    assert "ONE transfer calibration" in d["calibration"]
    assert "ONE label array" in d["labels"]
    assert d["p3"]["tested"] is False
    assert d["risk_identity"]["applied_to_learned_maps"] is False


def test_both_representations_share_identical_windows_and_labels(matched_wiring):
    s = matched_wiring["scored"]
    ww = s["per_representation_units"]["whole-window"]
    bd = s["per_representation_units"]["band"]
    assert ww["evaluation_window_index"] == bd["evaluation_window_index"]
    assert ww["label_sha256"] == bd["label_sha256"]
    assert ww["evaluation_window_index"] == s["shared"]["evaluation_window_index"]
    assert s["shared"]["same_units_for_both_representations"] is True
    # the two named P3 contrasts are reported on those same units
    for rep in ("whole-window", "band"):
        assert s["p3"]["weighted_residual_contrast"][rep]["n_windows"] == len(ww["evaluation_window_index"])
        assert s["p3"]["truth_referenced_contrast"][rep]["n_windows"] == len(ww["evaluation_window_index"])
    assert set(s["p3"]["ordering_agrees"]) == {"whole-window", "band"}
    assert s["p3"]["real_data_tested"] is False
    assert set(s["usefulness_against_baselines"]) == {"zero|empca", "calibration-mean|empca", "raw-bin|empca"}


def test_synthetic_fixture_wiring_is_explicitly_labelled_non_real(matched_wiring):
    s = matched_wiring["scored"]
    assert s["exposure_status"] == "synthetic-fixture"
    assert s["label"] == TM.NON_REAL_LABEL
    assert "not real data" in s["label"] and "not validated on real inputs" in s["label"]
    assert matched_wiring["b1"]["passed"] is True
    assert matched_wiring["b1"]["label"] == TM.NON_REAL_LABEL
    assert matched_wiring["cov"]["finite"] is True


def test_risk_identity_control_is_arithmetic_only():
    c = TM.control_risk_identity(seed=3, d=12, k=3, n=60_000)
    assert c["passed"] is True
    assert "not real data" in c["label"]
    assert set(c["arms"]) == {"rank_k_orthogonal", "identity"}


# --------------------------------------------------------------------------------------------
# 7. missing files / inconclusive gate block confirmatory execution
# --------------------------------------------------------------------------------------------
def _complete_calibration():
    return {"complete": True, "feasibility_table": {"4": {
        "type1_within_tolerance": True, "coverage_within_tolerance": True,
        "withholding_rate_max": 0.2, "power_medium_effect_min": 0.7}}}


def test_missing_files_block_confirmatory_execution(P):
    inv = TM.exposure_inventory()
    gate = TM.confirmatory_gate(P, inventory=inv, n_available_files=2,
                                calibration=_complete_calibration(), dependence_verdict="stabilised")
    assert gate["blocked"] and gate["status"] == RN.DATA_BLOCKED
    assert any("too few independent units" in r for r in gate["reasons"])


def test_unresolved_release_file_list_blocks_confirmatory_execution(P):
    inv = TM.exposure_inventory()
    check = TM.exposure_inventory_check(inv)
    assert check["blocked"] is True
    assert check["checks"]["independent_file_set_resolved"]["passed"] is False
    gate = TM.confirmatory_gate(P, inventory=inv, n_available_files=8,
                                calibration=_complete_calibration(), dependence_verdict="stabilised")
    assert gate["blocked"]
    assert any("release file list" in r for r in gate["reasons"])


def test_inconclusive_development_gate_blocks_confirmatory_execution(P):
    inv = TM.exposure_inventory()
    gate = TM.confirmatory_gate(P, inventory=inv, n_available_files=8,
                                calibration=_complete_calibration(), dependence_verdict="inconclusive")
    assert gate["blocked"]
    assert any("inconclusive" in r for r in gate["reasons"])


def test_absent_calibration_blocks_confirmatory_execution(P):
    inv = TM.exposure_inventory()
    gate = TM.confirmatory_gate(P, inventory=inv, n_available_files=8,
                                calibration=None, dependence_verdict="stabilised")
    assert gate["blocked"]
    assert any("no complete confirmation calibration" in r for r in gate["reasons"])


def test_preflight_and_check_inputs_return_data_blocked(P, capsys):
    assert RN.main(["preflight"]) == RN.EXIT_DATA_BLOCKED
    out = capsys.readouterr().out
    assert "DATA-BLOCKED" in out
    assert RN.main(["check-inputs"]) == RN.EXIT_DATA_BLOCKED
    out = capsys.readouterr().out
    assert "DATA-BLOCKED" in out
    assert '"real_data_validated": false' in out


# --------------------------------------------------------------------------------------------
# 8. no output claims real-data validation
# --------------------------------------------------------------------------------------------
def test_no_output_claims_real_data_validation(P, matched_wiring, capsys):
    prohibited = "validated on real data"
    assert prohibited not in json.dumps(P)
    claimed_text = "\n".join(P["explicitly_not_claimed"])
    for phrase in ("NOT a real-data result", "NOT a validated adapter", "NOT executed robustness"):
        assert phrase in claimed_text
    assert P["data_blocked"] is True and P["status"] == "blocked on data"
    assert P["budget"]["planned_cells_production"] == 0
    assert P["budget"]["planned_fits_production"] == 0

    assert RN.main(["validate"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["real_data_validated"] is False
    assert js["cresst"]["executable_here"] is False

    scored = matched_wiring["scored"]
    assert scored["label"] == TM.NON_REAL_LABEL
    assert prohibited not in json.dumps(scored, default=str)


def test_inventory_subcommand_reports_unresolved_release_list(capsys):
    assert RN.main(["inventory"]) == 0
    out = capsys.readouterr().out
    assert "release files not yet exposed" in out.lower()
    assert '"independent_file_set_resolved": false' in out


# --- A4: committed machine-readable inventory and refusal receipt --------------------
def test_committed_tidmad_inventory_and_input_gate_receipt_are_machine_readable():
    import json
    base = REPO / "experiments" / "mlst2026" / "followup_20260917" / "r6_instrument"
    inv = json.loads((base / "INPUT_READINESS_INVENTORY.json").read_text())
    assert inv["files_known_to_archive"] and inv["regions"]
    assert inv["release_file_list"]["derivable_from_archive"] is False
    # every region names a source run root and a file, so an "unexposed" claim is checkable
    assert all("file" in r and "run_tree" in r for r in inv["regions"])
    gate = json.loads((base / "INPUT_GATE_RECEIPT.json").read_text())
    assert gate["status"] == "DATA-BLOCKED" and gate["blocked"] is True
    assert any("cresst" in r for r in gate["reasons"])
    assert any("tidmad" in r for r in gate["reasons"])
    assert "real inputs" in " ".join(gate["reasons"]).lower()
