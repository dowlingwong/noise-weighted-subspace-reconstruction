"""Round 12 sweep: the gates must reject the failures they exist to catch.

Each test states the specific way the sweep could be wrong and then produces
that wrongness. A gate that has never rejected anything is a comment.

Nothing here runs the scientific sweep. The fixtures are the archived E10b and
E1 rows, which is the point: the anchor gate is only meaningful if it is
checked against the same bytes the paper cites.
"""

from __future__ import annotations

import copy

import pytest

from experiments.synthetic.nfpa_confirmatory import r10_common as rc
from experiments.synthetic.nfpa_confirmatory import r10_validation as rv
from experiments.synthetic.nfpa_confirmatory import r12_sweep as sw
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.r9_common import IntegrityError


# ------------------------------------------------------------------ identity


def test_anchor_condition_has_the_archived_e10b_config_hash():
    """The anchor is only an anchor if it is the same configuration."""
    assert sw.assert_anchor_hash_matches_archive() == sw.ANCHOR_CONFIG_HASH
    assert config_hash(sw.condition_config(sw.ANCHOR_N_CAL)) == config_hash(PRIMARY)


def test_non_anchor_conditions_get_distinct_hashes():
    hashes = {n: config_hash(sw.condition_config(n)) for n in sw.CALIBRATION_SIZES}
    assert len(set(hashes.values())) == len(hashes), hashes


def test_primary_is_never_mutated_by_condition_config():
    before = PRIMARY.N_calibration
    sw.condition_config(256)
    assert PRIMARY.N_calibration == before


@pytest.mark.parametrize("bad", [0, -1, -2048])
def test_nonpositive_calibration_sizes_are_refused(bad):
    with pytest.raises(IntegrityError):
        sw.condition_config(bad)


def test_a_sweep_without_the_anchor_is_refused():
    """Dropping 2048 would silently untether the sweep from archived evidence."""
    with pytest.raises(IntegrityError, match="anchor"):
        sw.validate_condition_set([256, 512, 1024, 8192])


def test_duplicate_sizes_are_refused():
    with pytest.raises(IntegrityError, match="duplicate"):
        sw.validate_condition_set([256, 2048, 2048])


# -------------------------------------------------------------- anchor gate


def _archived_e10b_nfpa_rows() -> list[dict]:
    return [
        r
        for r in rc.read_rows_csv(sw.ARCHIVED_E10B_DIR / "raw.csv")
        if r["method"] == "NFPA"
    ]


def test_archived_e10b_rows_satisfy_the_anchor_gate():
    """The gate must accept the very rows the paper cites, or it is miscalibrated."""
    record = sw.validate_anchor_reproduces_e10b(_archived_e10b_nfpa_rows())
    assert record["passed"] is True
    assert record["comparisons"] == 20  # 10 seeds x 2 metrics
    assert record["worst_relative_error"] == 0.0


@pytest.mark.parametrize("metric", ["test_raw_mse", "test_normalized_chi2"])
def test_a_drifted_anchor_is_rejected(metric):
    """Code drift is exactly what the anchor exists to catch."""
    rows = copy.deepcopy(_archived_e10b_nfpa_rows())
    rows[0][metric] = str(float(rows[0][metric]) * 1.000001)  # 1e-6 relative
    with pytest.raises(IntegrityError, match="does not reproduce"):
        sw.validate_anchor_reproduces_e10b(rows)


def test_an_anchor_row_with_no_archived_counterpart_is_rejected():
    rows = copy.deepcopy(_archived_e10b_nfpa_rows())
    rows[0]["seed"] = "424242"
    with pytest.raises(IntegrityError, match="no archived E10b counterpart"):
        sw.validate_anchor_reproduces_e10b(rows)


# ------------------------------------------------------------- control gate


def _control_rows(n_cal: int, *, scale: float = 1.0) -> list[dict]:
    """Iso-MPCA rows built from archived E1, optionally perturbed."""
    authoritative = rv.load_authoritative_metrics()
    rows = []
    for seed in range(10):
        rec = authoritative[(seed, "Iso-MPCA")]
        rows.append(
            {
                "seed": str(seed),
                "method": "Iso-MPCA",
                "n_calibration_events": str(n_cal),
                "test_raw_mse": str(float(rec["test_raw_mse"]) * scale),
                "test_normalized_chi2": str(float(rec["test_normalized_chi2"]) * scale),
            }
        )
    return rows


def test_a_covariance_independent_control_passes():
    by_condition = {n: _control_rows(n) for n in (256, 2048, 8192)}
    record = sw.validate_control_is_covariance_independent(by_condition)
    assert record["passed"] is True
    assert record["worst_relative_error_across_conditions"] == 0.0


def test_a_control_that_moves_with_calibration_size_is_rejected():
    """If the estimated covariance leaks into the control, every contrast is suspect."""
    by_condition = {
        256: _control_rows(256),
        2048: _control_rows(2048),
        8192: _control_rows(8192, scale=1.0 + 1e-6),
    }
    with pytest.raises(IntegrityError, match="leaking into the control arm|drifted from"):
        sw.validate_control_is_covariance_independent(by_condition)


def test_a_control_that_drifts_from_archived_e1_is_rejected():
    by_condition = {n: _control_rows(n, scale=1.0 + 1e-6) for n in (256, 2048)}
    with pytest.raises(IntegrityError, match="drifted from archived E1"):
        sw.validate_control_is_covariance_independent(by_condition)


# ------------------------------------------------------- structural gates


def _condition_rows(n_cal: int) -> list[dict]:
    """Archived E10b rows relabelled as an E12 condition."""
    cfg_hash = config_hash(sw.condition_config(n_cal))
    rows = []
    for row in rc.read_rows_csv(sw.ARCHIVED_E10B_DIR / "raw.csv"):
        new = dict(row)
        new["config_hash"] = cfg_hash
        new["analysis_label"] = sw.ANALYSIS_LABEL
        new["n_calibration_events"] = str(n_cal)
        rows.append(new)
    return rows


def _columns() -> tuple[str, ...]:
    from experiments.synthetic.nfpa_confirmatory.run_calibrated_estimated_covariance import (
        COLUMNS,
    )

    return COLUMNS


def test_well_formed_condition_rows_validate():
    record = sw.validate_condition_rows(
        _condition_rows(512), n_cal=512, columns=_columns(), seeds=range(10)
    )
    assert record["validated"] is True


def test_rows_recording_the_wrong_calibration_size_are_rejected():
    """A condition directory that silently holds another size's rows."""
    rows = _condition_rows(512)
    rows[0]["n_calibration_events"] = "2048"
    with pytest.raises(IntegrityError, match="records n_calibration_events"):
        sw.validate_condition_rows(rows, n_cal=512, columns=_columns(), seeds=range(10))


def test_rows_with_a_foreign_config_hash_are_rejected():
    rows = _condition_rows(512)
    for row in rows:
        row["config_hash"] = "deadbeefdeadbeef"
    with pytest.raises(IntegrityError, match="config_hash"):
        sw.validate_condition_rows(rows, n_cal=512, columns=_columns(), seeds=range(10))


def test_rows_labelled_confirmatory_are_rejected():
    """The evidence class is a gate, not a caption."""
    rows = _condition_rows(512)
    for row in rows:
        row["analysis_label"] = "single_condition_covariance_estimation_sensitivity"
    with pytest.raises(IntegrityError, match="analysis_label"):
        sw.validate_condition_rows(rows, n_cal=512, columns=_columns(), seeds=range(10))


def test_non_finite_metrics_are_rejected():
    rows = _condition_rows(512)
    rows[0]["test_normalized_chi2"] = "nan"
    with pytest.raises(IntegrityError, match="non-finite"):
        sw.validate_condition_rows(rows, n_cal=512, columns=_columns(), seeds=range(10))


# ------------------------------------------------------------ output safety


def test_writing_into_archived_e10b_or_e11_is_refused(tmp_path):
    """r10_common's refusal list predates E10b and E11 and does not cover them."""
    for archived in (
        sw.ARCHIVED_E10B_DIR / "sub",
        rc.RESULTS_ROOT / "E11_corrected_angles" / "sub",
    ):
        with pytest.raises(IntegrityError, match="read-only"):
            sw.refuse_archived_round10_dirs(archived)
    sw.refuse_archived_round10_dirs(tmp_path)  # a fresh directory is fine


# --------------------------------------------------------------- summaries


def _by_condition() -> dict[int, list[dict]]:
    return {n: _condition_rows(n) for n in (256, 2048)}


def test_sweep_summary_is_deterministic():
    a = sw.sweep_summary(_by_condition())
    b = sw.sweep_summary(_by_condition())
    assert rv.canonical_json_digest(a) == rv.canonical_json_digest(b)


def test_sweep_summary_carries_the_exploratory_labels():
    summary = sw.sweep_summary(_by_condition())
    assert summary["analysis_label"] == sw.ANALYSIS_LABEL
    assert summary["evidence_class"] == sw.EVIDENCE_CLASS
    assert "does not establish a robustness boundary" in summary["scope_note"]
    assert summary["per_condition"]["n_cal_2048"]["is_anchor"] is True
    assert summary["per_condition"]["n_cal_256"]["is_anchor"] is False


def test_a_tampered_summary_fails_the_regeneration_audit(tmp_path):
    by_condition = _by_condition()
    for n_cal, rows in by_condition.items():
        d = tmp_path / sw.condition_label(n_cal)
        d.mkdir(parents=True)
        rc.write_rows_csv(d / "raw.csv", rows, _columns())

    honest = sw.sweep_summary(by_condition)
    assert sw.assert_sweep_summary_regenerates(
        tmp_path, list(by_condition), honest
    )["passed"] is True

    tampered = copy.deepcopy(honest)
    tampered["per_condition"]["n_cal_256"][
        "contrast_nfpa_minus_iso.test_normalized_chi2"
    ]["mean_difference"] = -0.5
    with pytest.raises(IntegrityError, match="does not regenerate"):
        sw.assert_sweep_summary_regenerates(tmp_path, list(by_condition), tampered)


def test_regeneration_audit_refuses_a_missing_condition(tmp_path):
    with pytest.raises(IntegrityError, match="missing"):
        sw.assert_sweep_summary_regenerates(tmp_path, [2048], {})


# ------------------------------------------------------------- provenance


def test_a_non_smoke_sweep_requires_an_explicit_freeze_commit():
    from experiments.synthetic.nfpa_confirmatory import run_calibration_sweep as runner

    with pytest.raises(IntegrityError, match="required for a non-smoke run"):
        runner.main(["--out", "results/rps2026/E12_calibration_size_sweep"])


# ------------------------------------------------- end-to-end: builder into gate
#
# The gap that cost six hours of compute. Every test above builds rows by hand
# and *substitutes* the sweep's label -- the exact substitution the runner did
# not make. So the suite encoded the right contract, the pipeline violated it,
# and nothing ran the two together. These tests drive the real row builder into
# the real gate at smoke dimensions, which takes seconds.


def _smoke_cfg(n_cal: int):
    from dataclasses import replace

    return replace(rc.smoke_config(), N_calibration=n_cal)


def test_the_runner_itself_writes_rows_the_gate_accepts(tmp_path, monkeypatch):
    """Drive the actual runner, then the actual gate, over rows it wrote.

    This must NOT pass the label itself. Every earlier test did, which is the
    flaw that let the suite stay green against a runner that could not complete:
    the substitution under test was performed by the test. Here the runner is
    invoked as the runbook invokes it, and only then are its rows validated.
    """
    from experiments.synthetic.nfpa_confirmatory import run_calibration_sweep as runner
    from experiments.synthetic.nfpa_confirmatory.run_calibrated_estimated_covariance import (
        COLUMNS,
    )

    n_cal = 64
    monkeypatch.setenv("R10_RUNTIME_ROOT", str(tmp_path))
    out = tmp_path / "Round10_smoke" / "E12"

    rc_status = runner.main(["--smoke", "--sizes", str(n_cal), "--out", str(out)])
    assert rc_status == 0

    rows = list(rc.read_rows_csv(out / sw.condition_label(n_cal) / "raw.csv"))
    assert rows, "the runner wrote no rows"

    # The gate derives its expected config hash from the condition; point it at
    # the smoke configuration the runner actually used.
    from dataclasses import replace

    smoke_cfg = replace(rc.smoke_config(), N_calibration=n_cal)
    monkeypatch.setattr(sw, "condition_config", lambda _n: smoke_cfg)

    record = sw.validate_condition_rows(
        rows, n_cal=n_cal, columns=COLUMNS, seeds=list(rc.SMOKE_SEEDS)
    )
    assert record["validated"] is True


def test_forgetting_the_label_is_caught_immediately():
    """Reproduces the original defect: inheriting E10b's label must be rejected."""
    from experiments.synthetic.nfpa_confirmatory.run_calibrated_estimated_covariance import (
        ANALYSIS_LABEL as E10B_LABEL,
        run_seed,
    )

    rows, _ = run_seed(rc.SMOKE_SEEDS[0], _smoke_cfg(64), None)  # no label passed
    assert rows[0]["analysis_label"] == E10B_LABEL
    assert rows[0]["analysis_label"] != sw.ANALYSIS_LABEL, (
        "the default must remain E10b's, so that omitting the argument is a "
        "visible mistake rather than a silent one"
    )


def test_the_sweep_runner_passes_its_own_label():
    """Guards the call site itself, not just the callee's capability."""
    import inspect

    from experiments.synthetic.nfpa_confirmatory import run_calibration_sweep as runner

    src = inspect.getsource(runner.run_condition)
    assert "analysis_label=ANALYSIS_LABEL" in src, (
        "run_condition must pass the sweep's label into run_seed; without it "
        "every condition gate rejects rows the runner just wrote"
    )
