"""Round 10 hardening: provenance, fail-closed resume, archive audit, summaries.

These tests cover the four corrections applied after the first Round 10
handoff review:

1. an explicit, validated implementation-freeze commit (never inferred);
2. resume validation that re-derives every gate from the FINAL rows, so a
   corrupted reused shard cannot inherit a passing verdict;
3. an authoritative 15-file archive audit (the earlier tests pinned six while
   the prose said "E1-E10");
4. a deterministic summary-regeneration audit.
"""

from __future__ import annotations

import copy
import json
import subprocess

import pytest

from experiments.synthetic.nfpa_confirmatory import r10_provenance as prov
from experiments.synthetic.nfpa_confirmatory import r10_validation as rv
from experiments.synthetic.nfpa_confirmatory import run_calibrated_estimated_covariance as e10b
from experiments.synthetic.nfpa_confirmatory import run_corrected_angles as e11
from experiments.synthetic.nfpa_confirmatory.r9_common import IntegrityError, REPO_ROOT

TOL = 1e-9


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()


# --------------------------------------------------------------- provenance


def test_freeze_sha_is_required_and_never_inferred_from_head():
    with pytest.raises(IntegrityError, match="required for a non-smoke run"):
        prov.validate_freeze_commit(None, smoke=False)
    # Smoke may omit it, and it stays None — never backfilled from HEAD.
    freeze, head = prov.validate_freeze_commit(None, smoke=True)
    assert freeze is None
    assert head == _git("rev-parse", "HEAD")


def test_freeze_and_execution_shas_are_recorded_separately(monkeypatch):
    # This test is about SHA separation, not tree state; the dirty-tree gate
    # has its own test below.
    monkeypatch.setattr(prov, "worktree_is_dirty", lambda: False)
    head = _git("rev-parse", "HEAD")
    parent = _git("rev-parse", "HEAD~1")
    freeze, execution = prov.validate_freeze_commit(parent, smoke=False)
    assert freeze == parent
    assert execution == head
    assert freeze != execution, "a real freeze precedes execution; they must differ"

    block = prov.provenance_block(
        ["cmd"], __import__(
            "experiments.synthetic.nfpa_confirmatory.config", fromlist=["PRIMARY"]
        ).PRIMARY,
        freeze_sha=freeze, execution_head=execution, smoke=False,
        started_at_utc="2026-08-19T00:00:00Z",
    )
    assert block["implementation_freeze_commit"] == parent
    assert block["execution_content_commit"] == head
    assert block["evidence_class"] == prov.EVIDENCE_SCIENTIFIC
    for key in (
        "source_tree_dirty_at_start", "source_commit", "command", "config_hash",
        "primary_config_hash", "python_version", "numpy_version", "scipy_version",
        "platform", "hostname", "thread_limits", "started_at_utc",
        "completed_at_utc", "expected_output_changes_at_completion",
    ):
        assert key in block


@pytest.mark.parametrize(
    "bad",
    ["", "not-a-sha", "abc123", "z" * 40, "0" * 40],
)
def test_invalid_freeze_shas_fail_closed(bad):
    with pytest.raises(IntegrityError):
        prov.validate_freeze_commit(bad, smoke=False)


def test_non_ancestor_freeze_sha_fails_closed():
    """A commit that exists but is not an ancestor of HEAD is rejected."""
    # main diverges from the Round 10 line, so its tip is not an ancestor.
    other = _git("rev-parse", "main")
    if other == _git("rev-parse", "HEAD"):
        pytest.skip("HEAD is main; no divergent commit available in this checkout")
    with pytest.raises(IntegrityError, match="not an ancestor"):
        prov.validate_freeze_commit(other, smoke=False)


def test_dirty_worktree_fails_closed_for_non_smoke(monkeypatch):
    monkeypatch.setattr(prov, "worktree_is_dirty", lambda: True)
    parent = _git("rev-parse", "HEAD~1")
    with pytest.raises(IntegrityError, match="worktree is dirty"):
        prov.validate_freeze_commit(parent, smoke=False)
    # ... but smoke is an implementation check and may run on a dirty tree.
    assert prov.validate_freeze_commit(parent, smoke=True)[0] == parent


def test_smoke_provenance_cannot_be_mistaken_for_scientific_evidence():
    from experiments.synthetic.nfpa_confirmatory.config import PRIMARY

    smoke_block = prov.provenance_block(
        ["cmd"], PRIMARY, freeze_sha=None, execution_head="deadbeef" * 5,
        smoke=True, started_at_utc="2026-08-19T00:00:00Z",
    )
    assert smoke_block["evidence_class"] == prov.EVIDENCE_SMOKE
    assert smoke_block["evidence_class"] != prov.EVIDENCE_SCIENTIFIC
    assert smoke_block["implementation_freeze_commit"] is None


# ------------------------------------------------------------ archive audit


def test_authoritative_archive_covers_all_fifteen_preflight_paths():
    preflight = (REPO_ROOT / "results/rps2026/Round10_desktop/preflight.txt").read_text()
    listed = {
        line.split("  ", 1)[1].strip()
        for line in preflight.splitlines()
        if len(line) > 65 and line[:64].isalnum() and "  " in line
    }
    assert len(rv.AUTHORITATIVE_ARCHIVE) == 15
    assert set(rv.AUTHORITATIVE_ARCHIVE) == listed, (
        "the authoritative archive list must equal the Phase 0 preflight set exactly"
    )


def test_authoritative_archive_digests_are_unchanged():
    audit = rv.audit_authoritative_archive()
    assert audit["n_checked"] == 15, f"missing archive files: {audit['missing']}"
    assert audit["passed"], f"ARCHIVE MODIFIED: {audit['mismatched']}"


def test_archive_audit_detects_a_modified_file(monkeypatch):
    poisoned = dict(rv.AUTHORITATIVE_ARCHIVE)
    first = sorted(poisoned)[0]
    poisoned[first] = "0" * 64
    monkeypatch.setattr(rv, "AUTHORITATIVE_ARCHIVE", poisoned)
    audit = rv.audit_authoritative_archive()
    assert not audit["passed"] and audit["mismatched"][0]["path"] == first
    with pytest.raises(IntegrityError, match="AUTHORITATIVE ARCHIVE CHANGED"):
        rv.require_authoritative_archive_unchanged()


# ------------------------------------------------- fail-closed final rows


AUTH = rv.load_authoritative_metrics()


def _e10b_rows() -> list[dict]:
    """Rows consistent with the REAL archived E1 metrics.

    The validators recompute every archived value, reproduction error and delta
    from the authoritative table, so a synthetic fixture with invented numbers
    would (correctly) be rejected. Building from the archive is what lets the
    adversarial tests below isolate exactly one corruption at a time.
    """
    rows = []
    for seed in range(10):
        common = {
            "config_hash": "cfg", "seed": str(seed), "regime": "colored",
            "analysis_label": e10b.ANALYSIS_LABEL, "n_calibration_events": "2048",
            "v_c": "1.0", "v_t": "1.0", "v_hat": "1.0",
            "product_scale_ratio_vs_oracle": "1.0",
            "channel_shape_error": "0.01", "temporal_shape_error": "0.01",
            "kronecker_relative_frobenius_error": "0.02",
            "oracle_whitening_residual": "0.001",
            "channel_min_eigenvalue": "0.5", "channel_condition_number": "3.0",
            "temporal_min_eigenvalue": "0.4", "temporal_condition_number": "9.0",
            "runtime_seconds": "1.0",
        }
        nf = AUTH[(seed, "NFPA")]
        # A realistic small estimation degradation on top of the archived value.
        est_raw = nf["test_raw_mse"] + 3.5e-06
        est_chi2 = nf["test_normalized_chi2"] + 7.9e-05
        rows.append({**common, "method": "NFPA", "whitening": "estimated_calibrated_v2",
                     "test_raw_mse": repr(est_raw), "test_normalized_chi2": repr(est_chi2),
                     "archived_oracle_test_raw_mse": repr(nf["test_raw_mse"]),
                     "archived_oracle_test_normalized_chi2": repr(nf["test_normalized_chi2"]),
                     "delta_raw_mse_estimated_minus_oracle": repr(est_raw - nf["test_raw_mse"]),
                     "delta_normalized_chi2_estimated_minus_oracle":
                         repr(est_chi2 - nf["test_normalized_chi2"]),
                     "control_reproduction_rel_error_raw_mse": "",
                     "control_reproduction_rel_error_normalized_chi2": ""})
        iso = AUTH[(seed, "Iso-MPCA")]
        # The control reproduces the archive exactly => zero relative error.
        rows.append({**common, "method": "Iso-MPCA",
                     "whitening": "none_covariance_independent_control",
                     "test_raw_mse": repr(iso["test_raw_mse"]),
                     "test_normalized_chi2": repr(iso["test_normalized_chi2"]),
                     "archived_oracle_test_raw_mse": repr(iso["test_raw_mse"]),
                     "archived_oracle_test_normalized_chi2": repr(iso["test_normalized_chi2"]),
                     "delta_raw_mse_estimated_minus_oracle": "",
                     "delta_normalized_chi2_estimated_minus_oracle": "",
                     "control_reproduction_rel_error_raw_mse": "0.0",
                     "control_reproduction_rel_error_normalized_chi2": "0.0"})
    return [{k: r[k] for k in e10b.COLUMNS} for r in rows]


def _e11_rows() -> list[dict]:
    """Rows consistent with the REAL archived E1 (0-9) and E9 (10-29) metrics."""
    rows = []
    for seed in range(30):
        for method in ("CW-PCA", "NFPA", "Iso-MPCA"):
            is_cw = method == "CW-PCA"
            arch = AUTH[(seed, method)]
            rows.append({
                "config_hash": "cfg", "seed": str(seed),
                "cohort": "frozen_0_9" if seed < 10 else "replication_10_29",
                "regime": "colored", "method": method,
                "analysis_label": e11.ANALYSIS_LABEL,
                "archived_source": "E1" if seed < 10 else "E9",
                "reproduction_rel_error_raw_mse": "0.0",
                "reproduction_rel_error_normalized_chi2": "0.0",
                "test_raw_mse": repr(arch["test_raw_mse"]),
                "test_normalized_chi2": repr(arch["test_normalized_chi2"]),
                "angle_to_cw_pca_mean_deg_legacy": "" if is_cw else "82.2",
                "angle_to_cw_pca_max_deg_legacy": "" if is_cw else "89.1",
                "angle_to_cw_pca_mean_deg_v2": "" if is_cw else "59.4",
                "angle_to_cw_pca_max_deg_v2": "" if is_cw else "78.3",
                "runtime_seconds": "1.0",
            })
    return [{k: r[k] for k in e11.COLUMNS} for r in rows]


def test_clean_final_rows_pass_both_validators():
    g10 = rv.validate_e10b_final_rows(
        _e10b_rows(), columns=e10b.COLUMNS, config_hash="cfg",
        analysis_label=e10b.ANALYSIS_LABEL, tolerance=TOL)
    assert g10["control_gate_passed"] and g10["n_rows_validated"] == 20
    g11 = rv.validate_e11_final_rows(
        _e11_rows(), columns=e11.COLUMNS, config_hash="cfg",
        analysis_label=e11.ANALYSIS_LABEL, tolerance=TOL)
    assert g11["reproduction_gate_passed"] and g11["n_rows_validated"] == 90


@pytest.mark.parametrize(
    "corrupt,match",
    [
        (lambda r: r.__setitem__(1, {**r[1], "control_reproduction_rel_error_raw_mse": "1e-3"}),
         "disagrees with the recomputed value"),
        (lambda r: r.__setitem__(0, {**r[0], "test_normalized_chi2": "nan"}), "non-finite"),
        (lambda r: r.__setitem__(0, {**r[0], "whitening": "oracle"}), "whitening label altered"),
        (lambda r: r.__setitem__(0, {**r[0], "config_hash": "other"}), "config_hash"),
        (lambda r: r.__setitem__(0, {**r[0], "analysis_label": "confirmatory"}), "analysis_label"),
        (lambda r: r.__setitem__(0, {**r[0], "delta_raw_mse_estimated_minus_oracle": "0.5"}),
         "recomputed from the authoritative archive"),
        (lambda r: r.pop(), "expected 20 rows"),
        (lambda r: r.append(copy.deepcopy(r[0])), "duplicate"),
        (lambda r: r.__setitem__(0, {**r[0], "seed": "42"}), "seed set"),
    ],
)
def test_corrupted_e10b_rows_stop_the_run(corrupt, match):
    rows = _e10b_rows()
    corrupt(rows)
    with pytest.raises(IntegrityError, match=match):
        rv.validate_e10b_final_rows(
            rows, columns=e10b.COLUMNS, config_hash="cfg",
            analysis_label=e10b.ANALYSIS_LABEL, tolerance=TOL)


@pytest.mark.parametrize(
    "corrupt,match",
    [
        (lambda r: r.__setitem__(1, {**r[1], "reproduction_rel_error_raw_mse": "1e-3"}),
         "reproduction_rel_error_raw_mse"),
        (lambda r: r.__setitem__(0, {**r[0], "cohort": "replication_10_29"}), "cohort"),
        (lambda r: r.__setitem__(0, {**r[0], "archived_source": "E9"}), "archived_source"),
        (lambda r: r.__setitem__(1, {**r[1], "angle_to_cw_pca_mean_deg_v2": ""}),
         "non-finite corrected angle"),
        (lambda r: r.__setitem__(0, {**r[0], "angle_to_cw_pca_mean_deg_v2": "0.0"}),
         "self-reference"),
        (lambda r: r.pop(), "expected 90 rows"),
        (lambda r: r.append(copy.deepcopy(r[0])), "duplicate"),
        (lambda r: r.__setitem__(0, {**r[0], "seed": "42"}), "seed set"),
    ],
)
def test_corrupted_e11_rows_stop_the_run(corrupt, match):
    rows = _e11_rows()
    corrupt(rows)
    with pytest.raises(IntegrityError, match=match):
        rv.validate_e11_final_rows(
            rows, columns=e11.COLUMNS, config_hash="cfg",
            analysis_label=e11.ANALYSIS_LABEL, tolerance=TOL)


def test_control_gate_is_recomputed_not_inherited():
    """A resumed shard with a failing control cannot pass by inheriting a flag."""
    rows = _e10b_rows()
    rows[1] = {**rows[1], "control_reproduction_rel_error_normalized_chi2": "5e-8"}
    with pytest.raises(IntegrityError):
        rv.validate_e10b_final_rows(
            rows, columns=e10b.COLUMNS, config_hash="cfg",
            analysis_label=e10b.ANALYSIS_LABEL, tolerance=TOL)


# ------------------------------------------- deterministic summary audit


def test_summary_regeneration_audit_detects_a_tampered_summary(tmp_path):
    from experiments.synthetic.nfpa_confirmatory import r9_common as rc

    rows = _e11_rows()
    raw = tmp_path / "raw.csv"
    rc.write_rows_csv(raw, rows, e11.COLUMNS)
    summary = e11.build_summary(rows)
    audit = rv.assert_summary_regenerates(raw, summary, e11.build_summary)
    assert audit["passed"]

    tampered = json.loads(json.dumps(summary))
    tampered["frozen"]["n_seeds"] = 999
    with pytest.raises(IntegrityError, match="summary regeneration audit FAILED"):
        rv.assert_summary_regenerates(raw, tampered, e11.build_summary)


def test_summaries_are_pure_functions_of_rows():
    rows = _e10b_rows()
    assert rv.canonical_json_digest(e10b.summarize(rows)) == rv.canonical_json_digest(
        e10b.summarize(list(reversed(rows)))
    ), "summary must not depend on row order"


# --------------------------------------------- adversarial resume corruption
#
# The reviewer demonstrated that altering a stored METRIC while leaving the
# stored reproduction-error column small was accepted by both validators.
# These tests pin the fix: the errors are recomputed from the authoritative
# E1/E9 tables, so the corruption is caught by the recomputation itself.


def test_altered_e10b_control_metric_is_rejected_even_with_small_stored_error():
    rows = _e10b_rows()
    rows[1] = {**rows[1], "test_normalized_chi2": "999.0"}  # stored error left at 0.0
    with pytest.raises(IntegrityError, match="RECOMPUTED from the archive"):
        rv.validate_e10b_final_rows(
            rows, columns=e10b.COLUMNS, config_hash="cfg",
            analysis_label=e10b.ANALYSIS_LABEL, tolerance=TOL)


def test_altered_e11_metric_is_rejected_even_with_small_stored_error():
    rows = _e11_rows()
    rows[1] = {**rows[1], "test_normalized_chi2": "999.0",
               "angle_to_cw_pca_mean_deg_v2": "12.0"}
    with pytest.raises(IntegrityError, match="RECOMPUTED from the archive"):
        rv.validate_e11_final_rows(
            rows, columns=e11.COLUMNS, config_hash="cfg",
            analysis_label=e11.ANALYSIS_LABEL, tolerance=TOL)


def test_e10b_archived_column_must_match_the_authoritative_archive():
    rows = _e10b_rows()
    rows[0] = {**rows[0], "archived_oracle_test_raw_mse": "0.5"}
    with pytest.raises(IntegrityError, match="authoritative archive"):
        rv.validate_e10b_final_rows(
            rows, columns=e10b.COLUMNS, config_hash="cfg",
            analysis_label=e10b.ANALYSIS_LABEL, tolerance=TOL)


def test_e10b_delta_must_match_recomputation_from_the_archive():
    rows = _e10b_rows()
    rows[0] = {**rows[0], "delta_normalized_chi2_estimated_minus_oracle": "0.0"}
    with pytest.raises(IntegrityError, match="recomputed from the authoritative archive"):
        rv.validate_e10b_final_rows(
            rows, columns=e10b.COLUMNS, config_hash="cfg",
            analysis_label=e10b.ANALYSIS_LABEL, tolerance=TOL)


def test_stored_error_column_inconsistent_with_recomputation_is_rejected():
    """Even a *small* stored error must equal the recomputed one."""
    rows = _e10b_rows()
    rows[1] = {**rows[1], "control_reproduction_rel_error_raw_mse": "1e-10"}
    with pytest.raises(IntegrityError, match="disagrees with the recomputed value"):
        rv.validate_e10b_final_rows(
            rows, columns=e10b.COLUMNS, config_hash="cfg",
            analysis_label=e10b.ANALYSIS_LABEL, tolerance=TOL)


# ------------------------------- shard registry (resume provenance binding)


def _identity(freeze="a" * 40, execution="b" * 40, cfg="cfg"):
    return rv.run_identity(freeze, execution, cfg)


def test_registry_binds_shards_to_the_run_identity(tmp_path):
    from experiments.synthetic.nfpa_confirmatory import r9_common as rc

    shard = tmp_path / "seed_00.csv"
    rc.write_rows_csv(shard, [{"seed": "0", "method": "NFPA"}], ["seed", "method"])
    ident = _identity()
    rv.register_shard(tmp_path, 0, shard, ident)
    assert rv.validate_registered_shards(tmp_path, {0: shard}, ident) == {0: shard}


def test_resume_rejects_an_unregistered_shard(tmp_path):
    from experiments.synthetic.nfpa_confirmatory import r9_common as rc

    a, b = tmp_path / "seed_00.csv", tmp_path / "seed_01.csv"
    for p in (a, b):
        rc.write_rows_csv(p, [{"seed": "0", "method": "NFPA"}], ["seed", "method"])
    ident = _identity()
    rv.register_shard(tmp_path, 0, a, ident)          # b never registered
    with pytest.raises(IntegrityError, match="not registered"):
        rv.validate_registered_shards(tmp_path, {0: a, 1: b}, ident)


def test_resume_rejects_a_modified_shard(tmp_path):
    from experiments.synthetic.nfpa_confirmatory import r9_common as rc

    shard = tmp_path / "seed_00.csv"
    rc.write_rows_csv(shard, [{"seed": "0", "method": "NFPA"}], ["seed", "method"])
    ident = _identity()
    rv.register_shard(tmp_path, 0, shard, ident)
    rc.write_rows_csv(shard, [{"seed": "0", "method": "TAMPERED"}], ["seed", "method"])
    with pytest.raises(IntegrityError, match="digest mismatch"):
        rv.validate_registered_shards(tmp_path, {0: shard}, ident)


def test_resume_rejects_shards_from_a_different_freeze_or_execution(tmp_path):
    from experiments.synthetic.nfpa_confirmatory import r9_common as rc

    shard = tmp_path / "seed_00.csv"
    rc.write_rows_csv(shard, [{"seed": "0", "method": "NFPA"}], ["seed", "method"])
    rv.register_shard(tmp_path, 0, shard, _identity())
    for other in (
        _identity(freeze="c" * 40),          # different implementation freeze
        _identity(execution="d" * 40),       # different execution commit
        _identity(cfg="other"),              # different configuration
    ):
        with pytest.raises(IntegrityError, match="different run identity"):
            rv.validate_registered_shards(tmp_path, {0: shard}, other)


def test_resume_rejects_shards_with_no_registry_at_all(tmp_path):
    from experiments.synthetic.nfpa_confirmatory import r9_common as rc

    shard = tmp_path / "seed_00.csv"
    rc.write_rows_csv(shard, [{"seed": "0", "method": "NFPA"}], ["seed", "method"])
    with pytest.raises(IntegrityError, match="registry .* is missing|shard_registry"):
        rv.validate_registered_shards(tmp_path, {0: shard}, _identity())


# ---------------------------------------- dirty-state semantics at completion


def test_completion_does_not_reinterpret_expected_output_as_source_dirtiness(monkeypatch):
    from experiments.synthetic.nfpa_confirmatory.config import PRIMARY

    # Source tree was clean at start; by completion the run has written results,
    # so a naive git-status call would report dirty. The block must not.
    monkeypatch.setattr(prov, "worktree_is_dirty", lambda: True)
    block = prov.provenance_block(
        ["cmd"], PRIMARY, freeze_sha="a" * 40, execution_head="b" * 40, smoke=False,
        started_at_utc="2026-08-19T00:00:00Z",
        source_tree_dirty_at_start=False,
        expected_output_changes_at_completion=["results/rps2026/E11_corrected_angles"],
    )
    assert block["source_tree_dirty_at_start"] is False
    assert block["source_commit"] == "b" * 40
    assert block["expected_output_changes_at_completion"] == [
        "results/rps2026/E11_corrected_angles"
    ]


# ------------------------------------------------- out-of-tree runtime root


def test_runtime_root_allows_out_of_tree_execution(tmp_path, monkeypatch):
    from experiments.synthetic.nfpa_confirmatory import r10_common as r10c

    # Non-smoke output outside the worktree is always permitted...
    assert str(r10c.resolve_out_dir(tmp_path / "E11", smoke=False)).startswith(str(tmp_path))
    # ... and smoke may join it once a runtime root is declared.
    with pytest.raises(IntegrityError, match="R10_RUNTIME_ROOT"):
        r10c.resolve_out_dir(tmp_path / "Round10_smoke" / "E11", smoke=True)
    monkeypatch.setenv("R10_RUNTIME_ROOT", str(tmp_path))
    assert r10c.resolve_out_dir(tmp_path / "Round10_smoke" / "E11", smoke=True)
    # Archived directories stay refused regardless of the runtime root.
    for archived in r10c.ARCHIVED_DIRS:
        with pytest.raises(IntegrityError):
            r10c.resolve_out_dir(archived, smoke=False)


def test_demo_figure_can_be_redirected_away_from_the_tracked_png():
    """The full suite must not dirty the source tree via the demo's figure."""
    demo = (REPO_ROOT / "src/NFPA/nfpa_demo.py").read_text()
    assert "NFPA_DEMO_OUT_DIR" in demo
    test_src = (
        REPO_ROOT
        / "experiments/synthetic/nfpa_confirmatory/tests/test_correctness.py"
    ).read_text()
    assert "NFPA_DEMO_OUT_DIR" in test_src
