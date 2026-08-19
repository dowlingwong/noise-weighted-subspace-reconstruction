"""Fail-closed final-row validation and the authoritative archive audit.

Two independent problems this module closes.

1. **Resume validation was schema-only.** ``r9_common.validate_existing_shards``
   checks column names, seed uniqueness, config hash, regime and method set. A
   shard whose *numbers* were edited — a reproduction error pushed above
   tolerance, a cohort relabelled, an angle blanked — still passed and was
   reused, and the completed manifest then reported gates that had never been
   re-evaluated on the reused rows. The validators here re-derive every gate
   from the FINAL concatenated rows, immediately before the manifest is
   written, so a reused shard is held to exactly the same standard as a fresh
   one.

2. **The archive audit claimed more than it checked.** The Round 10 tests
   pinned six files while the code and comments spoke of "E1–E10 protected".
   :data:`AUTHORITATIVE_ARCHIVE` is the full 15-path set recorded at the
   Round 10 Phase 0 preflight, and it is the only list any Round 10 code,
   test, or document may cite.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .r9_common import IntegrityError, REPO_ROOT, read_rows_csv, sha256_of

ARCHIVED_E1_CSV = REPO_ROOT / "results/rps2026/E1_nfpa_blockA/block_A_colored_raw.csv"
ARCHIVED_E9_CSV = REPO_ROOT / "results/rps2026/E9_blockA_replication/replication_raw.csv"

# --------------------------------------------------------------------------
# Authoritative archive set (all 15 paths from Round10_desktop/preflight.txt)
# --------------------------------------------------------------------------

#: path -> SHA-256 recorded at Round 10 Phase 0. Any mismatch is a global stop.
AUTHORITATIVE_ARCHIVE: dict[str, str] = {
    "results/rps2026/E1_nfpa_blockA/block_A_colored_raw.csv":
        "e9778eb2d66366501e7016d0e3bfa3626aad6b0575322d94da995f583e6dbd7e",
    "results/rps2026/E2_nfpa_blockB/block_B_white_raw.csv":
        "9d1d65d92c140a9afb25646152e5ac628027b1efc1848db16be8e1d4697e8e33",
    "results/rps2026/E8_semantics/manifest.json":
        "e5a8f497b1d4eee1c736c39f3f02046f9b394313a5995582de4ba82e4e4a5042",
    "results/rps2026/E8_semantics/archived_fit_reproduction.csv":
        "a545925fc82e47a7f593731ed7b9f6f1df258d4662a64ed3704a4a48cca2a5f6",
    "results/rps2026/E8_semantics/primary_alpha_0p1/summary.json":
        "e0cee07e69f5189502cdf7325c77ee8e955f29e2a205ee5fb57524af06cd634e",
    "results/rps2026/E8_semantics/separable_alpha_0/summary.json":
        "b55b5260f3f0f431c27d34a1ffa1a70bd1f3f10a4b30ab56e005048e9c4da495",
    "results/rps2026/E9_blockA_replication/replication_raw.csv":
        "cb228581edc979e58dc8a152d2e8c484ff46d23b702141d5917f7e94d221121e",
    "results/rps2026/E9_blockA_replication/replication_summary.json":
        "769582050107b269361e35c2fddb8f5549a08e56f34d3736d3a8cc6f7df93bd7",
    "results/rps2026/E9_blockA_replication/pooled_summary.json":
        "845f66e13c1c24e9f3965b1f58fd5b8134b5d742169283a4e7577db28cbd3d92",
    "results/rps2026/E9_blockA_replication/manifest.json":
        "e74c49aa97dc0a940d551e73d4eeeec7c4dd9cedaa20abeb5686ec7e05e0fc20",
    "results/rps2026/E10_estimated_covariance/raw.csv":
        "9c17834614a67694986e026d0abaec69e3078b6d8747216c4bb98966383bcfae",
    "results/rps2026/E10_estimated_covariance/summary.json":
        "bd0e0336d03b4d3992199d5de71d97e6bbba4cde076ac90dc3fda586ce8c8737",
    "results/rps2026/E10_estimated_covariance/manifest.json":
        "fc2a8ac1c4518449df3b6940e79ece73e6fb053c83342571b7a187787e9b6fa1",
    "results/rps2026/E6_whitening_range.json":
        "7f85a6b5b7ebd14cf9324444c46be13e22da5fc25c399409e268d921fb63a433",
    "results/rps2026/E7_rung4_seed_replication.json":
        "87a181a86b3d405dc6b55cb0366d95a96e7f7b0909bb703b8ff6a0284a4cec18",
}


def audit_authoritative_archive() -> dict[str, Any]:
    """Digest every authoritative archive file; report mismatches and absences."""
    mismatched: list[dict[str, str]] = []
    missing: list[str] = []
    checked: dict[str, str] = {}
    for rel, want in sorted(AUTHORITATIVE_ARCHIVE.items()):
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        got = sha256_of(path)
        checked[rel] = got
        if got != want:
            mismatched.append({"path": rel, "expected": want, "observed": got})
    return {
        "n_files": len(AUTHORITATIVE_ARCHIVE),
        "n_checked": len(checked),
        "digests": checked,
        "missing": missing,
        "mismatched": mismatched,
        "passed": not missing and not mismatched,
    }


def require_authoritative_archive_unchanged() -> dict[str, Any]:
    audit = audit_authoritative_archive()
    if not audit["passed"]:
        raise IntegrityError(
            "AUTHORITATIVE ARCHIVE CHANGED — global stop.\n"
            f"  missing: {audit['missing']}\n  mismatched: {audit['mismatched']}"
        )
    return audit


# --------------------------------------------------------------------------
# Shared row helpers
# --------------------------------------------------------------------------


def _finite(value: str | float | None) -> bool:
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _f(value: str | float) -> float:
    return float(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _check_common(
    rows: Sequence[dict],
    *,
    experiment: str,
    seeds: Iterable[int],
    methods: Iterable[str],
    columns: Sequence[str],
    config_hash: str,
    regime: str,
    analysis_label: str,
) -> None:
    seeds = sorted(int(s) for s in seeds)
    methods = sorted(methods)
    _require(bool(rows), f"{experiment}: no rows to validate")

    got_seeds = sorted({int(r["seed"]) for r in rows})
    _require(got_seeds == seeds, f"{experiment}: seed set {got_seeds} != required {seeds}")

    pairs = [(int(r["seed"]), r["method"]) for r in rows]
    _require(len(pairs) == len(set(pairs)), f"{experiment}: duplicate (seed, method) rows")
    _require(
        len(rows) == len(seeds) * len(methods),
        f"{experiment}: expected {len(seeds) * len(methods)} rows, found {len(rows)}",
    )
    for seed in seeds:
        got = sorted(r["method"] for r in rows if int(r["seed"]) == seed)
        _require(got == methods, f"{experiment}: seed {seed} has methods {got} != {methods}")

    for row in rows:
        _require(
            list(row.keys()) == list(columns),
            f"{experiment}: seed {row['seed']} has an unexpected schema",
        )
        _require(
            row["config_hash"] == config_hash,
            f"{experiment}: seed {row['seed']} config_hash {row['config_hash']!r} != {config_hash!r}",
        )
        _require(row["regime"] == regime, f"{experiment}: seed {row['seed']} regime mismatch")
        _require(
            row["analysis_label"] == analysis_label,
            f"{experiment}: seed {row['seed']} analysis_label mismatch",
        )


# --------------------------------------------------------------------------
# Authoritative archived metrics — the ONLY trusted source for the gates
# --------------------------------------------------------------------------


def load_authoritative_metrics() -> dict[tuple[int, str], dict[str, Any]]:
    """``(seed, method) -> {test_raw_mse, test_normalized_chi2, source}``.

    Read straight from the archived E1 (seeds 0-9) and E9 (seeds 10-29) raw
    tables, whose digests are pinned in :data:`AUTHORITATIVE_ARCHIVE`. The
    validators recompute every reproduction error and every archived-value
    column against THIS table. A stored error column is treated as a claim to
    be checked, never as evidence: corrupting a metric while leaving a small
    error value in place must fail, and does.
    """
    table: dict[tuple[int, str], dict[str, Any]] = {}
    for path, source in ((ARCHIVED_E1_CSV, "E1"), (ARCHIVED_E9_CSV, "E9")):
        if not path.exists():
            raise IntegrityError(f"authoritative archive table missing: {path}")
        for row in read_rows_csv(path):
            table[(int(row["seed"]), row["method"])] = {
                "test_raw_mse": float(row["test_raw_mse"]),
                "test_normalized_chi2": float(row["test_normalized_chi2"]),
                "source": source,
            }
    return table


#: Agreement required between a stored column and its recomputation.
RECOMPUTE_RTOL = 1e-9
RECOMPUTE_ATOL = 1e-12


def _agrees(stored: float, recomputed: float) -> bool:
    return abs(stored - recomputed) <= max(
        RECOMPUTE_ATOL, RECOMPUTE_RTOL * max(abs(stored), abs(recomputed))
    )


def _rel_error(observed: float, archived: float) -> float:
    return abs(observed - archived) / max(abs(archived), 1e-15)


# --------------------------------------------------------------------------
# E10b
# --------------------------------------------------------------------------

E10B_REQUIRED_FINITE = (
    "test_raw_mse",
    "test_normalized_chi2",
    "v_c",
    "v_t",
    "v_hat",
    "product_scale_ratio_vs_oracle",
    "channel_shape_error",
    "temporal_shape_error",
    "kronecker_relative_frobenius_error",
    "oracle_whitening_residual",
    "channel_min_eigenvalue",
    "channel_condition_number",
    "temporal_min_eigenvalue",
    "temporal_condition_number",
)


def validate_e10b_final_rows(
    rows: Sequence[dict], *, columns: Sequence[str], config_hash: str,
    analysis_label: str, tolerance: float,
    authoritative: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-derive every E10b gate from the final rows. Returns the gate record.

    ``authoritative`` defaults to :func:`load_authoritative_metrics`. Pass an
    explicit table only in tests. Every archived value, reproduction error and
    delta is RECOMPUTED from it and compared with the stored column.
    """
    if authoritative is None:
        authoritative = load_authoritative_metrics()
    _check_common(
        rows, experiment="E10b", seeds=range(10), methods=("Iso-MPCA", "NFPA"),
        columns=columns, config_hash=config_hash, regime="colored",
        analysis_label=analysis_label,
    )

    worst_control = 0.0
    worst_recompute_disagreement = 0.0
    for row in rows:
        seed = int(row["seed"])
        arch = authoritative.get((seed, row["method"]))
        _require(
            arch is not None,
            f"E10b: seed {seed} method {row['method']} has no authoritative archived record",
        )
        _require(
            arch["source"] == "E1",
            f"E10b: seed {seed} must reference E1, authoritative table says {arch['source']}",
        )
        for col in E10B_REQUIRED_FINITE:
            _require(
                _finite(row[col]),
                f"E10b: seed {row['seed']} method {row['method']} has non-finite {col}={row[col]!r}",
            )
        if row["method"] == "NFPA":
            _require(
                row["whitening"] == "estimated_calibrated_v2",
                f"E10b: seed {row['seed']} NFPA whitening label altered",
            )
            for col in (
                "archived_oracle_test_raw_mse",
                "archived_oracle_test_normalized_chi2",
                "delta_raw_mse_estimated_minus_oracle",
                "delta_normalized_chi2_estimated_minus_oracle",
            ):
                _require(
                    _finite(row[col]),
                    f"E10b: seed {row['seed']} NFPA missing/non-finite {col}",
                )
            # Archived columns must equal the AUTHORITATIVE archive, not merely
            # be self-consistent, and the deltas are recomputed from it.
            for stored_col, key in (
                ("archived_oracle_test_raw_mse", "test_raw_mse"),
                ("archived_oracle_test_normalized_chi2", "test_normalized_chi2"),
            ):
                _require(
                    _agrees(_f(row[stored_col]), arch[key]),
                    f"E10b: seed {seed} {stored_col}={row[stored_col]} disagrees with the "
                    f"authoritative archive value {arch[key]!r}",
                )
            for got_col, cur_col, key in (
                ("delta_raw_mse_estimated_minus_oracle", "test_raw_mse", "test_raw_mse"),
                ("delta_normalized_chi2_estimated_minus_oracle", "test_normalized_chi2",
                 "test_normalized_chi2"),
            ):
                expect = _f(row[cur_col]) - arch[key]
                _require(
                    _agrees(_f(row[got_col]), expect),
                    f"E10b: seed {seed} {got_col}={row[got_col]} disagrees with the value "
                    f"recomputed from the authoritative archive ({expect!r})",
                )
                worst_recompute_disagreement = max(
                    worst_recompute_disagreement, abs(_f(row[got_col]) - expect)
                )
        else:
            _require(
                row["whitening"] == "none_covariance_independent_control",
                f"E10b: seed {row['seed']} control whitening label altered",
            )
            # RECOMPUTE the control reproduction errors from the authoritative
            # archive and the row's own metrics. The stored columns are claims;
            # this is the check. Corrupting a metric while leaving a small
            # stored error in place fails here.
            for col, metric_col, key in (
                ("control_reproduction_rel_error_raw_mse", "test_raw_mse", "test_raw_mse"),
                ("control_reproduction_rel_error_normalized_chi2", "test_normalized_chi2",
                 "test_normalized_chi2"),
            ):
                _require(
                    _finite(row[col]),
                    f"E10b: seed {seed} control missing/non-finite {col}",
                )
                recomputed = _rel_error(_f(row[metric_col]), arch[key])
                _require(
                    recomputed <= tolerance,
                    f"E10b: seed {seed} control reproduction RECOMPUTED from the archive is "
                    f"{recomputed:.3e} > {tolerance:.0e} (stored column said {row[col]})",
                )
                _require(
                    _agrees(_f(row[col]), recomputed),
                    f"E10b: seed {seed} stored {col}={row[col]} disagrees with the recomputed "
                    f"value {recomputed!r}",
                )
                worst_control = max(worst_control, recomputed)
                worst_recompute_disagreement = max(
                    worst_recompute_disagreement, abs(_f(row[col]) - recomputed)
                )
            for stored_col, key in (
                ("archived_oracle_test_raw_mse", "test_raw_mse"),
                ("archived_oracle_test_normalized_chi2", "test_normalized_chi2"),
            ):
                _require(_finite(row[stored_col]), f"E10b: seed {seed} control missing {stored_col}")
                _require(
                    _agrees(_f(row[stored_col]), arch[key]),
                    f"E10b: seed {seed} {stored_col} disagrees with the authoritative archive",
                )

    return {
        "control_gate_passed": True,  # recomputed above; any failure raised
        "worst_control_reproduction_rel_error": worst_control,
        "worst_stored_vs_recomputed_disagreement": worst_recompute_disagreement,
        "tolerance_rel": tolerance,
        "n_rows_validated": len(rows),
        "recomputed_from": (
            "final concatenated rows (fresh and resumed alike), with every archived "
            "value, reproduction error and delta recomputed from the authoritative "
            "E1 table rather than trusted from the stored columns"
        ),
    }


# --------------------------------------------------------------------------
# E11
# --------------------------------------------------------------------------


def validate_e11_final_rows(
    rows: Sequence[dict], *, columns: Sequence[str], config_hash: str,
    analysis_label: str, tolerance: float,
    authoritative: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-derive every E11 gate from the final rows. Returns the gate record.

    Reproduction errors are RECOMPUTED from the authoritative E1/E9 tables and
    compared with the stored columns; a corrupted metric fails even when the
    stored error column still looks small.
    """
    if authoritative is None:
        authoritative = load_authoritative_metrics()
    _check_common(
        rows, experiment="E11", seeds=range(30), methods=("CW-PCA", "Iso-MPCA", "NFPA"),
        columns=columns, config_hash=config_hash, regime="colored",
        analysis_label=analysis_label,
    )

    worst = 0.0
    worst_recompute_disagreement = 0.0
    for row in rows:
        seed = int(row["seed"])
        want_cohort = "frozen_0_9" if seed < 10 else "replication_10_29"
        want_source = "E1" if seed < 10 else "E9"
        arch = authoritative.get((seed, row["method"]))
        _require(
            arch is not None,
            f"E11: seed {seed} method {row['method']} has no authoritative archived record",
        )
        _require(
            arch["source"] == want_source,
            f"E11: seed {seed} authoritative source is {arch['source']}, expected {want_source}",
        )
        _require(
            row["cohort"] == want_cohort,
            f"E11: seed {seed} cohort {row['cohort']!r} != {want_cohort!r}",
        )
        _require(
            row["archived_source"] == want_source,
            f"E11: seed {seed} archived_source {row['archived_source']!r} != {want_source!r}",
        )
        for col in ("test_raw_mse", "test_normalized_chi2"):
            _require(_finite(row[col]), f"E11: seed {seed} {row['method']} non-finite {col}")
        # RECOMPUTE both reproduction errors from the authoritative archive.
        for col, metric_col, key in (
            ("reproduction_rel_error_raw_mse", "test_raw_mse", "test_raw_mse"),
            ("reproduction_rel_error_normalized_chi2", "test_normalized_chi2",
             "test_normalized_chi2"),
        ):
            _require(
                _finite(row[col]),
                f"E11: seed {seed} {row['method']} missing/non-finite {col}",
            )
            recomputed = _rel_error(_f(row[metric_col]), arch[key])
            _require(
                recomputed <= tolerance,
                f"E11: seed {seed} {row['method']} reproduction RECOMPUTED from the archive "
                f"is {recomputed:.3e} > {tolerance:.0e} (stored column said {row[col]})",
            )
            _require(
                _agrees(_f(row[col]), recomputed),
                f"E11: seed {seed} {row['method']} stored {col}={row[col]} disagrees with "
                f"the recomputed value {recomputed!r}",
            )
            worst = max(worst, recomputed)
            worst_recompute_disagreement = max(
                worst_recompute_disagreement, abs(_f(row[col]) - recomputed)
            )

        angle_cols = (
            "angle_to_cw_pca_mean_deg_v2",
            "angle_to_cw_pca_max_deg_v2",
            "angle_to_cw_pca_mean_deg_legacy",
            "angle_to_cw_pca_max_deg_legacy",
        )
        if row["method"] == "CW-PCA":
            # Declared convention: the self-reference angle fields are EMPTY,
            # never 0.0 — an angle of zero would be a measurement, not a
            # tautology.
            for col in angle_cols:
                _require(
                    row[col] in ("", None),
                    f"E11: seed {seed} CW-PCA self-reference {col} must be empty, got {row[col]!r}",
                )
        else:
            for col in ("angle_to_cw_pca_mean_deg_v2", "angle_to_cw_pca_max_deg_v2"):
                _require(
                    _finite(row[col]),
                    f"E11: seed {seed} {row['method']} non-finite corrected angle {col}",
                )

    return {
        "reproduction_gate_passed": True,  # recomputed above; any failure raised
        "worst_reproduction_rel_error": worst,
        "worst_stored_vs_recomputed_disagreement": worst_recompute_disagreement,
        "tolerance_rel": tolerance,
        "n_rows_validated": len(rows),
        "recomputed_from": (
            "final concatenated rows (fresh and resumed alike), with both reproduction "
            "errors recomputed from the authoritative E1/E9 tables rather than trusted "
            "from the stored columns"
        ),
    }


# --------------------------------------------------------------------------
# Deterministic summary regeneration audit
# --------------------------------------------------------------------------


def canonical_json_digest(payload: Any) -> str:
    """Digest of a canonical JSON rendering (sorted keys, fixed separators)."""
    import hashlib

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_summary_regenerates(raw_csv: Path, written_summary: dict, summarize_fn) -> dict[str, Any]:
    """Regenerate the summary from ``raw.csv`` and require an identical digest.

    This is the audit that makes "the summary is a deterministic function of
    the shards" a checked property rather than a claim in a docstring.
    """
    rows = read_rows_csv(raw_csv)
    regenerated = summarize_fn(rows)
    want = canonical_json_digest(written_summary)
    got = canonical_json_digest(regenerated)
    if want != got:
        raise IntegrityError(
            "summary regeneration audit FAILED: the summary written to disk is not "
            "reproducible from raw.csv\n"
            f"  written digest:     {want}\n  regenerated digest: {got}"
        )
    return {
        "passed": True,
        "canonical_digest": got,
        "regenerated_from": str(raw_csv.name),
    }


# --------------------------------------------------------------------------
# Shard registry — binds every shard to the run identity that produced it
# --------------------------------------------------------------------------
#
# Structural resume validation still allowed a partial directory produced by a
# DIFFERENT code revision to be reused and then attributed to the current
# execution commit. The registry closes that: the in-progress manifest carries
# the freeze/execution identity from the start, and each shard is registered
# with its SHA-256 immediately after it is written. Resume rejects an
# unregistered shard, a digest mismatch, or a freeze/execution mismatch.


def run_identity(freeze_sha: str | None, execution_head: str, config_hash: str) -> dict[str, Any]:
    return {
        "implementation_freeze_commit": freeze_sha,
        "execution_content_commit": execution_head,
        "config_hash": config_hash,
    }


def register_shard(out_dir: Path, seed: int, shard: Path, identity: dict[str, Any]) -> None:
    """Atomically add ``seed -> {digest, identity}`` to the shard registry."""
    from .r9_common import write_json

    registry_path = Path(out_dir) / "shard_registry.json"
    registry = (
        json.loads(registry_path.read_text()) if registry_path.exists()
        else {"identity": identity, "shards": {}}
    )
    if registry.get("identity") != identity:
        raise IntegrityError(
            "shard registry identity mismatch: this directory was started by a "
            f"different run.\n  registry: {registry.get('identity')}\n  current:  {identity}"
        )
    registry["shards"][str(seed)] = {
        "file": shard.name,
        "sha256": sha256_of(shard),
    }
    write_json(registry_path, registry)


def validate_registered_shards(
    out_dir: Path, existing: dict[int, Path], identity: dict[str, Any]
) -> dict[int, Path]:
    """Reject unregistered shards, digest mismatches and identity mismatches."""
    out_dir = Path(out_dir)
    if not existing:
        return {}
    registry_path = out_dir / "shard_registry.json"
    if not registry_path.exists():
        raise IntegrityError(
            f"{len(existing)} shard(s) present but {registry_path.name} is missing; "
            "this directory was produced before shard registration existed or was "
            "modified out of band. Refusing to reuse it."
        )
    registry = json.loads(registry_path.read_text())
    recorded = registry.get("identity")
    if recorded != identity:
        raise IntegrityError(
            "refusing to resume: the existing shards were produced by a different "
            f"run identity.\n  registry: {recorded}\n  current:  {identity}"
        )
    shards = registry.get("shards", {})
    for seed, path in sorted(existing.items()):
        entry = shards.get(str(seed))
        if entry is None:
            raise IntegrityError(
                f"shard {path.name} is not registered; it did not come from this run"
            )
        got = sha256_of(path)
        if got != entry["sha256"]:
            raise IntegrityError(
                f"shard {path.name} digest mismatch — it was modified after it was "
                f"written.\n  registered: {entry['sha256']}\n  observed:   {got}"
            )
    return existing
