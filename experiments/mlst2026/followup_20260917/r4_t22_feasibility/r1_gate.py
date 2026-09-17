"""A2: the R1-failure -> R4-cancellation gate (hard-wired, fail-closed).

R4's large-file production cells (n_files = 32/64) are only admissible if R1
yields a usable, *confirmed* interval.  The precondition is evaluated here, from
R1's frozen outcome and its acceptance receipt, before any R4 production cell is
planned, priced, drawn or charged.

Statuses
--------
``pass``
    R1 froze a primary-eligible candidate and a **complete, explicit** acceptance
    receipt confirms coverage and type-I within tolerance at every candidate file
    count, and neither withholding screen is universal.
``cancel``
    A scientific infeasibility: R1 froze ``"none"``, or its interval failed the
    coverage/type-I rules, or the withholding-bound screen ruled out every design.
    The large-file cells are omitted; the rest of R4 may run.
``blocked``
    A precondition is missing, malformed, partial, stale or mismatched: no
    acceptance receipt, an incomplete or development-only receipt, a candidate
    mismatch, an absent protocol/matrix identity, or a malformed Boolean set.
    This is **not** a scientific verdict; production must not start.
``unresolved``
    R1 has not yet produced a freeze.

Only ``pass`` is transferable.  Any other status removes the large-file cells
from the matrix, launch plan and cost report.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "mlst-followup-r4-infeasible/2"
R1_ACCEPTANCE_SCHEMAS = ("mlst-followup-r1-acceptance/2",)
DEFAULT_CANCEL_COUNTS = (32, 64)


def _read_json(path):
    p = Path(path) if path else None
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def evaluate_r1_precondition(freeze_path, acceptance_path=None, *,
                             cancel_large_file_counts=DEFAULT_CANCEL_COUNTS,
                             r4_withholding_screen=None,
                             expected_protocol_sha256: str | None = None,
                             expected_matrix_sha256: str | None = None) -> dict:
    """Machine-checkable R1 precondition for R4 (fail-closed)."""
    counts = [int(x) for x in cancel_large_file_counts]
    out = {"schema": SCHEMA, "freeze_path": str(freeze_path) if freeze_path else None,
           "acceptance_path": str(acceptance_path) if acceptance_path else None,
           "cancel_large_file_counts": counts, "status": "unresolved", "trigger": None,
           "reasons": [], "r1": {}, "cancel": False, "blocked": False}
    freeze = _read_json(freeze_path)
    if freeze is None:
        out["reasons"].append("R1 freeze is absent: R1 has not selected an interval")
        return out
    out["r1"]["chosen_candidate"] = freeze.get("chosen_candidate")
    out["r1"]["protocol_sha256"] = freeze.get("protocol_sha256")
    out["r1"]["matrix_sha256"] = freeze.get("matrix_sha256")
    out["r1"]["development_only"] = freeze.get("development_only")
    if freeze.get("schema") != "mlst-r1-frozen-candidate/1":
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append(f"R1 freeze schema is {freeze.get('schema')!r}, not the R1 freeze schema")
        return out
    if freeze.get("development_only"):
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 freeze is development_only and never citable")
        return out
    if freeze.get("chosen_candidate") == "none":
        out["status"], out["trigger"], out["cancel"] = "cancel", "r1_chose_none", True
        out["reasons"].append("R1 froze 'none': no prespecified interval was adequate")
        return out
    if freeze.get("chosen_candidate") not in freeze.get("candidate_family", []):
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append(f"R1 freeze names an unknown candidate {freeze.get('chosen_candidate')!r}")
        return out
    if freeze.get("chosen_candidate") not in freeze.get("primary_eligible", []):
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 froze a target-mismatched candidate that cannot replace the equal-file primary")
        return out
    acceptance = _read_json(acceptance_path)
    if acceptance is None:
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance receipt is absent or unreadable: the selected interval is not confirmed")
        return out
    if acceptance.get("schema") not in R1_ACCEPTANCE_SCHEMAS:
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append(f"R1 acceptance schema {acceptance.get('schema')!r} is not a recognised receipt")
        return out
    out["r1"]["acceptance"] = {k: acceptance.get(k) for k in
                               ("coverage_ok", "type1_ok", "frozen_candidate", "development_only",
                                "confirmation_complete", "candidate_primary_eligible", "decisions_explicit",
                                "protocol_sha256", "matrix_sha256")}
    if not acceptance.get("decisions_explicit"):
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance decisions are not explicit Booleans")
        return out
    if acceptance.get("development_only"):
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance receipt is development_only")
        return out
    if not acceptance.get("confirmation_complete"):
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance receipt is not from a complete confirmation reduction")
        return out
    if acceptance.get("candidate_primary_eligible") is not True:
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance candidate is not primary-eligible")
        return out
    if acceptance.get("frozen_candidate") != freeze.get("chosen_candidate"):
        out["status"], out["trigger"], out["cancel"] = "cancel", "r1_acceptance_candidate_mismatch", True
        out["reasons"].append("R1 acceptance receipt is for a different candidate than the freeze")
        return out
    if expected_protocol_sha256 and acceptance.get("protocol_sha256") != expected_protocol_sha256:
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance protocol identity does not match the R1 freeze")
        return out
    if expected_matrix_sha256 and acceptance.get("matrix_sha256") != expected_matrix_sha256:
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance matrix identity does not match the R1 freeze")
        return out
    cov, t1 = acceptance.get("coverage_ok"), acceptance.get("type1_ok")
    if not isinstance(cov, dict) or not cov or not isinstance(t1, dict) or not t1:
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance receipt has empty coverage/type-I decisions")
        return out
    if not all(isinstance(v, bool) for v in list(cov.values()) + list(t1.values())):
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance decisions are not all Booleans")
        return out
    if set(cov) != set(t1):
        out["status"], out["blocked"] = "blocked", True
        out["reasons"].append("R1 acceptance coverage and type-I file counts differ")
        return out
    bad_cov = sorted(k for k, v in cov.items() if v is False)
    bad_t1 = sorted(k for k, v in t1.items() if v is False)
    if bad_cov or bad_t1:
        out["status"], out["trigger"], out["cancel"] = "cancel", "r1_interval_failed", True
        out["reasons"].append(f"R1 interval failed coverage at {bad_cov} and/or type-I at {bad_t1}")
        return out
    if acceptance.get("withholding_screen_all_infeasible") is True:
        out["status"], out["trigger"], out["cancel"] = "cancel", "withholding_screen_infeasible", True
        out["reasons"].append("the withholding-bound screen ruled out every candidate design")
        return out
    if isinstance(r4_withholding_screen, dict) and r4_withholding_screen and all(r4_withholding_screen.values()):
        out["status"], out["trigger"], out["cancel"] = "cancel", "r4_withholding_screen_infeasible", True
        out["reasons"].append("R4's own withholding-bound screen ruled out every candidate file count")
        return out
    out["status"] = "pass"
    out["reasons"].append("R1 froze an accepted primary-eligible candidate and the screen is not universal")
    return out


def ready_to_transfer(gate: dict) -> bool:
    """R4 may only be called ready to transfer when the R1 precondition passes."""
    return bool(gate and gate.get("status") == "pass" and not gate.get("cancel") and not gate.get("blocked"))


def large_file_admissible(gate: dict) -> bool:
    """Large-file cells are admissible only on a clean pass."""
    return bool(gate and gate.get("status") == "pass")


def ineffective_receipt(gate: dict, *, r4_protocol_sha256: str | None = None) -> dict:
    """The emitted R4_INFEASIBLE receipt (machine-readable)."""
    return {"schema": SCHEMA, "kind": "R4_INFEASIBLE",
            "status": gate.get("status"), "trigger": gate.get("trigger"),
            "cancelled_large_file_counts": gate.get("cancel_large_file_counts") if gate.get("cancel") else [],
            "r1": gate.get("r1", {}), "reasons": gate.get("reasons", []),
            "r4_protocol_sha256": r4_protocol_sha256,
            "note": ("large-file production cells are omitted, not deferred; they are absent from the launch "
                     "plan and the cost report and are not staged as ready to transfer.  A 'blocked' status is a "
                     "missing/malformed precondition, not a scientific verdict.")}
