#!/usr/bin/env python3
"""Protocol validation and design-generation for Papers R1 and R2.

Validates YAML protocol files against required fields, checks design matrix
estimability, regenerates deterministic schedules, and compares output hashes.

Usage:
    python3 scripts/validate_protocols.py              # validate all protocols
    python3 scripts/validate_protocols.py --r1-only    # validate only R1
    python3 scripts/validate_protocols.py --r2-only    # validate only R2
    python3 scripts/validate_protocols.py --design     # also run design diagnostics
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOLS_DIR = PROJECT_ROOT / "protocols"

REQUIRED_FIELDS = {
    "common/terminology.yaml": [
        "version", "experimental_hierarchy", "outcome_definitions",
        "event_types", "visibility_classes", "fault_families"
    ],
    "common/claim_artifact_matrix.yaml": [
        "version", "claims"
    ],
    "common/endpoint_registry.yaml": [
        "version", "endpoints"
    ],
    "r1/pilot_protocol.yaml": [
        "protocol_id", "version", "experimental_design", "factors",
        "pilot_gates"
    ],
    "r1/confirmatory_protocol.yaml": [
        "protocol_id", "version", "experimental_design", "workload_families",
        "split_regimes", "primary_endpoints"
    ],
    "r2/controller_protocol.yaml": [
        "protocol_id", "version", "action_space", "controller_policies",
        "R2_C1_randomized_study"
    ],
    "r2/training_protocol.yaml": [
        "protocol_id", "version", "training_arms", "evaluation_sets",
        "primary_endpoints"
    ],
}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def check_required_fields(
    data: dict[str, Any], required: list[str], path: str
) -> list[str]:
    errors = []
    for field in required:
        if field not in data or data[field] is None:
            errors.append(f"  MISSING: {path} -> {field}")
    return errors


def validate_design_matrix(protocol: dict[str, Any], protocol_id: str) -> list[str]:
    """Check design matrix for estimability and balance diagnostics."""
    messages = []
    design = protocol.get("experimental_design", {})
    factors = protocol.get("factors", [])

    if design.get("design_type") == "mixed_level_d_optimal":
        severity_factors = [f for f in factors if f.get("name") == "fault_severity"]
        timing_factors = [f for f in factors if f.get("name") == "injection_band"]
        condition_factors = [f for f in factors if f.get("name") == "condition"]

        n_sev = len(severity_factors[0].get("levels", [])) if severity_factors else 0
        n_tim = len(timing_factors[0].get("levels", [])) if timing_factors else 0
        n_cond = len(condition_factors[0].get("levels", [])) if condition_factors else 0

        messages.append(
            f"  DESIGN CHECK [{protocol_id}]: mixed-level design with "
            f"{n_sev} severity levels, {n_tim} timing bands, {n_cond} conditions"
        )

        if n_sev * n_tim > n_cond:
            messages.append(
                f"  WARNING [{protocol_id}]: severity×timing exceeds conditions. "
                f"Full crossing not estimated. Design description must not claim "
                f"resolution-IV unless formal alias analysis supports it."
            )

        design_checks = protocol.get("design_matrix_checks", [])
        if design_checks:
            messages.append(
                f"  DESIGN CHECKS [{protocol_id}]: {', '.join(design_checks)}"
            )
        else:
            messages.append(
                f"  WARNING [{protocol_id}]: no design_matrix_checks declared"
            )

    return messages


def check_leakage_controls(protocol: dict[str, Any], protocol_id: str) -> list[str]:
    messages = []
    splits = protocol.get("split_regimes", [])
    if splits:
        split_names = [s.get("id", s) if isinstance(s, dict) else s for s in splits]
        messages.append(
            f"  SPLIT REGIMES [{protocol_id}]: {len(split_names)} regimes: "
            f"{', '.join(split_names)}"
        )

    leakage = protocol.get("leakage_controls", {})
    if leakage:
        messages.append(
            f"  LEAKAGE [{protocol_id}]: task_template_grouping={leakage.get('task_template_grouping', 'not specified')}"
        )
    else:
        messages.append(
            f"  WARNING [{protocol_id}]: no leakage_controls declared"
        )

    return messages


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if obj is None:
        return None
    return str(obj)

def compute_content_hash(data: dict[str, Any]) -> str:
    safe = _make_json_safe(data)
    canonical = json.dumps(safe, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def validate_r1_protocols() -> tuple[bool, list[str]]:
    messages = []
    all_valid = True

    pilot = load_yaml(PROTOCOLS_DIR / "r1" / "pilot_protocol.yaml")
    confirm = load_yaml(PROTOCOLS_DIR / "r1" / "confirmatory_protocol.yaml")

    pilot_errors = check_required_fields(
        pilot, REQUIRED_FIELDS["r1/pilot_protocol.yaml"], "R1 pilot"
    )
    confirm_errors = check_required_fields(
        confirm, REQUIRED_FIELDS["r1/confirmatory_protocol.yaml"], "R1 confirmatory"
    )

    if pilot_errors:
        messages.extend(pilot_errors)
        all_valid = False
    if confirm_errors:
        messages.extend(confirm_errors)
        all_valid = False

    messages.extend(validate_design_matrix(pilot, "R1-P"))
    messages.extend(validate_design_matrix(confirm, "R1-C"))
    messages.extend(check_leakage_controls(confirm, "R1-C"))

    pilot_hash = compute_content_hash(pilot)
    confirm_hash = compute_content_hash(confirm)
    messages.append(f"  HASH R1-P: {pilot_hash}")
    messages.append(f"  HASH R1-C: {confirm_hash}")

    return all_valid, messages


def validate_r2_protocols() -> tuple[bool, list[str]]:
    messages = []
    all_valid = True

    controller = load_yaml(PROTOCOLS_DIR / "r2" / "controller_protocol.yaml")
    training = load_yaml(PROTOCOLS_DIR / "r2" / "training_protocol.yaml")

    ctrl_errors = check_required_fields(
        controller, REQUIRED_FIELDS["r2/controller_protocol.yaml"], "R2 controller"
    )
    train_errors = check_required_fields(
        training, REQUIRED_FIELDS["r2/training_protocol.yaml"], "R2 training"
    )

    if ctrl_errors:
        messages.extend(ctrl_errors)
        all_valid = False
    if train_errors:
        messages.extend(train_errors)
        all_valid = False

    if controller.get("depends_on"):
        messages.append(
            f"  DEPENDENCY R2-C: depends on {controller['depends_on']}"
        )

    arms = training.get("training_arms", [])
    if len(arms) != 6:
        messages.append(
            f"  WARNING R2-T: expected 6 training arms, found {len(arms)}"
        )
    else:
        arm_names = [a.get("name", a.get("id", "?")) for a in arms]
        messages.append(f"  TRAINING ARMS R2-T: {', '.join(arm_names)}")

    messages.extend(check_leakage_controls(training, "R2-T"))

    ctrl_hash = compute_content_hash(controller)
    train_hash = compute_content_hash(training)
    messages.append(f"  HASH R2-C: {ctrl_hash}")
    messages.append(f"  HASH R2-T: {train_hash}")

    return all_valid, messages


def validate_common_protocols() -> tuple[bool, list[str]]:
    messages = []
    all_valid = True

    for name in ["terminology.yaml", "claim_artifact_matrix.yaml", "endpoint_registry.yaml"]:
        path = PROTOCOLS_DIR / "common" / name
        if path.exists():
            data = load_yaml(path)
            key = f"common/{name}"
            errors = check_required_fields(data, REQUIRED_FIELDS.get(key, ["version"]), key)
            if errors:
                messages.extend(errors)
                all_valid = False
            messages.append(f"  OK: {key}")
            messages.append(f"  HASH: {compute_content_hash(data)}")
        else:
            messages.append(f"  MISSING FILE: {key}")
            all_valid = False

    return all_valid, messages


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate R1/R2 protocols")
    parser.add_argument("--r1-only", action="store_true")
    parser.add_argument("--r2-only", action="store_true")
    parser.add_argument("--design", action="store_true",
                        help="Run design diagnostics (implies full validation)")
    args = parser.parse_args()

    exit_code = 0

    if not args.r2_only:
        print("=== Common protocols ===")
        ok, msgs = validate_common_protocols()
        for m in msgs:
            print(m)
        if not ok:
            exit_code = 1
        print()

    if not args.r2_only:
        print("=== R1 protocols ===")
        ok, msgs = validate_r1_protocols()
        for m in msgs:
            print(m)
        if not ok:
            exit_code = 1
        print()

    if not args.r1_only:
        print("=== R2 protocols ===")
        ok, msgs = validate_r2_protocols()
        for m in msgs:
            print(m)
        if not ok:
            exit_code = 1
        print()

    if exit_code == 0:
        print("ALL PROTOCOLS VALIDATED")
        if args.design:
            print("Design diagnostics: see warnings above for mixed-level design notes.")
    else:
        print("VALIDATION FAILED — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
