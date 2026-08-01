#!/usr/bin/env python3
"""Validate a grounded-research-ideation candidate and kill-switch integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_TOP = {
    "version",
    "direction",
    "evidence_status",
    "assumptions",
    "bottleneck",
    "pattern_composition",
    "title",
    "thesis",
    "motivation",
    "mechanism",
    "core_claim",
    "differentiation",
    "compute_budget",
    "falsification",
    "feasibility",
    "audit",
    "provenance",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be an object")
    return data


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def require_keys(errors: list[str], obj: Any, keys: set[str], label: str) -> None:
    if not isinstance(obj, dict):
        errors.append(f"{label} must be an object")
        return
    missing = sorted(keys - set(obj))
    if missing:
        errors.append(f"{label} missing keys: {', '.join(missing)}")


def validate(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    require_keys(errors, data, REQUIRED_TOP, "candidate")

    for key in ("direction", "title", "thesis", "motivation", "core_claim", "compute_budget"):
        if key in data and not nonempty_string(data[key]):
            errors.append(f"{key} must be a non-empty string")

    if data.get("evidence_status") not in {"grounded", "partial"}:
        errors.append("evidence_status must be 'grounded' or 'partial'")

    bottleneck = data.get("bottleneck")
    require_keys(errors, bottleneck, {"statement", "why_load_bearing", "lineage", "closest_work"}, "bottleneck")
    if isinstance(bottleneck, dict):
        if not nonempty_string(bottleneck.get("statement")):
            errors.append("bottleneck.statement must be non-empty")
        closest = bottleneck.get("closest_work")
        if not isinstance(closest, list) or not closest:
            errors.append("bottleneck.closest_work must contain at least one verified work")
        else:
            for index, work in enumerate(closest):
                require_keys(errors, work, {"work", "result", "remaining_gap", "source"}, f"bottleneck.closest_work[{index}]")

    patterns = data.get("pattern_composition")
    if not isinstance(patterns, list) or not 1 <= len(patterns) <= 3:
        errors.append("pattern_composition must contain 1 to 3 entries")
    else:
        pattern_keys = {"pattern", "role", "trigger_fit", "output_artifact", "guarded_failure"}
        for index, pattern in enumerate(patterns):
            require_keys(errors, pattern, pattern_keys, f"pattern_composition[{index}]")

    mechanism = data.get("mechanism")
    if not isinstance(mechanism, list) or not mechanism:
        errors.append("mechanism must contain at least one concrete step")
    else:
        step_keys = {"step", "name", "inputs", "operation", "outputs", "why"}
        for index, step in enumerate(mechanism):
            require_keys(errors, step, step_keys, f"mechanism[{index}]")

    differentiation = data.get("differentiation")
    if not isinstance(differentiation, list) or not differentiation:
        errors.append("differentiation must contain at least one prior-work comparison")
    else:
        diff_keys = {"prior_work", "shared", "delta", "source"}
        for index, item in enumerate(differentiation):
            require_keys(errors, item, diff_keys, f"differentiation[{index}]")
            if isinstance(item, dict) and not nonempty_string(item.get("source")):
                errors.append(f"differentiation[{index}].source must be non-empty")

    falsification = data.get("falsification")
    falsification_keys = {"load_bearing_variable", "prediction", "negative_control", "failure_condition"}
    require_keys(errors, falsification, falsification_keys, "falsification")
    if isinstance(falsification, dict):
        for key in falsification_keys:
            if not nonempty_string(falsification.get(key)):
                errors.append(f"falsification.{key} must be non-empty")

    feasibility = data.get("feasibility")
    require_keys(errors, feasibility, {"verdict", "requirements", "unknowns"}, "feasibility")

    audit = data.get("audit")
    require_keys(errors, audit, {"verdict", "checks", "threats", "revision_targets"}, "audit")
    if isinstance(audit, dict) and audit.get("verdict") not in {"advance", "revise", "abandon"}:
        errors.append("audit.verdict must be 'advance', 'revise', or 'abandon'")

    provenance = data.get("provenance")
    provenance_keys = {"retrieved_sources", "user_sources", "awareness_only", "retrieval_date"}
    require_keys(errors, provenance, provenance_keys, "provenance")
    if isinstance(provenance, dict) and not (
        provenance.get("retrieved_sources") or provenance.get("user_sources")
    ):
        errors.append("provenance must include at least one retrieved or user-provided source")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Final canonical candidate JSON")
    parser.add_argument("--initial", type=Path, help="Initial candidate JSON for kill-switch comparison")
    args = parser.parse_args()

    try:
        final = load_json(args.candidate)
        errors = validate(final)
        if args.initial:
            initial = load_json(args.initial)
            for key in ("compute_budget", "falsification"):
                if initial.get(key) != final.get(key):
                    errors.append(f"kill-switch field changed: {key}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {args.candidate} is structurally valid")
    if args.initial:
        print("OK: kill-switch commitments are preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
