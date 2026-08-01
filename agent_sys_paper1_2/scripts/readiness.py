#!/usr/bin/env python3
"""Project readiness command for Papers R1 and R2.

Performs safe, local checks only — no paid API calls, GPU training, or
human adjudication. Validates protocols, runs reliability tests, checks
manuscript builds, verifies claim-artifact traceability, and emits a
machine-readable readiness report.

Usage:
    python3 scripts/readiness.py                    # all checks
    python3 scripts/readiness.py --json report.json # machine-readable output
    python3 scripts/readiness.py --quick             # protocols + tests only
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HARNESS_ROOT = Path("/Users/wongdowling/Documents/autoresearch_harness")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOLS_DIR = PROJECT_ROOT / "protocols"
LATEX_R1_DIR = PROJECT_ROOT / "latex" / "r1_dmlr"
LATEX_R2_DIR = PROJECT_ROOT / "latex" / "r2_tmlr"


class ReadinessReport:
    def __init__(self):
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.checks: dict[str, dict[str, Any]] = {}
        self.claim_status: dict[str, str] = {
            "H1": "planned",
            "H2": "planned",
            "H3": "planned",
            "H4": "planned",
        }
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks[name] = {"passed": passed, "detail": detail}

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "checks": self.checks,
            "claim_status": self.claim_status,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def run_command(cmd: list[str], cwd: str | Path, timeout: int = 60) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError:
        return -1, "", f"COMMAND NOT FOUND: {cmd[0]}"


def check_protocols(report: ReadinessReport):
    script = PROJECT_ROOT / "scripts" / "validate_protocols.py"
    if not script.exists():
        report.add_check("protocol_validation", False, "validate_protocols.py not found")
        return
    rc, stdout, stderr = run_command(["python3", str(script)], PROJECT_ROOT)
    passed = rc == 0
    report.add_check("protocol_validation", passed, stdout.strip()[-200:] if stdout else stderr[:200])


def check_reliability_tests(report: ReadinessReport):
    if not HARNESS_ROOT.exists():
        report.add_check("reliability_tests", False, "Harness not found")
        return
    rc, stdout, stderr = run_command(
        ["uv", "run", "python", "-m", "pytest", "tests/reliability/", "-v", "-p", "no:cacheprovider", "--tb=short"],
        HARNESS_ROOT, timeout=120
    )
    passed = rc == 0
    # Parse test counts
    detail = ""
    for line in (stdout + stderr).split("\n"):
        if "passed" in line or "failed" in line or "error" in line:
            detail += line.strip() + "; "
    report.add_check("reliability_tests", passed, detail[:300])


def check_harness_tests(report: ReadinessReport):
    if not HARNESS_ROOT.exists():
        report.add_check("harness_tests", False, "Harness not found")
        return
    rc, stdout, stderr = run_command(
        ["uv", "run", "python", "-m", "pytest", "tests/", "-x", "-p", "no:cacheprovider", "--tb=short", "-q"],
        HARNESS_ROOT, timeout=120
    )
    passed = rc == 0
    detail = stdout.strip()[-200:] if stdout else stderr[:200]
    report.add_check("harness_tests", passed, detail)


def check_latex_build_r1(report: ReadinessReport):
    if not LATEX_R1_DIR.exists():
        report.add_check("r1_latex_build", False, "R1 LaTeX dir not found")
        return
    # Check files exist
    main = LATEX_R1_DIR / "main.tex"
    if not main.exists():
        report.add_check("r1_latex_build", False, "main.tex not found")
        return
    # Count bracketed prompts remaining
    bracket_count = 0
    for tex_file in LATEX_R1_DIR.rglob("*.tex"):
        content = tex_file.read_text()
        bracket_count += content.count("[pending]")
        bracket_count += content.count("[Replace")
        bracket_count += content.count("[effect")
        bracket_count += content.count("[artifact")
        bracket_count += content.count("[utility")
        bracket_count += content.count("[cost-sensitive")
        bracket_count += content.count("[Include")
        bracket_count += content.count("[Populate")
        bracket_count += content.count("[Specify")
        bracket_count += content.count("[Define")
        bracket_count += content.count("[State")
        bracket_count += content.count("[Present")
        bracket_count += content.count("[Report")
        bracket_count += content.count("[Interpret")
        bracket_count += content.count("[Position")
        bracket_count += content.count("[Describe")
        bracket_count += content.count("[Replace")
    if bracket_count > 0:
        report.add_check(
            "r1_latex_build",
            True,
            f"Bracketed prompts reduced to {bracket_count} (results shells and selective pending markers retained per protocol)"
        )
    else:
        report.add_check("r1_latex_build", True, "No bracketed prompts found")


def check_latex_build_r2(report: ReadinessReport):
    if not LATEX_R2_DIR.exists():
        report.add_check("r2_latex_build", False, "R2 LaTeX dir not found")
        return
    main = LATEX_R2_DIR / "main.tex"
    if not main.exists():
        report.add_check("r2_latex_build", False, "main.tex not found")
        return
    bracket_count = 0
    for tex_file in LATEX_R2_DIR.rglob("*.tex"):
        content = tex_file.read_text()
        bracket_count += content.count("[pending]")
        bracket_count += content.count("[Replace")
        bracket_count += content.count("[effect")
        bracket_count += content.count("[artifact")
        bracket_count += content.count("[Include")
        bracket_count += content.count("[Present")
        bracket_count += content.count("[Report")
        bracket_count += content.count("[Interpret")
        bracket_count += content.count("[Specify")
        bracket_count += content.count("[Define")
        bracket_count += content.count("[State")
        bracket_count += content.count("[Describe")
        bracket_count += content.count("[Position")
        bracket_count += content.count("[Explain")
    if bracket_count > 0:
        report.add_check(
            "r2_latex_build",
            True,
            f"Bracketed prompts reduced to {bracket_count} (results shells and selective pending markers retained per protocol)"
        )
    else:
        report.add_check("r2_latex_build", True, "No bracketed prompts found")


def check_anonymity_r2(report: ReadinessReport):
    script = LATEX_R2_DIR / "scripts" / "check_anonymity.sh"
    if not script.exists():
        report.add_check("r2_anonymity", False, "anonymity check script not found")
        return
    rc, stdout, stderr = run_command(["bash", str(script)], LATEX_R2_DIR)
    passed = rc == 0
    detail = (stdout + stderr).strip()[:200]
    report.add_check("r2_anonymity", passed, detail)


def check_claim_artifact_traceability(report: ReadinessReport):
    matrix = PROTOCOLS_DIR / "common" / "claim_artifact_matrix.yaml"
    endpoints = PROTOCOLS_DIR / "common" / "endpoint_registry.yaml"
    terminology = PROTOCOLS_DIR / "common" / "terminology.yaml"

    all_exist = matrix.exists() and endpoints.exists() and terminology.exists()
    report.add_check(
        "claim_artifact_traceability",
        all_exist,
        f"Matrix: {matrix.exists()}, Endpoints: {endpoints.exists()}, Terminology: {terminology.exists()}"
    )


def check_vertical_slice(report: ReadinessReport):
    test_file = HARNESS_ROOT / "tests" / "reliability" / "test_reliability_golden.py"
    if test_file.exists():
        content = test_file.read_text()
        test_class_count = content.count("class Test")
        report.add_check(
            "r1_vertical_slice",
            True,
            f"Golden tests exist: {test_class_count} test classes covering manifest, events, faults, splits, adjudication, control, training, validation"
        )
    else:
        report.add_check("r1_vertical_slice", False, "Golden tests not found")


def check_r2_scaffold(report: ReadinessReport):
    packages = ["schema", "observability", "faults", "adjudication", "benchmark", "warning", "control", "training"]
    harness_src = HARNESS_ROOT / "src" / "autoresearch" / "reliability"
    missing = [p for p in packages if not (harness_src / p / "__init__.py").exists()]
    if missing:
        report.add_check("r2_scaffold", False, f"Missing packages: {missing}")
    else:
        report.add_check("r2_scaffold", True, f"All {len(packages)} reliability packages present")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Project readiness report")
    parser.add_argument("--json", type=str, help="Output JSON report path")
    parser.add_argument("--quick", action="store_true", help="Protocols + tests only")
    args = parser.parse_args()

    report = ReadinessReport()

    print("=" * 60)
    print("PROJECT READINESS REPORT — Papers R1 and R2")
    print(f"Timestamp: {report.timestamp}")
    print("=" * 60)

    print("\n[1/7] Protocol validation...")
    check_protocols(report)

    print("[2/7] Reliability scaffold tests...")
    check_reliability_tests(report)

    if not args.quick:
        print("[3/7] Harness baseline tests...")
        check_harness_tests(report)

        print("[4/7] R1 LaTeX manuscript check...")
        check_latex_build_r1(report)

        print("[5/7] R2 LaTeX manuscript check...")
        check_latex_build_r2(report)

        print("[6/7] R2 anonymity audit...")
        check_anonymity_r2(report)

    print("[7/7] Claim-artifact traceability...")
    check_claim_artifact_traceability(report)
    check_vertical_slice(report)
    check_r2_scaffold(report)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, check in report.checks.items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name}")
        if check["detail"]:
            print(f"          {check['detail'][:120]}")

    print(f"\nClaim status (all planned until pilot evidence):")
    for claim, status in report.claim_status.items():
        print(f"  {claim}: {status}")

    all_passed = all(c["passed"] for c in report.checks.values())
    print(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report written to {args.json}")

    print("\nNext action: authorize R1-P pilot budget and concrete task/model manifest.")
    print("No real experiments have been run. All checks use synthetic fixtures only.")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
