#!/usr/bin/env python3
"""Project readiness command for Papers R1 and R2 (Phase 7).

Performs safe, local checks only — no paid API calls, GPU training, or
human adjudication. The readiness report classifies every H1-H4 claim as
planned, instrumented, pilot-ready, pilot-observed, confirmatory-observed,
failed, or blocked. Only real, audited evidence may advance a claim beyond
pilot-ready.

Checks performed:
  1. Validate common, R1, and R2 protocols.
  2. Regenerate deterministic example schedules and compare their hashes.
  3. Run targeted reliability tests in the harness.
  4. Run the synthetic R1 vertical slice (and report R2 slice status).
  5. Verify claim-artifact traceability and missing generated inputs.
  6. Build both manuscripts against their pinned official templates.
  7. Run the TMLR anonymity audit.
  8. Emit a machine-readable and human-readable readiness report.

Usage:
    python3 scripts/readiness.py                     # all checks
    python3 scripts/readiness.py --json report.json  # machine-readable output
    python3 scripts/readiness.py --quick             # protocols + tests + hashes only
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HARNESS_ROOT = Path("/Users/wongdowling/Documents/autoresearch_harness")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOLS_DIR = PROJECT_ROOT / "protocols"
LATEX_R1_DIR = PROJECT_ROOT / "latex" / "r1_dmlr"
LATEX_R2_DIR = PROJECT_ROOT / "latex" / "r2_tmlr"
GOLDEN_DESIGNS = PROTOCOLS_DIR / "designs" / "golden_outputs.yaml"

CLAIM_STATUS_LABELS = [
    "planned", "instrumented", "pilot-ready", "pilot-observed",
    "confirmatory-observed", "failed", "blocked",
]


class ReadinessReport:
    def __init__(self):
        self.timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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

    def advance_claim(self, claim: str, status: str):
        if status not in CLAIM_STATUS_LABELS:
            raise ValueError(f"invalid claim status: {status}")
        current = self.claim_status[claim]
        if status == "planned":
            return
        if status == "instrumented" and current == "planned":
            self.claim_status[claim] = "instrumented"
        elif status in ("pilot-ready", "pilot-observed", "confirmatory-observed",
                        "failed", "blocked"):
            self.claim_status[claim] = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "checks": self.checks,
            "claim_status": self.claim_status,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def run_command(cmd: list[str], cwd: str | Path, timeout: int = 120) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError:
        return -1, "", f"COMMAND NOT FOUND: {cmd[0]}"


# --- Check 1: protocol validation -------------------------------------------


def check_protocols(report: ReadinessReport):
    script = PROJECT_ROOT / "scripts" / "validate_protocols.py"
    rc, stdout, stderr = run_command(["python3", str(script)], PROJECT_ROOT)
    passed = rc == 0
    detail = stdout.strip().splitlines()
    summary = detail[-1] if detail else (stderr[:200] or "no output")
    report.add_check("protocol_validation", passed, summary)


# --- Check 2: deterministic schedule regeneration + hash comparison ---------


def check_design_regeneration(report: ReadinessReport):
    script = PROJECT_ROOT / "scripts" / "design_generate.py"
    if not GOLDEN_DESIGNS.exists():
        report.add_check("design_regeneration", False, "golden_outputs.yaml not found")
        return

    golden = yaml.safe_load(GOLDEN_DESIGNS.read_text())
    synth_golden = golden["synthetic"]["output_hash"]

    rc, stdout, stderr = run_command(
        ["python3", str(script), "--synthetic"], PROJECT_ROOT
    )
    if rc != 0:
        report.add_check("design_regeneration", False, stderr[:200])
        return

    for line in stdout.splitlines():
        if line.startswith("output_hash:"):
            actual = line.split(":", 1)[1].strip()
            break
    else:
        report.add_check("design_regeneration", False, "no output_hash emitted")
        return

    match = actual == synth_golden
    report.add_check(
        "design_regeneration",
        match,
        f"golden={synth_golden} regenerated={actual}",
    )
    if not match:
        report.errors.append(
            "synthetic design hash drifted from golden_outputs.yaml; "
            "review the materialization change and update golden with rationale"
        )


# --- Check 3: targeted reliability tests ------------------------------------


def check_reliability_tests(report: ReadinessReport):
    rc, stdout, stderr = run_command(
        [
            "uv", "run", "python", "-m", "pytest", "tests/reliability/",
            "-q", "-p", "no:cacheprovider",
        ],
        HARNESS_ROOT,
        timeout=180,
    )
    passed = rc == 0
    last_line = (stdout.strip().splitlines() or ["no output"])[-1]
    report.add_check("reliability_tests", passed, last_line)


# --- Check 4: synthetic vertical slices -------------------------------------


def check_synthetic_r1_slice(report: ReadinessReport):
    script = HARNESS_ROOT / "scripts" / "run_r1_synthetic_readiness.py"
    if not script.exists():
        report.add_check("synthetic_r1_slice", False, "script not found")
        return
    rc, stdout, stderr = run_command(
        ["uv", "run", "python3", str(script), "--out", "/tmp/r1_slice_check"],
        HARNESS_ROOT,
        timeout=120,
    )
    passed = rc == 0
    detail = stdout.strip().splitlines()[-1] if stdout.strip() else stderr[:200]
    report.add_check("synthetic_r1_slice", passed, detail)
    if passed:
        report.advance_claim("H1", "instrumented")
        report.advance_claim("H2", "instrumented")


def check_synthetic_r2_slice(report: ReadinessReport):
    # The R2 controller/training-view path is exercised by the reliability
    # test suite (Phase 5); a dedicated R2 synthetic slice runner is planned
    # once intervention data exists. Report the test-level status honestly.
    rc, stdout, stderr = run_command(
        [
            "uv", "run", "python", "-m", "pytest",
            "tests/reliability/test_reliability_vertical_slice.py",
            "-q", "-p", "no:cacheprovider",
        ],
        HARNESS_ROOT,
        timeout=120,
    )
    passed = rc == 0
    last_line = (stdout.strip().splitlines() or ["no output"])[-1]
    report.add_check(
        "synthetic_r2_slice",
        passed,
        f"controller/training-view interface tests: {last_line}",
    )
    if passed:
        report.advance_claim("H3", "instrumented")
        report.advance_claim("H4", "instrumented")


# --- Check 5: claim-artifact traceability ------------------------------------


def check_traceability(report: ReadinessReport):
    matrix = PROTOCOLS_DIR / "common" / "claim_artifact_matrix.yaml"
    endpoints = PROTOCOLS_DIR / "common" / "endpoint_registry.yaml"
    terminology = PROTOCOLS_DIR / "common" / "terminology.yaml"

    missing = [str(p) for p in (matrix, endpoints, terminology) if not p.exists()]
    report.add_check("claim_artifact_traceability", not missing, f"missing: {missing}" if missing else "matrix, endpoints, terminology present")

    # Verify every claim maps to a protocol analysis ID in the endpoint registry
    if matrix.exists() and endpoints.exists():
        claims = yaml.safe_load(matrix.read_text()).get("claims", {})
        endpoints_data = yaml.safe_load(endpoints.read_text())
        endpoint_ids = set()
        analysis_codes = set()
        for group in endpoints_data.get("endpoints", {}).values():
            if isinstance(group, list):
                for ep in group:
                    if isinstance(ep, dict) and ep.get("id"):
                        endpoint_ids.add(ep["id"])
                    if isinstance(ep, dict) and ep.get("analysis"):
                        analysis_codes.add(ep["analysis"])
        unmapped = []
        for claim_id, claim in claims.items():
            analysis_id = claim.get("analysis_id", "")
            if not analysis_id:
                unmapped.append(f"{claim_id}->(missing)")
                continue
            # The claim matrix references analysis IDs (R1-E2-primary); the
            # endpoint registry stores full endpoint IDs and analysis codes
            # (R1-E2-primary-warning-lead-time / R1-E2). Accept a match on
            # either the analysis code prefix or a registered endpoint id.
            base_analysis = analysis_id.rsplit("-", 1)[0]
            matched = (
                analysis_id in endpoint_ids
                or base_analysis in analysis_codes
                or any(eid.startswith(analysis_id) for eid in endpoint_ids)
            )
            if not matched:
                unmapped.append(f"{claim_id}->{analysis_id}")
        if unmapped:
            report.add_check(
                "claim_analysis_mapping", False, f"unmapped: {unmapped}"
            )
        else:
            report.add_check(
                "claim_analysis_mapping", True,
                "all H1-H4 analysis IDs resolve to registered endpoints",
            )


# --- Check 6: manuscript builds ---------------------------------------------


def check_latex_build(report: ReadinessReport, name: str, latex_dir: Path):
    rc, stdout, stderr = run_command(["make", "pdf"], latex_dir, timeout=300)
    log_path = latex_dir / "main.log"
    undefined = 0
    overfull = 0
    if log_path.exists():
        text = log_path.read_text()
        undefined = sum(1 for _ in text.splitlines() if "undefined" in _)
        overfull = sum(1 for _ in text.splitlines() if "Overfull" in _)

    pdf_exists = (latex_dir / "main.pdf").exists()
    passed = rc == 0 and pdf_exists and undefined == 0
    detail = (
        f"undefined={undefined} overfull={overfull} pdf={pdf_exists}"
    )
    report.add_check(f"{name}_latex_build", passed, detail)
    if not passed:
        report.errors.append(f"{name} LaTeX build failed or has undefined refs")


# --- Check 7: TMLR anonymity audit ------------------------------------------


def check_anonymity_r2(report: ReadinessReport):
    script = LATEX_R2_DIR / "scripts" / "check_anonymity.sh"
    rc, stdout, stderr = run_command(["bash", str(script)], LATEX_R2_DIR)
    passed = rc == 0
    detail = (stdout.strip() or "no output")
    report.add_check("r2_anonymity", passed, detail)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Project readiness report")
    parser.add_argument("--json", type=str, help="Output JSON report path")
    parser.add_argument("--quick", action="store_true", help="checks 1-5 only")
    args = parser.parse_args()

    report = ReadinessReport()

    print("=" * 60)
    print("PROJECT READINESS REPORT — Papers R1 and R2")
    print(f"Timestamp: {report.timestamp}")
    print("=" * 60)

    print("\n[1/8] Protocol validation...")
    check_protocols(report)

    print("[2/8] Design schedule regeneration + hash comparison...")
    check_design_regeneration(report)

    print("[3/8] Reliability scaffold tests...")
    check_reliability_tests(report)

    print("[4/8] Synthetic vertical slices...")
    check_synthetic_r1_slice(report)
    check_synthetic_r2_slice(report)

    print("[5/8] Claim-artifact traceability...")
    check_traceability(report)

    if not args.quick:
        print("[6/8] Manuscript builds (R1 DMLR, R2 TMLR)...")
        check_latex_build(report, "r1", LATEX_R1_DIR)
        check_latex_build(report, "r2", LATEX_R2_DIR)

        print("[7/8] R2 anonymity audit...")
        check_anonymity_r2(report)

    print("[8/8] Claim status classification...")
    for claim, status in report.claim_status.items():
        print(f"  {claim}: {status}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, check in report.checks.items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name}: {check['detail'][:110]}")

    if report.errors:
        print("\nErrors:")
        for err in report.errors:
            print(f"  - {err}")

    all_passed = all(c["passed"] for c in report.checks.values())
    print(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Report written to {args.json}")

    print("\nNext action: authorize R1-P pilot budget and concrete task/model manifest.")
    print("No real experiments have been run. All checks use synthetic fixtures only.")
    print("Claims advanced to 'instrumented' reflect tested interfaces, not evidence.")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
