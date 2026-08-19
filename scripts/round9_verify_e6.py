"""Round 9 / Phase 1: recover and verify the E6 whitening-range artifact.

``E6_whitening_range.json`` was produced by the Round 8 remote run and committed
on the unmerged ``main`` side branch (8ee9eb1). The Round 9 handoff commit
descends from a different parent, so the tracked destination is absent. This
script recovers the artifact from the Git object store -- which is a
content-addressed source, not a hand copy -- verifies every property the Round 9
runbook requires, and only then installs it.

Verified in code:

* ``excluding_dc.cv_inverse_J`` (not ``all_bands.cv_inverse_J``) rounds to 0.69;
* the stored verdict is ``collinearity_supported: false``;
* 328 bands are recorded before DC exclusion;
* the frozen PSD estimator is recorded (boxcar window, median average, 512
  calibration windows);
* the source and destination SHA-256 digests are recorded and equal.

Any mismatch exits nonzero with ``E6_VARIANT_MISMATCH`` and installs nothing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "8ee9eb1"
SOURCE_PATH = "results/rps2026/E6_whitening_range.json"
DEST = REPO / "results" / "rps2026" / "E6_whitening_range.json"
RECORD = REPO / "results" / "rps2026" / "Round9_remote" / "E6_verification.json"

EXPECTED_CV_ROUNDED = 0.69
EXPECTED_N_BANDS = 328
EXPECTED_WINDOW = "boxcar"
EXPECTED_AVERAGE = "median"
EXPECTED_CAL_WINDOWS = 512


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    blob = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{SOURCE_PATH}"],
        cwd=str(REPO), capture_output=True, check=True,
    ).stdout
    source_digest = sha256_bytes(blob)
    payload = json.loads(blob)

    failures: list[str] = []
    checks: dict[str, object] = {}

    cv_excl = payload.get("excluding_dc", {}).get("cv_inverse_J")
    checks["excluding_dc_cv_inverse_J"] = cv_excl
    checks["all_bands_cv_inverse_J"] = payload.get("all_bands", {}).get("cv_inverse_J")
    if cv_excl is None or round(float(cv_excl), 2) != EXPECTED_CV_ROUNDED:
        failures.append(f"excluding_dc.cv_inverse_J={cv_excl!r} does not round to {EXPECTED_CV_ROUNDED}")

    verdict_flag = payload.get("collinearity_supported")
    checks["collinearity_supported"] = verdict_flag
    if verdict_flag is not False:
        failures.append(f"collinearity_supported={verdict_flag!r}, expected False")

    n_bands = payload.get("all_bands", {}).get("n_bands")
    checks["n_bands_before_dc_exclusion"] = n_bands
    checks["n_bands_used"] = payload.get("n_bands_used")
    if n_bands != EXPECTED_N_BANDS:
        failures.append(f"all_bands.n_bands={n_bands!r}, expected {EXPECTED_N_BANDS}")

    estimator = {
        "psd_window": payload.get("psd_window"),
        "psd_average": payload.get("psd_average"),
        "n_calibration_windows": payload.get("n_calibration_windows"),
    }
    checks["estimator"] = estimator
    if estimator["psd_window"] != EXPECTED_WINDOW:
        failures.append(f"psd_window={estimator['psd_window']!r}, expected {EXPECTED_WINDOW!r}")
    if estimator["psd_average"] != EXPECTED_AVERAGE:
        failures.append(f"psd_average={estimator['psd_average']!r}, expected {EXPECTED_AVERAGE!r}")
    if estimator["n_calibration_windows"] != EXPECTED_CAL_WINDOWS:
        failures.append(
            f"n_calibration_windows={estimator['n_calibration_windows']!r}, expected {EXPECTED_CAL_WINDOWS}"
        )

    checks["data_dir"] = payload.get("data_dir")
    checks["source_commit"] = subprocess.run(
        ["git", "rev-parse", SOURCE_COMMIT], cwd=str(REPO), capture_output=True, text=True, check=True
    ).stdout.strip()
    checks["source_path"] = SOURCE_PATH
    checks["source_sha256"] = source_digest

    if failures:
        record = {"status": "E6_VARIANT_MISMATCH", "failures": failures, "checks": checks}
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print("E6_VARIANT_MISMATCH", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 2

    if DEST.exists():
        existing = DEST.read_bytes()
        if sha256_bytes(existing) != source_digest:
            print("E6_VARIANT_MISMATCH: destination exists with different content", file=sys.stderr)
            return 2
        installed = "already_present_identical"
    else:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_bytes(blob)
        installed = "installed_from_git_object_store"

    dest_digest = sha256_bytes(DEST.read_bytes())
    record = {
        "status": "E6_VERIFIED",
        "installed": installed,
        "destination": str(DEST.relative_to(REPO)),
        "destination_sha256": dest_digest,
        "digests_match": dest_digest == source_digest,
        "checks": checks,
        "note": (
            "Recovered from the unmerged Round 8 side branch 8ee9eb1 via the Git object "
            "store; not recomputed, because the Round 7 data root and frozen estimator "
            "were not re-derived in Round 9."
        ),
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
