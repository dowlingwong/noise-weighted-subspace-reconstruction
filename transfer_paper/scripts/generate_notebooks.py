"""Generate dependency-light Jupyter notebooks for the Paper 1 handoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFER_ROOT = SCRIPT_DIR.parent
NOTEBOOK_ROOT = TRANSFER_ROOT / "notebooks"


def markdown(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


SETUP = """
from pathlib import Path
import sys

TRANSFER_ROOT = Path.cwd()
if TRANSFER_ROOT.name == "notebooks":
    TRANSFER_ROOT = TRANSFER_ROOT.parent
elif (TRANSFER_ROOT / "transfer_paper").is_dir():
    TRANSFER_ROOT = TRANSFER_ROOT / "transfer_paper"

sys.path.insert(0, str(TRANSFER_ROOT / "scripts"))
from paper_support import *
"""


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write(name: str, cells: list[dict[str, Any]]) -> None:
    path = NOTEBOOK_ROOT / name
    path.write_text(
        json.dumps(notebook(cells), indent=1) + "\n",
        encoding="utf-8",
    )
    print(path)


def main() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    _write(
        "00_evidence_inventory.ipynb",
        [
            markdown(
                """
# Evidence inventory and claim status

This notebook is the entry point for the paper-writing agent. It verifies the
transferred file manifest, displays the archived run history, and separates
verified claims from implemented-but-pending experiments.

Before drafting Results, also read `PENDING_RESULT_PLACEHOLDERS.md`. Pending,
absent, diagnostic-only, or not-solidly-passed experiments must appear as
explicit placeholders in working drafts.
"""
            ),
            code(SETUP),
            code(
                """
import hashlib
import json
import pandas as pd

manifest = json.loads(
    (TRANSFER_ROOT / "data/transfer_manifest.json").read_text()
)
checks = []
for item in manifest["files"]:
    path = TRANSFER_ROOT / item["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checks.append({
        "path": item["path"],
        "checksum_ok": digest == item["sha256"],
        "source": item["source"],
    })
check_table = pd.DataFrame(checks)
assert check_table["checksum_ok"].all()
check_table
"""
            ),
            code(
                """
pd.read_csv(TRANSFER_ROOT / "data/derived/gwosc_run_history.csv")
"""
            ),
            code(
                """
pd.read_csv(TRANSFER_ROOT / "data/derived/claim_status.csv")
"""
            ),
            code(
                """
pd.read_csv(TRANSFER_ROOT / "data/derived/paper_implications.csv")
"""
            ),
            code(
                """
pd.read_csv(TRANSFER_ROOT / "data/derived/method_traceability.csv")
"""
            ),
            code(
                """
fig = plot_gwosc_run_history()
save_figure(fig, "gwosc_run_history")
fig
"""
            ),
            code(
                """
fig = plot_paper_claim_support_matrix()
save_figure(fig, "paper_claim_support_matrix")
fig
"""
            ),
            markdown(
                """
The claim-status table is a writing constraint. A pending or negative result
must not be rewritten as a positive result. Consult
`PAPER_WRITING_HANDOFF.md`, `WRITING_AGENT_BRIEF.md`, and
`MANUSCRIPT_EVIDENCE_MAP.md` before drafting Results or Discussion text. Use
`PENDING_RESULT_PLACEHOLDERS.md` wherever a missing or not-solidly-passed
experiment would otherwise be described as a result.
"""
            ),
        ],
    )
    _write(
        "01_synthetic_validation.ipynb",
        [
            markdown(
                """
# Synthetic validation ladder

This notebook summarizes the interval-backed S1–S9 synthetic validation suite.
The four-panel overview shows representative normalization, subspace,
metric-reversal, and residual-calibration gates. The complete numerical table
below remains the source for exact values and confidence intervals.
"""
            ),
            code(SETUP),
            code(
                """
import pandas as pd
summary = pd.read_csv(
    TRANSFER_ROOT / "data/derived/synthetic_gate_summary.csv"
)
summary
"""
            ),
            code(
                """
fig = plot_synthetic_validation_overview()
save_figure(fig, "synthetic_validation_overview")
fig
"""
            ),
            markdown(
                """
These controlled results support mathematical and implementation claims. They
must not be used as evidence that the real GWOSC null statistic is calibrated.
"""
            ),
        ],
    )
    _write(
        "02_gwosc_baseline_validation.ipynb",
        [
            markdown(
                """
# GWOSC baseline validation

This notebook analyzes the latest archived enhanced GWOSC run. It shows the
failed random and chronological held-out calibration gate, exact PSD agreement
with GWpy, and the earlier diagnostic comparison of non-identical score paths.
"""
            ),
            code(SETUP),
            code(
                """
import pandas as pd
primary = pd.read_csv(
    TRANSFER_ROOT / "data/derived/gwosc_primary_results.csv"
)
quality = pd.read_csv(
    TRANSFER_ROOT / "data/derived/gwosc_data_quality.csv"
)
reference = pd.read_csv(
    TRANSFER_ROOT / "data/derived/gwosc_reference_summary.csv"
)
display(primary)
display(quality)
display(reference)
"""
            ),
            code(
                """
fig = plot_gwosc_null_calibration()
save_figure(fig, "gwosc_null_calibration")
fig
"""
            ),
            code(
                """
fig = plot_gwosc_reference_comparison()
save_figure(fig, "gwosc_reference_comparison")
fig
"""
            ),
            markdown(
                """
Interpretation: the PSD estimator is independently verified, but the global
PSD does not predict the held-out score spread. The event and injection values
are therefore uncalibrated diagnostics and cannot support significance or
sensitivity claims.
"""
            ),
        ],
    )
    _write(
        "03_gwosc_filter_equivalence.ipynb",
        [
            markdown(
                """
# GWOSC filtering/statistic equivalence

This notebook consumes the predeclared shared-FIR experiment. It remains in a
pending state until `filter_equivalence.json` is synchronized from a controlled
remote evidence run.
"""
            ),
            code(SETUP),
            code(
                """
record = followup_record("filter_equivalence.json")
if record is None:
    print("PENDING: no archived remote filter-equivalence evidence")
else:
    print(record.get("status"))
    display(record["metrics"]["acceptance"])
"""
            ),
            code(
                """
fig = plot_filter_equivalence()
if fig is not None:
    save_figure(fig, "gwosc_filter_equivalence")
fig
"""
            ),
            markdown(
                """
The identity-error panels answer whether the explicit FFT-convolution and GWpy
paths compute the same predeclared statistic. The GLS/FIR correlation panels
answer a separate methodological question and must not be used as the software
identity acceptance criterion.
"""
            ),
        ],
    )
    _write(
        "04_gwosc_time_local_psd.ipynb",
        [
            markdown(
                """
# GWOSC time-local PSD modelling

This notebook consumes the predeclared global-versus-local PSD experiment. The
64-second radius is primary; 32 and 96 seconds are sensitivity analyses. The
notebook remains pending until `time_local_noise.json` is synchronized.
"""
            ),
            code(SETUP),
            code(
                """
record = followup_record("time_local_noise.json")
if record is None:
    print("PENDING: no archived remote time-local-noise evidence")
else:
    print(record.get("status"))
    display(record["metrics"]["acceptance"])
"""
            ),
            code(
                """
fig = plot_time_local_psd()
if fig is not None:
    save_figure(fig, "gwosc_time_local_psd")
fig
"""
            ),
            markdown(
                """
First verify that the stationary synthetic control passes and that the primary
model covers at least 90% of windows. Only then compare H1/L1 score spread and
chronological stability. Template-projected and narrow-band diagnostics are
explanatory; they cannot retroactively define new cuts in the same run.
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()
