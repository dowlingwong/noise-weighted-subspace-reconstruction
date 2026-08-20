"""The Round 10 appendix table must fail closed on *any* altered input.

The generator prints numbers that a reader will treat as evidence, so every
archived file it reads has to be authenticated -- not just the two summaries.
The relative-degradation percentages are computed from E10b's per-seed rows, so
`raw.csv` is evidence in exactly the same sense as `summary.json`.

These tests never touch the real archive: each one copies the two archived
directories into a temporary tree and repoints the module at the copy.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "scripts" / "make_rps_round10_table.py"

E10B_NAME = "E10b_calibrated_estimated_covariance"
E11_NAME = "E11_corrected_angles"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_rps_round10_table", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sandboxed(tmp_path: Path):
    """A module instance pointed at a disposable copy of the archived inputs."""
    module = _load_module()
    results = REPO / "results" / "rps2026"
    for name in (E10B_NAME, E11_NAME):
        shutil.copytree(results / name, tmp_path / name)

    e10b, e11 = tmp_path / E10B_NAME, tmp_path / E11_NAME
    remap = {
        e10b / "summary.json": module.EXPECTED_DIGESTS[module.E10B_DIR / "summary.json"],
        e10b / "raw.csv": module.EXPECTED_DIGESTS[module.E10B_DIR / "raw.csv"],
        e11 / "summary.json": module.EXPECTED_DIGESTS[module.E11_DIR / "summary.json"],
    }
    module.E10B_DIR, module.E11_DIR = e10b, e11
    module.EXPECTED_DIGESTS = remap
    return module


def test_untampered_inputs_build_and_carry_the_archived_numbers(tmp_path):
    module = _sandboxed(tmp_path)
    tex = module.build()

    # The two figures the main text quotes, and the sub-degree E11 correction.
    assert "0.0078" in tex and "0.0345" in tex
    assert "-0.019122" in tex
    assert "62.89" in tex and "86.71" in tex
    assert "0.81" in tex and "0.73" in tex


def test_generation_is_deterministic(tmp_path):
    module = _sandboxed(tmp_path)
    assert module.build() == module.build()


def test_tampered_summary_is_refused(tmp_path):
    module = _sandboxed(tmp_path)
    path = module.E11_DIR / "summary.json"
    payload = json.loads(path.read_text())
    payload["experiment"] = "TAMPERED"
    path.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="digest"):
        module.build()


def test_tampered_raw_rows_are_refused(tmp_path):
    """The regression this test exists for.

    An earlier revision verified only the summaries. Editing a single
    denominator cell in `raw.csv` moved the printed MSE degradation from
    0.0345% to 0.0341% and the generator still succeeded, because the summary
    digests were untouched.
    """
    module = _sandboxed(tmp_path)
    path = module.E10B_DIR / "raw.csv"
    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    col = header.index("archived_oracle_test_raw_mse")
    cells = lines[1].split(",")
    cells[col] = str(float(cells[col]) * 1.10)
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(SystemExit, match="digest"):
        module.build()


def test_a_whitespace_only_edit_to_raw_is_refused(tmp_path):
    """Byte-level authentication, not value-level: no silent normalization."""
    module = _sandboxed(tmp_path)
    path = module.E10B_DIR / "raw.csv"
    path.write_text(path.read_text() + "\n")

    with pytest.raises(SystemExit, match="digest"):
        module.build()


def test_reading_an_unpinned_path_is_refused(tmp_path):
    """Adding an archived input without pinning it must not silently succeed."""
    module = _sandboxed(tmp_path)
    stray = tmp_path / "unpinned.json"
    stray.write_text("{}")

    with pytest.raises(SystemExit, match="not pinned"):
        module.read_verified(stray)


def test_the_live_archive_still_matches_every_pinned_digest():
    """Guards the pins themselves against drifting away from the archive."""
    module = _load_module()
    for path in module.EXPECTED_DIGESTS:
        assert path.exists(), path
        module.read_verified(path)
