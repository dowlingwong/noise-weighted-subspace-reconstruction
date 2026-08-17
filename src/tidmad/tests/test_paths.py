"""Gates for the dataset-path resolver.

The failure this guards against is subtle and has already happened: a directory
exists at the expected path but does not hold the files the run needs, so the
run starts, reads nothing, and fails somewhere downstream where the cause is no
longer obvious. Existence of the directory is therefore never the test --
presence of the *contents* is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tidmad.paths import TidmadPathError, _resolve, resolve_raw_root


def _make(root: Path, name: str, files: tuple[str, ...] = ()) -> Path:
    d = root / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).touch()
    return d


def test_an_empty_directory_at_the_expected_path_is_skipped(tmp_path, monkeypatch):
    """The first candidate exists but is empty; the second holds the data."""
    monkeypatch.delenv("TIDMAD_DATA_ROOT", raising=False)
    _make(tmp_path, "planned")
    real = _make(tmp_path, "actual", ("abra_science_0000.h5", "abra_science_0001.h5"))

    candidates = (str(tmp_path / "planned"), str(tmp_path / "actual"))
    assert _resolve("TIDMAD_DATA_ROOT", candidates, "abra_science_*.h5", "x") == real


def test_candidate_order_is_respected(tmp_path, monkeypatch):
    monkeypatch.delenv("TIDMAD_DATA_ROOT", raising=False)
    first = _make(tmp_path, "first", ("abra_science_0000.h5",))
    _make(tmp_path, "second", ("abra_science_0000.h5",))

    candidates = (str(first), str(tmp_path / "second"))
    assert _resolve("TIDMAD_DATA_ROOT", candidates, "abra_science_*.h5", "x") == first


def test_env_override_wins_but_is_still_content_checked(tmp_path, monkeypatch):
    """An explicit path that holds nothing must fail loudly, not fall back.

    Falling back would silently read a different tree than the one the operator
    named, which is exactly how two rounds end up incomparable.
    """
    good = _make(tmp_path, "good", ("abra_science_0000.h5",))
    empty = _make(tmp_path, "empty")

    monkeypatch.setenv("TIDMAD_DATA_ROOT", str(good))
    assert resolve_raw_root() == good

    monkeypatch.setenv("TIDMAD_DATA_ROOT", str(empty))
    with pytest.raises(TidmadPathError, match="contains no"):
        resolve_raw_root()


def test_no_candidate_matches_reports_every_path_tried(tmp_path, monkeypatch):
    monkeypatch.delenv("TIDMAD_DATA_ROOT", raising=False)
    candidates = (str(tmp_path / "nope_a"), str(tmp_path / "nope_b"))
    with pytest.raises(TidmadPathError) as exc:
        _resolve("TIDMAD_DATA_ROOT", candidates, "abra_science_*.h5", "x")
    for candidate in candidates:
        assert candidate in str(exc.value)
