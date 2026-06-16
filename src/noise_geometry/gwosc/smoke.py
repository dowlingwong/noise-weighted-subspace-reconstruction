"""Minimal GWOSC dependency checks."""

from __future__ import annotations

import importlib.util


def dependency_status() -> dict[str, bool]:
    """Report whether optional GWOSC analysis packages are importable."""
    return {
        "gwosc": importlib.util.find_spec("gwosc") is not None,
        "gwpy": importlib.util.find_spec("gwpy") is not None,
    }
