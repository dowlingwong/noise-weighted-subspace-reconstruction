"""Shared pytest setup for the modular noise package."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# The test modules still import the package under its legacy name
# ``wk7.noise_module`` (an earlier repo layout). Alias it to the current
# location so the suite runs from this repository.
import sys as _sys
import types as _types
from pathlib import Path as _Path

_QP_DIR = _Path(__file__).resolve().parents[2]
if str(_QP_DIR) not in _sys.path:
    _sys.path.insert(0, str(_QP_DIR))
import noise_module as _nm  # noqa: E402

_wk7 = _types.ModuleType("wk7")
_wk7.noise_module = _nm
_sys.modules.setdefault("wk7", _wk7)
_sys.modules.setdefault("wk7.noise_module", _nm)
for _sub in ("NoiseGenerator", "temporal_noise", "artifact_injector",
             "multichannel_noise", "templates", "utils"):
    try:
        _mod = __import__(f"noise_module.{_sub}", fromlist=[_sub])
        _sys.modules.setdefault(f"wk7.noise_module.{_sub}", _mod)
    except ImportError:
        pass
