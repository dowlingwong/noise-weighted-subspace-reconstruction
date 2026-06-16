"""Minimal GWOSC smoke check.

This intentionally does not download data. It reports whether optional GWOSC
packages are installed and points users to the download/preprocess scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.gwosc import dependency_status


if __name__ == "__main__":
    status = dependency_status()
    status["downloads_attempted"] = False
    print(json.dumps(status, indent=2))
