"""tidmad — TIDMAD rung-4 package (PLAN_04).

Self-contained: the reusable backbone lives in ``tidmad/backbone/``, vendored from
there and the Paper 1 noise/PSD/h5 machinery from ``noise_geometry`` (parent
repo ``src/``), which is placed on ``sys.path`` here when present.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # .../transformer/tidmad
_TRANSFORMER_ROOT = _THIS_DIR.parent  # .../transformer
_PARENT_REPO = _TRANSFORMER_ROOT.parent  # .../noise-weighted-subspace-reconstruction
for _candidate in (_PARENT_REPO / "src",):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

__version__ = "0.1.0"
