"""Shared pytest setup for the standalone modular-noise distribution."""

from __future__ import annotations

import sys
from pathlib import Path


DISTRIBUTION_ROOT = Path(__file__).resolve().parents[2]
if str(DISTRIBUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(DISTRIBUTION_ROOT))

QP_SIMULATOR_ROOT = Path(__file__).resolve().parents[3] / "QP_simulator"
if str(QP_SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(QP_SIMULATOR_ROOT))
