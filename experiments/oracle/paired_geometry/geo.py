"""Deprecated shim.

The canonical implementation now lives in ``src/prometheus_simulation``, which
also owns the injection, the run, the readout and the physics record. This
module re-exports the geometry helpers so existing imports keep working.

    from prometheus_simulation.geometry import load_geo, recentre_delta, ...
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from prometheus_simulation.geometry import (  # noqa: E402,F401
    Geometry, load_geo, load_geometries, recentre_delta, injection_cylinder, survey,
    NUBENCH_GEOFILES, NUBENCH_MEDIA,
)


def shared_cylinder(a: Geometry, b: Geometry, radius_margin: float = 200.0,
                    height_margin: float = 200.0) -> dict:
    """Pairwise wrapper kept for the Tier-2 pair tables; prefer
    ``prometheus_simulation.geometry.injection_cylinder``."""
    return injection_cylinder({a.name: a, b.name: b}, mode="union",
                              margin_r=radius_margin, margin_h=height_margin)
