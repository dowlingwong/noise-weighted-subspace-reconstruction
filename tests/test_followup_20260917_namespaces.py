"""Cross-package seed-namespace collision checks for the follow-up round.

Every new protocol must declare development/pilot/confirmation namespaces that
are disjoint from every older package's non-development namespaces and from each
other.  The historical packages share one development range (900000-900999); the
new packages must not reuse it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

FOLLOWUP = REPO / "experiments" / "mlst2026" / "followup_20260917"
ARCHIVED = REPO / "experiments" / "mlst2026" / "successor_20260913"
NAMESPACES = ("development", "pilot", "confirmation")


def _protocols(root: Path):
    out = {}
    for p in sorted(root.glob("*/PROTOCOL_*.json")):
        P = json.loads(p.read_text())
        out[P["package"]] = P["seed_namespaces"]
    return out


def _overlap(a, b) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def test_new_packages_are_pairwise_disjoint_on_every_namespace():
    new = _protocols(FOLLOWUP)
    assert len(new) >= 6, new
    names = sorted(new)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for ns in NAMESPACES:
                assert not _overlap(new[names[i]][ns], new[names[j]][ns]), \
                    f"{names[i]}.{ns} overlaps {names[j]}.{ns}"


def test_new_namespaces_avoid_archived_non_development_namespaces():
    new = _protocols(FOLLOWUP)
    old = _protocols(ARCHIVED)
    for npkg, nns in new.items():
        for pkg, ons in old.items():
            for ns in ("pilot", "confirmation"):
                assert not _overlap(nns[ns], ons[ns]), f"{npkg}.{ns} overlaps archived {pkg}.{ns}"


def test_new_development_ranges_avoid_the_shared_archived_development_range():
    new = _protocols(FOLLOWUP)
    archived_dev = [900000, 900999]
    for pkg, ns in new.items():
        assert not _overlap(ns["development"], archived_dev), f"{pkg}.development reuses the archived range"
