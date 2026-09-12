"""Run matrices: every cell the package will execute, enumerated before execution.

A *cell* is one unit of work with a stable ``cell_id``.  The matrix records,
per cell, its role (``pilot``, ``confirmation``, ``control``, ``algebraic``,
``development``), its seed, its arm parameters and the number of estimator
fits and algebraic solves it will charge.  Fit counts are therefore derived
from the matrix, never from the number of seeds.

    fits  = estimator invocations on drawn data (charged against the cap)
    solves = eigen/least-squares solves that fit nothing (engineered
             perturbations, analytic checks); reported, costed, not capped
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .protocol import canonical_json

ROLES = ("development", "pilot", "confirmation", "control", "algebraic")


def cell_id(payload: dict) -> str:
    """Deterministic id from the cell's defining parameters (not from its role
    label, so a role relabel would be visible as the same id with a different
    role)."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def make_cell(*, role: str, seed: int | None, params: dict, n_fits: int, n_solves: int = 0,
              family: str | None = None, notes: str | None = None) -> dict:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; allowed {ROLES}")
    if n_fits < 0 or n_solves < 0:
        raise ValueError("fit and solve counts must be non-negative")
    defining = {"seed": seed, "params": params, "family": family}
    c = {"cell_id": cell_id(defining), "role": role, "seed": seed, "family": family,
         "params": params, "n_fits": int(n_fits), "n_solves": int(n_solves)}
    if notes:
        c["notes"] = notes
    return c


class RunMatrix:
    def __init__(self, package: str, protocol_sha256: str, cells: list[dict]):
        self.package = package
        self.protocol_sha256 = protocol_sha256
        ids = [c["cell_id"] for c in cells]
        dup = [k for k, v in Counter(ids).items() if v > 1]
        if dup:
            raise ValueError(f"duplicate cell ids in matrix: {dup[:5]}")
        self.cells = sorted(cells, key=lambda c: (ROLES.index(c["role"]), c["family"] or "", c["seed"] if c["seed"] is not None else -1, c["cell_id"]))

    # -- accounting ---------------------------------------------------------
    def counts(self) -> dict:
        by_role = {}
        for role in ROLES:
            cs = [c for c in self.cells if c["role"] == role]
            if cs:
                by_role[role] = {"cells": len(cs), "fits": sum(c["n_fits"] for c in cs),
                                 "solves": sum(c["n_solves"] for c in cs),
                                 "seeds": sorted({c["seed"] for c in cs if c["seed"] is not None})}
        by_family = {}
        for fam in sorted({c["family"] or "-" for c in self.cells}):
            cs = [c for c in self.cells if (c["family"] or "-") == fam]
            by_family[fam] = {"cells": len(cs), "fits": sum(c["n_fits"] for c in cs), "solves": sum(c["n_solves"] for c in cs)}
        return {"cells": len(self.cells), "fits": sum(c["n_fits"] for c in self.cells),
                "solves": sum(c["n_solves"] for c in self.cells), "by_role": by_role, "by_family": by_family}

    def matrix_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.cells).encode("utf-8")).hexdigest()

    def subset(self, *, roles=None, cell_ids=None, families=None) -> "RunMatrix":
        cs = self.cells
        if roles:
            cs = [c for c in cs if c["role"] in set(roles)]
        if families:
            cs = [c for c in cs if c["family"] in set(families)]
        if cell_ids:
            wanted = set(cell_ids)
            unknown = wanted - {c["cell_id"] for c in self.cells}
            if unknown:
                raise ValueError(f"unknown cell ids requested: {sorted(unknown)[:5]}")
            cs = [c for c in cs if c["cell_id"] in wanted]
        return RunMatrix(self.package, self.protocol_sha256, list(cs))

    # -- io ---------------------------------------------------------------------
    def write(self, out_dir: Path, *, extra: dict | None = None) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_matrix.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in self.cells))
        summary = {"package": self.package, "protocol_sha256": self.protocol_sha256,
                   "matrix_sha256": self.matrix_hash(), **(extra or {}), **self.counts()}
        (out_dir / "run_matrix_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        return summary

    @classmethod
    def read(cls, out_dir: Path) -> "RunMatrix":
        out_dir = Path(out_dir)
        cells = [json.loads(l) for l in (out_dir / "run_matrix.jsonl").read_text().splitlines() if l.strip()]
        summ = json.loads((out_dir / "run_matrix_summary.json").read_text())
        m = cls(summ["package"], summ["protocol_sha256"], cells)
        if m.matrix_hash() != summ["matrix_sha256"]:
            raise ValueError("run_matrix.jsonl does not match run_matrix_summary.json (matrix hash differs)")
        return m


def check_seed_roles(matrix: RunMatrix, P: dict) -> list[str]:
    """Every seeded cell's seed must lie in the namespace of its role; the
    development namespace is the only one a fixture run may use."""
    from .protocol import namespace_of_seed
    problems = []
    for c in matrix.cells:
        if c["seed"] is None:
            continue
        ns = namespace_of_seed(P, c["seed"])
        role = c["role"]
        expected = {"pilot": "pilot", "confirmation": "confirmation", "development": "development"}.get(role)
        if expected is None:
            # controls and algebraic cells may live in pilot or confirmation namespaces but never in development
            if ns == "development":
                problems.append(f"{c['cell_id']}: role {role} uses a development seed {c['seed']}")
            continue
        if ns != expected:
            problems.append(f"{c['cell_id']}: role {role} but seed {c['seed']} is in namespace {ns!r}")
    return problems
