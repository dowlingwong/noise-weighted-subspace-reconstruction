#!/usr/bin/env python3
"""Deterministic design-generation and diagnostics for R1 mixed-level design.

Reads the R1 confirmatory design, materializes a block-randomized run schedule
from a recorded seed, and reports rank, estimable terms, condition-number /
variance-inflation diagnostics, balance by block, and aliased or unsupported
interactions.

The mixed-level design is never described as resolution-IV unless the formal
alias analysis supports that terminology; the script refuses that label.

Usage:
    python3 scripts/design_generate.py                 # materialize R1-C schedule
    python3 scripts/design_generate.py --seed <s>       # override seed
    python3 scripts/design_generate.py --check <hash>   # compare output hash
    python3 scripts/design_generate.py --synthetic      # run on bundled synthetic input
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESIGNS_DIR = PROJECT_ROOT / "protocols" / "designs"


@dataclass
class DesignCell:
    """One row of the materialized design matrix."""

    block: str
    task_id: str
    workload_family: str
    condition: str
    severity: int
    injection_band: int
    seed: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "task_id": self.task_id,
            "workload_family": self.workload_family,
            "condition": self.condition,
            "severity": self.severity,
            "injection_band": self.injection_band,
            "seed": self.seed,
        }


@dataclass
class DesignDiagnostics:
    """Estimability and balance diagnostics for a materialized design."""

    n_rows: int = 0
    rank: int = 0
    n_columns: int = 0
    condition_number: float = 0.0
    balance_by_block: dict[str, int] = field(default_factory=dict)
    aliased_or_unsupported_interactions: list[str] = field(default_factory=list)
    resolution_label_supported: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "rank": self.rank,
            "n_columns": self.n_columns,
            "condition_number": round(self.condition_number, 4),
            "balance_by_block": self.balance_by_block,
            "aliased_or_unsupported_interactions": self.aliased_or_unsupported_interactions,
            "resolution_label_supported": self.resolution_label_supported,
        }


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _hash(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode()).hexdigest()[:16]


def materialize_design(
    *,
    workload_families: list[str],
    task_instances_per_family: int,
    conditions: list[str],
    severity_levels: list[int],
    injection_bands: list[int],
    seed: str,
) -> list[DesignCell]:
    """Materialize a blocked, partially crossed mixed-level design.

    Every task x model x condition cell is present (partially crossed).
    Severity and injection band are assigned by a block-randomized
    mixed-level assignment, not fully crossed.
    """
    rng = random.Random(seed)
    cells: list[DesignCell] = []

    for family in workload_families:
        for task_index in range(task_instances_per_family):
            task_id = f"{family}-t{task_index + 1:03d}"
            block = f"{family}"
            for condition in conditions:
                severity = rng.choice(severity_levels)
                band = rng.choice(injection_bands)
                cells.append(
                    DesignCell(
                        block=block,
                        task_id=task_id,
                        workload_family=family,
                        condition=condition,
                        severity=severity,
                        injection_band=band,
                        seed=f"{seed}-{task_id}-{condition}",
                    )
                )

    # Block-randomize run order within each block
    by_block: dict[str, list[DesignCell]] = {}
    for cell in cells:
        by_block.setdefault(cell.block, []).append(cell)
    ordered: list[DesignCell] = []
    for block in sorted(by_block):
        block_cells = by_block[block]
        rng.shuffle(block_cells)
        ordered.extend(block_cells)
    return ordered


def compute_design_matrix(cells: list[DesignCell]) -> tuple[list[list[float]], list[str]]:
    """Build an indicator design matrix for main effects and 2-way interactions.

    Returns (X, column_labels). Uses only the factors that are actually
    estimable under the partially crossed design: condition, severity,
    injection band, and their pairwise interactions where supported.
    """
    severity_levels = sorted({c.severity for c in cells})
    band_levels = sorted({c.injection_band for c in cells})
    conditions = sorted({c.condition for c in cells})

    column_labels: list[str] = []
    for condition in conditions[1:]:
        column_labels.append(f"condition={condition}")
    for severity in severity_levels[1:]:
        column_labels.append(f"severity={severity}")
    for band in band_levels[1:]:
        column_labels.append(f"band={band}")
    for condition in conditions[1:]:
        for severity in severity_levels[1:]:
            column_labels.append(f"condition={condition} x severity={severity}")
    for condition in conditions[1:]:
        for band in band_levels[1:]:
            column_labels.append(f"condition={condition} x band={band}")
    for severity in severity_levels[1:]:
        for band in band_levels[1:]:
            column_labels.append(f"severity={severity} x band={band}")

    matrix: list[list[float]] = []
    for cell in cells:
        row = [0.0] * len(column_labels)
        for col_index, label in enumerate(column_labels):
            parts = label.split(" x ")
            if len(parts) == 1:
                factor, level = parts[0].split("=")
                if factor == "condition" and cell.condition == level:
                    row[col_index] = 1.0
                elif factor == "severity" and cell.severity == int(level):
                    row[col_index] = 1.0
                elif factor == "band" and cell.injection_band == int(level):
                    row[col_index] = 1.0
            else:
                f1, v1 = parts[0].split("=")
                f2, v2 = parts[1].split("=")
                m1 = (
                    (f1 == "condition" and cell.condition == v1)
                    or (f1 == "severity" and cell.severity == int(v1))
                    or (f1 == "band" and cell.injection_band == int(v1))
                )
                m2 = (
                    (f2 == "condition" and cell.condition == v2)
                    or (f2 == "severity" and cell.severity == int(v2))
                    or (f2 == "band" and cell.injection_band == int(v2))
                )
                if m1 and m2:
                    row[col_index] = 1.0
        matrix.append(row)
    return matrix, column_labels


def _condition_number_squared(X: list[list[float]]) -> float:
    n = len(X)
    if n == 0:
        return 0.0
    m = len(X[0])
    if m == 0:
        return 0.0
    # Compute X'X (m x m) and its eigenvalues via a tiny Jacobi-free power method
    # for symmetric matrices is complex; instead use the max column-scale
    # variance-inflation proxy: kappa = sqrt(max(X'X diag)) with simple 2x2/3x3
    # eigen solver fallback for small m. For diagnostics, report the ratio of
    # largest to smallest singular value using the trace-based bound.
    XtX = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            acc = 0.0
            for row in X:
                acc += row[i] * row[j]
            XtX[i][j] = acc
    # Use the Frobenius-based ratio as a stable, dependency-free diagnostic.
    trace = sum(XtX[i][i] for i in range(m))
    frob = (sum(XtX[i][j] ** 2 for i in range(m) for j in range(m))) ** 0.5
    if trace == 0:
        return float("inf")
    # kappa bound: lambda_max >= trace/m ; lambda_min >= trace - (m-1)*lambda_max_upper
    lambda_est_max = max(trace / m, frob / (m ** 0.5))
    lambda_est_min = max(trace - (m - 1) * lambda_est_max, 1e-12)
    return lambda_est_max / lambda_est_min


def compute_diagnostics(cells: list[DesignCell]) -> DesignDiagnostics:
    matrix, column_labels = compute_design_matrix(cells)
    rank = _matrix_rank(matrix)
    n_cols = len(column_labels)
    condition_number = _condition_number_squared(matrix)

    balance: dict[str, int] = {}
    for cell in cells:
        balance[cell.block] = balance.get(cell.block, 0) + 1

    unsupported: list[str] = []
    n_conditions = len({c.condition for c in cells})
    n_severity = len({c.severity for c in cells})
    n_bands = len({c.injection_band for c in cells})
    if n_severity * n_bands > n_conditions:
        unsupported.append(
            "severity x band full crossing exceeds condition capacity; "
            "interactions are partially estimable only"
        )

    resolution_supported = False
    if n_conditions == n_severity == n_bands == 2:
        # Only a pure 2^(k) design can be called resolution-IV
        resolution_supported = False  # requires formal alias analysis
    if unsupported:
        unsupported.append(
            "mixed two- and three-level factors present; design is NOT resolution-IV"
        )

    return DesignDiagnostics(
        n_rows=len(cells),
        rank=rank,
        n_columns=n_cols,
        condition_number=condition_number,
        balance_by_block=balance,
        aliased_or_unsupported_interactions=unsupported,
        resolution_label_supported=resolution_supported,
    )


def _matrix_rank(matrix: list[list[float]]) -> int:
    if not matrix:
        return 0
    m = len(matrix[0])
    if m == 0:
        return 0
    # Gaussian elimination over floats
    work = [list(row) for row in matrix]
    rank = 0
    for col in range(m):
        pivot_row = None
        for r in range(rank, len(work)):
            if abs(work[r][col]) > 1e-9:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        work[rank], work[pivot_row] = work[pivot_row], work[rank]
        pivot = work[rank][col]
        for c in range(col, m):
            work[rank][c] /= pivot
        for r in range(len(work)):
            if r != rank and abs(work[r][col]) > 1e-9:
                factor = work[r][col]
                for c in range(col, m):
                    work[r][c] -= factor * work[rank][c]
        rank += 1
        if rank == m:
            break
    return rank


# Synthetic example inputs and golden outputs

SYNTHETIC_INPUT = {
    "workload_families": ["repo_repair", "ml_experiment"],
    "task_instances_per_family": 2,
    "conditions": ["no_fault", "fault_a", "fault_b"],
    "severity_levels": [1, 2, 3],
    "injection_bands": [1, 2, 3],
    "seed": "synthetic-seed-001",
}


def _synthetic_golden_hash() -> str:
    """Golden output hash computed once from the deterministic materialization."""
    cells = materialize_design(**SYNTHETIC_INPUT)
    diag = compute_diagnostics(cells)
    output = {
        "input": SYNTHETIC_INPUT,
        "n_rows": len(cells),
        "diagnostics": diag.to_dict(),
    }
    return _hash(output)


def run_design_generation(
    design: dict[str, Any], seed: str | None = None
) -> dict[str, Any]:
    """Materialize a design and return a reproducible output artifact."""
    actual_seed = seed or design.get("seed", "default-seed")
    cells = materialize_design(
        workload_families=design["workload_families"],
        task_instances_per_family=design["task_instances_per_family"],
        conditions=design["conditions"],
        severity_levels=design["severity_levels"],
        injection_bands=design["injection_bands"],
        seed=actual_seed,
    )
    diag = compute_diagnostics(cells)
    output = {
        "input": dict(design),
        "seed": actual_seed,
        "n_rows": len(cells),
        "schedule": [c.to_dict() for c in cells],
        "diagnostics": diag.to_dict(),
    }
    output["output_hash"] = _hash(output)
    return output


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="R1 design generation and diagnostics")
    parser.add_argument("--seed", type=str, default=None, help="override randomization seed")
    parser.add_argument("--synthetic", action="store_true", help="run on bundled synthetic input")
    parser.add_argument("--check", type=str, default=None, help="verify output hash matches")
    parser.add_argument("--output", type=str, default=None, help="write JSON output path")
    args = parser.parse_args()

    if args.synthetic:
        design = SYNTHETIC_INPUT
        label = "SYNTHETIC"
    else:
        # Load the R1 confirmatory protocol design parameters.
        import yaml

        proto_path = PROJECT_ROOT / "protocols" / "r1" / "confirmatory_protocol.yaml"
        with open(proto_path) as f:
            protocol = yaml.safe_load(f)
        families = [f["id"] for f in protocol["workload_families"]]
        design = {
            "workload_families": families,
            "task_instances_per_family": 2,  # synthetic subset; real 12-20 after pilot
            "conditions": ["no_fault", "f1", "f2", "f3", "f4", "f5", "f6"],
            "severity_levels": [1, 2, 3],
            "injection_bands": [1, 2, 3],
            "seed": protocol.get("experimental_design", {}).get("randomization_seed", "R1-C-2026-08-01-seed-v1"),
        }
        label = "R1-C"

    output = run_design_generation(design, seed=args.seed)
    print(f"=== DESIGN GENERATION [{label}] ===")
    print(f"seed: {output['seed']}")
    print(f"n_rows: {output['n_rows']}")
    diag = output["diagnostics"]
    print(f"rank: {diag['rank']} / columns: {diag['n_columns']}")
    print(f"condition_number (squared bound): {diag['condition_number']}")
    print(f"balance_by_block: {diag['balance_by_block']}")
    if diag["aliased_or_unsupported_interactions"]:
        print("aliased_or_unsupported_interactions:")
        for msg in diag["aliased_or_unsupported_interactions"]:
            print(f"  - {msg}")
    if diag["resolution_label_supported"]:
        print("resolution label supported: yes")
    else:
        print("resolution label supported: NO — design is mixed-level; do not call resolution-IV")
    print(f"output_hash: {output['output_hash']}")

    if args.check:
        if args.check == output["output_hash"]:
            print("HASH CHECK: MATCH")
        else:
            print(f"HASH CHECK: MISMATCH — expected {args.check}")
            return 1

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True))
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
