"""Round 10 output-safety parameters, delegating mechanics to ``r9_common``.

Round 10 adds two corrected runners (E10b, E11). They inherit the entire
Round 9 output contract — atomic shards, manifest provenance, resume
validation, archive refusal — and change only the parameters that must not
collide with Round 9:

* smoke output lives below ``results/rps2026/Round10_smoke/``;
* smoke seeds are 910000/910001 (disjoint from the scientific 0–29 and from
  Round 9's 900000/900001);
* the archive-refusal list covers every archived evidence directory E1–E10,
  not only Block A — Round 10 exists because archived columns must never be
  redefined in place.

Nothing here makes a scientific choice.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import r9_common as rc
from .r9_common import (  # re-exported so runners import one module
    IntegrityError,
    REPRODUCTION_RTOL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    bootstrap_paired,
    concat_shards,
    paired_contrast,
    prepare_output_dir,
    provenance,
    read_rows_csv,
    sha256_of,
    shard_path,
    smoke_config,
    write_json,
    write_rows_csv,
)

RESULTS_ROOT = rc.RESULTS_ROOT

#: Round 10 smoke confinement.
SMOKE_ROOT = RESULTS_ROOT / "Round10_smoke"
SMOKE_SEEDS = (910000, 910001)

#: Optional stable runtime root OUTSIDE the Git worktree.
#:
#: The non-smoke runners require a clean source tree, so a flow that writes
#: results into the repository before they start deadlocks itself. Setting
#: ``R10_RUNTIME_ROOT`` lets tests, logs, audits, smoke and the scientific runs
#: all land outside the worktree; the completed directories are copied into the
#: repository once both scientific runs have finished.
def runtime_root() -> Path | None:
    value = os.environ.get("R10_RUNTIME_ROOT")
    return Path(value).expanduser().resolve() if value else None


def _permitted_smoke_roots() -> tuple[Path, ...]:
    rt = runtime_root()
    return (SMOKE_ROOT,) if rt is None else (SMOKE_ROOT, rt / "Round10_smoke")

#: Every archived evidence directory is read-only for Round 10 runners.
ARCHIVED_DIRS = (
    RESULTS_ROOT / "E1_nfpa_blockA",
    RESULTS_ROOT / "E2_nfpa_blockB",
    RESULTS_ROOT / "E8_semantics",
    RESULTS_ROOT / "E9_blockA_replication",
    RESULTS_ROOT / "E10_estimated_covariance",
)

#: Archived raw tables used by the Round 10 reproduction gates.
ARCHIVED_E1_CSV = rc.ARCHIVED_E1_CSV
ARCHIVED_E9_CSV = RESULTS_ROOT / "E9_blockA_replication" / "replication_raw.csv"
ARCHIVED_E10_CSV = RESULTS_ROOT / "E10_estimated_covariance" / "raw.csv"


def resolve_out_dir(out_dir: str | os.PathLike[str], *, smoke: bool) -> Path:
    """Round 10 variant of :func:`r9_common.resolve_out_dir`.

    Refuses every archived evidence directory (not only Block A) and confines
    smoke below ``Round10_smoke/``.
    """
    path = Path(out_dir).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    resolved = path.resolve()

    for archived in ARCHIVED_DIRS:
        if rc._is_within(resolved, archived):
            raise IntegrityError(
                f"output directory {resolved} resolves into the archived evidence "
                f"directory {archived}; Round 10 archives are read-only"
            )
    permitted = _permitted_smoke_roots()
    if smoke and not any(rc._is_within(resolved, root) for root in permitted):
        raise IntegrityError(
            f"--smoke output must live below one of {[str(p) for p in permitted]}, "
            f"got {resolved} (set R10_RUNTIME_ROOT to allow an out-of-tree smoke root)"
        )
    if not smoke and any(rc._is_within(resolved, root) for root in permitted):
        raise IntegrityError(
            f"non-smoke output must not be written below a smoke root {[str(p) for p in permitted]}"
        )
    # Round 9's smoke root is equally off-limits in either mode.
    if rc._is_within(resolved, rc.SMOKE_ROOT):
        raise IntegrityError(
            f"Round 10 output must not land in Round 9's smoke root {rc.SMOKE_ROOT}"
        )
    return resolved


def guard_seed_set(seeds, expected, *, smoke: bool, allow_subset: bool = False) -> list[int]:
    """Round 10 variant of :func:`r9_common.guard_seed_set` (910000/910001 smoke)."""
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise IntegrityError(f"duplicate seeds requested: {seeds}")
    if smoke:
        if sorted(seeds) != sorted(SMOKE_SEEDS):
            raise IntegrityError(f"--smoke requires seeds {list(SMOKE_SEEDS)}, got {seeds}")
        return sorted(seeds)
    forbidden = set(SMOKE_SEEDS) | set(rc.SMOKE_SEEDS)
    if any(s in forbidden for s in seeds):
        raise IntegrityError("smoke seeds may not be used without --smoke")
    expected_sorted = sorted(int(s) for s in expected)
    if allow_subset:
        extra = sorted(set(seeds) - set(expected_sorted))
        if extra:
            raise IntegrityError(f"seeds {extra} are outside the registered set {expected_sorted}")
        return sorted(seeds)
    if sorted(seeds) != expected_sorted:
        raise IntegrityError(
            f"unexpected seed set: got {sorted(seeds)}, registered set is {expected_sorted}"
        )
    return sorted(seeds)
