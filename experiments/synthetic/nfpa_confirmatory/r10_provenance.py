"""Round-10-specific provenance: an explicit, fail-closed implementation freeze.

Why this exists
---------------
Round 9's ``r9_common.provenance`` records ``implementation_freeze_commit`` as
whatever ``HEAD`` happened to be at execution time. That is not a freeze: it
cannot distinguish "the code was frozen before any full result" from "the code
was edited and run in the same breath", and a dirty tree still produced a
manifest that looked authoritative.

Round 10 therefore requires the operator to *state* the freeze commit, and
validates that statement against the repository before any scientific
computation:

* the freeze SHA must be a full 40-character SHA of an existing commit;
* it must be an ancestor of the current execution ``HEAD`` (so the running
  code descends from the frozen implementation);
* for a non-smoke run the worktree must be clean;
* the execution ``HEAD`` is recorded **separately** from the freeze SHA, so the
  two can never be confused after the fact;
* nothing is inferred from ``HEAD``: omitting the option fails closed.

Round 9 semantics are deliberately left untouched; this module is additive and
is used only by the Round 10 runners (E10b, E11).

Smoke runs
----------
Smoke output is an implementation check, never scientific evidence. A smoke
manifest carries ``evidence_class = "implementation_check_not_scientific_evidence"``
and, if the operator supplies no freeze SHA, ``implementation_freeze_commit =
None`` — it is never silently backfilled from ``HEAD``. A full-run manifest
always carries ``evidence_class = "scientific_evidence"`` and a validated
freeze SHA.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import re
import socket
import subprocess
import sys
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from .config import PRIMARY_HASH, PrimaryConfig, config_hash
from .r9_common import REPO_ROOT, IntegrityError

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

EVIDENCE_SCIENTIFIC = "scientific_evidence"
EVIDENCE_SMOKE = "implementation_check_not_scientific_evidence"


def _git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120
    )
    if check and proc.returncode != 0:
        raise IntegrityError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_execution_head() -> str:
    return _git("rev-parse", "HEAD")


def worktree_is_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def validate_freeze_commit(freeze_sha: str | None, *, smoke: bool) -> tuple[str | None, str]:
    """Validate the declared freeze SHA; return ``(freeze_sha, execution_head)``.

    Fails closed: a missing, malformed, unknown, or non-ancestor SHA raises,
    as does a dirty worktree for a non-smoke run. Nothing is inferred from
    ``HEAD``.
    """
    execution_head = resolve_execution_head()

    if freeze_sha is None:
        if not smoke:
            raise IntegrityError(
                "--implementation-freeze-commit is required for a non-smoke run: "
                "the freeze commit must be stated explicitly and is never inferred "
                "from HEAD"
            )
        return None, execution_head

    freeze_sha = freeze_sha.strip().lower()
    if not SHA_RE.match(freeze_sha):
        raise IntegrityError(
            f"implementation freeze commit {freeze_sha!r} is not a 40-character hex SHA"
        )
    kind = _git("cat-file", "-t", freeze_sha, check=False)
    if kind != "commit":
        raise IntegrityError(
            f"implementation freeze commit {freeze_sha} does not resolve to a commit "
            f"(git cat-file -t returned {kind!r})"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze_sha, execution_head],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    if ancestor.returncode != 0:
        raise IntegrityError(
            f"implementation freeze commit {freeze_sha} is not an ancestor of the "
            f"execution HEAD {execution_head}; the running code does not descend "
            "from the declared freeze"
        )
    if not smoke and worktree_is_dirty():
        raise IntegrityError(
            "worktree is dirty; a non-smoke run must execute a committed tree so "
            "that the recorded provenance is reproducible. Commit or stash "
            "deliberately, then re-run."
        )
    return freeze_sha, execution_head


def provenance_block(
    command: Sequence[str],
    cfg: PrimaryConfig,
    *,
    freeze_sha: str | None,
    execution_head: str,
    smoke: bool,
    started_at_utc: str,
) -> dict[str, Any]:
    """The Round 10 manifest provenance block."""
    try:
        import scipy  # noqa: F401 — version only

        scipy_version = scipy.__version__
    except Exception:  # pragma: no cover - scipy is a hard dep of the runners
        scipy_version = "unavailable"

    return {
        "evidence_class": EVIDENCE_SMOKE if smoke else EVIDENCE_SCIENTIFIC,
        "implementation_freeze_commit": freeze_sha,
        "execution_content_commit": execution_head,
        "freeze_is_ancestor_of_execution": bool(freeze_sha) or None,
        "git_tree_dirty": worktree_is_dirty(),
        "command": list(command),
        "config_hash": config_hash(cfg),
        "primary_config_hash": PRIMARY_HASH,
        "config": asdict(cfg),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "thread_limits": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "started_at_utc": started_at_utc,
        "completed_at_utc": None,  # filled by the final atomic manifest write
    }


def add_freeze_argument(parser) -> None:
    """Register the required freeze option on a runner's argument parser."""
    parser.add_argument(
        "--implementation-freeze-commit",
        dest="implementation_freeze_commit",
        default=None,
        metavar="SHA40",
        help=(
            "REQUIRED for non-smoke runs: the 40-character SHA of the commit that "
            "froze the implementation. Must exist and be an ancestor of HEAD; the "
            "worktree must be clean. Never inferred from HEAD."
        ),
    )
