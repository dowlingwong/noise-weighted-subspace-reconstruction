"""Locate the TIDMAD data trees on whatever host we happen to be running on.

Why this module exists
----------------------
The experiment checklist specifies ``/ceph/dwong/datasets/tidmad/raw``. That
directory does not exist on ``deepthought``; the release data live in the
archived Paper-1 tree at ``/ceph/dwong/paper1_dataset/tidmad/raw``, and the
denoised outputs beside them. This discrepancy has already cost time in three
separate rounds -- each runbook has carried a "resolve and record the directory"
instruction that a human had to satisfy by hand.

So resolve it in code, once, by the only criterion that matters: a raw root is a
directory that actually contains the noise-only science files ``J(f)`` is fitted
on, and a denoised root is one that contains ``abra_validation_denoised_*.h5``.
Existence of the *directory* is not the test -- an empty directory at the
expected path is worse than no directory, because it fails later and less
clearly.

Candidates are tried in order and the first that satisfies its content test
wins. An explicit path (``TIDMAD_DATA_ROOT`` / ``TIDMAD_DENOISED_DIR``, or the
``--data-dir`` flag of a caller) always overrides the search.

Usage
-----
As a command, to print what this host has::

    PYTHONPATH=src python -m tidmad.paths

As a library::

    from tidmad.paths import resolve_raw_root, resolve_denoised_dir
"""

from __future__ import annotations

import os
from pathlib import Path

#: Raw release tree: training, validation and noise-only science files.
RAW_CANDIDATES: tuple[str, ...] = (
    "/ceph/dwong/datasets/tidmad/raw",          # the checklist's planned root
    "/ceph/dwong/paper1_dataset/tidmad/raw",    # where the data actually are
    "~/datasets/tidmad/raw",
)

#: Denoised model outputs, written by ``tidmad.infer``.
DENOISED_CANDIDATES: tuple[str, ...] = (
    "/ceph/dwong/datasets/tidmad/denoised",
    "/ceph/dwong/paper1_dataset/tidmad/denoised",
    "~/datasets/tidmad/denoised",
)

#: Upstream TIDMAD checkout, for ``benchmark.py``.
UPSTREAM_CANDIDATES: tuple[str, ...] = (
    "~/src/TIDMAD",
    "/ceph/dwong/src/TIDMAD",
    "~/TIDMAD",
)


class TidmadPathError(RuntimeError):
    """Raised when no candidate satisfies the content test."""


def _first_matching(candidates: tuple[str, ...], pattern: str) -> Path | None:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        try:
            if path.is_dir() and any(path.glob(pattern)):
                return path
        except OSError:
            # An unreadable mount is a miss, not a crash: keep looking.
            continue
    return None


def _resolve(env_var: str, candidates: tuple[str, ...], pattern: str, what: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        path = Path(override).expanduser()
        if not any(path.glob(pattern)):
            raise TidmadPathError(
                f"{env_var}={override} contains no {pattern!r}. Refusing to use it: "
                f"the {what} would not be the one previous rounds used."
            )
        return path

    found = _first_matching(candidates, pattern)
    if found is None:
        listed = "\n".join(f"    {c}" for c in candidates)
        raise TidmadPathError(
            f"No directory containing {pattern!r} found among:\n{listed}\n"
            f"Set {env_var} explicitly, or search:\n"
            f"    find /ceph -maxdepth 6 -name '{pattern}' -printf '%h\\n' 2>/dev/null | sort -u"
        )
    return found


def resolve_raw_root() -> Path:
    """Directory holding ``abra_science_*.h5`` -- the noise-only files J(f) uses."""
    return _resolve("TIDMAD_DATA_ROOT", RAW_CANDIDATES, "abra_science_*.h5", "PSD estimate")


def resolve_denoised_dir() -> Path:
    """Directory holding ``abra_validation_denoised_*.h5``."""
    return _resolve(
        "TIDMAD_DENOISED_DIR", DENOISED_CANDIDATES,
        "abra_validation_denoised_*.h5", "scored outputs",
    )


def resolve_upstream() -> Path:
    """The upstream TIDMAD checkout containing ``benchmark.py``."""
    return _resolve("TIDMAD_UPSTREAM", UPSTREAM_CANDIDATES, "benchmark.py", "scorer")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shell", action="store_true",
                    help="emit `export VAR=path` lines instead of a report")
    args = ap.parse_args()

    probes = (
        ("TIDMAD_DATA_ROOT", resolve_raw_root, "abra_science_*.h5", "abra_validation_00??.h5"),
        ("TIDMAD_DENOISED_DIR", resolve_denoised_dir, "abra_validation_denoised_*.h5", None),
        ("TIDMAD_UPSTREAM", resolve_upstream, "benchmark.py", None),
    )

    failures = 0
    for var, resolver, pattern, extra in probes:
        try:
            path = resolver()
        except TidmadPathError as exc:
            failures += 1
            if args.shell:
                print(f"# {var}: UNRESOLVED")
            else:
                print(f"{var}: NOT FOUND\n{exc}\n")
            continue

        if args.shell:
            print(f"export {var}={path}")
            continue

        counts = [f"{len(list(path.glob(pattern)))} x {pattern}"]
        if extra:
            counts.append(f"{len(list(path.glob(extra)))} x {extra}")
        print(f"{var}={path}")
        print(f"    {', '.join(counts)}")

    if not args.shell and failures:
        print(f"\n{failures} path(s) unresolved. Do not start a run until they are.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
