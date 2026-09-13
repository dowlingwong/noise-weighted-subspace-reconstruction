#!/usr/bin/env python3
r"""Build the remote handoff package for the 13 September 2026 successor round.

What this produces is a *transfer* artifact, not a result.  A package inside it is
at most `ready to transfer` in the round's status vocabulary
(`configured` -> `locally tested` -> `ready to transfer` -> `executed` ->
`results validated` -> `claim supported`).  "Ready for remote execution" means the
implementation passed local acceptance.  It does not mean any scientific claim is
supported, and none of the historical verdicts is affected by the existence of this
package.

Contents of the archive
-----------------------
    HANDOFF.md              what this is, how to run it, what to send back
    ENVIRONMENT.md          the environment actually measured here, and every
                            setting that is deliberately UNRESOLVED
    MANIFEST.json           SHA-256 of every file, plus source commit and counts
    verify_handoff.py       runs in the extracted tree: checks the manifest, the
                            imports, validate/enumerate for all five packages, and
                            (with --tests) the local test suite
    src/                    source snapshot: the code the packages import
    experiments/            "
    tests/                  "
    pyproject.toml          "
    packages/<pkg>/         per-package protocol, commands, schema, findings,
                            requirements, resource estimate and EXPECTED_CELLS.json
    cresst_archive/         registered CRESST inputs the CRESST-S package recomputes
                            from (read-only)
    results/                read-only archived run records the T22-S exposure inventory
                            reads
    provenance/             the branch tip commit, the diff of this round against
                            its merge base, and the git bundle of just this round's
                            commits (so history is recoverable without the 107 MB
                            full-repository bundle)

What this deliberately does NOT contain
---------------------------------------
No host name, no scheduler directive, no queue, no account, no wall-clock request,
no node or core count, no filesystem path on any remote machine.  Those are the
receiving environment's to supply; every one of them appears in ENVIRONMENT.md as an
UNRESOLVED item with the reason it cannot be filled in from here.  The raw TIDMAD data
root is likewise unresolved: this repository carries no TIDMAD bytes, and a T22-S
matched-stage run on real data takes that path from the operator.  The archived
CRESST inputs the CRESST-S package recomputes from ARE carried (`cresst_archive/`),
because a clean extraction without them skipped every CRESST-S scientific test -- and a
package whose scientific checks all skip has not been shown to work.

    python3 experiments/mlst2026/successor_20260913/make_handoff.py --out DIR
    python3 experiments/mlst2026/successor_20260913/make_handoff.py --out DIR --verify
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent                     # repository root
ROUND = "successor_20260913"
PKGS = ["t19s", "t20s", "t21s", "t22s", "cresst_s"]

#: Source paths copied into the snapshot.  `experiments` and `src` are taken whole
#: because the successor packages import audited historical modules from both
#: (t19_coupling, t20_population_accounting, t13a_cresst_channel_bootstrap,
#: t22_tidmad_narrowband, tidmad_band, tidmad_endpoint, src.canonical.empca);
#: pruning them to a computed closure would be a second thing to keep correct.
SNAPSHOT = ["experiments", "src", "tests", "pyproject.toml"]

#: Archived run trees the T22-S matched-stage package READS (never writes).  A clean
#: extraction without them silently produced an empty exposure inventory, which is the
#: one artifact that records which instrument regions have already been examined -- so
#: leaving them out would ship a package that cannot tell the remote operator which
#: hold-outs are already spent.  They are copied read-only and their digests are in
#: MANIFEST.json; the runners refuse to write anywhere under `results/`.
ARCHIVE_INPUTS = [
    "results/t22_2026-09-11",
    "results/mlst_2026-09-10",
]

#: Directories excluded from the snapshot: caches, and the historical result and
#: evidence trees, which are immutable inputs the remote round must not receive a
#: second copy of and must not write to.
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}

#: The CRESST-S package recomputes the archived per-channel residuals from registered
#: bytes that live in the MANUSCRIPT repository's evidence root, not in this one.  A
#: clean extraction without them skipped 16 of the 112 tests -- a package whose
#: scientific checks are all skipped is not shown to work by a green test run, so the
#: files are copied in under `cresst_archive/` and `verify_handoff.py` points
#: `CRESST_S_ARCHIVE_ROOT` at them.  Every one is a registered entry of the manuscript
#: repository's evidence manifest; their digests go into this archive's manifest too.
CRESST_ARCHIVE_FILES = [
    "cresst/sensitivity/20260713_channel_robustness_v1/event_residuals_lossless.npz",
    "cresst/sensitivity/20260713_channel_robustness_v1/event_residuals_metadata.json",
    "cresst/sensitivity/20260713_channel_robustness_v1/channel_summary.csv",
    "cresst/sensitivity/20260713_channel_robustness_v1/estimand_summary.csv",
    "cresst/sensitivity/20260713_channel_robustness_v1/summary.json",
    "cresst/sensitivity/20260713_channel_robustness_v1/analysis_config.yaml",
    "cresst/sensitivity/20260713_channel_robustness_v1/analysis_script.py",
    "paper1/mlst_2026-09-10/T13a_cresst_channel_bootstrap_v1/t13a_results.json",
]

#: Where the manuscript repository's evidence root is looked for.  Overridable with
#: --ms-evidence; never invented, and its absence is reported, not silently tolerated.
DEFAULT_MS_EVIDENCE = ROOT.parent / "MS" / "evidence" / "data"


def sh(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_snapshot(dst: Path) -> None:
    for rel in SNAPSHOT + ARCHIVE_INPUTS:
        src = ROOT / rel
        if src.is_file():
            shutil.copy2(src, dst / rel)
            continue
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            d = Path(dirpath)
            out = dst / d.relative_to(ROOT)
            out.mkdir(parents=True, exist_ok=True)
            for fn in filenames:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                shutil.copy2(d / fn, out / fn)


def copy_cresst_archive(dst: Path, ms_evidence: Path) -> list[str]:
    """Copy the registered CRESST inputs; return the list of any that are missing."""
    missing = []
    for rel in CRESST_ARCHIVE_FILES:
        src = ms_evidence / rel
        if not src.is_file():
            missing.append(rel)
            continue
        out = dst / "cresst_archive" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    return missing


def package_materials(dst: Path, matrices: dict) -> None:
    for pkg in PKGS:
        srcd = HERE / pkg
        outd = dst / "packages" / pkg
        outd.mkdir(parents=True, exist_ok=True)
        for pat in ("PROTOCOL_*.json", "REMOTE_COMMANDS.md", "SCHEMA.md", "FINDINGS_*.md",
                    "requirements.txt", "RESOURCE_ESTIMATE.json", "BENCHMARK.json", "FROZEN_*.json",
                    "EXPOSURE_INVENTORY.md"):
            for p in sorted(srcd.glob(pat)):
                shutil.copy2(p, outd / p.name)
        (outd / "EXPECTED_CELLS.json").write_text(
            json.dumps(matrices[pkg], indent=1) + "\n")


class HandoffFactError(RuntimeError):
    """A per-package fact could not be discovered, or contradicts the package."""


def _priced_counts(pkg: str, est: dict | None, summary: dict) -> dict | None:
    """The cells the package's resource projection actually covers, checked against
    the enumerated production matrix.

    A projected CPU total printed beside a cell count that covers a different set of
    cells is a silent disagreement, so it is refused here rather than shipped: the
    handoff prints one pair of numbers and they must be the same pair.
    """
    if est is None:
        return None
    priced = est.get("priced_counts") or est.get("production_projection", {}).get("production_counts")
    if priced is None:
        raise HandoffFactError(
            f"{pkg}: RESOURCE_ESTIMATE.json does not say which cells its total covers "
            "(no 'priced_counts'); regenerate it with the current `estimate` subcommand")
    for key in ("cells", "fits", "solves"):
        if int(priced.get(key, -1)) != int(summary.get(key, -2)):
            raise HandoffFactError(
                f"{pkg}: the resource estimate prices {priced.get('cells')} cells / "
                f"{priced.get('fits')} fits but `enumerate` gives {summary.get('cells')} / "
                f"{summary.get('fits')}; the projection and the printed counts would "
                "disagree.  Regenerate RESOURCE_ESTIMATE.json from a fixture run root.")
    if "development" in (priced.get("by_role") or {}):
        raise HandoffFactError(f"{pkg}: the resource estimate prices development cells; "
                               "development work is fixture work and is never transferred")
    _check_measured_estimate(pkg, est)
    return priced


def _check_measured_estimate(pkg: str, est: dict) -> None:
    """The estimate must be the measured, scaling-aware one, not the flat product.

    The v1 schema multiplied one fixture wall time by a constant declared in the
    protocol.  That is a linear extrapolation of a tiny fixture with no dimension
    and no algorithmic scaling in it, and shipping it as the estimate is what this
    guard exists to prevent.  The checks are structural, so a regression that
    quietly reinstates the flat product fails the build rather than the reader.
    """
    if est.get("schema") != "mlst-local-package-resource-estimate/2":
        raise HandoffFactError(
            f"{pkg}: RESOURCE_ESTIMATE.json is schema {est.get('schema')!r}, not the measured "
            "'mlst-local-package-resource-estimate/2'; regenerate it with the current `estimate` subcommand "
            "after running run_benchmarks.py")
    if not est.get("measured_projection_available"):
        raise HandoffFactError(f"{pkg}: the resource estimate has no measured projection "
                               f"({est.get('why_no_measured_projection', 'BENCHMARK.json missing')})")
    missing = [k for k in ("hardware", "cpu_seconds", "wall_time_hours_by_workers", "parallelism_assumption",
                           "peak_memory", "storage", "transfer", "setup", "unknowns", "reduction_work",
                           "work_accounting", "families")
               if k not in est]
    if missing:
        raise HandoffFactError(f"{pkg}: the resource estimate is missing required fields {missing}")
    if est.get("unmeasured_families"):
        raise HandoffFactError(f"{pkg}: families {est['unmeasured_families']} have no measured benchmark; "
                               "an unbenchmarked family may not be shipped inside a priced total")
    for fam, row in (est.get("families") or {}).items():
        if not row.get("measured"):
            raise HandoffFactError(f"{pkg}: family {fam!r} is unmeasured in the resource estimate")
        if row.get("basis") not in ("measured", "projected", "measured-on-substitute"):
            raise HandoffFactError(f"{pkg}: family {fam!r} does not label its basis measured/projected")
        if row.get("basis") == "projected" and row.get("fitted_exponent") is None:
            raise HandoffFactError(f"{pkg}: family {fam!r} is projected but names no fitted exponent")
    flat = (est.get("cross_check_flat_fixture_scale") or {}).get("total_cpu_seconds_point")
    point = (est.get("cpu_seconds") or {}).get("cell_compute_point")
    if flat and point and abs(point - flat) < 1e-9:
        raise HandoffFactError(f"{pkg}: the projected total equals the flat fixture x declared-scale product "
                               "exactly; the flat product is a cross-check, not the estimate")


#: Names of module-level constants that hold the run-root file a pilot reduce freezes.
#: Matched by NAME, not by a per-package lookup of file names, so a renamed freeze file
#: is picked up automatically.
_FREEZE_NAME_RE = re.compile(r"(FROZEN|SELECTION)", re.IGNORECASE)


def discover_freeze_file(pkg: str) -> str | None:
    """The file this package's pilot reduce freezes into the run root, or ``None``.

    Read out of the package's own source: a module-level ``<...FROZEN...> = "X.json"``
    or ``<...SELECTION...> = "X.json"``.  Cross-checked against the package's
    ``REMOTE_COMMANDS.md`` (the authoritative document), so a drift between the code
    and the commands fails the build instead of being printed as fact.
    """
    srcd = HERE / pkg
    found: dict[str, str] = {}
    for py in sorted(srcd.glob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            val = node.value.value
            if not isinstance(val, str) or not val.endswith(".json"):
                continue
            for t in node.targets:
                if isinstance(t, ast.Name) and _FREEZE_NAME_RE.search(t.id):
                    found[val] = f"{py.name}:{t.id}"
    commands = (srcd / "REMOTE_COMMANDS.md").read_text()
    if len(found) > 1:
        raise HandoffFactError(f"{pkg}: more than one freeze-file constant ({found}); "
                               "the handoff cannot name one of them as THE freeze file")
    if not found:
        if re.search(r"\bFROZEN_[A-Z_]+\.json\b", commands):
            raise HandoffFactError(f"{pkg}: REMOTE_COMMANDS.md names a frozen file but no "
                                   "freeze-file constant was found in the package source")
        return None
    name, where = next(iter(found.items()))
    if name not in commands:
        raise HandoffFactError(f"{pkg}: the source freezes {name!r} ({where}) but "
                               "REMOTE_COMMANDS.md never mentions it; the two have drifted")
    return name


_SUBCOMMANDS = ("validate", "enumerate", "run", "reduce", "estimate")


def _fenced_command_lines(md: str) -> list[list[str]]:
    """Tokenised command lines from the fenced code blocks of a commands document,
    with ``\\`` continuations joined and comment lines dropped."""
    lines: list[str] = []
    inside = False
    for raw in md.splitlines():
        if raw.strip().startswith("```"):
            inside = not inside
            continue
        if inside:
            lines.append(raw)
    joined: list[str] = []
    buf = ""
    for ln in lines:
        s = ln.strip()
        if s.endswith("\\"):
            buf += s[:-1].strip() + " "
            continue
        joined.append((buf + s).strip())
        buf = ""
    if buf:
        joined.append(buf.strip())
    out = []
    for s in joined:
        if not s or s.startswith("#"):
            continue
        out.append(s.split())
    return out


def discover_run_selectors(pkg: str) -> list[str]:
    """The phase selectors this package's own commands document actually uses.

    Returned in the order ``REMOTE_COMMANDS.md`` gives them, de-duplicated.  Fixture
    runs and shell-substituted shard lists are not part of the production sequence and
    are skipped.  Packages do not agree on the flag: some select a phase by ``--roles``
    and some by ``--families``, which is exactly why this is read out rather than
    templated.
    """
    md = (HERE / pkg / "REMOTE_COMMANDS.md").read_text()
    seen, out = set(), []
    for toks in _fenced_command_lines(md):
        if "run" not in toks or "--protocol" not in toks:
            continue
        if toks.index("run") > toks.index("--protocol"):
            continue
        if "--fixture" in toks or any("$(" in t for t in toks):
            continue
        sel = []
        for flag in ("--roles", "--families"):
            if flag in toks:
                i = toks.index(flag)
                vals = []
                j = i + 1
                while j < len(toks) and not toks[j].startswith("--"):
                    vals.append(toks[j])
                    j += 1
                if vals:
                    sel.append(" ".join([flag, *vals]))
        s = "  ".join(sel) if sel else "(no selector: every cell not yet completed)"
        if s not in seen:
            seen.add(s)
            out.append(s)
    if not out:
        raise HandoffFactError(f"{pkg}: no production `run` command found in REMOTE_COMMANDS.md")
    return out


def enumerate_matrix(pkg: str, workdir: Path) -> dict:
    """Enumerate the production matrix and return what the remote must reproduce."""
    proto = next((HERE / pkg).glob("PROTOCOL_*.json"))
    out = workdir / pkg
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, "-m",
                    f"experiments.mlst2026.{ROUND}.{pkg}", "enumerate",
                    "--protocol", str(proto), "--out", str(out)],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    summary = json.loads((out / "run_matrix_summary.json").read_text())
    rows = [json.loads(l) for l in (out / "run_matrix.jsonl").read_text().splitlines() if l]
    by_family: dict[str, int] = {}
    by_role: dict[str, int] = {}
    for r in rows:
        by_family[r.get("family", "?")] = by_family.get(r.get("family", "?"), 0) + 1
        by_role[r.get("role", "?")] = by_role.get(r.get("role", "?"), 0) + 1
    est_p = (HERE / pkg / "RESOURCE_ESTIMATE.json")
    est = json.loads(est_p.read_text()) if est_p.exists() else None
    priced = _priced_counts(pkg, est, summary)
    return {
        "package": pkg,
        "protocol_file": proto.name,
        "freeze_file": discover_freeze_file(pkg),
        "run_selectors": discover_run_selectors(pkg),
        "priced_counts": priced,
        "protocol_sha256": summary.get("protocol_sha256"),
        "matrix_sha256": summary.get("matrix_sha256"),
        "cells": summary.get("cells"),
        "fits": summary.get("fits"),
        "solves": summary.get("solves"),
        "cells_by_family": dict(sorted(by_family.items())),
        "cells_by_role": dict(sorted(by_role.items())),
        "cell_ids_sha256": hashlib.sha256(
            "\n".join(sorted(r["cell_id"] for r in rows)).encode()).hexdigest(),
        "resource_estimate": est,
        "note": "The remote round must reproduce matrix_sha256 and cell_ids_sha256 exactly. "
                "A different matrix_sha256 is a different design and must not be run.",
    }


HANDOFF_MD = """# Remote handoff — MLST successor round, {date}

Source repository: `noise-weighted-subspace-reconstruction`, branch `{branch}`,
commit `{commit}`.

## Status of what is in here

Round status vocabulary: `configured` -> `locally tested` -> **`ready to transfer`** ->
`executed` -> `results validated` -> `claim supported`.

Every package in this archive is at **`ready to transfer`**. That means the
implementation passed local acceptance: protocols validate, matrices enumerate to fixed
digests, fixture end-to-end runs complete, controls behave, reducers refuse missing,
duplicated, unattributed and tampered cells, injected failures are charged and not
refunded, and injected incorrect scientific outputs are rejected by the packages' own
checks. It does **not** mean any scientific claim is supported. The historical verdicts
stand unchanged until a production run returns and is validated:

* T19 — the registered slope prediction is **not met**; the observable-by-observable
  reading is post hoc.
* T20 — the registered `c/N` form is **refuted**; `N*` is **bracketed, not located**.
* T21 — **five of five** registered predictions not met.
* T22 — the dependence assessment did not stabilise; **all three verdicts are withheld**.
* CRESST — an **archive-specific** same-rank residual effect; rank/shrinkage robustness
  and time-cut sensitivity are **deferred, not established**.

Nothing in this archive may be quoted as a result, and no fixture output is citable.

## What is in the archive

| path | what it is |
|---|---|
| `ENVIRONMENT.md` | the environment measured here, and every setting left UNRESOLVED |
| `MANIFEST.json` | SHA-256 of every file, source commit, per-package expected counts |
| `verify_handoff.py` | run this first in the extracted tree |
| `experiments/`, `src/`, `tests/`, `pyproject.toml` | source snapshot |
| `results/t22_2026-09-11/`, `results/mlst_2026-09-10/` | **read-only** archived run records the T22-S exposure inventory reads. Do not write here; the runners refuse `<RUN_ROOT>` under `results/` |
| `packages/<pkg>/` | protocol, commands, schema, findings, requirements, resource estimate, `BENCHMARK.json`, `EXPECTED_CELLS.json` |
| `provenance/` | branch commit, this round's diff against its merge base, and a git bundle of this round's commits |
| `experiments/mlst2026/successor_20260913/RESOURCE_REPORT.md` | the measured benchmark tables, fitted vs theoretical scaling exponents, wall-time ranges per worker count, storage, and the list of what remains unknown about the receiving machine |
| `experiments/mlst2026/successor_20260913/CONDITIONAL_EXPERIMENTS.md` | which experiments were conditional on retaining a stronger claim, what happened to that claim, and whether each was built, not built, or is not buildable |
| `experiments/mlst2026/successor_20260913/README.md` | the round's binding conventions: layout, seed namespaces, accounting rules, required tests, status vocabulary |

## First step, before anything is computed

```
tar xzf {tarname}
cd {stem}
python3 verify_handoff.py                 # manifest, imports, validate + enumerate (computes no cells)
python3 verify_handoff.py --tests         # adds the local test suite (a few minutes)
```

`verify_handoff.py` fails if any file's digest differs, if any package's enumerated
`matrix_sha256` differs from `EXPECTED_CELLS.json`, or if an import is missing. A
digest or matrix mismatch means the transfer or the environment differs from the one
that was locally accepted; stop and report it rather than running production cells.

## Expected work

{expected_table}

Every row's cells/fits/solves are the enumerated production matrix, and the projected
CPU seconds cover **exactly those cells**: the projection excludes `development` cells
(fixture-only work, never transferred), and the build refuses to produce this archive if
a package's `RESOURCE_ESTIMATE.json` prices a different set of cells from the one printed
here. `packages/<pkg>/RESOURCE_ESTIMATE.json` states the same set in `priced_counts`.

CPU-second figures are point projections from measured fixture timings scaled by each
protocol's declared fixture-to-production factor, single-threaded BLAS. They carry no
variance, exclude reduce time, and are upper bounds where the fixture cells are
overhead-dominated. They are **not** a wall-clock request: this archive names no node,
core count or queue (see `ENVIRONMENT.md`).

## Running a package

Each package's exact command sequence, in order, with its phase gates, is in
`packages/<pkg>/REMOTE_COMMANDS.md`. Read that file; it is authoritative over this
summary. The shape common to all five:

```
export PYTHONPATH=<DATA_ROOT>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PKG="python -m experiments.mlst2026.successor_20260913.<pkg>"
PROTO=packages/<pkg>/<PROTOCOL FILE from the table below>

$PKG validate  --protocol $PROTO
$PKG enumerate --protocol $PROTO --out <RUN_ROOT>
$PKG run       --protocol $PROTO --out <RUN_ROOT> <PHASE 1 SELECTOR>
$PKG reduce    --protocol $PROTO --out <RUN_ROOT> --allow-partial   # writes the freeze, where there is one
$PKG run       --protocol $PROTO --out <RUN_ROOT> <PHASE 2 SELECTOR>
$PKG reduce    --protocol $PROTO --out <RUN_ROOT>
$PKG estimate  --protocol $PROTO --out <RUN_ROOT>
```

The protocol file name, the phase selectors and the freeze file are **not** the same in
every package, and the selector flags are not interchangeable: some packages select a
phase by `--roles` and some by `--families`. The table below is read out of each package
at build time — its protocol file is the one on disk, its selectors are the ones its own
`REMOTE_COMMANDS.md` issues, in that order, and its freeze file is the one its reducer
writes. Some packages need extra flags (CRESST-S takes `--archive-root`, and
`--cresst-data-root` for its robustness grid); those are in the package's own document.

{package_table}

Three rules the runners enforce rather than document:

* `<RUN_ROOT>` may not be inside `results/` or `evidence/`. Those trees are immutable
  historical evidence; the runner refuses to write there.
* {freeze_rule}
* `--fixture` runs the development seed namespace only and marks its outputs
  `development_only`. It is refused on pilot and confirmation cells. Development
  outcomes are never citable.

## Failure, interruption and resume

* **Resume:** `run` is resumable and skips cells already completed in `<RUN_ROOT>`.
  Re-running the same command after any interruption is the correct recovery.
* **Charging:** every attempt is charged to `EXECUTION_LEDGER.jsonl` **at start**.
  Failed, retried and interrupted attempts are **never refunded**. This is deliberate:
  the ledger records what was actually spent, not what succeeded.
* **Budget:** exceeding the protocol's fit cap refuses the attempt with **exit 2**. Do
  not raise a cap to get past this; report it. Each protocol reserves retry headroom
  under its cap.
* **Retries:** a failed cell may be retried at most twice; each retry is charged.
* **Interrupted cells:** `sweep_interrupted()` marks attempts that never completed, so a
  resumed run does not silently treat them as absent.
* **Duplicate completion:** a cell completing twice is **exit 5**, not a merge.
* **Reduce refusals:** missing, extra, duplicated, unattributed or tampered cells are
  **exit 3**; a failed scientific self-check on a cell is **exit 4**; a missing or
  mismatched frozen selection is **exit 6**. `--allow-partial` is only for the pilot
  reduce and labels its output partial.
* **Parallelism:** workers may take disjoint cell subsets sharing one `<RUN_ROOT>`; the
  ledger is appended under an advisory `flock` on a sidecar lock file. **That lock was
  verified on a local disk only.** On a network filesystem, either run one worker per
  ledger or give each worker its own `<RUN_ROOT>` and reduce them separately. Do not
  merge cell files by hand: a separate root is a separate run.

## What to send back

Return, per package, the whole `<RUN_ROOT>` except nothing — it is small and its value
is that it is complete:

1. `run_matrix_summary.json` and `run_matrix.jsonl` (so the design can be re-checked)
2. `protocol_snapshot.json`
3. every file under `cells/`
4. `EXECUTION_LEDGER.jsonl` and `LEDGER_SUMMARY.json` (**unedited** — the failures and
   retries are part of the record, not noise to be cleaned)
5. the freeze file the package writes, copied out at the moment the pilot reduce wrote
   it — it is a different file in every package, and some write none:
   {freeze_return_list}
6. `REDUCED.json` from **both** the pilot reduce and the confirmation reduce
7. `RESOURCE_ESTIMATE.json` from the production root
8. a `RETURN_NOTES.md` recording: the resolved values of every UNRESOLVED item in
   `ENVIRONMENT.md`; the `python`, `numpy` and `scipy` versions actually used; the
   `verify_handoff.py` output; any deviation from `REMOTE_COMMANDS.md` and why; and any
   cell that failed, with its exit code

Also return a SHA-256 manifest of what you send, produced on the remote side.

Do **not** edit, prune or "clean" any returned file, and do not re-run a phase to
replace an unfavourable outcome. A refuted prediction, a failed gate and an exhausted
budget are results; a rerun to improve them is not.
"""

ENVIRONMENT_MD = """# Environment

## Measured here

These are the versions the local acceptance run actually used. They are a record, not a
requirement; the receiving environment should report what it used in `RETURN_NOTES.md`.

| item | value |
|---|---|
| python | {python} |
| numpy | {numpy} |
| scipy | {scipy} |
| platform | {platform} |
| BLAS threading during timing | single-threaded (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) |

Per-package pins are in `packages/<pkg>/requirements.txt`. The packages import `numpy`;
`scipy` is required because the repository's test collection and the audited historical
modules import it. `h5py` is imported by the archived TIDMAD path and is needed only if
a T22-S matched-stage run is given real data.

## UNRESOLVED — the receiving environment must supply these

Nothing below can be filled in from here without inventing it, so nothing below is
filled in. Record the resolved value of each in `RETURN_NOTES.md`.

| item | why it is unresolved |
|---|---|
| `<RUN_ROOT>` | a writable path on the remote machine; must be outside `results/` and `evidence/`, which the runners refuse |
| `<DATA_ROOT>` | the path of the extracted source snapshot on the remote machine (`PYTHONPATH`) |
| `<WORKERS>` | depends on available cores and on the filesystem's lock behaviour (see below) |
| scheduler, queue, account, partition | no scheduler is assumed; this archive contains no job script |
| wall-clock request, memory request, node count | derived from `<WORKERS>` and the site's policy, neither of which is known here |
| raw instrument data location | **partly resolved.** The archived CRESST per-channel residuals the CRESST-S package recomputes from are carried in `cresst_archive/` (registered evidence, digests in `MANIFEST.json`), so CRESST-S needs nothing external. The raw TIDMAD `.h5` files are **not** carried and are not in this repository; only a T22-S *matched-stage run on real data* needs them, and it refuses rather than fabricates when they are absent. The T22-S inference-calibration path is fully synthetic and needs no instrument bytes. |
| network filesystem lock behaviour | the ledger's advisory `flock` was verified on a **local disk only**. Until it is checked on the remote filesystem, run one worker per ledger, or one `<RUN_ROOT>` per worker |
| python/numpy/scipy versions available remotely | if they differ from the pins, run `verify_handoff.py --tests` before any production cell and report the versions |

## Read-only archived inputs carried in this archive

`results/t22_2026-09-11/` and `results/mlst_2026-09-10/` are copied in because the T22-S
matched-stage package reads them to build its **exposure inventory** — the record of
which instrument regions the earlier stages already examined. Without it, a remote run
could re-examine a spent hold-out without noticing. They are inputs, not outputs:
nothing writes to them, and the runners refuse a `<RUN_ROOT>` anywhere under `results/`.
Their digests are in `MANIFEST.json`, so a modified copy is detected by
`verify_handoff.py`.

## Resource projections

`packages/<pkg>/RESOURCE_ESTIMATE.json` holds, per package, a **measured** projection:
each package's dominant kernel was benchmarked with its own code at four or more problem
sizes that bracket or reach the production dimension, the scaling was fitted as
`t = a n^b` and reported beside the algorithm's exponent, and the per-cell seconds were
multiplied by the enumerated production matrix. `packages/<pkg>/BENCHMARK.json` holds the
raw timings, the memory probes and the fits; `RESOURCE_REPORT.md` prints them as tables
with the arithmetic shown.

Every family says whether its number is `measured` or `projected`; a projected one names
the fitted exponent and the extrapolation factor in the dimension. The totals are ranges
taken from the measured min/max over repeats and the fit residual, not from an invented
margin. The estimates include estimator calls, restarts, replicates and the reduction
step, and report peak memory, output storage, transfer size and setup time separately
from computation.

The old flat product (one fixture wall time times a constant declared in the protocol)
is kept only as `cross_check_flat_fixture_scale`, labelled as not being the estimate.
Where it disagrees with the measurement by more than a factor two, the estimate says so.

Wall-time rows at 1, 8 and 32 workers are **conditional**: the server's core count,
per-core speed, BLAS threading policy, filesystem behaviour and link bandwidth are not
known here and have not been invented. Each estimate carries its own `unknowns` list.

## Cost of a failure

Retries are charged and not refunded, and each protocol reserves retry headroom under
its fit cap. If the cap is reached the attempt is refused (exit 2) rather than the cap
being raised. Report an exhausted budget; do not edit the protocol.
"""

VERIFY_PY = r'''#!/usr/bin/env python3
"""Verify this extracted handoff before anything is computed.

Checks, in order:
  1. every file's SHA-256 against MANIFEST.json (and that no file is missing or extra);
  2. python version and the imports the packages need;
  3. for each package: `validate` succeeds, and `enumerate` reproduces the
     `matrix_sha256` and `cell_ids_sha256` recorded in packages/<pkg>/EXPECTED_CELLS.json;
  4. with --tests, the package test suite.

`enumerate` computes no cells and draws no data.  Exit 0 iff everything passes; any
mismatch is a reason to stop and report, not to continue into production cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND = "successor_20260913"


def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true", help="also run the package test suite")
    args = ap.parse_args()
    man = json.loads((HERE / "MANIFEST.json").read_text())
    problems = []

    print("[1/4] manifest")
    recorded = man["files"]
    for rel, digest in sorted(recorded.items()):
        p = HERE / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(p) != digest:
            problems.append(f"digest differs: {rel}")
    on_disk = {
        str(p.relative_to(HERE)) for p in HERE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
        and p.name not in ("MANIFEST.json",) and not p.name.endswith((".pyc", ".pyo"))
    }
    for extra in sorted(on_disk - set(recorded)):
        problems.append(f"not in manifest: {extra}")
    print(f"      {len(recorded)} files recorded, {len(problems)} problem(s)")

    print("[2/4] environment")
    print(f"      python {sys.version.split()[0]} (locally accepted on {man['environment']['python']})")
    for mod in ("numpy", "scipy"):
        try:
            m = __import__(mod)
            print(f"      {mod} {m.__version__} (locally accepted on {man['environment'][mod]})")
        except Exception as e:
            problems.append(f"cannot import {mod}: {e}")

    print("[3/4] validate and enumerate (computes no cells)")
    env = dict(os.environ, PYTHONPATH=str(HERE))
    cresst = HERE / "cresst_archive"
    if cresst.is_dir():
        env["CRESST_S_ARCHIVE_ROOT"] = str(cresst)
    with tempfile.TemporaryDirectory() as td:
        for pkg in man["packages"]:
            exp = json.loads((HERE / "packages" / pkg / "EXPECTED_CELLS.json").read_text())
            proto = HERE / "packages" / pkg / exp["protocol_file"]
            mod = f"experiments.mlst2026.{ROUND}.{pkg}"
            r = subprocess.run([sys.executable, "-m", mod, "validate", "--protocol", str(proto)],
                               cwd=HERE, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                problems.append(f"{pkg}: validate failed: {r.stderr.strip()[:200]}")
                continue
            out = Path(td) / pkg
            r = subprocess.run([sys.executable, "-m", mod, "enumerate",
                                "--protocol", str(proto), "--out", str(out)],
                               cwd=HERE, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                problems.append(f"{pkg}: enumerate failed: {r.stderr.strip()[:200]}")
                continue
            s = json.loads((out / "run_matrix_summary.json").read_text())
            rows = [json.loads(l) for l in (out / "run_matrix.jsonl").read_text().splitlines() if l]
            ids = hashlib.sha256("\n".join(sorted(r_["cell_id"] for r_ in rows)).encode()).hexdigest()
            ok = (s.get("matrix_sha256") == exp["matrix_sha256"]
                  and ids == exp["cell_ids_sha256"]
                  and s.get("cells") == exp["cells"] and s.get("fits") == exp["fits"])
            print(f"      {pkg:9s} cells={s.get('cells')} fits={s.get('fits')} "
                  f"matrix={str(s.get('matrix_sha256'))[:12]} {'ok' if ok else 'MISMATCH'}")
            if not ok:
                problems.append(f"{pkg}: enumerated matrix differs from EXPECTED_CELLS.json "
                                f"-- this is a different design; do not run it")

    if args.tests:
        print("[4/4] test suite")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                            f"tests/test_{ROUND}_common.py"]
                           + [f"tests/test_{ROUND}_{p}.py" for p in man["packages"]],
                           cwd=HERE, env=env, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        print("      " + last)
        if r.returncode != 0:
            problems.append("test suite failed")
        # A skipped scientific test is not a passing one.  The CRESST-S checks skip
        # when the archived inputs are unreachable; that must be visible here, because
        # otherwise a green summary line would stand for tests that never ran.
        if "skipped" in last:
            print("      note: skipped tests are tests that did NOT run.")
            if not cresst.is_dir():
                problems.append("cresst_archive/ is absent, so the CRESST-S scientific "
                                "tests skipped; this extraction has not been shown to "
                                "reproduce the archived CRESST recomputation")
    else:
        print("[4/4] test suite skipped (pass --tests to run it)")

    print()
    if problems:
        print(f"HANDOFF VERIFY: {len(problems)} problem(s)")
        for p in problems:
            print("  " + p)
        print("\nDo not run production cells until these are explained.")
        return 1
    print("HANDOFF VERIFY: ok -- manifest, imports and all five enumerated matrices match.")
    print("This verifies the transfer, not any scientific claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _package_table(matrices: dict) -> str:
    rows = ["| package | protocol file | `run` phase selectors, in the order its "
            "`REMOTE_COMMANDS.md` issues them | freeze written by the pilot reduce |",
            "|---|---|---|---|"]
    for pkg in PKGS:
        m = matrices[pkg]
        sels = "<br>".join(f"{i}. `{s}`" for i, s in enumerate(m["run_selectors"], 1))
        fz = f"`{m['freeze_file']}`" if m["freeze_file"] else "none"
        rows.append(f"| `{pkg}` | `{m['protocol_file']}` | {sels} | {fz} |")
    return "\n".join(rows)


def _freeze_rule(matrices: dict) -> str:
    have = [(p, matrices[p]["freeze_file"]) for p in PKGS if matrices[p]["freeze_file"]]
    none = [p for p in PKGS if not matrices[p]["freeze_file"]]
    s = ("A confirmation cell is refused, before anything is charged, until the pilot reduce "
         "has written that package's freeze for this protocol and this matrix: "
         + ", ".join(f"`{f}` for `{p}`" for p, f in have) + ".")
    if none:
        s += ("  " + ", ".join(f"`{p}`" for p in none)
              + " freezes nothing (no pilot-selected quantity), so it has no such gate;"
                " see its `REMOTE_COMMANDS.md`.")
    return textwrap.fill(s, width=88, subsequent_indent="  ")


def _freeze_return_list(matrices: dict) -> str:
    s = "; ".join(f"`{p}` -> " + (f"`{matrices[p]['freeze_file']}`" if matrices[p]["freeze_file"] else "none")
                  for p in PKGS)
    return textwrap.fill(s, width=88, initial_indent="   ", subsequent_indent="   ").lstrip()


def build(out_dir: Path, verify: bool, ms_evidence: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"mlst-{ROUND}-handoff-{stamp}"
    staging = out_dir / stem
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    commit = sh("git", "rev-parse", "HEAD")
    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD")
    dirty = sh("git", "status", "--porcelain", "--untracked-files=no")

    print(f"[handoff] {branch} @ {commit[:12]}" + ("  (WORKING TREE DIRTY)" if dirty else ""))

    print("[handoff] enumerating production matrices")
    matrices = {}
    with tempfile.TemporaryDirectory() as td:
        for pkg in PKGS:
            matrices[pkg] = enumerate_matrix(pkg, Path(td))
            m = matrices[pkg]
            print(f"           {pkg:9s} cells={m['cells']} fits={m['fits']} "
                  f"solves={m['solves']} matrix={str(m['matrix_sha256'])[:12]}")

    print("[handoff] copying source snapshot")
    copy_snapshot(staging)
    package_materials(staging, matrices)

    missing_cresst = copy_cresst_archive(staging, ms_evidence)
    if missing_cresst:
        print(f"[handoff] WARNING: {len(missing_cresst)} CRESST archive input(s) not found "
              f"under {ms_evidence}; CRESST-S scientific tests will SKIP in the extraction",
              file=sys.stderr)
        for rel in missing_cresst:
            print("           missing: " + rel, file=sys.stderr)
    else:
        print(f"[handoff] copied {len(CRESST_ARCHIVE_FILES)} registered CRESST inputs")

    # provenance: the round's own commits, not the whole 107 MB history
    prov = staging / "provenance"
    prov.mkdir()
    base = sh("git", "merge-base", "HEAD", "main") if _has_ref("main") else None
    (prov / "COMMIT").write_text(commit + "\n")
    (prov / "BRANCH").write_text(branch + "\n")
    if dirty:
        (prov / "WORKING_TREE_DIRTY.txt").write_text(dirty + "\n")
    if base:
        (prov / "round.diff").write_text(sh("git", "diff", f"{base}..HEAD"))
        (prov / "round.log").write_text(sh("git", "log", "--oneline", f"{base}..HEAD"))
        subprocess.run(["git", "bundle", "create", str(prov / "round.bundle"),
                        f"{base}..HEAD"], cwd=ROOT, check=True, capture_output=True)

    import numpy
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "not installed"
    envinfo = {"python": sys.version.split()[0], "numpy": numpy.__version__,
               "scipy": scipy_v, "platform": sys.platform}

    (staging / "ENVIRONMENT.md").write_text(ENVIRONMENT_MD.format(**envinfo))
    (staging / "verify_handoff.py").write_text(VERIFY_PY)
    (staging / "verify_handoff.py").chmod(0o755)

    rows = ["| package | cells | fits | solves | measured CPU-seconds (range, same cells) | "
            "wall h @ 8 workers | peak RSS/cell | output |",
            "|---|---|---|---|---|---|---|---|"]
    for pkg in PKGS:
        m = matrices[pkg]
        est = m.get("resource_estimate") or {}
        cpu = est.get("cpu_seconds") or {}
        w8 = (est.get("wall_time_hours_by_workers") or {}).get("8") or {}
        mem = ((est.get("peak_memory") or {}).get("per_cell_peak") or {}).get("rss_high_water_bytes")
        store = (est.get("storage") or {}).get("total_mb")
        lo, hi = cpu.get("total_low"), cpu.get("total_high")
        cpu_s = ("%.0f - %.0f" % (lo, hi)) if isinstance(lo, (int, float)) else "see RESOURCE_ESTIMATE.json"
        w8_s = ("%.2f - %.2f" % (w8["hours_low"], w8["hours_high"])) if w8 else "-"
        mem_s = ("%.0f MB" % (mem / 1e6)) if isinstance(mem, (int, float)) else "-"
        st_s = ("%.0f MB" % store) if isinstance(store, (int, float)) else "-"
        rows.append(f"| `{pkg}` | {m['cells']} | {m['fits']} | {m['solves']} | {cpu_s} | {w8_s} | "
                    f"{mem_s} | {st_s} |")
    rows.append("")
    rows.append("CPU-seconds are measured on the benchmark host recorded in each "
                "`RESOURCE_ESTIMATE.json` (`hardware`), single-threaded BLAS. The 8-worker column is "
                "**conditional**: this round does not know the server's core count, per-core speed or BLAS "
                "policy. See `RESOURCE_REPORT.md` for the per-package benchmark tables and the list of what "
                "remains unknown.")
    (staging / "HANDOFF.md").write_text(HANDOFF_MD.format(
        date=datetime.now(timezone.utc).strftime("%d %B %Y"),
        branch=branch, commit=commit, tarname=f"{stem}.tar.gz", stem=stem,
        expected_table="\n".join(rows),
        package_table=_package_table(matrices),
        freeze_rule=_freeze_rule(matrices),
        freeze_return_list=_freeze_return_list(matrices)))

    print("[handoff] writing manifest")
    files = {}
    for p in sorted(staging.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".pyc", ".pyo")):
            files[str(p.relative_to(staging))] = sha256_file(p)
    manifest = {
        "schema": "mlst-remote-handoff/1",
        "round": ROUND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": "noise-weighted-subspace-reconstruction",
        "source_branch": branch,
        "source_commit": commit,
        "source_working_tree_dirty": bool(dirty),
        "status": "ready to transfer",
        "status_note": "Local acceptance passed.  Nothing here has been executed, and no "
                       "scientific claim is supported by this archive.",
        "environment": envinfo,
        "packages": PKGS,
        "cresst_archive_present": not missing_cresst,
        "cresst_archive_missing": missing_cresst,
        "expected": {p: {k: matrices[p][k] for k in
                         ("cells", "fits", "solves", "matrix_sha256", "protocol_sha256",
                          "cell_ids_sha256")} for p in PKGS},
        "n_files": len(files),
        "files": files,
    }
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")

    tarpath = out_dir / f"{stem}.tar.gz"
    print(f"[handoff] writing {tarpath.name}")
    with tarfile.open(tarpath, "w:gz") as tf:
        tf.add(staging, arcname=stem)
    size_mb = tarpath.stat().st_size / 1048576
    print(f"[handoff] {tarpath} ({size_mb:.1f} MB, {len(files)} files)")

    if verify:
        print("\n[handoff] verifying from a clean extraction")
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(tarpath) as tf:
                tf.extractall(td)
            ex = Path(td) / stem
            r = subprocess.run([sys.executable, "verify_handoff.py", "--tests"],
                               cwd=ex, capture_output=True, text=True)
            print(r.stdout)
            if r.stderr.strip():
                print(r.stderr[-2000:], file=sys.stderr)
            if r.returncode != 0:
                print("[handoff] VERIFICATION FAILED", file=sys.stderr)
                return 1
    return 0


def _has_ref(name: str) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", name],
                          cwd=ROOT, capture_output=True).returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="directory to write the archive into")
    ap.add_argument("--verify", action="store_true",
                    help="extract into a clean temporary directory and run verify_handoff.py --tests")
    ap.add_argument("--ms-evidence", type=Path, default=DEFAULT_MS_EVIDENCE,
                    help="the manuscript repository's evidence/data root, source of the "
                         "registered CRESST inputs the CRESST-S package recomputes from")
    a = ap.parse_args(argv)
    return build(a.out.resolve(), a.verify, a.ms_evidence.resolve())


if __name__ == "__main__":
    sys.exit(main())
