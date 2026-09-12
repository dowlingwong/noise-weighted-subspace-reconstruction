"""Common CLI skeleton for the local-build packages.

Subcommands every package exposes::

    validate   --protocol P                       check protocol + matrix; no compute
    enumerate  --protocol P --out DIR             write run_matrix.jsonl; no compute
    run        --protocol P --out DIR [--roles ..] [--cells ..] [--fixture]
    reduce     --protocol P --out DIR [--allow-partial]
    estimate   --protocol P --out DIR             resource estimate from timings

Safety rules implemented here and inherited by all packages:

* ``--help``, ``validate`` and ``enumerate`` never fit anything;
* ``run`` requires an explicit ``--out``; it refuses any path under
  ``results/`` or ``evidence/`` of the repository (the frozen archives);
* ``--fixture`` forces the development seed namespace, marks every output
  ``development_only: true`` and refuses confirmation cells;
* exit codes: 0 ok, 1 failure/exception, 2 budget refusal, 3 reconciliation,
  4 schema, 5 duplicate completion, 6 protocol/usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from .ledger import BudgetExceeded, DuplicateCompletion, ExecutionLedger
from .matrix import RunMatrix, check_seed_roles
from .protocol import ProtocolError, load_protocol, namespace_of_seed, protocol_hash
from .results import (ReconciliationError, SchemaError, check_cell_schema, code_provenance, read_cells, reconcile,
                      require_reconciled, sha256_file, write_json_atomic)

REPO_ROOT = Path(__file__).resolve().parents[4]
FORBIDDEN_OUT_ROOTS = (REPO_ROOT / "results", REPO_ROOT / "evidence")


class UsageError(RuntimeError):
    exit_code = 6


def build_parser(package: str, description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"python -m experiments.mlst2026.successor_20260913.{package}", description=description)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("validate", "check the protocol and the enumerated matrix; computes nothing"),
                        ("enumerate", "write the run matrix and its counts; computes nothing"),
                        ("run", "execute matrix cells into --out (explicit root required)"),
                        ("reduce", "deterministically reduce a run root; refuses incomplete/duplicate runs unless told"),
                        ("estimate", "resource estimate from the matrix and the timings recorded in --out")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--protocol", required=True, type=Path, help="frozen protocol JSON")
        if name != "validate":
            sp.add_argument("--out", required=True, type=Path, help="run root (input for reduce/estimate, output for enumerate/run)")
        if name == "run":
            sp.add_argument("--roles", nargs="+", default=None, help="restrict to roles (development pilot confirmation control algebraic)")
            sp.add_argument("--cells", nargs="+", default=None, help="restrict to these cell ids")
            sp.add_argument("--families", nargs="+", default=None)
            sp.add_argument("--fixture", action="store_true", help="development-only reduced run: development seeds, small sizes, never confirmation")
            sp.add_argument("--allow-repeat", action="store_true", help="re-run an already completed cell (charged again, flagged)")
            sp.add_argument("--max-cells", type=int, default=None, help="stop after this many cells (partial run; reduce will say so)")
            sp.add_argument("--fail-cell", default=None, help="TEST ONLY: raise inside this cell id to exercise failure accounting")
        if name == "reduce":
            sp.add_argument("--allow-partial", action="store_true", help="reduce a run with missing cells (output is labelled partial)")
    return ap


def _refuse_forbidden_out(out: Path) -> None:
    r = out.resolve()
    for root in FORBIDDEN_OUT_ROOTS:
        try:
            r.relative_to(root.resolve())
        except ValueError:
            continue
        raise UsageError(f"refusing to write under the frozen archive root {root}; choose another --out")


def main_template(*, package: str, description: str, build_matrix, run_cell, reduce_run, result_required: dict,
                  source_files: list[Path], fixture_overrides=None, argv=None) -> int:
    """Generic main.  ``build_matrix(P, fixture: bool) -> RunMatrix``;
    ``run_cell(P, cell, fixture: bool, ctx: dict) -> (result: dict, fits_used, solves_used)``;
    ``reduce_run(P, matrix, cells: dict[cell_id -> dict], out: Path, partial: bool) -> dict``."""
    ap = build_parser(package, description)
    args = ap.parse_args(argv)
    try:
        return _dispatch(args, package=package, build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run,
                         result_required=result_required, source_files=source_files, fixture_overrides=fixture_overrides)
    except (ProtocolError, UsageError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 6
    except BudgetExceeded as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 2
    except DuplicateCompletion as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 5
    except ReconciliationError as e:
        print(f"reduce refused: {e}", file=sys.stderr)
        return 3
    except SchemaError as e:
        print(f"schema error: {e}", file=sys.stderr)
        return 4


def _dispatch(args, *, package, build_matrix, run_cell, reduce_run, result_required, source_files, fixture_overrides) -> int:
    P = load_protocol(args.protocol)
    if P["package"] != package:
        raise ProtocolError(f"protocol is for package {P['package']!r}, runner is {package!r}")
    fixture = bool(getattr(args, "fixture", False))
    if args.cmd in ("reduce", "estimate"):
        # the run root records whether it was built as a fixture; reduce with the same matrix
        summ_path = Path(args.out) / "run_matrix_summary.json"
        if summ_path.exists():
            fixture = bool(json.loads(summ_path.read_text()).get("fixture", False))
    matrix = build_matrix(P, fixture)
    problems = check_seed_roles(matrix, P)
    if problems:
        raise ProtocolError("seed/role problems: " + "; ".join(problems[:5]))
    counts = matrix.counts()
    cap = P["budget"]["estimator_fits_cap"]
    if args.cmd == "validate":
        print(json.dumps({"package": package, "protocol_sha256": P["_sha256"], "protocol_hash": protocol_hash(P),
                          "matrix_sha256": matrix.matrix_hash(), "counts": counts, "cap": cap,
                          "fits_within_cap": counts["fits"] <= cap}, indent=1))
        if counts["fits"] > cap and not fixture:
            raise ProtocolError(f"matrix plans {counts['fits']} fits above the cap {cap}")
        return 0
    out = Path(args.out)
    if args.cmd in ("enumerate", "run"):
        _refuse_forbidden_out(out)
    if args.cmd == "enumerate":
        summ = matrix.write(out, extra={"fixture": fixture})
        write_json_atomic(out / "protocol_snapshot.json", {k: v for k, v in P.items()})
        print(json.dumps(summ, indent=1))
        return 0
    if args.cmd == "run":
        return _run(args, P, matrix, out, package, run_cell, result_required, source_files, fixture, fixture_overrides)
    if args.cmd == "reduce":
        return _reduce(args, P, matrix, out, package, reduce_run, result_required)
    if args.cmd == "estimate":
        # timings come from the run root (a fixture); the priced matrix is always the production one
        return _estimate(P, build_matrix(P, False), out, fixture_matrix=matrix if fixture else None)
    raise UsageError(f"unknown command {args.cmd}")


def _run(args, P, matrix, out, package, run_cell, result_required, source_files, fixture, fixture_overrides) -> int:
    if fixture:
        # a fixture run covers the development cells plus the (cheap) controls and algebraic checks
        sub = matrix.subset(roles=["development", "control", "algebraic"])
        if args.roles and any(r in ("pilot", "confirmation") for r in args.roles):
            raise UsageError("a fixture run may not touch pilot or confirmation cells")
    else:
        sub = matrix
    sub = sub.subset(roles=args.roles, cell_ids=args.cells, families=args.families) if (args.roles or args.cells or args.families) else sub
    if fixture:
        sub = sub.subset(roles=[r for r in ("development", "control", "algebraic") if any(c["role"] == r for c in sub.cells)])
        bad = [c for c in sub.cells if c["seed"] is not None and namespace_of_seed(P, c["seed"]) != "development"]
        if bad:
            raise UsageError(f"fixture run selected {len(bad)} cells outside the development seed namespace")
    if not sub.cells:
        raise UsageError("no cells selected")
    out.mkdir(parents=True, exist_ok=True)
    # the matrix on disk is the matrix of record for this run root
    if (out / "run_matrix_summary.json").exists():
        prev = RunMatrix.read(out)
        if prev.matrix_hash() != matrix.matrix_hash():
            raise UsageError("the run root holds a different matrix (hash mismatch); use a new --out for a new design")
    else:
        matrix.write(out, extra={"fixture": fixture})
        write_json_atomic(out / "protocol_snapshot.json", {k: v for k, v in P.items()})
    ledger = ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=P["budget"]["estimator_fits_cap"], package=package, protocol_sha256=P["_sha256"])
    swept = ledger.sweep_interrupted()
    if swept:
        print(f"swept {swept} interrupted attempt(s) into the ledger (charged, not refunded)")
    prov = code_provenance(source_files, REPO_ROOT)
    ctx = {"fixture_overrides": fixture_overrides(P) if (fixture and fixture_overrides) else None, "out": out}
    done = ledger.completed_by_cell()
    n_run = 0
    for cell in sub.cells:
        if cell["cell_id"] in done and not args.allow_repeat:
            print(f"skip {cell['cell_id']} ({cell['role']}): already completed")
            continue
        if args.max_cells is not None and n_run >= args.max_cells:
            print(f"stopping after {n_run} cells (--max-cells); the run is partial")
            break
        u = ledger.start(cell, allow_repeat=args.allow_repeat)
        t0 = time.time()
        try:
            if args.fail_cell and cell["cell_id"] == args.fail_cell:
                raise RuntimeError("injected failure (--fail-cell)")
            result, fits_used, solves_used = run_cell(P, cell, fixture, ctx)
            js = {"cell_id": cell["cell_id"], "attempt_uuid": u, "role": cell["role"], "seed": cell["seed"], "family": cell["family"],
                  "params": cell["params"], "fits_used": int(fits_used), "solves_used": int(solves_used), "result": result,
                  "development_only": bool(fixture), "wall_s": time.time() - t0, "provenance": {**prov, "protocol_sha256": P["_sha256"],
                  "matrix_sha256": matrix.matrix_hash(), "fixture_overrides": ctx["fixture_overrides"]}}
            check_cell_schema(js, result_required, where=cell["cell_id"])
            if int(fits_used) != int(cell["n_fits"]) or int(solves_used) != int(cell.get("n_solves", 0)):
                raise SchemaError(f"{cell['cell_id']}: accounting mismatch: planned fits/solves {cell['n_fits']}/{cell.get('n_solves', 0)} "
                                  f"but the cell reports {fits_used}/{solves_used}")
            path = out / "cells" / f"{cell['cell_id']}.json"
            if path.exists() and not args.allow_repeat:
                raise DuplicateCompletion(f"{path} already exists")
            digest = write_json_atomic(path, js)
            ledger.complete(u, cell, fits_used=fits_used, solves_used=solves_used, output_path=path, output_sha256=digest, wall_s=time.time() - t0)
            n_run += 1
            print(f"done {cell['cell_id']} ({cell['role']}, {cell['family']}) fits={fits_used} {time.time()-t0:.2f}s", flush=True)
        except (KeyboardInterrupt, SystemExit):
            ledger.fail(u, cell, reason="interrupted by signal", wall_s=time.time() - t0)
            raise
        except SchemaError:
            ledger.fail(u, cell, reason="schema/accounting error: " + traceback.format_exc(limit=1), wall_s=time.time() - t0)
            raise
        except Exception:
            ledger.fail(u, cell, reason=traceback.format_exc(limit=3), wall_s=time.time() - t0)
            print(f"FAILED {cell['cell_id']}: see ledger", file=sys.stderr, flush=True)
            return 1
    write_json_atomic(out / "LEDGER_SUMMARY.json", ledger.summary())
    print(json.dumps(ledger.summary(), indent=1))
    return 0


def _reduce(args, P, matrix, out, package, reduce_run, result_required) -> int:
    ledger = ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=P["budget"]["estimator_fits_cap"], package=package, protocol_sha256=P["_sha256"])
    if not (out / "run_matrix_summary.json").exists():
        raise ReconciliationError(f"{out} has no run matrix; nothing to reduce")
    on_disk = RunMatrix.read(out)
    if on_disk.matrix_hash() != matrix.matrix_hash():
        raise ReconciliationError("the protocol's matrix and the run root's matrix differ")
    observed = read_cells(out)
    # only reconcile roles that were part of this run root (fixture runs hold development cells only)
    roles_present = sorted({js["role"] for files in observed.values() for _, js in files})
    fixture_run = any(js.get("development_only") for files in observed.values() for _, js in files)
    roles = ["development", "control", "algebraic"] if fixture_run else None
    if fixture_run and any(r not in ("development", "control", "algebraic") for r in roles_present):
        raise ReconciliationError("a run root mixes development-only outputs with pilot/confirmation outputs")
    report = reconcile(matrix.cells, observed, ledger_completed=ledger.completed_by_cell(), roles=roles)
    if fixture_run:
        # a fixture may legitimately cover a subset of the development cells
        report["missing"] = [] if args.allow_partial else report["missing"]
    require_reconciled(report, allow_partial=args.allow_partial)
    cells = {}
    for k, files in observed.items():
        p, js = files[0]
        check_cell_schema(js, result_required, where=str(p))
        if sha256_file(p) != _ledger_digest(ledger, js["attempt_uuid"]):
            raise ReconciliationError(f"{p}: bytes differ from the digest recorded at completion")
        cells[k] = js
    summary = reduce_run(P, matrix, cells, out, bool(report["missing"]))
    summary = {"package": package, "protocol_sha256": P["_sha256"], "matrix_sha256": matrix.matrix_hash(),
               "reconciliation": report, "ledger": ledger.summary(), "partial": bool(report["missing"]),
               "development_only": fixture_run, **summary}
    write_json_atomic(out / "REDUCED.json", summary)
    print(json.dumps({k: summary[k] for k in ("package", "partial", "development_only", "reconciliation")}, indent=1))
    return 0


def _ledger_digest(ledger: ExecutionLedger, attempt_uuid: str) -> str | None:
    a = ledger.attempts().get(attempt_uuid)
    return a.get("output_sha256") if a else None


def _estimate(P, matrix, out, *, fixture_matrix=None) -> int:
    """Resource estimate: measured per-cell wall time by family from whatever
    cells exist under --out (a fixture run), scaled by the declared
    ``budget.fixture_to_production_scale`` per family, times the PRODUCTION
    matrix (``matrix``).  Nothing here is a guess: absent timings are reported
    as absent, and the root's own (fixture) counts are recorded beside the
    production counts so the two are never confused."""
    observed = read_cells(out)
    timings: dict[str, list[float]] = {}
    for files in observed.values():
        for _, js in files:
            timings.setdefault(js["family"] or "-", []).append(float(js.get("wall_s", 0.0)))
    scale = P["budget"].get("fixture_to_production_scale", {})
    est = {"families": {}, "total_cpu_seconds_point": 0.0, "unmeasured_families": []}
    for fam, info in matrix.counts()["by_family"].items():
        t = timings.get(fam)
        if not t:
            est["unmeasured_families"].append(fam)
            est["families"][fam] = {"cells": info["cells"], "measured": False}
            continue
        per = float(sum(t) / len(t)) * float(scale.get(fam, scale.get("default", 1.0)))
        est["families"][fam] = {"cells": info["cells"], "fits": info["fits"], "solves": info["solves"], "measured": True,
                                "fixture_seconds_per_cell": float(sum(t) / len(t)), "scale": float(scale.get(fam, scale.get("default", 1.0))),
                                "production_seconds_per_cell": per, "production_cpu_seconds": per * info["cells"]}
        est["total_cpu_seconds_point"] += per * info["cells"]
    est["counts"] = matrix.counts()
    est["priced_matrix"] = "production"
    est["timing_source_matrix"] = "fixture" if fixture_matrix is not None else "production"
    if fixture_matrix is not None:
        est["fixture_counts"] = fixture_matrix.counts()
    est["retry_allowance"] = {"policy": P["budget"].get("retry_policy", "unspecified"),
                              "fits_headroom_under_cap": P["budget"]["estimator_fits_cap"] - matrix.counts()["fits"]}
    write_json_atomic(out / "RESOURCE_ESTIMATE.json", est)
    print(json.dumps(est, indent=1))
    return 0
