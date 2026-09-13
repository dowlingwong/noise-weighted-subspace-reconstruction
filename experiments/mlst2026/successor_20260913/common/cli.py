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
* fit accounting is *observed*: every cell runs inside a ``common.fitcount``
  counting region, the packages count each estimator invocation at its call
  site, and the cell is refused (exit 4) unless the declared count, the
  reported count and the instrumented count all agree.  The observed
  breakdown is written into the cell file as ``fits_observed``;
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

from . import fitcount
from .ledger import BudgetExceeded, DuplicateCompletion, ExecutionLedger
from .matrix import ROLES, RunMatrix, check_seed_roles
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
            # Estimator invocations are instrumented at the call sites (common.fitcount),
            # so the count the cell reports is observed rather than restated from the
            # constant that produced the matrix declaration.
            with fitcount.counting() as observed_fits:
                result, fits_used, solves_used = run_cell(P, cell, fixture, ctx)
            js = {"cell_id": cell["cell_id"], "attempt_uuid": u, "role": cell["role"], "seed": cell["seed"], "family": cell["family"],
                  "params": cell["params"], "fits_used": int(fits_used), "solves_used": int(solves_used), "result": result,
                  "fits_observed": observed_fits.as_dict(),
                  "development_only": bool(fixture), "wall_s": time.time() - t0, "provenance": {**prov, "protocol_sha256": P["_sha256"],
                  "matrix_sha256": matrix.matrix_hash(), "fixture_overrides": ctx["fixture_overrides"]}}
            check_cell_schema(js, result_required, where=cell["cell_id"])
            if int(fits_used) != int(observed_fits.total):
                raise SchemaError(f"{cell['cell_id']}: instrumentation mismatch: the cell reports {fits_used} fits but "
                                  f"{observed_fits.total} estimator invocations were instrumented "
                                  f"({observed_fits.as_dict()['by_label']}); the reported count must be the observed one")
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


def _production_only(matrix: RunMatrix) -> RunMatrix:
    """The cells a remote production run will execute: everything but ``development``.

    A resource projection and the cell counts printed beside it must cover the
    same set of cells.  Development cells are fixture work, priced at the
    production scale they were never meant to carry, so they are excluded here
    and the exclusion is recorded in the estimate."""
    return matrix.subset(roles=[r for r in ROLES if r != "development"])


def _flat_cross_check(P, matrix, out) -> dict:
    """The OLD estimate, kept only as a cross-check.

    It is one *fixture* per-cell wall time multiplied by the constant
    ``budget.fixture_to_production_scale`` declared in the protocol.  That is a
    linear extrapolation of a tiny fixture with no account of the production
    dimension or of the algorithmic scaling of the kernel, which is exactly what
    the governing brief forbids as an estimate.  It is retained because a large
    disagreement between it and the measurement is itself informative: it says
    how wrong the declared scale factor was.
    """
    observed = read_cells(out)
    timings: dict[str, list[float]] = {}
    for files in observed.values():
        for _, js in files:
            timings.setdefault(js["family"] or "-", []).append(float(js.get("wall_s", 0.0)))
    scale = P["budget"].get("fixture_to_production_scale", {})
    fams: dict[str, dict] = {}
    total = 0.0
    unmeasured = []
    for fam, info in matrix.counts()["by_family"].items():
        t = timings.get(fam)
        if not t:
            unmeasured.append(fam)
            fams[fam] = {"cells": info["cells"], "measured": False}
            continue
        sc = float(scale.get(fam, scale.get("default", 1.0)))
        per = float(sum(t) / len(t)) * sc
        fams[fam] = {"cells": info["cells"], "fits": info["fits"], "solves": info["solves"], "measured": True,
                     "fixture_seconds_per_cell": float(sum(t) / len(t)), "fixture_cells_timed": len(t),
                     "scale": sc, "production_seconds_per_cell": per,
                     "production_cpu_seconds": per * info["cells"]}
        total += per * info["cells"]
    return {"what_this_is": ("NOT the estimate.  One fixture wall time per family multiplied by the constant "
                            "budget.fixture_to_production_scale, i.e. a flat linear extrapolation of the "
                            "fixture with no dimension and no algorithmic scaling in it.  Kept so that the "
                            "size of its error against the measurement is visible."),
            "families": fams, "total_cpu_seconds_point": total, "unmeasured_families": unmeasured}


def _load_benchmark(P) -> tuple[dict | None, str]:
    """The measured benchmark that lives beside the protocol.  Absent is reported,
    never silently replaced by a guess."""
    path = Path(P["_path"]).resolve().parent / "BENCHMARK.json"
    try:
        shown = str(path.relative_to(REPO_ROOT))
    except ValueError:
        shown = str(path)
    if not path.exists():
        return None, shown
    return json.loads(path.read_text()), shown


def _kernel_summary(k: dict) -> dict:
    """The part of a kernel benchmark that belongs in the estimate: the measured
    points, the fitted exponent beside the theoretical one, and how the production
    number was obtained."""
    out = {"kernel": k.get("kernel"), "call_path": k.get("call_path", ""),
           "size_symbol": k.get("size_symbol"), "notes": k.get("notes", ""),
           "points": [{"size": r.get("size"), "median_s": r.get("median_s"), "min_s": r.get("min_s"),
                       "max_s": r.get("max_s"), "repeats": r.get("repeats"),
                       "tracemalloc_peak_bytes": (r.get("memory") or {}).get("tracemalloc_peak_bytes"),
                       "rss_high_water_bytes": (r.get("memory") or {}).get("rss_high_water_bytes"),
                       "memory_measured_in": r.get("memory_measured_in")}
                      for r in k.get("points", [])]}
    if "fit" in k:
        out["fit"] = {kk: k["fit"][kk] for kk in ("a", "b", "residual_log_rms", "max_abs_log_residual",
                                                  "r_squared", "n_points") if kk in k["fit"]}
    if "theory" in k:
        out["theoretical_exponent"] = k["theory"]["exponent"]
        out["theory_note"] = k["theory"]["note"]
    if "exponent_check" in k:
        out["exponent_check"] = k["exponent_check"]
    if "projection" in k:
        out["projection"] = k["projection"]
    return out


def _reduce_projection(bench: dict, counts: dict, pilot_cells: int) -> dict:
    """Reduction work.  ``reduce`` reads every cell file of the run root, so its
    cost is priced per cell at the measured rate, and the number of times it runs
    is the number of reduces the protocol's flow performs."""
    r = (bench or {}).get("reduce") or {}
    if not r.get("measured"):
        return {"measured": False, "why": r.get("why", "no fixture run root was available to time reduce")}
    per = float(r.get("seconds_per_cell") or 0.0)
    n_all = int(counts["cells"])
    runs = [("pilot reduce", pilot_cells)] if pilot_cells else []
    runs.append(("final reduce over the whole run root", n_all))
    total = sum(per * n for _, n in runs)
    return {"measured": True,
            "measured_on_cells": r.get("cells_read"),
            "measured_seconds_of_reduce_work": r.get("wall_seconds_reduce_work"),
            "interpreter_and_import_seconds": r.get("interpreter_and_import_seconds_measured"),
            "seconds_per_cell_measured": per,
            "throughput_bytes_per_second_measured": r.get("throughput_bytes_per_second"),
            "bound": r.get("bound"), "bound_note": r.get("bound_note", ""),
            "reduces_priced": [{"what": w, "cells": n, "seconds": per * n} for w, n in runs],
            "total_seconds_projected": total,
            "how": ("measured seconds per cell on the fixture run root (%s cells) x the production cell count; "
                    "linear because reduce reads each cell file exactly once"
                    % r.get("cells_read")),
            "basis": "projected from a measured per-cell rate"}


def _storage_projection(bench: dict, matrix, counts: dict) -> dict:
    by_fam = ((bench or {}).get("storage") or {}).get("bytes_per_cell_by_family") or {}
    r = (bench or {}).get("reduce") or {}
    rows = {}
    total = 0
    for fam, info in counts["by_family"].items():
        e = by_fam.get(fam)
        if not e:
            rows[fam] = {"cells": info["cells"], "bytes_per_cell": None, "basis": "unmeasured"}
            continue
        b = int(e["bytes"]) * int(info["cells"])
        total += b
        rows[fam] = {"cells": info["cells"], "bytes_per_cell": int(e["bytes"]), "bytes_total": b,
                     "basis": e.get("basis", "measured"), "how": e.get("how", "")}
    ledger_per_cell = float(r.get("ledger_bytes_per_cell") or 0.0)
    ledger = ledger_per_cell * counts["cells"] * 2   # one start line and one completion line per attempt
    fx_cells = int(r.get("cells_read") or 0)
    fx_reduced = float(r.get("reduced_output_bytes") or 0.0)
    reduced = (fx_reduced / fx_cells * counts["cells"]) if fx_cells else None
    grand = total + ledger + (reduced or 0.0)
    return {"cells": rows,
            "cell_files_bytes_total": total,
            "ledger_bytes_projected": ledger,
            "ledger_basis": ("measured %.0f bytes per completed cell on the fixture ledger, x 2 because every "
                             "attempt writes a start line and a completion line" % ledger_per_cell),
            "reduced_output_bytes_projected": reduced,
            "reduced_basis": ("projected linearly from the measured fixture REDUCED.json (%.0f bytes over %d "
                              "cells)" % (fx_reduced, fx_cells)) if fx_cells else "unmeasured",
            "total_bytes": grand, "total_mb": grand / 1e6,
            "note": "cell-file bytes are MEASURED on a production-configuration result object per family; the "
                    "multiplication by the cell count and the ledger/reduced lines are projections"}


def _peak_memory(bench: dict) -> dict:
    """Peak memory per cell at the production dimension, from the benchmark."""
    best = None
    rows = []
    for name, k in ((bench or {}).get("kernels") or {}).items():
        for r in k.get("points", []):
            m = r.get("memory")
            if not m:
                continue
            row = {"kernel": name, "size": r.get("size"),
                   "measured_in": r.get("memory_measured_in", "this process"),
                   "tracemalloc_peak_bytes": m.get("tracemalloc_peak_bytes"),
                   "rss_high_water_bytes": m.get("rss_high_water_bytes")}
            rows.append(row)
            # a fresh-subprocess measurement is the only one that belongs to a single
            # cell, so it is preferred over an in-process high-water mark even when the
            # in-process number is larger (that number includes everything the
            # benchmark process did before it)
            key = (row["measured_in"] == "subprocess", row["rss_high_water_bytes"] or 0)
            if best is None or key > (best["measured_in"] == "subprocess", best["rss_high_water_bytes"] or 0):
                best = row
    if best is None:
        return {"measured": False, "why": "no memory point in BENCHMARK.json"}
    return {"measured": True, "per_cell_peak": best, "all_points": rows,
            "per_cell_peak_chosen_by": ("the largest fresh-subprocess RSS measurement among the kernels, "
                                        "because only a fresh process attributes the high-water mark to one "
                                        "cell; in-process points are listed in all_points and are upper "
                                        "bounds on the whole benchmark process, not on one cell"),
            "tracemalloc_is": "peak Python-allocator bytes attributable to the call",
            "rss_is": ("ru_maxrss of the process running the cell, including the interpreter and numpy; "
                       "measured in a fresh subprocess where 'measured_in' says 'subprocess', which is the "
                       "only way the high-water mark belongs to one problem size"),
            "per_worker_note": "one worker holds one cell, so W workers need about W x the RSS figure above; "
                               "this machine has 7 GiB, the server's memory is UNKNOWN"}


def _estimate(P, matrix, out, *, fixture_matrix=None) -> int:
    """Resource estimate for the PRODUCTION matrix, from measured, scaling-aware
    benchmarks of the package's own kernels.

    The headline is ``cpu_seconds`` / ``wall_time_hours_by_workers``: per family,
    the number of production cells times the seconds per cell that
    ``BENCHMARK.json`` measured at (or projected to, with a fitted exponent) the
    production dimension.  Every family says whether its number is ``measured``
    or ``projected``, and a projected one carries the fitted exponent and the
    extrapolation factor in the dimension it used.

    The old flat product (one fixture time x a constant declared in the protocol)
    is still computed, but only under ``cross_check_flat_fixture_scale`` and
    labelled as not being the estimate.

    The priced set is the production matrix with ``role == development`` removed.
    Development cells are fixture work and are never transferred, so pricing them
    would make the projected total cover a different set of cells from the counts
    printed beside it.  ``priced_counts`` is the set the total covers, and it is
    what the handoff prints.
    """
    matrix = _production_only(matrix)
    counts = matrix.counts()
    bench, bench_path = _load_benchmark(P)
    flat = _flat_cross_check(P, matrix, out)

    est: dict = {
        "schema": "mlst-local-package-resource-estimate/2",
        "package": P["package"],
        "estimate_basis": (
            "measured benchmarks of the package's own kernels at four or more problem sizes that bracket or "
            "reach the production dimension, fitted as t = a n^b and reported beside the algorithm's "
            "exponent, multiplied by the enumerated production matrix.  NOT a fixture time multiplied by a "
            "declared constant; that product is kept separately under "
            "'cross_check_flat_fixture_scale' and is not the estimate."),
        "hardware": (bench or {}).get("hardware", "UNKNOWN: BENCHMARK.json is absent"),
        "thread_settings": (bench or {}).get("thread_settings"),
        "benchmark_source": {"path": bench_path, "present": bench is not None,
                             "generated_utc": (bench or {}).get("generated_utc"),
                             "benchmark_wall_seconds": (bench or {}).get("benchmark_wall_seconds")},
    }
    if bench is None:
        est["measured_projection_available"] = False
        est["why_no_measured_projection"] = (
            f"no BENCHMARK.json beside the protocol at {bench_path}; run "
            "experiments/mlst2026/successor_20260913/run_benchmarks.py to produce one.  Only the flat "
            "cross-check is available, and a flat cross-check is not an estimate.")
    else:
        est["measured_projection_available"] = True

    bfam = ((bench or {}).get("families") or {})
    fams: dict[str, dict] = {}
    unmeasured: list[str] = []
    lo = mid = hi = 0.0
    for fam, info in counts["by_family"].items():
        e = bfam.get(fam)
        if not e:
            unmeasured.append(fam)
            fams[fam] = {"cells": info["cells"], "fits": info["fits"], "solves": info["solves"],
                         "measured": False,
                         "why": "no benchmark for this family in BENCHMARK.json"}
            continue
        s = e["seconds_per_cell"]
        row = {"cells": info["cells"], "fits": info["fits"], "solves": info["solves"], "measured": True,
               "basis": e.get("basis"), "kernel": e.get("kernel"),
               "calls_per_cell": e.get("calls_per_cell"),
               "benchmark_size": e.get("size"), "size_symbol": e.get("size_symbol"),
               "fitted_exponent": e.get("fitted_exponent"),
               "theoretical_exponent": e.get("theoretical_exponent"),
               "extrapolation_factor_in_dimension": e.get("extrapolation_factor_in_dimension"),
               "seconds_per_cell_median": s["median"], "seconds_per_cell_low": s["low"],
               "seconds_per_cell_high": s["high"],
               "cpu_seconds_point": s["median"] * info["cells"],
               "cpu_seconds_low": s["low"] * info["cells"],
               "cpu_seconds_high": s["high"] * info["cells"],
               "how": e.get("how", ""), "projection_label": e.get("projection_label", "")}
        fams[fam] = row
        lo += row["cpu_seconds_low"]
        mid += row["cpu_seconds_point"]
        hi += row["cpu_seconds_high"]
    est["families"] = fams
    est["unmeasured_families"] = unmeasured
    est["per_cell_central_statistic"] = (
        "seconds_per_cell_median is the measured median over repeats where the family's cells are identical, "
        "and the mean over the family's own cells of their measured medians where they are not (t20s N grid, "
        "cresst_s constructions / reports / ranks / designs, t22s n_files census).  The low/high bounds are "
        "the corresponding measured minima and maxima; they are measurements, not margins.")

    pilot_cells = int(counts["by_role"].get("pilot", {}).get("cells", 0))
    red = _reduce_projection(bench, counts, pilot_cells)
    est["reduction_work"] = red
    reduce_s = float(red.get("total_seconds_projected") or 0.0)

    est["cpu_seconds"] = {
        "cell_compute_low": lo, "cell_compute_point": mid, "cell_compute_high": hi,
        "reduction": reduce_s,
        "total_low": lo + reduce_s, "total_point": mid + reduce_s, "total_high": hi + reduce_s,
        "excludes": "transfer, environment setup and per-worker interpreter start-up, which are reported "
                    "separately below",
    }
    est["total_cpu_seconds_point"] = mid + reduce_s
    est["total_cpu_hours_point"] = (mid + reduce_s) / 3600.0

    workers = (1, 8, 32)
    est["parallelism_assumption"] = (
        "cells are independent processes that share only the append-only ledger (advisory flock), so the "
        "cell compute is taken as perfectly parallel over W workers; the reduction is a single process and "
        "is added serially.  wall = cell_compute / W + reduction.")
    est["wall_time_hours_by_workers"] = {
        str(w): {"workers": w,
                 "hours_low": (lo / w + reduce_s) / 3600.0,
                 "hours_high": (hi / w + reduce_s) / 3600.0,
                 "hours_point": (mid / w + reduce_s) / 3600.0,
                 "conditional": ("CONDITIONAL on the server having %d usable cores at a per-core speed "
                                 "comparable to the benchmark Xeon.  The server specification is NOT known; "
                                 "this is not a claim that %d cores exist." % (w, w))}
        for w in workers}

    wa = dict((bench or {}).get("work_accounting") or {})
    wa.update({
        "priced_cells": counts["cells"],
        "estimator_fits_planned_from_matrix": counts["fits"],
        "solves_planned_from_matrix": counts["solves"],
        "cells_by_role": {r: v["cells"] for r, v in counts["by_role"].items()},
        "cells_by_family": {f: v["cells"] for f, v in counts["by_family"].items()},
    })
    est["work_accounting"] = wa
    est["peak_memory"] = _peak_memory(bench)
    est["storage"] = _storage_projection(bench, matrix, counts)

    setup = ((bench or {}).get("setup_and_transfer") or {})
    est["transfer"] = dict(setup.get("transfer") or {"archive_bytes": None})
    est["transfer"]["separate_from_computation"] = True
    est["setup"] = {k: v for k, v in setup.items() if k != "transfer"}
    est["setup"]["separate_from_computation"] = True

    est["unknowns"] = [
        "server core count (the wall-time rows are conditional on 1, 8 and 32; none of them is asserted)",
        "server per-core speed relative to the benchmark Intel Xeon @ 2.10GHz",
        "server BLAS threading policy (these numbers are single-threaded; a multithreaded BLAS changes the "
        "per-cell time and the useful worker count in opposite directions)",
        "server filesystem speed and whether advisory flock on the ledger works there (a network filesystem "
        "falls back to serial ledger appends, which caps useful parallelism)",
        "link bandwidth for the transfer (only the archive size is known)",
        "whether the pinned requirements are already installed on the server",
        "server memory per worker (the per-cell RSS figures above are measured; the budget is not known)",
    ]

    est["cross_check_flat_fixture_scale"] = flat
    if flat["total_cpu_seconds_point"] > 0 and mid > 0:
        est["cross_check_flat_fixture_scale"]["measured_over_flat_ratio"] = mid / flat["total_cpu_seconds_point"]
        est["cross_check_flat_fixture_scale"]["verdict"] = (
            "the declared flat scale is within a factor 2 of the measurement"
            if 0.5 <= mid / flat["total_cpu_seconds_point"] <= 2.0 else
            "the declared flat scale disagrees with the measurement by a factor %.3g; the flat factor is "
            "wrong, not the measurement" % (max(mid, flat["total_cpu_seconds_point"])
                                            / max(1e-12, min(mid, flat["total_cpu_seconds_point"]))))

    est["counts"] = counts
    est["priced_counts"] = counts
    est["priced_matrix"] = "production"
    est["priced_roles"] = sorted({c["role"] for c in matrix.cells})
    est["priced_cells_note"] = ("the total covers exactly the cells in 'priced_counts': the production matrix "
                                "with role 'development' (fixture-only work, never transferred) excluded")
    est["timing_source_matrix"] = "fixture" if fixture_matrix is not None else "production"
    if fixture_matrix is not None:
        est["fixture_counts"] = fixture_matrix.counts()
    est["retry_allowance"] = {"policy": P["budget"].get("retry_policy", "unspecified"),
                              "fits_headroom_under_cap": P["budget"]["estimator_fits_cap"] - counts["fits"],
                              "cpu_seconds_per_retried_cell": ("the family's seconds_per_cell above; a retry "
                                                               "costs a whole cell and is charged at start")}
    if bench is not None:
        est["kernels"] = {k: _kernel_summary(v) for k, v in (bench.get("kernels") or {}).items()}
        est["benchmark_notes"] = bench.get("notes", [])
    write_json_atomic(out / "RESOURCE_ESTIMATE.json", est)
    print(json.dumps({k: est[k] for k in ("package", "estimate_basis", "cpu_seconds",
                                          "wall_time_hours_by_workers", "unmeasured_families")
                      if k in est}, indent=1))
    return 0
