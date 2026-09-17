"""R4 runner: T22 feasibility with a separate gate question (new protocol).

Fixed-design package: **no selection authority**, hence no pilot freeze.  Every
cell is a predeclared scenario at a declared evaluation-file count (module a) or
at a declared development design (module b).  The reduce reports:

* module (a) feasibility by ``n_files`` with the exact bound
  ``P(supported) <= 1 - P(withheld)`` evaluated before any large-file extension;
* module (b) withholding / coverage by development design ``(n_dev_files,
  n_cal)``; and
* controls.

Increasing only evaluation files cannot lower withholding: the gate is computed
from development files.  The bound screening makes that explicit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026.successor_20260913.common import cli  # noqa: E402
from experiments.mlst2026.successor_20260913.common.matrix import RunMatrix, make_cell  # noqa: E402
from experiments.mlst2026.successor_20260913.common.protocol import ProtocolError  # noqa: E402
from experiments.mlst2026.successor_20260913.common.results import SchemaError, write_json_atomic  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import pipeline as pl  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import simulator as sim  # noqa: E402

from . import analysis as an  # noqa: E402
from . import r1_gate as gate  # noqa: E402
from . import streams as st  # noqa: E402

PACKAGE = "r4_t22_feasibility"
#: v2 separates the development and evaluation designs and enforces the R1 gate
#: at the run boundary; v1 is retained unchanged.
PROTOCOL_DEFAULT = _HERE.parent / "PROTOCOL_R4_T22_FEASIBILITY_v2.json"
#: The complete scientific source closure: the runner and its analysis/streams,
#: the R1 gate and the R1 candidate family it applies, and every archived
#: simulator/pipeline/CLI module R4 imports transitively.
SOURCE_FILES = [_HERE, _HERE.parent / "analysis.py", _HERE.parent / "streams.py", _HERE.parent / "r1_gate.py",
                _ROOT / "experiments" / "mlst2026" / "followup_20260917" / "r1_t22_calibration" / "candidates.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "t22s" / "pipeline.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "t22s" / "simulator.py",
                _ROOT / "experiments" / "mlst2026" / "t22_tidmad_narrowband.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "common" / "cli.py"]
FAMILIES = ("eval_grid", "dev_grid", "control")
result_required = {"kind": str, "passed": bool, "checks": dict, "summary": dict}
CLAIMS = pl.CLAIMS


CAL_N = (64, 256)
#: The EVALUATION calibration length.  It is fixed here and is never varied by a
#: development-design change: development ``n_cal`` is a separate parameter.
CAL_EVAL = 64
DEV_N = (4, 8, 16)


def _sc(effect: str, n_files: int, phi: float, *, n_dev: int = 4, n_cal: int = CAL_EVAL, tau: float = 0.01,
        drift_sd: float = 0.0, n_cal_dev: int | None = None):
    lam_e, lam_p, lam_z = sim.EFFECTS[effect]
    tags = [effect, f"n{n_files}", f"phi{phi:g}", f"tau{tau:g}", f"ndev{n_dev}", f"ncal{n_cal}"]
    if n_cal_dev is not None:
        tags.append(f"ncaldev{n_cal_dev}")
    if drift_sd > 0:
        tags.append(f"drift{drift_sd:g}")
    return sim.Scenario(name="-".join(tags), n_files=n_files, lam_emp=lam_e, lam_pca=lam_p, lam_zero=lam_z,
                        phi=phi, tau=tau, n_dev_files=n_dev, n_cal=n_cal, drift_sd=drift_sd)


EVAL_N = (4, 8, 16, 32, 64)


def eval_scenarios() -> list:
    out = []
    for n in EVAL_N:
        out.append(_sc("alt-medium", n, 0.0))
        out.append(_sc("alt-medium", n, 0.6))
        out.append(_sc("null-partial", n, 0.6))
    out.append(_sc("alt-large", 4, 0.6, drift_sd=0.15))
    names = [s.name for s in out]
    assert len(names) == len(set(names))
    return out


def dev_design_pairs() -> list:
    """(scenario, n_cal_dev) for module (b).  The evaluation calibration is fixed
    at ``CAL_EVAL``; only the development calibration varies here."""
    out = []
    for nd in DEV_N:
        for nc in CAL_N:
            out.append((_sc("alt-medium", 16, 0.6, n_dev=nd, n_cal=CAL_EVAL, n_cal_dev=nc), nc))
            out.append((_sc("null-partial", 16, 0.6, n_dev=nd, n_cal=CAL_EVAL, n_cal_dev=nc), nc))
    names = [s.name for s, _ in out]
    assert len(names) == len(set(names))
    return out


def dev_scenarios() -> list:
    return [s for s, _ in dev_design_pairs()]


def fixture_eval_scenarios() -> list:
    return [s for s in eval_scenarios() if s.n_files in (4, 8)][:4]


def fixture_dev_scenarios() -> list:
    return [s for s, nc in dev_design_pairs() if s.n_dev_files in (4, 8) and nc == 64][:4]


# --- A2: the R1-failure -> R4-cancellation gate --------------------------------------
def _repo_relative(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_ROOT / p)


def r1_gate_config(P: dict) -> dict:
    return P["matrix"]["production"].get("r1_gate", {})


def _withholding_screen(P: dict):
    """The optional prespecified development-gate screen, wired into production.

    If the protocol names ``withholding_screen_path`` and that file exists, its
    ``{n_files: bool_infeasible}`` mapping is applied to the gate now, so an
    infeasible large-file extension is omitted from the matrix and the launch plan
    before it is priced — not only at reduction.  A missing screen is simply
    absent; a malformed one is an error rather than a silent pass.
    """
    path = r1_gate_config(P).get("withholding_screen_path")
    if not path:
        return None
    p = _repo_relative(path)
    if not p.exists():
        return None
    import json as _json
    got = _json.loads(p.read_text())
    if not isinstance(got, dict) or not all(isinstance(v, bool) for v in got.values()):
        raise ProtocolError(f"withholding screen {p} is malformed: expected {{n_files: boolean}}")
    return got


def evaluate_gate(P: dict, *, freeze_override: Path | None = None, acceptance_override: Path | None = None) -> dict:
    import os
    cfg = r1_gate_config(P)
    freeze = freeze_override or (Path(os.environ["R4_R1_FREEZE"]) if os.environ.get("R4_R1_FREEZE")
                                 else _repo_relative(cfg["freeze_path"]))
    acceptance = acceptance_override or (Path(os.environ["R4_R1_ACCEPTANCE"]) if os.environ.get("R4_R1_ACCEPTANCE")
                                         else _repo_relative(cfg["acceptance_path"]))
    return gate.evaluate_r1_precondition(freeze, acceptance,
                                         cancel_large_file_counts=cfg.get("cancel_large_file_counts", (32, 64)),
                                         r4_withholding_screen=_withholding_screen(P))


def r1_inference(P: dict) -> dict:
    """The inference R4 must apply: the frozen accepted R1 candidate, or the
    archived baseline when no R1 replacement is available (with the reason)."""
    g = evaluate_gate(P)
    if gate.large_file_admissible(g):
        cand_name = g["r1"].get("chosen_candidate")
        acc_path = Path(g["acceptance_path"]) if g.get("acceptance_path") else None
        receipt_sha = None
        if acc_path and acc_path.exists():
            import hashlib
            receipt_sha = hashlib.sha256(acc_path.read_bytes()).hexdigest()
        return {"candidate": cand_name, "version": "r1-t22-calibration-v1",
                "applies_r1_candidate": True, "r1_receipt": str(acc_path) if acc_path else None,
                "r1_receipt_sha256": receipt_sha, "r1_protocol_sha256": g["r1"].get("protocol_sha256"),
                "note": f"R1 gate status {g['status']}: the selected R1 candidate is applied"}
    return {"candidate": "two_stage", "version": "archived-t22s-baseline", "applies_r1_candidate": False,
            "r1_receipt": None, "r1_receipt_sha256": None,
            "note": (f"R1 gate status {g['status']}: no accepted R1 replacement is applied; this experiment tests "
                     f"the ARCHIVED baseline only and is not a feasibility test of the corrected interval")}


def effective_eval_scenarios(P: dict) -> list:
    """The eval scenarios after the R1 gate.

    Large-file cells (32/64) are admissible only on a clean ``pass``.  A
    ``blocked`` or ``unresolved`` gate removes them too, so an unresolved
    precondition can never be silently drawn as if it were accepted.
    """
    scs = eval_scenarios()
    g = evaluate_gate(P)
    if not gate.large_file_admissible(g):
        keep = set(g["cancel_large_file_counts"])
        return [s for s in scs if s.n_files not in keep]
    return scs


# --- matrix --------------------------------------------------------------------------
def _cells(scenarios, *, role, family, seeds, mc, chunks, n_cal_dev_map=None):
    out = []
    for sc in scenarios:
        for chunk in range(chunks):
            params = {"scenario": sc.as_dict(), "mc_replicates": int(mc), "chunk": chunk}
            if n_cal_dev_map is not None and sc.name in n_cal_dev_map:
                params["n_cal_dev"] = int(n_cal_dev_map[sc.name])
            out.append(make_cell(role=role, seed=int(seeds[chunk]), params=params, n_fits=0, family=family))
    return out


def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    _check(P)
    M, prod = P["matrix"], P["matrix"]["production"]
    cells = []
    if fixture:
        fx = M["fixture"]
        dev_lo = P["seed_namespaces"]["development"][0]
        cells += _cells(fixture_eval_scenarios(), role="development", family="eval_grid",
                        seeds=[dev_lo + i for i in range(fx["chunks"])], mc=fx["mc_per_cell"], chunks=fx["chunks"])
        cells += _cells(fixture_dev_scenarios(), role="development", family="dev_grid",
                        seeds=[dev_lo + 100 + i for i in range(fx["chunks"])], mc=fx["mc_per_cell"], chunks=fx["chunks"],
                        n_cal_dev_map={s.name: 64 for s in fixture_dev_scenarios()})
    else:
        conf_lo = P["seed_namespaces"]["confirmation"][0]
        n_e = prod["mc_replicates_per_scenario"] // prod["mc_per_cell"]
        cells += _cells(effective_eval_scenarios(P), role="confirmation", family="eval_grid",
                        seeds=[conf_lo + i for i in range(n_e)], mc=prod["mc_per_cell"], chunks=n_e)
        dev_pairs = dev_design_pairs()
        cells += _cells([s for s, _ in dev_pairs], role="confirmation", family="dev_grid",
                        seeds=[conf_lo + 100 + i for i in range(n_e)], mc=prod["mc_per_cell"], chunks=n_e,
                        n_cal_dev_map={s.name: nc for s, nc in dev_pairs})
    for name, cfg in M["production"]["controls"].items():
        cells.append(make_cell(role="control", seed=None, params={"control": name, **cfg}, n_fits=0, family="control"))
    return RunMatrix(PACKAGE, P["_sha256"], cells)


def _check(P: dict) -> None:
    cat = P["matrix"]["scenario_catalogue"]
    got = {"eval": [s.name for s in eval_scenarios()], "dev": [s.name for s in dev_scenarios()],
           "fixture_eval": [s.name for s in fixture_eval_scenarios()], "fixture_dev": [s.name for s in fixture_dev_scenarios()]}
    for k, names in got.items():
        if cat.get(k) != names:
            raise ProtocolError(f"R4 scenario catalogue {k!r} differs from the code")
    if tuple(P["inference"]["feasibility"]["eval_n_files"]) != EVAL_N:
        raise ProtocolError("R4 eval n_files grid differs from the code")
    if tuple(P["inference"]["feasibility"]["dev_n_files"]) != DEV_N or tuple(P["inference"]["feasibility"]["dev_n_cal"]) != CAL_N:
        raise ProtocolError("R4 development design grid differs from the code")


# --- cells ---------------------------------------------------------------------------
def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict):
    if cell["family"] in ("eval_grid", "dev_grid"):
        return _run_sim(P, cell, fixture), 0, 0
    if cell["family"] == "control":
        return _run_control(P, cell), 0, 0
    raise RuntimeError(f"unknown family {cell['family']!r}")


def _run_sim(P: dict, cell: dict, fixture: bool) -> dict:
    sc = sim.Scenario(**cell["params"]["scenario"])
    reps = dict(P["matrix"]["fixture" if fixture else "production"]["replicates"])
    inference = r1_inference(P)
    n_cal_dev = cell["params"].get("n_cal_dev")
    records = []
    for j in range(int(cell["params"]["mc_replicates"])):
        seed = int(cell["seed"]) * 100_000 + j
        rec = st.run_replicate_streams(seed, sc, file_replicates=reps["file"], hier_replicates=reps["hier"],
                                       curve_replicates=reps["curve"], inference=inference, n_cal_dev=n_cal_dev)
        rec["problems"] = pl.check_record(rec, sc)
        records.append(rec)
    problems = [p for r in records for p in r["problems"]]
    return {"kind": "simulation", "passed": not problems,
            "checks": {"invariant_violations": len(problems), "first": problems[:3]},
            "summary": {"scenario_name": sc.name, "n_mc": len(records), "replicates": reps,
                        "inference": inference,
                        "stream_source_change": "separate development/evaluation streams"},
            "records": records}


def _run_control(P: dict, cell: dict) -> dict:
    name = cell["params"]["control"]
    checks = {}
    if name == "streams_are_separate":
        sc = _sc("alt-medium", 8, 0.6)
        dev_a, ev_a, _ = st.simulate_scenario_streams(12345, sc)
        sc2 = _sc("alt-medium", 64, 0.6)
        dev_b, ev_b, _ = st.simulate_scenario_streams(12345, sc2)
        # changing n_files must not change the development realization
        same_dev = all(a.N == b.N and abs(a.lam_emp - b.lam_emp) < 1e-15 for a, b in zip(dev_a, dev_b))
        # changing n_files may change evaluation (more files), but the first 8 files' seeds are the same stream
        checks["development_fixed_under_n_files_change"] = {"passed": bool(same_dev),
                                                            "n_files_A": 8, "n_files_B": 64, "n_dev": sc.n_dev_files}
        checks["distinct_stream_seeds"] = {"passed": True,
                                           "seeds": st.simulate_scenario_streams(12345, sc)[2]["stream_seeds"]}
    elif name == "support_bound_arithmetic":
        recs = [{"withheld": v} for v in ([True] * 80 + [False] * 20)]
        b = an.support_bound(recs)
        checks["bound"] = {"passed": abs(b["observed_max_unconditional_support"] - 0.2) < 1e-12
                           and b["plausible_max_unconditional_support"] >= 0.2,
                           "detail": b}
    elif name == "estimand_identity":
        sc = _sc("alt-medium", 4, 0.0, tau=0.0)
        recs = [st.run_replicate_streams(777_000 + j, sc, file_replicates=40, hier_replicates=200, curve_replicates=30)
                for j in range(10)]
        dev = [r["contrasts"]["raw-bin|empca"]["bias"] for r in recs]
        checks["bias_small"] = {"passed": abs(np.mean(dev)) < 5e-5, "mean_bias": float(np.mean(dev))}
        checks["invariants"] = {"passed": all(not pl.check_record(r, sc) for r in recs)}
    else:
        raise RuntimeError(f"unknown control {name!r}")
    return {"kind": "control", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"control": name}}


# --- reduce ---------------------------------------------------------------------------
def _pool(family_cells):
    groups = {}
    for js in family_cells:
        key = js["result"]["summary"]["scenario_name"]
        g = groups.setdefault(key, {"records": [], "cells": [],
                                    "scenario": sim.Scenario(**js["params"]["scenario"])})
        g["records"].extend(js["result"]["records"])
        g["cells"].append(js["cell_id"])
    return groups


def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    fixture = any(js.get("development_only") for js in cells.values())
    tol = P["calibration_tolerances"]
    by_family = {f: [js for js in cells.values() if js["family"] == f] for f in FAMILIES}
    violations = []
    for js in cells.values():
        r = js["result"]
        if r["kind"] == "simulation":
            sc = sim.Scenario(**js["params"]["scenario"])
            for rec in r["records"]:
                probs = pl.check_record(rec, sc)
                if probs:
                    violations.append({"cell_id": js["cell_id"], "problems": probs[:3]})
        elif not r["passed"]:
            violations.append({"cell_id": js["cell_id"], "problems": ["control failed"]})
    if violations:
        raise SchemaError("scientific check failed: " + str(violations[:3]))
    summary = {"controls": {js["result"]["summary"]["control"]: {"passed": js["result"]["passed"], "checks": js["result"]["checks"]}
                            for js in by_family["control"]},
               "tolerances": tol, "feasibility_rules": P["inference"]["feasibility"],
               "module_a_eval_grid": {}, "module_b_dev_grid": {}}
    for fam, target in (("eval_grid", "module_a_eval_grid"), ("dev_grid", "module_b_dev_grid")):
        groups = _pool(by_family[fam])
        for name, g in sorted(groups.items()):
            summary[target][name] = an.scenario_summary(g["scenario"], g["records"], tol, claims=CLAIMS)
    # module (a) feasibility by n_files
    expected_by_n: dict[int, int] = {}
    for s in eval_scenarios():
        expected_by_n[s.n_files] = expected_by_n.get(s.n_files, 0) + 1
    rows_by_n = {}
    for name, s in summary["module_a_eval_grid"].items():
        rows_by_n.setdefault(s["n_files"], []).append(s)
    summary["feasibility_by_n_files"] = {
        str(n): an.feasibility_row(rows, P["inference"]["feasibility"], tol,
                                   expected_scenarios=expected_by_n.get(n), complete=(len(rows) == expected_by_n.get(n)))
        for n, rows in sorted(rows_by_n.items())}
    # module (b) development design table.  The key keeps the full scenario
    # identity (effect too): a key that omitted it overwrote a null row with an
    # alternative row at the same (n_dev_files, n_cal).
    dev_rows = {}
    for name, s in summary["module_b_dev_grid"].items():
        key = name  # the scenario name already carries effect, n_files, phi, tau, ndev and ncal
        dev_rows[key] = {
            "scenario": s["scenario"], "n_dev_files": s["n_dev_files"], "n_cal": s["n_cal"],
            "withholding": s["withholding"], "support_bound": s["support_bound"],
            "any_null_family_within_tolerance": s.get("any_null_claim_supported", {}).get("within_tolerance")
            if s.get("any_null_claim_supported") else None,
            "coverage_primary": s["coverage_primary"]["raw-bin|empca"]}
    summary["development_design_table"] = dev_rows
    summary["inference"] = r1_inference(P)
    summary["screening_conclusion"] = {
        "rule": ("if the development gate fixes withholding above 0.5, unconditional medium-effect power cannot reach "
                 "0.5 and no evaluation-file extension can repair it"),
        "by_n_files": {n: {"screen_power_0_5_possible": r["screen_power_0_5_possible_min"],
                           "screened_infeasible": r["screened_infeasible_by_withholding_bound"]}
                       for n, r in summary["feasibility_by_n_files"].items()},
        "gate_question": ("development-data adequacy is a separate question from evaluation-file count; a gate candidate "
                          "or a development-dimension change requires a new protocol version with full type-I/coverage "
                          "validation and fresh confirmation"),
    }
    summary["statement"] = (
        "T22 feasibility with a separate development-gate question.  It reports evaluation-file power/coverage at a "
        "fixed development design (module a) and development-data adequacy at a fixed evaluation design (module b), and "
        "screens infeasible designs by the exact bound P(supported) <= 1 - P(withheld).  Nothing here establishes "
        "adequacy on unexamined instrument data; the archived verdicts remain withheld."
        + ("  DEVELOPMENT ONLY: fixture output, never citable." if fixture else ""))
    # A2: the R1-failure -> R4-cancellation gate
    g = evaluate_gate(P)
    summary["r1_precondition"] = g
    summary["r4_ready_to_transfer"] = gate.ready_to_transfer(g)
    if g["cancel"]:
        summary["r4_infeasible"] = gate.ineffective_receipt(g, r4_protocol_sha256=P["_sha256"])
        write_json_atomic(Path(out) / "R4_INFEASIBLE.json", summary["r4_infeasible"])
    return summary


# --- entry point ---------------------------------------------------------------------
def _gate_command(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility gate")
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--freeze", type=Path, default=None)
    ap.add_argument("--acceptance", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the decision / R4_INFEASIBLE receipt here")
    a = ap.parse_args(argv)
    from experiments.mlst2026.successor_20260913.common.protocol import load_protocol
    P = load_protocol(a.protocol)
    g = evaluate_gate(P, freeze_override=a.freeze, acceptance_override=a.acceptance)
    payload = {"decision": g,
               "r4_ready_to_transfer": gate.ready_to_transfer(g),
               "large_file_cells_cancelled": ([s.name for s in eval_scenarios() if s.n_files in g["cancel_large_file_counts"]]
                                               if g["cancel"] else [])}
    if g["cancel"]:
        payload["R4_INFEASIBLE"] = gate.ineffective_receipt(g, r4_protocol_sha256=P["_sha256"])
    print(json.dumps(payload, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(a.out, payload)
    return 0 if gate.ready_to_transfer(g) else (3 if g["status"] == "unresolved" else 2)


def _precheck_gate(argv) -> str | None:
    """Refuse a production run while the R1 precondition is unresolved/blocked.

    This is the execution-boundary enforcement: the standalone ``gate`` command
    is not enough, because the ordinary ``run`` path must itself refuse to draw
    or charge large-file cells without a clean pass.  A ``cancel`` is a
    scientific infeasibility and does *not* block the rest of R4; the large-file
    cells are simply absent from the matrix.
    """
    if not argv or argv[0] != "run" or "--fixture" in argv:
        return None
    from experiments.mlst2026.successor_20260913.common.protocol import load_protocol
    proto = None
    for i, a in enumerate(argv):
        if a == "--protocol" and i + 1 < len(argv):
            proto = Path(argv[i + 1])
    if proto is None:
        return None
    P = load_protocol(proto)
    g = evaluate_gate(P)
    if g["status"] in ("unresolved", "blocked"):
        return (f"R1 precondition is {g['status']}: " + "; ".join(g["reasons"][:2]) +
                ".  Resolve R1 (freeze + complete acceptance receipt) before running R4 production cells.")
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "gate":
        return _gate_command(argv[1:])
    msg = _precheck_gate(argv)
    if msg:
        print("error: " + msg, file=sys.stderr)
        return 6
    return cli.main_template(package=PACKAGE,
                             description="R4: T22 feasibility (evaluation-file power and development-data adequacy)",
                             build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run,
                             result_required=result_required, source_files=SOURCE_FILES,
                             fixture_overrides=lambda P: {"scale": "fixture", **P["matrix"]["fixture"]}, argv=argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
