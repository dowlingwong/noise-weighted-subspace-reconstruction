"""R6 runner: data-blocked instrument readiness CLI.

Subcommands
-----------
``validate``      load and structurally check the R6 protocol, re-enumerate the
                  archived 350-cell grid and confirm the protocol pins it.  Computes
                  nothing and opens no data.
``enumerate``     write the enumerated CRESST grid cells and the declared matched
                  TIDMAD design.  Computes nothing and opens no data.
``inventory``     generate the TIDMAD exposure inventory from the archived run trees
                  (reuses ``matched_stage.exposure_inventory``); opens no instrument
                  data.
``check-inputs``  show the per-input readiness table (CRESST raw-pulse gate, TIDMAD
                  exposure gate, calibration and dependence gates) and refuse.
``preflight``     the confirmatory readiness verdict; returns DATA-BLOCKED (exit 3)
                  whenever any gate cannot be satisfied.

None of these subcommands requires real data, and none fabricates a cell.  When the
inputs are absent (their state here) the status is the string ``DATA-BLOCKED`` and
the exit code is ``3`` for the gate commands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.mlst2026.successor_20260913.common.protocol import (ProtocolError, load_protocol,  # noqa: E402
                                                                     protocol_hash)
from experiments.mlst2026.successor_20260913.common.results import write_json_atomic  # noqa: E402
from experiments.mlst2026.successor_20260913.cresst_s.archive_recompute import InputNotAvailable  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import matched_stage as ms  # noqa: E402

from . import cresst_readiness as CR  # noqa: E402
from . import tidmad_readiness as TM  # noqa: E402

PACKAGE = "r6_instrument"
PROTOCOL_DEFAULT = _HERE.parent / "PROTOCOL_R6_INSTRUMENT_v1.json"
EXIT_OK = 0
EXIT_DATA_BLOCKED = 3
EXIT_USAGE = 6
DATA_BLOCKED = "DATA-BLOCKED"


def _load(args) -> dict:
    P = load_protocol(args.protocol)
    if P["package"] != PACKAGE:
        raise ProtocolError(f"protocol is for package {P['package']!r}, runner is {PACKAGE!r}")
    return P


def _status(P: dict) -> dict:
    return {"package": PACKAGE, "version": P["version"], "status": P["status"],
            "data_blocked": True, "computes": False, "opens_real_data": False,
            "executes_production_cells": False, "real_data_validated": False,
            "protocol_sha256": P["_sha256"], "protocol_hash": protocol_hash(P),
            "label": CR.NON_REAL_LABEL}


# --- subcommands ----------------------------------------------------------------------------
def cmd_validate(args) -> int:
    P = _load(args)
    rows = CR.enumerate_grid(P)
    cost = CR.cost_grid(P, rows)
    design = TM.matched_design(P)
    out = {**_status(P),
           "cresst": {"cells_enumerated": cost["n_cells"], "fits_if_executed": cost["n_fits"],
                      "executable_here": cost["executable_here"],
                      "grid_matches_archived": True,
                      "contrast_groups": cost["n_contrast_groups"],
                      "paired_contrasts_total": cost["n_paired_contrasts_total"]},
           "tidmad": {"matched_design_declared": True, "executable_here": design["executable_here"]},
           "statement": ("interface implemented and fail-closed; NOT validated on data and no cell executed")}
    print(json.dumps(out, indent=1))
    return EXIT_OK


def cmd_enumerate(args) -> int:
    P = _load(args)
    rows = CR.enumerate_grid(P)
    cost = CR.cost_grid(P, rows)
    design = TM.matched_design(P)
    payload = {**_status(P), "computes": False,
               "cresst": {**cost, "cells": rows},
               "tidmad": design}
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "R6_ENUMERATION.json", payload)
        print(json.dumps({"package": PACKAGE, "out": str(out), "cresst_cells": cost["n_cells"],
                          "cresst_fits_if_executed": cost["n_fits"],
                          "tidmad_executable_here": design["executable_here"],
                          "data_blocked": True}, indent=1))
    else:
        print(json.dumps(payload, indent=1))
    return EXIT_OK


def cmd_inventory(args) -> int:
    roots = args.archive_root or None
    inv = TM.exposure_inventory(roots)
    md = ms.inventory_markdown(inv)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md)
        write_json_atomic(p.with_suffix(".json"), inv)
        print(f"wrote {p} ({len(inv['regions'])} regions over "
              f"{len(inv['files_known_to_archive'])} files)")
    else:
        print(md)
    print(json.dumps({"release_file_list_derivable": inv["release_file_list"]["derivable_from_archive"],
                      "files_not_yet_exposed": inv["release_file_list"]["files_not_yet_exposed"],
                      "independent_file_set_resolved": False}, indent=1))
    return EXIT_OK


def _gate(args, P: dict) -> dict:
    reasons: list[str] = []
    result: dict = {**_status(P), "cresst": None, "tidmad": None, "reasons": reasons}
    try:
        c = CR.check_inputs(P, args.cresst_data_root)
        result["cresst"] = c
    except InputNotAvailable as e:
        result["cresst"] = {"status": DATA_BLOCKED, "reason": str(e)}
        reasons.append(f"cresst: {e}")
    inv = TM.exposure_inventory(args.archive_root or None)
    gate = TM.confirmatory_gate(P, inventory=inv, n_available_files=args.n_files,
                                calibration=None, dependence_verdict=args.dependence_verdict)
    result["tidmad"] = {"inventory": {"files_known_to_archive": len(inv["files_known_to_archive"]),
                                      "regions_recorded": len(inv["regions"]),
                                      "release_file_list_derivable": inv["release_file_list"]["derivable_from_archive"],
                                      "independent_file_set_resolved": False},
                        "gate": gate}
    reasons.extend(f"tidmad: {r}" for r in gate["reasons"])
    result["blocked"] = bool(reasons)
    result["status"] = DATA_BLOCKED if result["blocked"] else "ready"
    result["statement"] = ("interface implemented and fail-closed; a passing gate would still be an input check, "
                           "never a real-data result")
    return result


def cmd_check_inputs(args) -> int:
    P = _load(args)
    result = _gate(args, P)
    print(json.dumps(result, indent=1))
    return EXIT_DATA_BLOCKED if result["blocked"] else EXIT_OK


def cmd_preflight(args) -> int:
    P = _load(args)
    result = _gate(args, P)
    result["check"] = "preflight"
    print(json.dumps(result, indent=1))
    return EXIT_DATA_BLOCKED if result["blocked"] else EXIT_OK


# --- parser ---------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m experiments.mlst2026.followup_20260917.r6_instrument",
                                 description="R6: data-blocked instrument readiness (fail-closed interfaces only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT, help="R6 protocol JSON")

    for name in ("validate",):
        common(sub.add_parser(name, help="check the protocol and the archived grid; computes nothing"))
    sp = sub.add_parser("enumerate", help="write the enumerated grid and matched design; computes nothing")
    common(sp)
    sp.add_argument("--out", type=Path, default=None)
    sp = sub.add_parser("inventory", help="TIDMAD exposure inventory from the archived run trees")
    sp.add_argument("--out", type=Path, default=None)
    sp.add_argument("--archive-root", type=Path, nargs="*", default=None)
    for name, helptext in (("check-inputs", "per-input readiness table; refuses when inputs are absent"),
                           ("preflight", "confirmatory readiness verdict; DATA-BLOCKED when not ready")):
        sp = sub.add_parser(name, help=helptext)
        common(sp)
        sp.add_argument("--cresst-data-root", type=Path, default=None)
        sp.add_argument("--archive-root", type=Path, nargs="*", default=None)
        sp.add_argument("--n-files", type=int, default=None,
                        help="independent unexposed files available (none: do not guess)")
        sp.add_argument("--dependence-verdict", default=None,
                        choices=["stabilised", "stabilised-at-endpoint", "inconclusive"])
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return {"validate": cmd_validate, "enumerate": cmd_enumerate, "inventory": cmd_inventory,
                "check-inputs": cmd_check_inputs, "preflight": cmd_preflight}[args.cmd](args)
    except (ProtocolError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
