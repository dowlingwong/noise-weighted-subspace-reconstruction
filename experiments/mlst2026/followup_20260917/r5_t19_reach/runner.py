"""R5 runner: optional T19 full-covariance theorem-domain reachability.

Fixed-design, no selection authority.  Each cell fixes one calibration size
``m`` and draws independent seeds; the reduce reports the eta distribution,
the fraction reaching the domain, conditional in-domain violations, first-order
agreement, and the fitted eta-versus-m slope.  "Not reachable" is a valid result.
"""
from __future__ import annotations

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
from experiments.mlst2026.successor_20260913.common.results import SchemaError  # noqa: E402
from experiments.mlst2026.successor_20260913.t19s import observables as O  # noqa: E402
from experiments.mlst2026.successor_20260913.t19s import synthetic as S  # noqa: E402

from . import covariance as cov  # noqa: E402
from . import reachability as reach  # noqa: E402

PACKAGE = "r5_t19_reach"
#: v2 makes the production path genuinely nested (one cell per seed group over
#: the full m ladder); v1 is retained unchanged.
PROTOCOL_DEFAULT = _HERE.parent / "PROTOCOL_R5_T19_REACH_v2.json"
SOURCE_FILES = [_HERE, _HERE.parent / "reachability.py", _HERE.parent / "covariance.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "t19s" / "synthetic.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "t19s" / "observables.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "common" / "cli.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "common" / "matrix.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "common" / "results.py"]
FAMILIES = ("reach", "control")
result_required = {"kind": str, "passed": bool, "checks": dict, "summary": dict}


def _design(P: dict, fixture: bool) -> dict:
    d = P["matrix"]["fixture" if fixture else "production"]
    return {"d": int(d["d"]), "m_grid": list(d["m_grid"]), "n_train": int(d["n_train"]),
            "chunk": int(d["chunk"]), "n_seeds": int(d["n_seeds"]),
            "n_groups": int(d.get("n_groups", 1))}


# --- matrix --------------------------------------------------------------------------
def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    """One cell per seed GROUP, each running the FULL m ladder.

    The nested design requires the covariance prefixes across m to come from one
    stream.  A cell per m with an independent seed (the previous design) breaks
    nesting: no m is a prefix of another.  Here each group's cell consumes the
    full ladder in ascending m order, and the fit count is the true
    ``n_seeds * len(m_grid)``.
    """
    _check(P, fixture)
    des = _design(P, fixture)
    cells = []
    role = "development" if fixture else "confirmation"
    lo = P["seed_namespaces"]["development" if fixture else "confirmation"][0]
    for i in range(des["n_groups"]):
        cells.append(make_cell(role=role, seed=lo + i,
                               params={"kind": "reach", "d": des["d"], "m_grid": [int(m) for m in des["m_grid"]],
                                       "n_train": des["n_train"], "chunk": des["chunk"],
                                       "n_seeds": des["n_seeds"], "group": i},
                               n_fits=des["n_seeds"] * len(des["m_grid"]), family="reach"))
    for name in P["matrix"]["controls"]:
        cells.append(make_cell(role="control", seed=None, params={"kind": "control", "control": name}, n_fits=0,
                               family="control"))
    return RunMatrix(PACKAGE, P["_sha256"], cells)


def _check(P: dict, fixture: bool) -> None:
    if P.get("version") != "R5-v2":
        raise ProtocolError(
            f"R5 protocol version {P.get('version')!r} is superseded by R5-v2 (nested production path); "
            "use PROTOCOL_R5_T19_REACH_v2.json. v1 is retained for the record and must not be executed")
    for scale in ("production", "fixture"):
        des = P["matrix"][scale]
        if tuple(int(x) for x in des["m_grid"]) != tuple(int(x) for x in P["calibration_grids"][scale]):
            raise ProtocolError(f"R5 {scale} m_grid differs from calibration_grids")
        if int(des.get("n_groups", 1)) < 1:
            raise ProtocolError(f"R5 {scale} n_groups must be >= 1")
    if not fixture and P["calibration_grids"]["production"] != sorted(P["calibration_grids"]["production"]):
        raise ProtocolError("R5 production m grid must be ascending")
    if set(P["controls"]) != set(P["matrix"]["controls"]):
        raise ProtocolError("R5 controls differ from the code's controls")


# --- cells ---------------------------------------------------------------------------
def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict):
    params = cell["params"]
    if params["kind"] == "reach":
        g = S.make_generator(int(params["d"]))
        m_grid = [int(m) for m in params["m_grid"]]
        records = reach.reachability_records(int(cell["seed"]), g, m_grid=m_grid,
                                             n_train=int(params["n_train"]), chunk=int(params["chunk"]),
                                             n_seeds=int(params["n_seeds"]))
        # one subspace fit per (seed, m); the archived estimator is instrumented at its call site
        return {"kind": "reachability", "passed": True,
                "checks": {"n_records": len(records), "nested_prefixes": True},
                "summary": {"d": int(params["d"]), "m_grid": m_grid, "group": int(params.get("group", 0)),
                            "n_seeds": int(params["n_seeds"]), "parent_seed": int(cell["seed"]),
                            "n_in_domain": int(sum(r["in_domain"] for r in records)),
                            "estimator_arm": "S_full"},
                "records": records}, int(params["n_seeds"]) * len(m_grid), 0
    if params["kind"] == "control":
        return _run_control(params["control"]), 0, 0
    raise RuntimeError(f"unknown cell kind {params['kind']!r}")


def _run_control(name: str) -> dict:
    checks = {}
    if name == "streaming_matches_direct":
        rng = np.random.default_rng(5)
        d, m, chunk = 6, 1000, 128
        draws = rng.standard_normal((m, d))
        direct = cov.direct_covariance(draws)
        it = iter(draws)
        streamed = cov.streaming_covariance(lambda n: np.array([next(it) for _ in range(n)]), m, d, chunk)
        checks["match"] = {"passed": float(np.max(np.abs(direct - streamed))) < 1e-10,
                           "max_abs_diff": float(np.max(np.abs(direct - streamed)))}
    elif name == "nested_arms_are_prefixes":
        rng = np.random.default_rng(6)
        d, chunk = 5, 64
        draws = rng.standard_normal((1000, d))
        it = iter(draws)
        covs = cov.nested_streaming_covariances(lambda n: np.array([next(it) for _ in range(n)]), [200, 1000], d, chunk)
        direct_200 = cov.direct_covariance(draws[:200])
        checks["prefix"] = {"passed": float(np.max(np.abs(covs[200] - direct_200))) < 1e-10}
    elif name == "out_of_domain_retained":
        recs = [{"m": 1, "eta": 5.0, "in_domain": False}, {"m": 1, "eta": 0.05, "in_domain": True,
                "bound": {"violation": 0.0}, "first_order_projector_ratio_op": 1.01}]
        s = reach.summarise(recs, [1])
        checks["retained"] = {"passed": s["by_m"]["1"]["n"] == 2 and s["by_m"]["1"]["n_in_domain"] == 1,
                              "detail": s["by_m"]["1"]}
    else:
        raise RuntimeError(f"unknown control {name!r}")
    return {"kind": "control", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"control": name}}


# --- reduce ---------------------------------------------------------------------------
def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    fixture = any(js.get("development_only") for js in cells.values())
    by_family = {f: [js for js in cells.values() if js["family"] == f] for f in FAMILIES}
    violations = []
    for js in cells.values():
        if js["family"] == "control" and not js["result"]["passed"]:
            violations.append({"cell_id": js["cell_id"], "problems": ["control failed"]})
    if violations:
        raise SchemaError("scientific check failed: " + str(violations[:3]))
    des = _design(P, fixture)
    records = [r for js in by_family["reach"] for r in js["result"]["records"]]
    summary = {"controls": {js["result"]["summary"]["control"]: {"passed": js["result"]["passed"], "checks": js["result"]["checks"]}
                            for js in by_family["control"]},
               "design": des, "reachability": reach.summarise(records, des["m_grid"]),
               "eta_scaling": reach.fit_eta_scaling(reach.summarise(records, des["m_grid"])),
               "expected_eta_at_anchor": None}
    # the archived extrapolation anchor: median eta 0.226 at m=524288 gave eta=0.1 at ~2.7e6
    summary["statement"] = (
        "Optional T19 full-covariance theorem-domain reachability.  It reports the eta distribution, the fraction of "
        "realisations reaching eta <= 0.1, conditional in-domain bound violations and first-order agreement at a fixed "
        "calibration allocation.  Out-of-domain realisations are never discarded.  Not reachable is a valid result; the "
        "archived T19 prediction remains NOT met."
        + ("  DEVELOPMENT ONLY: fixture output, never citable." if fixture else ""))
    return summary


def main(argv=None) -> int:
    return cli.main_template(package=PACKAGE,
                             description="R5 (optional): T19 S_full theorem-domain reachability at a larger calibration allocation",
                             build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run,
                             result_required=result_required, source_files=SOURCE_FILES,
                             fixture_overrides=lambda P: {"scale": "fixture", **P["matrix"]["fixture"]}, argv=argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
