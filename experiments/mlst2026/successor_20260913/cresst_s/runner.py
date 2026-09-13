"""CRESST-S runner (L5): archive reporting, cluster-aware description, a prepared
robustness grid and a calibration of the archive's own hierarchical construction.

Built on ``common.cli.main_template``.  Subcommands: ``validate``, ``enumerate``,
``run``, ``reduce``, ``estimate``.

Extra options handled by this runner before the common CLI sees them
--------------------------------------------------------------------
``--archive-root PATH``
    The manuscript repository's ``evidence/data`` directory, which holds the archived
    CRESST inputs.  **There is no default.**  The archived inputs live in MS, not in
    this repository, and are addressed in the protocol by relative path plus SHA-256.
``--cresst-data-root PATH``
    ``<CRESST_DATA_ROOT>``: the raw CRESST pulse traces the robustness grid needs.
    Absent from this environment; production robustness cells fail cleanly without it.
``--fixture-data-root PATH``
    Where the synthetic development pulse fixture is written (default: ``<out>/fixture_data``).

``validate``, ``enumerate`` and ``--help`` need none of them and open no data.

Matrix families
---------------
``archive_point``       algebraic; the exact point estimates recomputed from the archived
                        residuals and compared to the archive at 1e-12.
``archive_interval``    control; the ten archived interval constructions, recomputed and
                        compared endpoint by endpoint at a declared Monte-Carlo tolerance.
``cluster_report``      control; per-channel effects, leave-one-channel-out, leave-one-run-out,
                        run-stratified estimates and the variance decomposition.
``science_control``     control; block-1 identity, all-zero-signal identity, archived weight
                        constancy, and the uniform-weight EMPCA/PCA limit of the estimator.
``calib_control``       control; the calibration's own null-dependence control.
``robustness_pilot``    pilot; the rank scan.  **Not executable here** (needs raw CRESST pulses).
``robustness_confirm``  confirmation; the rank x shrinkage x time-cut grid.  **Not executable here.**
``calib_pilot``         pilot; the coverage grid at pilot Monte-Carlo size.
``calib_confirm``       confirmation; the coverage grid at confirmation Monte-Carlo size.

Development (fixture) cells appear in ``robustness_pilot``/``robustness_confirm`` (run on the
synthetic pulse fixture, the only cells here that fit anything besides the estimator control)
and in ``calib_pilot``/``calib_confirm``.

Accounting
----------
Only the robustness cells and the ``empca_uniform_weight_limit`` control invoke an estimator;
every other cell fits nothing and solves nothing, because the CRESST residuals were already
computed by the frozen validation and this package only reduces and resamples them.  Bootstrap
and Monte-Carlo draws are reported as ``replicates``, never as fits.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.mlst2026 import t13a_cresst_channel_bootstrap as t13a  # noqa: E402
from experiments.mlst2026.successor_20260913.common import cli  # noqa: E402
from experiments.mlst2026.successor_20260913.common import fitcount  # noqa: E402
from experiments.mlst2026.successor_20260913.common.matrix import RunMatrix, make_cell  # noqa: E402
from experiments.mlst2026.successor_20260913.common.protocol import ProtocolError  # noqa: E402
from experiments.mlst2026.successor_20260913.common.results import SchemaError  # noqa: E402

from . import archive_recompute as AR  # noqa: E402
from . import calibration as CB  # noqa: E402
from . import cluster_report as CR  # noqa: E402
from . import robustness_design as RD  # noqa: E402

PACKAGE = "cresst_s"
PROTOCOL_DEFAULT = _HERE.parent / "PROTOCOL_CRESST_S_v1.json"
SOURCE_FILES = [_HERE, _HERE.parent / "archive_recompute.py", _HERE.parent / "cluster_report.py",
                _HERE.parent / "calibration.py", _HERE.parent / "robustness_design.py",
                _ROOT / "experiments" / "mlst2026" / "t13a_cresst_channel_bootstrap.py",
                _ROOT / "src" / "canonical" / "empca.py"]

FAMILIES = ("archive_point", "archive_interval", "cluster_report", "science_control", "calib_control",
            "robustness_pilot", "robustness_confirm", "calib_pilot", "calib_confirm")
CLUSTER_REPORTS = ("per_channel_effects", "leave_one_channel_out", "leave_one_run_out",
                   "run_stratified", "variance_decomposition")
SCIENCE_CONTROLS = ("block1_equals_iid", "zero_effect_identity", "event_weight_constant",
                    "empca_uniform_weight_limit")

result_required = {"kind": str, "passed": bool, "checks": dict, "summary": dict}

#: cost constants measured in this container on the archived functions; used only to
#: derive ``budget.fixture_to_production_scale`` mechanically.
SEC_PER_EVENT_DRAW = 9.2e-9        # bootstrap resample + mean of one event, per draw
SEC_PER_EMPCA_UNIT = 2.1e-7        # per (train event x component x rank x iteration) per EM fit
SEC_ARCHIVE_LOAD = 0.10            # loading and reducing the archived residual table
N_ARCHIVE_EVENTS = 19335           # evaluation events in the archive (analysis_config.yaml)

# --- extra CLI state (set by main(), read by run_cell) -----------------------------------------
_ARCHIVE_ROOT: Path | None = None
_CRESST_DATA_ROOT: Path | None = None
_FIXTURE_DATA_ROOT: Path | None = None
_CLUSTER_CACHE: dict = {}


def set_roots(*, archive_root=None, cresst_data_root=None, fixture_data_root=None) -> None:
    """Used by the CLI and by tests to point the runner at the archived inputs."""
    global _ARCHIVE_ROOT, _CRESST_DATA_ROOT, _FIXTURE_DATA_ROOT
    _ARCHIVE_ROOT = Path(archive_root) if archive_root else None
    _CRESST_DATA_ROOT = Path(cresst_data_root) if cresst_data_root else None
    _FIXTURE_DATA_ROOT = Path(fixture_data_root) if fixture_data_root else None
    _CLUSTER_CACHE.clear()


def _mutation() -> str | None:
    """TEST ONLY: an injected incorrect scientific output, so that the package's own
    scientific checks can be shown to reject it (not merely the digest check)."""
    return os.environ.get("CRESST_S_TEST_MUTATION") or None


def _archive():
    """Resolve, hash and load the archive once per process."""
    key = str(_ARCHIVE_ROOT)
    if key not in _CLUSTER_CACHE:
        resolved = AR.resolve_archive(_ARCHIVE_ROOT)
        labels, runs, clusters, times = AR.load_clusters(resolved)
        _CLUSTER_CACHE[key] = {"resolved": resolved, "labels": labels, "runs": runs,
                               "clusters": clusters, "times": times,
                               "reference": AR.load_reference_results(resolved),
                               "channel_csv": AR.load_channel_summary(resolved)}
    return _CLUSTER_CACHE[key]


# --- cell specification (shared by the matrix, the cells and the cost model) --------------------
def _scale_block(P: dict, fixture: bool) -> dict:
    return P["matrix"]["fixture" if fixture else "production"]


def cells_spec(P: dict, fixture: bool) -> list[dict]:
    """Every cell as ``{family, role, seed, params}``, in one place so that the matrix,
    the runner and the cost model can never disagree."""
    S = _scale_block(P, fixture)
    scale = "fixture" if fixture else "production"
    dev_lo = P["seed_namespaces"]["development"][0]
    pilot_lo = P["seed_namespaces"]["pilot"][0]
    conf_lo = P["seed_namespaces"]["confirmation"][0]
    out: list[dict] = []

    # -- A. archive recomputation ---------------------------------------------------------
    out.append({"family": "archive_point", "role": "algebraic", "seed": None,
                "params": {"analysis": "archive_point_estimates", "scale": scale}})
    for name in sorted(AR.CONSTRUCTIONS):
        out.append({"family": "archive_interval", "role": "control", "seed": None,
                    "params": {"analysis": "archive_interval", "construction": name,
                               "replicates": int(S["archive_replicates"]),
                               "archive_seed": int(AR.ARCHIVE_SEED), "scale": scale}})
    # -- B. cluster-aware descriptive reporting -------------------------------------------
    for report in CLUSTER_REPORTS:
        out.append({"family": "cluster_report", "role": "control", "seed": None,
                    "params": {"analysis": "cluster_report", "report": report,
                               "replicates": int(S["archive_replicates"]),
                               "archive_seed": int(AR.ARCHIVE_SEED), "scale": scale}})
    # -- scientific controls ----------------------------------------------------------------
    for name in SCIENCE_CONTROLS:
        cfg = dict(S["controls"][name])
        out.append({"family": "science_control", "role": "control", "seed": None,
                    "params": {"analysis": "science_control", "control": name, "scale": scale, **cfg}})
    out.append({"family": "calib_control", "role": "control", "seed": None,
                "params": {"analysis": "calibration_control", "control": "null_dependence",
                           "scale": scale, **dict(S["controls"]["calibration_null_dependence"])}})

    # -- C. prepared, unexecuted robustness grid --------------------------------------------
    for arm, family, lo in (("pilot", "robustness_pilot", pilot_lo),
                            ("confirmation", "robustness_confirm", conf_lo)):
        for row in RD.grid_rows(P, arm):
            seed = int(lo + row["time_cut_index"] * len(RD.ARCHIVE_CHANNELS) + row["channel_index"])
            out.append({"family": family, "role": arm, "seed": seed,
                        "params": {"analysis": "robustness", "scale": "production",
                                   "niter": int(P["robustness"]["empca_iterations"]), **row}})
    # -- D. calibration of the proposed inference -------------------------------------------
    for arm, family, lo in (("pilot", "calib_pilot", pilot_lo + 50),
                            ("confirmation", "calib_confirm", conf_lo + 50)):
        for i, d in enumerate(_calibration_designs(P, arm, "production")):
            out.append({"family": family, "role": arm, "seed": int(lo + i),
                        "params": {"analysis": "calibration", "scale": "production", **d.as_dict()}})

    if fixture:
        F = P["matrix"]["fixture"]
        for i, row in enumerate(_fixture_robustness_rows(P)):
            fam = "robustness_pilot" if i < 2 else "robustness_confirm"
            out.append({"family": fam, "role": "development", "seed": int(dev_lo + i),
                        "params": {"analysis": "robustness", "scale": "fixture",
                                   "niter": int(F["empca_iterations"]), **row}})
        for i, d in enumerate(_calibration_designs(P, "pilot", "fixture")):
            out.append({"family": "calib_pilot", "role": "development", "seed": int(dev_lo + 100 + i),
                        "params": {"analysis": "calibration", "scale": "fixture", **d.as_dict()}})
        for i, d in enumerate(_calibration_designs(P, "confirmation", "fixture")):
            out.append({"family": "calib_confirm", "role": "development", "seed": int(dev_lo + 110 + i),
                        "params": {"analysis": "calibration", "scale": "fixture", **d.as_dict()}})
    return out


def _fixture_robustness_rows(P: dict) -> list[dict]:
    """Development robustness cells: the primary setting and one contrast, on the two
    synthetic fixture channels, so both robustness families are exercised and timed."""
    prim = RD.primary_setting(P)
    F = P["matrix"]["fixture"]
    rows = []
    for (run, ch) in RD.ARCHIVE_CHANNELS[:2]:
        for rank in F["robustness_ranks"]:
            rows.append({"run": run, "channel": ch, "channel_label": f"{run}_{ch}",
                         "rank": int(rank), "shrinkage": float(prim["shrinkage"]),
                         "time_cut": prim["time_cut"], "covariance_model": prim["covariance_model"],
                         "contrast_group": f"{run}_{ch}|{prim['time_cut']}",
                         "contrast_group_index": 0, "channel_index": 0, "time_cut_index": 0,
                         "is_group_reference": bool(int(rank) == int(prim["rank"])),
                         "is_primary_setting": bool(int(rank) == int(prim["rank"])),
                         "time_cut_paired_against": None,
                         "paired_against": None if int(rank) == int(prim["rank"]) else
                         f"{run}_{ch}|{prim['time_cut']}|rank{prim['rank']}|shrink{prim['shrinkage']}",
                         "dependence_unit": "synthetic fixture channel",
                         "within_run_dependence": "not applicable: synthetic development fixture",
                         "executable_here": True,
                         "requires": "synthetic development pulse fixture written by the runner"})
    return rows


def _calibration_designs(P: dict, arm: str, scale: str) -> list[CB.CalibrationDesign]:
    C = P["calibration"]
    g = C["grid"][arm][scale]
    gen = C["generator"]
    out = []
    for n_runs in g["n_runs"]:
        for cpr in g["channels_per_run"]:
            for sd_run in g["sd_run"]:
                out.append(CB.CalibrationDesign(
                    n_runs=int(n_runs), channels_per_run=int(cpr), n_events=int(g["n_events"]),
                    sd_run=float(sd_run), sd_channel=float(gen["sd_channel"]), theta=float(gen["theta"]),
                    sigma_p=float(gen["sigma_p"]), sigma_e=float(gen["sigma_e"]),
                    mc_replicates=int(g["mc_replicates"]), bootstrap_replicates=int(g["bootstrap_replicates"]),
                    alpha=float(C["alpha"])))
    return out


# --- accounting ---------------------------------------------------------------------------------
def cell_fits(spec: dict) -> int:
    if spec["family"] in ("robustness_pilot", "robustness_confirm"):
        return RD.FITS_PER_CELL
    if spec["family"] == "science_control" and spec["params"].get("control") == "empca_uniform_weight_limit":
        return 2 * RD.FITS_PER_CELL   # the flat-PSD arm and the shaped-PSD arm
    return 0


def cell_replicates(spec: dict) -> int:
    """Inference replicates the cell will draw (reported, never charged as fits)."""
    p = spec["params"]
    fam = spec["family"]
    if fam == "archive_interval":
        return int(p["replicates"])
    if fam == "cluster_report":
        r = int(p["replicates"])
        return {"per_channel_effects": 14 * r, "leave_one_channel_out": 7 * r,
                "leave_one_run_out": 2 * r, "run_stratified": 4 * r,
                "variance_decomposition": 7 * r}[p["report"]]
    if fam == "science_control":
        return int(p.get("replicates", 0))
    if fam == "calib_control":
        return 2 * int(p["mc_replicates"]) * int(p["bootstrap_replicates"])
    if fam in ("calib_pilot", "calib_confirm"):
        return int(p["mc_replicates"]) * int(p["bootstrap_replicates"])
    return 0


def work_proxy_seconds(spec: dict) -> float:
    """Per-cell cost proxy in seconds, from constants measured in this container.
    Used only to derive ``budget.fixture_to_production_scale``."""
    fam, p = spec["family"], spec["params"]
    if fam == "archive_point":
        return SEC_ARCHIVE_LOAD
    if fam in ("archive_interval", "cluster_report"):
        return SEC_ARCHIVE_LOAD + SEC_PER_EVENT_DRAW * cell_replicates(spec) * N_ARCHIVE_EVENTS
    if fam == "science_control":
        name = p["control"]
        if name == "block1_equals_iid":
            return 1e-8 * float(p["n"]) * float(p["k"])
        if name == "zero_effect_identity":
            return SEC_ARCHIVE_LOAD + SEC_PER_EVENT_DRAW * int(p["replicates"]) * N_ARCHIVE_EVENTS
        if name == "event_weight_constant":
            return SEC_ARCHIVE_LOAD
        n_train = int(p["n_events"]) // 2
        n_comp = 2 * (int(p["n_samples"]) // 2 + 1)
        return 2.0 * RD.EMPCA_RESTARTS * SEC_PER_EMPCA_UNIT * n_train * n_comp * int(p["rank"]) * int(p["niter"])
    if fam in ("calib_control", "calib_pilot", "calib_confirm"):
        if fam == "calib_control":
            n_clusters = int(p["n_runs"]) * int(p["channels_per_run"])
            reps = 2 * int(p["mc_replicates"]) * int(p["bootstrap_replicates"])
        else:
            n_clusters = int(p["n_runs"]) * int(p["channels_per_run"])
            reps = int(p["mc_replicates"]) * int(p["bootstrap_replicates"])
        return SEC_PER_EVENT_DRAW * reps * n_clusters * int(p["n_events"])
    if fam in ("robustness_pilot", "robustness_confirm"):
        R = _protocol_cache.get("robustness_dimensions") or {}
        n_events = int(p.get("_n_events", R.get("n_events", 400)))
        n_samples = int(p.get("_n_samples", R.get("n_samples", 128)))
        n_train = n_events // 4 if p["time_cut"] != "full" else n_events // 2
        n_comp = 2 * (n_samples // 2 + 1)
        return RD.EMPCA_RESTARTS * SEC_PER_EMPCA_UNIT * n_train * n_comp * int(p["rank"]) * int(p["niter"])
    raise ProtocolError(f"no cost proxy for family {fam!r}")


_protocol_cache: dict = {}


def declared_scales(P: dict) -> dict:
    """``mean production seconds per cell / mean fixture seconds per cell``, by family."""
    _protocol_cache["robustness_dimensions"] = P["robustness"]["production_dimensions"]
    prod = cells_spec(P, False)
    fx_all = cells_spec(P, True)
    fixture_roles = {"development", "control", "algebraic"}
    fx = [s for s in fx_all if s["role"] in fixture_roles]
    out = {}
    for fam in FAMILIES:
        pc = [s for s in prod if s["family"] == fam]
        fc = [s for s in fx if s["family"] == fam]
        if not pc or not fc:
            continue
        if fam in ("robustness_pilot", "robustness_confirm"):
            fdim = P["matrix"]["fixture"]["robustness_dimensions"]
            fw = []
            for s in fc:
                n_train = int(fdim["n_events"]) // 2
                n_comp = 2 * (int(fdim["n_samples"]) // 2 + 1)
                fw.append(RD.EMPCA_RESTARTS * SEC_PER_EMPCA_UNIT * n_train * n_comp
                          * int(s["params"]["rank"]) * int(s["params"]["niter"]))
        else:
            fw = [work_proxy_seconds(s) for s in fc]
        pw = [work_proxy_seconds(s) for s in pc]
        out[fam] = float(f"{float(np.mean(pw) / np.mean(fw)):.4g}")
    return out


def _check_declared_scales(P: dict) -> None:
    want = declared_scales(P)
    got = P["budget"].get("fixture_to_production_scale") or {}
    bad = [k for k, v in want.items() if abs(float(got.get(k, -1)) - v) > 1e-3 * max(1.0, abs(v))]
    if bad:
        raise ProtocolError(f"budget.fixture_to_production_scale is not the mechanically derived value for "
                            f"{bad}: protocol has {[got.get(k) for k in bad]}, derived {[want[k] for k in bad]}")


# --- matrix -----------------------------------------------------------------------------------
def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    _check_grid_matches_protocol(P)
    _check_declared_scales(P)
    cells = []
    for spec in cells_spec(P, fixture):
        # ``replicates`` stays the per-construction count the cell's code uses;
        # ``replicates_total`` is what the cell will actually draw, for accounting.
        params = dict(spec["params"])
        params["replicates_total"] = cell_replicates(spec)
        cells.append(make_cell(role=spec["role"], seed=spec["seed"], params=params,
                               n_fits=cell_fits(spec), n_solves=0, family=spec["family"],
                               notes=("NOT EXECUTABLE IN THIS ENVIRONMENT: needs <CRESST_DATA_ROOT> raw pulses"
                                      if spec["family"].startswith("robustness") and spec["role"] != "development"
                                      else None)))
    return RunMatrix(PACKAGE, P["_sha256"], cells)


def _check_grid_matches_protocol(P: dict) -> None:
    """The protocol pins the catalogues; the code must match them."""
    if sorted(P["archive"]["constructions"]) != sorted(AR.CONSTRUCTIONS):
        raise ProtocolError("the protocol's interval-construction catalogue differs from the code's")
    if list(P["archive"]["stream_order"]) != list(AR.ALL_STREAM_ORDER):
        raise ProtocolError("the protocol's RNG stream order differs from the code's; every number would change")
    if list(P["archive"]["stream_order"])[:len(t13a.STREAM_ORDER)] != list(t13a.STREAM_ORDER):
        raise ProtocolError("the first substreams must be the archived ones, in the archived order")
    if sorted(P["cluster_reporting"]["reports"]) != sorted(CLUSTER_REPORTS):
        raise ProtocolError("the protocol's cluster-report catalogue differs from the code's")
    got = [f"{r}_{c}" for r, c in RD.ARCHIVE_CHANNELS]
    if list(P["robustness"]["channels"]) != got:
        raise ProtocolError("the protocol's channel list differs from the archived channel order")
    if int(P["robustness"]["empca_restarts"]) != RD.EMPCA_RESTARTS:
        raise ProtocolError("the protocol's EM restart count differs from the code's fit accounting")
    if int(P["archive"]["replicates"]) != AR.ARCHIVE_REPLICATES or int(P["archive"]["seed"]) != AR.ARCHIVE_SEED:
        raise ProtocolError("the protocol's archived seed/replicate constants differ from the archived runner's")


# --- cells --------------------------------------------------------------------------------------
def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict):
    fam = cell["family"]
    if fam == "archive_point":
        return _cell_archive_point(P, cell), 0, 0
    if fam == "archive_interval":
        return _cell_archive_interval(P, cell), 0, 0
    if fam == "cluster_report":
        return _cell_cluster_report(P, cell), 0, 0
    if fam == "science_control":
        return _cell_science_control(P, cell), fitcount.observed(), 0
    if fam == "calib_control":
        return _cell_calibration_control(P, cell), 0, 0
    if fam in ("calib_pilot", "calib_confirm"):
        return _cell_calibration(P, cell), 0, 0
    if fam in ("robustness_pilot", "robustness_confirm"):
        return _cell_robustness(P, cell, fixture, ctx), fitcount.observed(), 0
    raise RuntimeError(f"unknown family {fam!r}")


def _archive_provenance(P: dict, A: dict) -> dict:
    return AR.verify_declared_digests(A["resolved"], P["provenance"]["archived_inputs"])


def _cell_archive_point(P: dict, cell: dict) -> dict:
    A = _archive()
    prov = _archive_provenance(P, A)
    rec = AR.point_estimates(A["labels"], A["runs"], A["clusters"], A["times"])
    tol = float(P["tolerances"]["point_estimate_absolute"])
    cmp = AR.compare_point_estimates(rec, A["reference"], A["channel_csv"], tol)
    checks = {
        "archived_inputs_match_protocol_sha256": {"passed": bool(prov["all_match"]), "detail": prov["mismatched"]},
        "point_estimates_reproduce_the_archive": {"passed": bool(cmp["all_within_tolerance"]),
                                                  "max_abs_deviation": cmp["max_abs_deviation"],
                                                  "tolerance": tol},
        "seven_clusters_from_two_runs": {"passed": bool(rec["n_channels"] == 7 and rec["n_runs"] == 2),
                                         "n_channels": rec["n_channels"], "n_runs": rec["n_runs"]},
        "event_count_matches_analysis_config": {"passed": bool(rec["n_evaluation_events"] == N_ARCHIVE_EVENTS),
                                                "n_evaluation_events": rec["n_evaluation_events"]},
    }
    return {"kind": "archive_point", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"point_estimates": rec, "comparison": cmp, "archived_inputs": prov["per_input"],
                        "label": ("recomputation of an archived result; a deviation beyond tolerance is a failure "
                                  "of this package, not a new result")}}


def _cell_archive_interval(P: dict, cell: dict) -> dict:
    A = _archive()
    p = cell["params"]
    name = p["construction"]
    mut = _mutation()
    force = False if (mut == "events_only_hierarchical" and AR.CONSTRUCTIONS[name][0]) else None
    rec = AR.recompute_interval(name, A["clusters"], replicates=int(p["replicates"]),
                                seed=int(p["archive_seed"]), force_hierarchical=force)
    arch = A["reference"]["intervals"][name]
    T = P["tolerances"]
    cmp = AR.compare_interval(rec, arch, replicates_archive=int(P["archive"]["replicates"]),
                              z_target=float(T["interval_endpoint_mc_standard_errors"]),
                              shape_inflation=float(T["interval_endpoint_shape_inflation"]))
    diag = rec["channel_layer_diagnostic"]
    checks = {
        "endpoints_reproduce_the_archive": {"passed": bool(cmp["all_within_tolerance"]),
                                            "max_abs_deviation": cmp["max_abs_deviation"],
                                            "tolerance": cmp["tolerance"]},
        "channel_layer_matches_its_label": {
            "passed": bool(diag["channel_layer_resamples"] == rec["declared_hierarchical"]),
            "declared_hierarchical": rec["declared_hierarchical"],
            "channel_layer_resamples": diag["channel_layer_resamples"],
            "mean_distinct_channels_per_replicate": diag["mean_distinct_channels_per_replicate"],
            "expected_under_declared_rule": diag["expected_under_declared_rule"]},
        "point_estimate_is_the_archived_one": {
            "passed": bool(abs(rec["point_estimate"] - float(arch["point_estimate"]))
                           <= float(T["point_estimate_absolute"])),
            "abs_deviation": abs(rec["point_estimate"] - float(arch["point_estimate"]))},
    }
    return {"kind": "archive_interval", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"construction": name, "recomputed": rec, "archived": arch, "comparison": cmp,
                        "replicates": int(p["replicates"]), "mutation": mut,
                        "label": "recomputation of an archived interval construction"}}


def _cell_cluster_report(P: dict, cell: dict) -> dict:
    A = _archive()
    p = cell["params"]
    report = p["report"]
    R = int(p["replicates"])
    seed = int(p["archive_seed"])
    args = (A["labels"], A["runs"], A["clusters"])
    T = P["tolerances"]
    mut = _mutation()
    checks: dict = {}
    payload: dict

    def endpoint_tol(archived_row):
        return AR.percentile_mc_tolerance(float(archived_row["bootstrap_sd"]) / 100.0, R,
                                          int(P["archive"]["replicates"]),
                                          z_target=float(T["interval_endpoint_mc_standard_errors"]),
                                          shape_inflation=float(T["interval_endpoint_shape_inflation"])) * 100.0

    if report == "per_channel_effects":
        payload = CR.per_channel_effects(*args, replicates=R, seed=seed)
        ref = {r["channel_label"]: r for r in A["reference"]["per_channel_effects"]}
        rows, ok = {}, True
        for row in payload["rows"]:
            a = ref[row["channel_label"]]
            d_pt = abs(row["fraction"] - float(a["fraction"]))
            tol = AR.percentile_mc_tolerance(float(a["event_iid"]["bootstrap_sd"]), R,
                                             int(P["archive"]["replicates"]),
                                             z_target=float(T["interval_endpoint_mc_standard_errors"]),
                                             shape_inflation=float(T["interval_endpoint_shape_inflation"]))
            eps = {k: abs(row["event_iid"][k] - float(a["event_iid"][k])) for k in AR.ENDPOINT_KEYS}
            row_ok = bool(d_pt <= float(T["point_estimate_absolute"]) and max(eps.values()) <= tol)
            rows[row["channel_label"]] = {"fraction_abs_deviation": d_pt, "endpoint_abs_deviations": eps,
                                          "endpoint_tolerance": tol, "within_tolerance": row_ok}
            ok = ok and row_ok
        checks["reproduces_archived_per_channel_effects"] = {"passed": ok, "per_channel": rows}
        checks["every_channel_shows_a_positive_reduction"] = {
            "passed": bool(all(r["fraction"] > 0 for r in payload["rows"])),
            "n_positive": int(sum(r["fraction"] > 0 for r in payload["rows"])), "n_channels": len(payload["rows"]),
            "note": "descriptive property of the archive, not an inference"}
    elif report == "leave_one_channel_out":
        payload = CR.leave_one_channel_out(*args, replicates=R, seed=seed)
        cmp = CR.compare_leave_one_out(payload["rows"], A["reference"]["leave_one_channel_out"],
                                       "dropped_channel", point_tol=float(T["point_estimate_absolute"]),
                                       endpoint_tol_fn=endpoint_tol)
        payload["comparison"] = cmp
        checks["reproduces_archived_leave_one_channel_out"] = {"passed": cmp["all_within_tolerance"],
                                                               "max_point_abs_deviation": cmp["max_point_abs_deviation"]}
    elif report == "leave_one_run_out":
        payload = CR.leave_one_run_out(*args, replicates=R, seed=seed)
        cmp = CR.compare_leave_one_out(payload["rows"], A["reference"]["leave_one_run_out"],
                                       "dropped_run", point_tol=float(T["point_estimate_absolute"]),
                                       endpoint_tol_fn=endpoint_tol)
        payload["comparison"] = cmp
        checks["reproduces_archived_leave_one_run_out"] = {"passed": cmp["all_within_tolerance"],
                                                           "max_point_abs_deviation": cmp["max_point_abs_deviation"]}
        checks["two_runs_only"] = {"passed": bool(payload["n_runs"] == 2), "n_runs": payload["n_runs"],
                                   "note": "with two runs a leave-one-run-out is a single-run analysis"}
    elif report == "run_stratified":
        payload = CR.run_stratified(*args, replicates=R, seed=seed)
        checks["run_33_has_two_channels"] = {
            "passed": bool(any(s["run"] == "33" and s["n_channels"] == 2 for s in payload["strata"])),
            "strata": {s["run"]: s["n_channels"] for s in payload["strata"]}}
        checks["cluster_interval_flagged_as_coarse"] = {
            "passed": bool(all(s["cluster_interval_is_coarse"] == (s["n_channels"] <= 3) for s in payload["strata"]))}
    elif report == "variance_decomposition":
        claim = bool(mut == "between_run_from_two_runs")
        payload = CR.variance_decomposition(*args, replicates=R, seed=seed,
                                            min_runs_for_between_run=int(P["cluster_reporting"]["min_runs_for_between_run_variance"]),
                                            claim_between_run_identified=claim)
        problems = CR.check_variance_decomposition(
            payload, min_runs=int(P["cluster_reporting"]["min_runs_for_between_run_variance"]))
        checks["between_run_component_not_reported_as_identified"] = {
            "passed": not problems, "problems": problems,
            "n_runs": payload["n_runs"],
            "min_runs_required": int(P["cluster_reporting"]["min_runs_for_between_run_variance"])}
        checks["within_and_between_channel_components_reported"] = {
            "passed": bool(payload["within_channel"]["mean_bootstrap_variance"] >= 0.0
                           and payload["between_channel"]["moment_estimate"] >= 0.0),
            "within_channel": payload["within_channel"]["mean_bootstrap_variance"],
            "between_channel": payload["between_channel"]["moment_estimate"]}
    else:  # pragma: no cover - guarded by _check_grid_matches_protocol
        raise RuntimeError(f"unknown cluster report {report!r}")
    return {"kind": "cluster_report", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"report": report, "replicates": R, "mutation": mut, "payload": payload,
                        "label": CR.DESCRIPTIVE_LABEL}}


def _cell_science_control(P: dict, cell: dict) -> dict:
    p = cell["params"]
    name = p["control"]
    if name == "block1_equals_iid":
        r = AR.control_block1_equals_iid(seed=int(AR.ARCHIVE_SEED), n=int(p["n"]), k=int(p["k"]))
        checks = {"block1_identical_to_iid": {"passed": r["block1_identical_to_iid"]},
                  "block64_differs_from_iid": {"passed": r["block64_differs_from_iid"]}}
    elif name == "zero_effect_identity":
        A = _archive()
        r = AR.control_zero_effect(A["clusters"], replicates=int(p["replicates"]), seed=int(AR.ARCHIVE_SEED))
        checks = {"all_fractions_exactly_zero": {"passed": bool(r["max_abs_fraction"] == 0.0),
                                                 "max_abs_fraction": r["max_abs_fraction"]},
                  "interval_degenerate_at_zero": {"passed": bool(r["ci95_lo"] == 0.0 and r["ci95_hi"] == 0.0
                                                                 and r["bootstrap_sd"] == 0.0)}}
    elif name == "event_weight_constant":
        A = _archive()
        r = AR.control_event_weight_constant(A["resolved"])
        checks = {"archived_event_weight_constant": {"passed": r["event_weight_constant"],
                                                     "value": r["event_weight_value"]},
                  "perturbed_weights_detected": {"passed": r["perturbed_copy_detected"]}}
    elif name == "empca_uniform_weight_limit":
        r = RD.control_uniform_weight_limit(n_events=int(p["n_events"]), n_samples=int(p["n_samples"]),
                                            rank=int(p["rank"]), niter=int(p["niter"]),
                                            tolerance=float(p["tolerance"]))
        checks = {"uniform_weights_reduce_empca_to_pca": {"passed": r["uniform_weight_limit_holds"],
                                                          "relative_difference": r["relative_difference_flat"],
                                                          "tolerance": r["tolerance"]},
                  "weighting_changes_the_problem": {"passed": r["weighting_changes_the_problem"]}}
        if int(r["n_fits"]) != int(fitcount.observed()):
            raise SchemaError(f"{cell['cell_id']}: the control's own tally is {r['n_fits']} fits but "
                              f"{fitcount.observed()} estimator invocations were instrumented")
    else:  # pragma: no cover
        raise RuntimeError(f"unknown control {name!r}")
    return {"kind": "science_control", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"control": name, "detail": r, "replicates": int(p.get("replicates", 0))}}


def _cell_calibration_control(P: dict, cell: dict) -> dict:
    p = cell["params"]
    gen = P["calibration"]["generator"]
    r = CB.null_dependence_control(
        n_runs=int(p["n_runs"]), channels_per_run=int(p["channels_per_run"]), n_events=int(p["n_events"]),
        sd_channel=float(gen["sd_channel"]), sd_run_alt=float(p["sd_run_alt"]), theta=float(gen["theta"]),
        sigma_p=float(gen["sigma_p"]), sigma_e=float(gen["sigma_e"]), mc_replicates=int(p["mc_replicates"]),
        bootstrap_replicates=int(p["bootstrap_replicates"]), seed=int(p["control_seed"]),
        coverage_band=list(P["calibration"]["null_arm_coverage_band"]))
    checks = {"null_arm_covers_near_nominal": {"passed": r["null_arm_in_band"],
                                               "coverage": r["null_arm"]["coverage"],
                                               "band": r["coverage_band_for_null_arm"]},
              "between_run_variance_degrades_coverage": {"passed": r["dependence_degrades_coverage"],
                                                         "coverage_drop": r["coverage_drop"]}}
    return {"kind": "calibration_control", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"control": "null_dependence", "detail": r, "label": CB.CALIBRATION_LABEL}}


def _cell_calibration(P: dict, cell: dict) -> dict:
    p = cell["params"]
    d = CB.CalibrationDesign(**{k: p[k] for k in ("n_runs", "channels_per_run", "n_events", "sd_run",
                                                  "sd_channel", "theta", "sigma_p", "sigma_e",
                                                  "mc_replicates", "bootstrap_replicates", "alpha")})
    r = CB.coverage(d, seed=int(cell["seed"]))
    checks = {"coverage_estimated_with_monte_carlo_uncertainty": {
        "passed": bool(r["coverage_of_population_theta"]["n"] == int(d.mc_replicates)
                       and r["coverage_of_population_theta"]["half_width"] is not None),
        "wilson": r["coverage_of_population_theta"]},
        "is_calibration_of_a_procedure_not_evidence_about_cresst": {"passed": True,
                                                                    "note": CB.CALIBRATION_LABEL}}
    return {"kind": "calibration", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"coverage": r, "label": CB.CALIBRATION_LABEL}}


def _cell_robustness(P: dict, cell: dict, fixture: bool, ctx: dict) -> dict:
    p = cell["params"]
    if cell["role"] == "development":
        root = _FIXTURE_DATA_ROOT or (Path(ctx["out"]) / "fixture_data")
        F = P["matrix"]["fixture"]["robustness_dimensions"]
        if not (Path(root) / RD.INDEX_NAME).exists():
            RD.write_fixture_data_root(root, seed=int(cell["seed"]), channels=RD.ARCHIVE_CHANNELS[:2],
                                       n_events=int(F["n_events"]), n_noise=int(F["n_noise"]),
                                       n_samples=int(F["n_samples"]))
        data_root = root
    else:
        data_root = _CRESST_DATA_ROOT
    scored = RD.run_robustness_cell(P, p, data_root=data_root, seed=int(cell["seed"]), niter=int(p["niter"]))
    if int(scored["n_fits"]) != int(fitcount.observed()):
        raise SchemaError(f"{cell['cell_id']}: the cell's own tally is {scored['n_fits']} fits but "
                          f"{fitcount.observed()} estimator invocations were instrumented")
    checks = {"residuals_finite_and_positive": {
        "passed": bool(np.isfinite(scored["weighted_residual_pca"]) and scored["weighted_residual_pca"] > 0
                       and np.isfinite(scored["weighted_residual_empca"]) and scored["weighted_residual_empca"] > 0)},
        "held_out_events_present": {"passed": bool(scored["n_eval_events"] > 0 and scored["n_train_events"] > 0)},
        "development_fixture_is_not_cresst_data": {
            "passed": bool(scored["synthetic_fixture_data"] == (cell["role"] == "development")),
            "synthetic": scored["synthetic_fixture_data"]}}
    return {"kind": "robustness", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"scored": scored, "arm": p.get("arm", "development"),
                        "label": ("DEVELOPMENT ONLY: synthetic pulse fixture, never citable"
                                  if cell["role"] == "development" else
                                  "production robustness cell: requires <CRESST_DATA_ROOT>")}}


# --- reduce ----------------------------------------------------------------------------------------
def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    fixture = any(js.get("development_only") for js in cells.values())
    by_family = {f: [js for js in cells.values() if js["family"] == f] for f in FAMILIES}
    violations: list[str] = []

    # (1) every cell's own scientific checks
    for js in sorted(cells.values(), key=lambda j: j["cell_id"]):
        r = js["result"]
        if not r["passed"]:
            failed = sorted(k for k, v in r["checks"].items() if not v.get("passed", True))
            violations.append(f"{js['cell_id']} ({r['kind']}): failed checks {failed}")

    # (2) the cluster layer must add variance to the channel-fixed construction
    sd = {}
    for js in by_family["archive_interval"]:
        sd[js["result"]["summary"]["construction"]] = float(js["result"]["summary"]["recomputed"]["bootstrap_sd"])
    factor = float(P["tolerances"]["min_hierarchical_to_channel_fixed_sd_ratio"])
    pair_rows = {}
    for hier, fixed in AR.HIERARCHICAL_PAIRS:
        if hier in sd and fixed in sd:
            ratio = sd[hier] / sd[fixed] if sd[fixed] > 0 else float("inf")
            pair_rows[f"{hier}/{fixed}"] = {"sd_hierarchical": sd[hier], "sd_channel_fixed": sd[fixed],
                                            "ratio": float(ratio), "min_ratio": factor,
                                            "passed": bool(ratio >= factor)}
            if ratio < factor:
                violations.append(f"cluster layer adds no variance: sd({hier})/sd({fixed}) = {ratio:.3f} "
                                  f"< {factor} - a construction labelled hierarchical resampled only events")

    # (3) declared label vs the channel-layer fingerprint recorded inside each cell
    for js in by_family["archive_interval"]:
        c = js["result"]["checks"]["channel_layer_matches_its_label"]
        if not c["passed"]:
            violations.append(f"{js['result']['summary']['construction']}: declared "
                              f"channel_resampling does not match the channel layer actually used "
                              f"(mean distinct channels per replicate = "
                              f"{c['mean_distinct_channels_per_replicate']:.3f}, expected "
                              f"{c['expected_under_declared_rule']:.3f})")

    # (4) leave-one-run-out must equal the run-stratified estimate (two runs)
    reports = {js["result"]["summary"]["report"]: js["result"]["summary"]["payload"]
               for js in by_family["cluster_report"]}
    loro_identity = None
    if "leave_one_run_out" in reports and "run_stratified" in reports:
        loro_identity = CR.check_loro_matches_stratified(reports["leave_one_run_out"], reports["run_stratified"],
                                                         tol=float(P["tolerances"]["point_estimate_absolute"]))
        if not loro_identity["passed"]:
            violations.append("leave-one-run-out does not equal the run-stratified estimate of the retained run")

    # (5) the between-run variance gate
    if "variance_decomposition" in reports:
        probs = CR.check_variance_decomposition(
            reports["variance_decomposition"],
            min_runs=int(P["cluster_reporting"]["min_runs_for_between_run_variance"]))
        violations.extend(probs)

    if violations:
        raise SchemaError("scientific check failed: " + "; ".join(violations[:3])
                          + (f" (+{len(violations) - 3} more)" if len(violations) > 3 else ""))

    # --- summaries -----------------------------------------------------------------------------
    summary: dict = {"families_present": {f: len(v) for f, v in by_family.items() if v},
                     "replicates_by_role": _replicates_by_role(cells),
                     "scientific_checks": {"cluster_layer_adds_variance": pair_rows,
                                           "leave_one_run_out_equals_run_stratified": loro_identity}}
    ap = by_family["archive_point"]
    if ap:
        s = ap[0]["result"]["summary"]
        summary["archive_recomputation"] = {
            "point_estimates": s["point_estimates"], "comparison": s["comparison"],
            "archived_inputs": s["archived_inputs"],
            "verdict": ("every recomputed point estimate agrees with both archived sources within "
                        f"{s['comparison']['tolerance_absolute']:g}")}
    if by_family["archive_interval"]:
        rows = {}
        for js in sorted(by_family["archive_interval"], key=lambda j: j["result"]["summary"]["construction"]):
            s = js["result"]["summary"]
            rows[s["construction"]] = {"recomputed": {k: s["recomputed"][k] for k in AR.ENDPOINT_KEYS},
                                       "archived": {k: float(s["archived"][k]) for k in AR.ENDPOINT_KEYS},
                                       "max_abs_deviation": s["comparison"]["max_abs_deviation"],
                                       "tolerance": s["comparison"]["tolerance"],
                                       "bit_identical_to_archive": s["comparison"]["bit_identical_to_archive"],
                                       "replicates": s["replicates"],
                                       "within_tolerance": s["comparison"]["all_within_tolerance"]}
        summary["archive_intervals"] = {
            "constructions": rows, "n_constructions": len(rows),
            "all_within_tolerance": all(r["within_tolerance"] for r in rows.values()),
            "n_bit_identical": int(sum(r["bit_identical_to_archive"] for r in rows.values())),
            "max_abs_deviation": max((r["max_abs_deviation"] for r in rows.values()), default=None)}
    if reports:
        summary["cluster_report"] = {k: reports[k] for k in sorted(reports)}
        summary["cluster_report"]["label"] = CR.DESCRIPTIVE_LABEL
    if by_family["science_control"] or by_family["calib_control"]:
        summary["controls"] = {js["result"]["summary"]["control"]:
                               {"passed": js["result"]["passed"], "checks": js["result"]["checks"]}
                               for js in by_family["science_control"] + by_family["calib_control"]}
    # calibration curves
    curves = {}
    for fam in ("calib_pilot", "calib_confirm"):
        groups: dict[str, list] = {}
        for js in by_family[fam]:
            cov = js["result"]["summary"]["coverage"]
            d = cov["design"]
            key = f"{fam}|C{int(d['channels_per_run'])}|sd_run{float(d['sd_run']):g}"
            groups.setdefault(key, []).append(cov)
        for key, rows in groups.items():
            curves[key] = CB.coverage_curve(rows)
    if curves:
        summary["calibration"] = {
            "coverage_curves": curves, "label": CB.CALIBRATION_LABEL,
            "reference_arm": P["calibration"]["reference_arm"],
            "nominal_level": 1.0 - float(P["calibration"]["alpha"]),
            "generator": P["calibration"]["generator"],
            "statement": ("Coverage of the archive's own hierarchical construction under a declared "
                          "between-run variance.  It is a property of the procedure; it is not evidence "
                          "about CRESST and it does not license a new-run claim.")}
    # robustness: what was enumerated, what ran, what could not
    rb = {}
    for fam in ("robustness_pilot", "robustness_confirm"):
        planned = [c for c in matrix.cells if c["family"] == fam and c["role"] != "development"]
        rows = RD.grid_rows(P, "pilot" if fam == "robustness_pilot" else "confirmation")
        rb[fam] = {"cells_enumerated": len(planned), "fits_planned": sum(c["n_fits"] for c in planned),
                   "executed_here": 0, "contrast_plan": RD.contrast_plan(rows),
                   "status": "not executed: requires <CRESST_DATA_ROOT> raw CRESST pulse traces, "
                             "which are not present in this environment"}
        dev = [js for js in by_family[fam] if js["role"] == "development"]
        if dev:
            rb[fam]["development_fixture_cells"] = [
                {"cell_id": js["cell_id"], "scored": js["result"]["summary"]["scored"],
                 "label": js["result"]["summary"]["label"]} for js in dev]
    summary["robustness_design"] = {**rb, "primary_setting": RD.primary_setting(P),
                                    "label": "prepared and unexecuted"}
    summary["statement"] = _statement(fixture, partial)
    return summary


def _replicates_by_role(cells: dict) -> dict:
    out: dict = {}
    for js in cells.values():
        role = js["role"]
        row = out.setdefault(role, {"cells": 0, "replicates": 0, "fits": 0})
        row["cells"] += 1
        row["replicates"] += int(js["params"].get("replicates_total", 0))
        row["fits"] += int(js["fits_used"])
    return out


def _statement(fixture: bool, partial: bool) -> str:
    s = ("CRESST-S is archive reporting.  It recomputes the archived channel-robustness analysis from the "
         "archived event residuals, reports the archive's cluster structure descriptively, prepares (but does "
         "not execute) a rank/shrinkage/time-cut robustness grid that needs raw CRESST pulse data absent from "
         "this environment, and calibrates the coverage of the archive's own hierarchical construction as a "
         "procedure under a declared dependence model.  There are seven run/channel clusters from TWO runs: "
         "nothing in this package establishes coverage for a new run or a new detector, no between-run variance "
         "component is identifiable, and every cluster-aware number here is a descriptive sensitivity of the "
         "archive, not population inference.  No new detector data exists here and none was sought.")
    if fixture:
        s += "  DEVELOPMENT ONLY: fixture output at reduced size, never citable."
    if partial:
        s += "  PARTIAL: not every enumerated cell is present."
    return s


# --- entry point ---------------------------------------------------------------------------------
_EXTRA_HELP = """
CRESST-S extra options (consumed by this runner before the common CLI):

  --archive-root PATH        MS evidence/data directory holding the archived CRESST inputs.
                             No default: the archive lives in the manuscript repository and is
                             addressed in the protocol by relative path plus SHA-256.
  --cresst-data-root PATH    <CRESST_DATA_ROOT>: raw CRESST pulse traces for the robustness grid.
                             Absent from this environment; those cells fail with
                             "input not available: ..." and are charged in the ledger.
  --fixture-data-root PATH   where to write the synthetic development pulse fixture
                             (default: <out>/fixture_data).
"""


def _take_option(argv: list[str], name: str):
    """Remove ``--name VALUE`` (or ``--name=VALUE``) from argv and return VALUE."""
    val = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == name:
            if i + 1 >= len(argv):
                raise cli.UsageError(f"{name} needs a path")
            val = argv[i + 1]
            del argv[i:i + 2]
            continue
        if a.startswith(name + "="):
            val = a.split("=", 1)[1]
            del argv[i]
            continue
        i += 1
    return val


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print(_EXTRA_HELP)
    try:
        archive = _take_option(argv, "--archive-root")
        cresst = _take_option(argv, "--cresst-data-root")
        fixture_root = _take_option(argv, "--fixture-data-root")
    except cli.UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 6
    set_roots(archive_root=archive, cresst_data_root=cresst, fixture_data_root=fixture_root)
    return cli.main_template(
        package=PACKAGE,
        description=("CRESST-S (L5): recomputation of the archived CRESST channel analysis, cluster-aware "
                     "descriptive reporting, a prepared and unexecuted robustness grid, and calibration of the "
                     "archive's hierarchical construction.  " + _EXTRA_HELP),
        build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run,
        result_required=result_required, source_files=SOURCE_FILES,
        fixture_overrides=lambda P: {"scale": "fixture", **P["matrix"]["fixture"]}, argv=argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
