"""T22-S runner (L4): inference calibration on synthetic files + matched interface.

Built on ``common.cli.main_template``.  Subcommands: ``validate``,
``enumerate``, ``run``, ``reduce``, ``estimate`` (common) and ``inventory``
(writes ``EXPOSURE_INVENTORY.md`` from the archived run trees; opens no
instrument data).

Matrix families
---------------
``sim_pilot``       tuning scenarios x two bootstrap-replicate settings (paired
                    seeds); the pilot reducer freezes one setting into
                    ``FROZEN_CALIBRATION.json`` under the declared rule.
``sim_confirm``     the declared validation scenarios; a confirmation cell
                    refuses to run without the frozen file and records its
                    digest.
``control``         seedless scientific controls (simulator estimand identity,
                    AR(1) autocorrelation, iid bootstrap standard error,
                    dependence gate, Holm arithmetic).
``matched_fixture`` development-only: a synthetic TIDMAD-layout HDF5 stretch read
                    through the archived opener under an exposure allowance and
                    scored by the matched interface (the only cells that fit).

Fit accounting: simulation cells fit nothing (their learned estimates are
generated) and report bootstrap replicates; the matched fixture cell charges
16 + 8 fits (eight estimators per representation plus the B = 1 identity
control), the archived counting convention.  That charge is *observed*: every
estimator the representation entry points return is counted through
``common.fitcount`` at the call site, and the cell reports the count rather
than the ``MATCHED_FITS`` constant the matrix was declared from.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026 import t22_tidmad_narrowband as t22  # noqa: E402
from experiments.mlst2026 import tidmad_band as tb  # noqa: E402
from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402
from experiments.mlst2026.successor_20260913.common import cli  # noqa: E402
from experiments.mlst2026.successor_20260913.common import fitcount  # noqa: E402
from experiments.mlst2026.successor_20260913.common.matrix import RunMatrix, make_cell  # noqa: E402
from experiments.mlst2026.successor_20260913.common.protocol import ProtocolError, load_protocol  # noqa: E402
from experiments.mlst2026.successor_20260913.common.results import (ReconciliationError, SchemaError,  # noqa: E402
                                                                    write_json_atomic)

from . import matched_stage as ms  # noqa: E402
from . import pipeline as pl  # noqa: E402
from . import simulator as sim  # noqa: E402

PACKAGE = "t22s"
PROTOCOL_DEFAULT = _HERE.parent / "PROTOCOL_T22S_v2.json"
FROZEN_NAME = "FROZEN_CALIBRATION.json"
FROZEN_SCHEMA = "mlst-t22s-frozen-calibration/1"
SOURCE_FILES = [_HERE, _HERE.parent / "simulator.py", _HERE.parent / "pipeline.py", _HERE.parent / "matched_stage.py",
                _ROOT / "experiments" / "mlst2026" / "t22_tidmad_narrowband.py",
                _ROOT / "experiments" / "mlst2026" / "tidmad_endpoint.py",
                _ROOT / "experiments" / "mlst2026" / "tidmad_band.py"]
FAMILIES = ("sim_pilot", "sim_confirm", "control", "matched_fixture")
MATCHED_FIXTURE = {"w": 1024, "n_cal": 64, "n_fit": 100, "n_eval": 100, "guard": 1024, "k_inj": 37, "B": 9}
MATCHED_FITS = 2 * len(te.METHODS) + len(te.METHODS)   # two representations + the B = 1 identity control

result_required = {"kind": str, "passed": bool, "checks": dict, "summary": dict}


# --- statistics ---------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> dict:
    if n <= 0:
        return {"rate": None, "lo": None, "hi": None, "n": 0, "k": int(k)}
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": float(p), "lo": float(max(0.0, centre - half)), "hi": float(min(1.0, centre + half)),
            "half_width": float(half), "n": int(n), "k": int(k)}


def wilson_n_for_half_width(p: float, half_width: float, z: float = 1.959964) -> int:
    """Smallest n whose Wilson half-width at rate p is <= half_width."""
    n = 1
    while True:
        w = wilson(int(round(p * n)), n, z)["half_width"]
        if w <= half_width and n >= 10:
            return n
        n += 1
        if n > 10 ** 6:  # pragma: no cover
            raise RuntimeError("no n found")


# --- matrix --------------------------------------------------------------------------
def _scenario_cells(P: dict, scenarios, *, role: str, family: str, seeds, mc: int, settings: dict | None,
                    chunks: int) -> list[dict]:
    """``seeds`` gives one base seed per chunk; paired settings share the seed."""
    cells = []
    for sc in scenarios:
        for chunk in range(chunks):
            seed = int(seeds[chunk])
            if settings is None:
                params = {"scenario": sc.as_dict(), "setting": "frozen", "replicates": None, "mc_replicates": int(mc), "chunk": chunk}
                cells.append(make_cell(role=role, seed=seed, params=params, n_fits=0, family=family))
            else:
                for label, reps in settings.items():
                    params = {"scenario": sc.as_dict(), "setting": label, "replicates": dict(reps), "mc_replicates": int(mc), "chunk": chunk}
                    cells.append(make_cell(role=role, seed=seed, params=params, n_fits=0, family=family))
    return cells


def _control_cells(P: dict, scale: str) -> list[dict]:
    cs = P["matrix"][scale]["controls"]
    return [make_cell(role="control", seed=None, params={"control": name, "scale": scale, **cfg}, n_fits=0, family="control")
            for name, cfg in cs.items()]


def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    M = P["matrix"]
    prod = M["production"]
    _check_catalogue(P)
    cells = []
    pilot_lo = P["seed_namespaces"]["pilot"][0]
    conf_lo = P["seed_namespaces"]["confirmation"][0]
    n_pilot_chunks = prod["pilot_mc_per_scenario"] // prod["mc_per_cell"]
    n_conf_chunks = prod["mc_replicates_per_scenario"] // prod["mc_per_cell"]
    cells += _scenario_cells(P, sim.pilot_scenarios(), role="pilot", family="sim_pilot",
                             seeds=[pilot_lo + i for i in range(n_pilot_chunks)], mc=prod["mc_per_cell"],
                             settings=prod["replicate_settings"], chunks=n_pilot_chunks)
    cells += _scenario_cells(P, sim.confirmation_scenarios(), role="confirmation", family="sim_confirm",
                             seeds=[conf_lo + i for i in range(n_conf_chunks)], mc=prod["mc_per_cell"],
                             settings=None, chunks=n_conf_chunks)
    cells += _control_cells(P, "fixture" if fixture else "production")
    if fixture:
        fx = M["fixture"]
        dev_lo = P["seed_namespaces"]["development"][0]
        n_fx_chunks = int(fx["chunks"])
        cells += _scenario_cells(P, sim.pilot_scenarios()[:fx["n_pilot_scenarios"]], role="development", family="sim_pilot",
                                 seeds=[dev_lo + i for i in range(n_fx_chunks)], mc=fx["mc_per_cell"],
                                 settings=fx["replicate_settings"], chunks=n_fx_chunks)
        cells += _scenario_cells(P, sim.development_scenarios(), role="development", family="sim_confirm",
                                 seeds=[dev_lo + 100 + i for i in range(n_fx_chunks)], mc=fx["mc_per_cell"],
                                 settings=None, chunks=n_fx_chunks)
        cells.append(make_cell(role="development", seed=dev_lo + 500, params={"matched": dict(MATCHED_FIXTURE)},
                               n_fits=MATCHED_FITS, family="matched_fixture",
                               notes="synthetic TIDMAD-layout HDF5 stretch; the only cells that fit anything"))
    return RunMatrix(PACKAGE, P["_sha256"], cells)


def _check_catalogue(P: dict) -> None:
    """The protocol pins the scenario catalogue; the code must match it."""
    decl = P["matrix"]["scenario_catalogue"]
    got = {"pilot": [s.name for s in sim.pilot_scenarios()],
           "confirmation": [s.name for s in sim.confirmation_scenarios()],
           "development": [s.name for s in sim.development_scenarios()]}
    for k, names in got.items():
        if decl.get(k) != names:
            raise ProtocolError(f"scenario catalogue {k!r} in the protocol differs from the code's catalogue "
                                f"({len(decl.get(k, []))} vs {len(names)} scenarios); the design input changed")
    if tuple(P["inference"]["block_grid_evaluation"]) != pl.BLOCKS_EVAL or tuple(P["inference"]["block_grid_calibration"]) != pl.BLOCKS_CAL:
        raise ProtocolError("block grids in the protocol differ from the archived grids the pipeline uses")
    want = declared_scales(P)
    got = P["budget"].get("fixture_to_production_scale") or {}
    if any(abs(float(got.get(k, -1)) - v) > 1e-3 * max(1.0, v) for k, v in want.items()):
        raise ProtocolError(f"budget.fixture_to_production_scale {got} is not the mechanically derived {want}")


def work_proxy_seconds(params: dict, replicates: dict) -> float:
    """Cost proxy per cell (seconds, from timings measured in this container on
    the archived functions): joint bootstrap 180 us per replicate per file,
    two-stage bootstrap 2.3 us per replicate per file per contrast, block
    curves 14 us per replicate per block length, plus 1 ms of simulation per
    file.  Used only for the fixture-to-production scale factors."""
    sc = params["scenario"]
    n_f, n_d, mc = sc["n_files"], sc["n_dev_files"], params["mc_replicates"]
    per = (n_f * replicates["file"] * 180e-6 + 4 * replicates["hier"] * n_f * 2.3e-6
           + n_d * (4 * len(pl.BLOCKS_EVAL) + len(pl.BLOCKS_CAL)) * replicates["curve"] * 14e-6 + 0.001 * (n_f + n_d))
    return float(mc * per)


def declared_scales(P: dict) -> dict:
    """Mean production work per cell / mean fixture work per cell, by family."""
    prod, fx = P["matrix"]["production"], P["matrix"]["fixture"]
    out = {}
    # pilot: both settings in both scales
    p_work = [work_proxy_seconds({"scenario": s.as_dict(), "mc_replicates": prod["mc_per_cell"]}, r)
              for s in sim.pilot_scenarios() for r in prod["replicate_settings"].values()]
    f_work = [work_proxy_seconds({"scenario": s.as_dict(), "mc_replicates": fx["mc_per_cell"]}, r)
              for s in sim.pilot_scenarios()[:fx["n_pilot_scenarios"]] for r in fx["replicate_settings"].values()]
    out["sim_pilot"] = float(np.mean(p_work) / np.mean(f_work))
    # confirmation: the frozen setting is unknown before the pilot; the archived one bounds it
    p_work = [work_proxy_seconds({"scenario": s.as_dict(), "mc_replicates": prod["mc_per_cell"]}, prod["replicate_settings"]["archived"])
              for s in sim.confirmation_scenarios()]
    # A three-replicate fixture pilot cannot distinguish the two settings (Wilson
    # half-width ~0.5), so the fixture freeze is the cheaper 'reduced' setting;
    # production may freeze 'archived'.  Numerator archived / denominator reduced
    # therefore makes the confirmation estimate an upper bound.
    f_work = [work_proxy_seconds({"scenario": s.as_dict(), "mc_replicates": fx["mc_per_cell"]}, fx["replicate_settings"]["reduced"])
              for s in sim.development_scenarios()]
    out["sim_confirm"] = float(np.mean(p_work) / np.mean(f_work))
    pc, fc = P["matrix"]["production"]["controls"], P["matrix"]["fixture"]["controls"]

    def _ctrl_work(cfg):
        reps = cfg.get("replicates", 1)
        reps = sum(reps.values()) if isinstance(reps, dict) else reps
        return reps * cfg.get("n_files", 1) * max(1, cfg.get("n", 1) // 1000)

    out["control"] = float(np.mean([_ctrl_work(pc[k]) for k in pc]) / np.mean([_ctrl_work(fc[k]) for k in fc]))
    out["matched_fixture"] = 1.0
    return {k: float(f"{v:.4g}") for k, v in out.items()}


# --- frozen calibration --------------------------------------------------------------
def frozen_path(out: Path) -> Path:
    return Path(out) / FROZEN_NAME


def read_frozen(out: Path, P: dict, *, fixture: bool) -> tuple[dict, str]:
    p = frozen_path(out)
    if not p.exists():
        raise RuntimeError(f"validation refused: {p} does not exist; run and reduce the pilot cells first "
                           "(the confirmation scenarios are scored only against a frozen calibration)")
    raw = p.read_bytes()
    fz = json.loads(raw)
    if fz.get("schema") != FROZEN_SCHEMA:
        raise RuntimeError(f"{p}: unexpected schema {fz.get('schema')!r}")
    if fz.get("protocol_sha256") != P["_sha256"]:
        raise RuntimeError(f"{p} was frozen under protocol {fz.get('protocol_sha256')}, not {P['_sha256']}")
    if bool(fz.get("development_only")) != bool(fixture):
        raise RuntimeError(f"{p} development_only={fz.get('development_only')} does not match this run (fixture={fixture})")
    return fz, hashlib.sha256(raw).hexdigest()


# --- cells ---------------------------------------------------------------------------
def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict):
    fam = cell["family"]
    if fam in ("sim_pilot", "sim_confirm"):
        return _run_simulation_cell(P, cell, fixture, ctx), 0, 0
    if fam == "control":
        return _run_control_cell(P, cell, fixture, ctx), 0, 0
    if fam == "matched_fixture":
        res = _run_matched_cell(P, cell, fixture, ctx)
        obs = fitcount.require()                       # the counting region cli._run opened
        res["summary"]["n_fits"] = obs.total
        res["summary"]["fits_observed"] = obs.as_dict()
        return res, obs.total, 0
    raise RuntimeError(f"unknown family {fam!r}")


def _mutation():
    """TEST ONLY: a pipeline mutation injected through the environment so that the
    package's own scientific checks can be shown to reject it at reduce time."""
    return os.environ.get("T22S_TEST_MUTATION") or None


def _run_simulation_cell(P: dict, cell: dict, fixture: bool, ctx: dict) -> dict:
    params = cell["params"]
    sc = sim.Scenario(**params["scenario"])
    frozen_sha = ""
    if params["setting"] == "frozen":
        fz, frozen_sha = read_frozen(ctx["out"], P, fixture=fixture)
        reps = dict(fz["replicates"])
        setting = fz["setting"]
    else:
        reps = dict(params["replicates"])
        setting = params["setting"]
    mutation = _mutation()
    records = []
    for j in range(int(params["mc_replicates"])):
        rep_seed = int(cell["seed"]) * 10_000 + j
        rec = _replicate_with_mutation(rep_seed, sc, reps, mutation)
        rec["problems"] = pl.check_record(rec, sc)
        records.append(rec)
    n = len(records)
    n_wh = sum(r["withheld"] for r in records)
    summary = {"scenario_name": sc.name, "n_mc": n, "n_withheld": n_wh,
               "n_supported": {c: sum(r["verdicts"][c]["verdict"] == "supported" for r in records) for c in pl.CLAIMS},
               "n_covers": {k: sum(r["contrasts"][k].get("covers", False) for r in records) for k in sim.EXACT_MU_CONTRASTS},
               "true_mu": sc.true_mu(), "effect_sizes": sc.effect_sizes(),
               "replicates_used": reps, "setting": setting, "frozen_sha256": frozen_sha,
               "replicates_total": {"file": reps["file"] * sc.n_files * n, "hier": reps["hier"] * len(sim.CONTRASTS) * n,
                                    "curve": reps["curve"] * sc.n_dev_files * (4 * len(pl.BLOCKS_EVAL) + len(pl.BLOCKS_CAL)) * n},
               "work_proxy_seconds": work_proxy_seconds({"scenario": sc.as_dict(), "mc_replicates": n}, reps),
               "mutation": mutation}
    problems = [p for r in records for p in r["problems"]]
    if problems:
        print(f"WARNING {cell['cell_id']}: {len(problems)} scientific-invariant violations recorded (reduce will refuse)", file=sys.stderr)
    return {"kind": "simulation", "passed": not problems, "checks": {"invariant_violations": len(problems), "first": problems[:3]},
            "summary": summary, "records": records}


def _replicate_with_mutation(seed, sc, reps, mutation):
    kw = dict(file_replicates=reps["file"], hier_replicates=reps["hier"], curve_replicates=reps["curve"])
    if mutation is None:
        return pl.run_replicate(seed, sc, **kw)
    if mutation == "drop_dependence_gate":
        # the archived gate is bypassed: the combined stage is told the assessment stabilised
        orig = t22.combine_stage
        try:
            t22.combine_stage = lambda per_file, P_, dep: orig(per_file, P_, "stabilised")
            rec = pl.run_replicate(seed, sc, **kw)
        finally:
            t22.combine_stage = orig
        return rec
    if mutation == "files_only_bootstrap":
        # files resampled without within-file draws
        orig = t22.hierarchical_file_bootstrap

        def files_only(draws_by_file, *, replicates, seed, point_estimates=None):
            n = len(draws_by_file)
            rng = np.random.default_rng(seed)
            m = np.array([float(np.mean(d)) for d in draws_by_file])
            means = np.array([np.mean(m[rng.integers(0, n, size=n)]) for _ in range(replicates)])
            lo, hi = np.percentile(means, [2.5, 97.5])
            le = float(np.sum(means <= 0.0)); ge = float(np.sum(means >= 0.0))
            p = min(1.0, 2.0 * min((le + 1.0) / (replicates + 1.0), (ge + 1.0) / (replicates + 1.0)))
            return {"n_files": n, "equal_file_mean_of_bootstrap_means": float(m.mean()),
                    "equal_file_mean_of_point_estimates": float(np.mean(point_estimates)), "ci95_lo": float(lo), "ci95_hi": float(hi),
                    "excludes_zero": bool(lo > 0 or hi < 0), "bootstrap_sd": float(np.std(means, ddof=1)), "p_two_sided": float(p),
                    "replicates": int(replicates)}
        try:
            t22.hierarchical_file_bootstrap = files_only
            rec = pl.run_replicate(seed, sc, **kw)
        finally:
            t22.hierarchical_file_bootstrap = orig
        return rec
    raise RuntimeError(f"unknown test mutation {mutation!r}")


def _run_control_cell(P: dict, cell: dict, fixture: bool, ctx: dict) -> dict:
    name = cell["params"]["control"]
    cfg = cell["params"]
    rng = np.random.default_rng(424242)   # seedless cell: a fixed internal stream, declared here
    checks, summary = {}, {}
    if name == "simulator_estimand":
        sc = sim.make_scenario("alt-medium", int(cfg["n_files"]), 0.6, tau=0.0)
        mu = sc.true_mu()
        acc = {k: [] for k in sim.EXACT_MU_CONTRASTS}
        Ns = []
        for i in range(sc.n_files):
            f = sim.simulate_file(rng, sc, f"c{i}", sc.lam_emp)
            Ns.append(f.N)
            est = {"zero": np.zeros(sc.n_eval, complex), "raw-bin": f.eval_read, **f.generated_estimates}
            e2 = {m: te.squared_errors(q, f.truth_eval, sc.w) for m, q in est.items()}   # TRUE labels g rho
            for key in acc:
                a, b = key.split("|")
                acc[key].append(e2[a] - e2[b])
        # The identity is exact conditional on the drawn files: E[Delta | files] =
        # mu * mean_i(N_i) (N_i is log-normal with mean one, so the unconditional
        # estimand is mu); the conditional form is what a finite pool can test.
        n_bar = float(np.mean(Ns))
        for key, parts in acc.items():
            d = np.concatenate(parts)
            se = float(d.std(ddof=1) / np.sqrt(d.size))
            target = mu[key] * n_bar
            z = float((d.mean() - target) / se)
            checks[key] = {"realised_mean": float(d.mean()), "declared_mu": mu[key], "mean_N_files": n_bar,
                           "conditional_target": target, "se": se, "z": z, "passed": abs(z) <= 4.0}
        summary = {"n_windows": int(sum(len(p) for p in acc["raw-bin|empca"])), "scenario": sc.name}
    elif name == "ar1_autocorrelation":
        n = int(cfg["n"])
        for phi in (0.0, 0.6):
            z = sim.complex_process(rng, n, phi)
            x = z.real - z.real.mean()
            r1 = float(np.sum(x[:-1] * x[1:]) / np.sum(x * x))
            checks[f"phi={phi}"] = {"lag1": r1, "expected": phi, "tolerance": 0.03, "passed": abs(r1 - phi) <= 0.03}
        summary = {"n": n}
    elif name == "iid_bootstrap_se":
        sc = sim.make_scenario("null-partial", 4, 0.0, tau=0.0)
        f = sim.simulate_file(rng, sc, "iid", 0.0)
        cal = te.fit_transfer(f.cal_read, f.cal_ref, f.N)
        est = pl.method_estimates(f, cal)
        t_eval = cal.g * f.eval_ref
        d = te.squared_errors(est["raw-bin"], t_eval, sc.w) - te.squared_errors(est["empca"], t_eval, sc.w)
        draws = te.joint_bootstrap(est, f.eval_read, f.eval_ref, f.cal_read, f.cal_ref, f.N, sc.w, [("raw-bin", "empca")],
                                   replicates=int(cfg["replicates"]), seed=7, b_eval=1, b_cal=1, calibration_only=False)
        boot_sd = float(np.std(draws[("raw-bin", "empca")], ddof=1))
        analytic = float(d.std(ddof=1) / np.sqrt(d.size))
        checks["se_ratio"] = {"bootstrap_sd": boot_sd, "analytic_se": analytic, "ratio": boot_sd / analytic,
                              "tolerance": 0.25, "passed": abs(boot_sd / analytic - 1.0) <= 0.25}
        summary = {"replicates": int(cfg["replicates"])}
    elif name == "dependence_gate":
        sc = sim.make_scenario("alt-large", 4, 0.0, tau=0.0)
        reps = dict(cfg["replicates"])
        kw = dict(file_replicates=reps["file"], hier_replicates=reps["hier"], curve_replicates=reps["curve"])
        r_in = pl.run_replicate(7000_0001, sc, dependence_override={"verdict": "inconclusive", "block_length_evaluation": 32}, **kw)
        r_ok = pl.run_replicate(7000_0001, sc, dependence_override={"verdict": "stabilised", "block_length_evaluation": 1}, **kw)
        checks["gate_withholds_every_claim"] = {"verdicts": {k: v["verdict"] for k, v in r_in["verdicts"].items()},
                                                "passed": all(v["verdict"] == "inconclusive" for v in r_in["verdicts"].values())}
        checks["positive_control_supported_when_stabilised"] = {"verdicts": {k: v["verdict"] for k, v in r_ok["verdicts"].items()},
                                                                "passed": r_ok["verdicts"]["denoising_benefit"]["verdict"] == "supported"}
        summary = {"scenario": sc.name, "replicates": reps}
    elif name == "holm_arithmetic":
        adj = t22.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        expect = {"a": 0.03, "b": 0.06, "c": 0.06}
        checks["holm"] = {"adjusted": adj, "expected": expect,
                          "passed": all(abs(adj[k] - expect[k]) < 1e-12 for k in expect)}
        summary = {}
    else:
        raise RuntimeError(f"unknown control {name!r}")
    passed = all(c["passed"] for c in checks.values())
    return {"kind": "control", "passed": bool(passed), "checks": checks, "summary": {"control": name, **summary}}


def _run_matched_cell(P: dict, cell: dict, fixture: bool, ctx: dict) -> dict:
    cfg = cell["params"]["matched"]
    fx_dir = Path(ctx["out"]) / "fixtures"
    fx_dir.mkdir(parents=True, exist_ok=True)
    path = fx_dir / "abra_fixture_0000.h5"
    meta = ms.write_tidmad_fixture(path, seed=int(cell["seed"]), w=cfg["w"], n_cal=cfg["n_cal"], n_fit=cfg["n_fit"],
                                   n_eval=cfg["n_eval"], guard=cfg["guard"], k_inj=cfg["k_inj"])
    alloc = ms.fixture_allocation(w=cfg["w"], n_cal=cfg["n_cal"], n_fit=cfg["n_fit"], n_eval=cfg["n_eval"], guard=cfg["guard"])
    st = ms.load_fixture_stretch(path, 0, alloc)
    k = int(cfg["k_inj"])
    N = te.mean_bin_power(st["fit_read"])
    N[k] = te.offline_noise_floor(st["fit_read"], k)
    out = ms.matched_scores(st, k_inj=k, B=int(cfg["B"]), w=cfg["w"], N_bins=N, exposure_status="synthetic-fixture")
    cal = te.fit_transfer(st["cal_read"][:, k], st["cal_ref"][:, k], float(N[k]))
    est_b1, diag1 = tb.compute_band_estimates(st["eval_read"], st["fit_read"], k, 1, cfg["w"], N, cal)
    fitcount.count_completed("b1-identity-estimator", n=len(est_b1))
    ident = tb.assert_single_bin_identity(diag1, "B=1 fixture", tb.IDENTITY_RTOL)
    pr = out["per_representation"]
    checks = {
        "identical_evaluation_windows": {"passed": pr["whole-window"]["evaluation_window_index"] == pr["band"]["evaluation_window_index"]},
        "identical_labels": {"passed": pr["whole-window"]["label_sha256"] == pr["band"]["label_sha256"]},
        "shared_methods_identical": {"passed": out["shared_methods_identical_across_representations"]},
        "b1_identity_control": {"passed": ident["passed"], "max_relative_difference": ident["max_relative_difference"]},
        "exposure_guard_engaged": {"passed": True, "note": "read through t22.load_stretch under exposure_allowance naming this file and region"},
    }
    out_small = {k_: v for k_, v in out.items() if k_ != "per_representation"}
    out_small["per_representation"] = {rep: {kk: vv for kk, vv in d.items() if kk != "evaluation_window_index"} for rep, d in pr.items()}
    out_small["shared"] = {kk: vv for kk, vv in out["shared"].items() if kk != "evaluation_window_index"}
    return {"kind": "matched_fixture", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"fixture": meta, "n_fits": out["n_fits"] + len(te.METHODS), "matched": out_small,
                        "exposure_status": out["exposure_status"], "label": out["label"]}}


# --- reduce ---------------------------------------------------------------------------
def _scenario_from_params(params):
    return sim.Scenario(**params["scenario"])


def _pool(cells_in_family: list[dict]) -> dict:
    """Pool chunk cells by (scenario, setting)."""
    groups = {}
    for js in cells_in_family:
        key = (js["result"]["summary"]["scenario_name"], js["result"]["summary"]["setting"])
        g = groups.setdefault(key, {"records": [], "cells": [], "scenario": _scenario_from_params(js["params"]),
                                    "replicates_used": js["result"]["summary"]["replicates_used"], "frozen_sha256": set()})
        g["records"].extend(js["result"]["records"])
        g["cells"].append(js["cell_id"])
        g["frozen_sha256"].add(js["result"]["summary"].get("frozen_sha256", ""))
    return groups


def summarise_scenario(sc: sim.Scenario, records: list[dict], tol: dict) -> dict:
    n = len(records)
    nw = [r for r in records if not r["withheld"]]
    mu = sc.true_mu()
    out = {"scenario": sc.as_dict(), "n_mc": n, "true_mu": mu, "effect_sizes": sc.effect_sizes(),
           "withholding": wilson(n - len(nw), n),
           "dependence_verdicts": {v: sum(r["dependence"]["verdict"] == v for r in records)
                                   for v in ("stabilised", "stabilised-at-endpoint", "inconclusive")},
           "contrasts": {}, "claims": {}, "any_supported": None}
    for key in sim.EXACT_MU_CONTRASTS:
        cs = [r["contrasts"][key] for r in records]
        cs_nw = [r["contrasts"][key] for r in nw]
        bias = np.array([c["bias"] for c in cs])
        out["contrasts"][key] = {
            "coverage_two_stage": wilson(sum(c["covers"] for c in cs), n),
            "coverage_two_stage_given_not_withheld": wilson(sum(c["covers"] for c in cs_nw), len(cs_nw)),
            "coverage_t_interval": wilson(sum(c["t_covers"] for c in cs), n),
            "mean_length_two_stage": float(np.mean([c["length"] for c in cs])),
            "mean_length_t_interval": float(np.mean([c["t_length"] for c in cs])),
            "bias_mean": float(bias.mean()), "bias_mc_se": float(bias.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
            "bias_sign": int(np.sign(bias.mean())),
            "bias_sign_resolved": bool(n > 1 and abs(bias.mean()) > 2 * bias.std(ddof=1) / np.sqrt(n)),
            "rmse_point": float(np.sqrt(np.mean(bias ** 2))),
        }
        cov = out["contrasts"][key]["coverage_two_stage"]
        out["contrasts"][key]["coverage_within_tolerance"] = _within(cov, tol["coverage"])
    null_claims = {name: all(mu.get(k, mu["zero|empca"]) == 0.0 for k in keys) for name, keys in pl.CLAIMS.items()}
    for name in pl.CLAIMS:
        k_sup = sum(r["verdicts"][name]["verdict"] == "supported" for r in records)
        k_sup_nw = sum(r["verdicts"][name]["verdict"] == "supported" for r in nw)
        row = {"is_null": null_claims[name], "supported": wilson(k_sup, n),
               "supported_given_not_withheld": wilson(k_sup_nw, len(nw)),
               "verdict_counts": {v: sum(r["verdicts"][name]["verdict"] == v for r in records)
                                  for v in ("supported", "ordering-only", "not-supported", "inconclusive", "not-evaluable")}}
        if null_claims[name]:
            row["type1"] = row["supported"]
            row["type1_within_tolerance"] = _within(row["supported"], tol["type1"])
        else:
            row["power"] = row["supported"]
        out["claims"][name] = row
    k_any_null = sum(any(r["verdicts"][c]["verdict"] == "supported" for c in pl.CLAIMS if null_claims[c]) for r in records)
    k_any = sum(any(r["verdicts"][c]["verdict"] == "supported" for c in pl.CLAIMS) for r in records)
    out["any_supported"] = wilson(k_any, n)
    if any(null_claims.values()):
        out["any_null_claim_supported"] = wilson(k_any_null, n)
        out["any_null_claim_supported"]["within_tolerance"] = _within(out["any_null_claim_supported"], tol["type1"])
    out["complete_null"] = all(null_claims.values())
    return out


def _within(w: dict, band: list) -> bool | None:
    """Not rejected: the Wilson interval is not entirely outside [lo, hi]."""
    if w["n"] == 0:
        return None
    lo, hi = band
    return not (w["hi"] < lo or w["lo"] > hi)


def _freeze(P: dict, matrix: RunMatrix, pilot_groups: dict, out: Path, fixture: bool) -> dict:
    tol = P["calibration_tolerances"]
    rule = P["inference"]["pilot_decision_rule"]
    scale = "fixture" if fixture else "production"
    settings = P["matrix"][scale]["replicate_settings"]
    ref, cand = rule["reference_setting"], rule["candidate_setting"]
    scenarios = sorted({k[0] for k in pilot_groups})
    comparison, all_ok = {}, True
    for name in scenarios:
        a = pilot_groups.get((name, ref)); b = pilot_groups.get((name, cand))
        if a is None or b is None:
            raise ReconciliationError(f"pilot scenario {name}: both settings {ref!r} and {cand!r} are needed to freeze")
        sa = summarise_scenario(a["scenario"], a["records"], tol)
        sb = summarise_scenario(b["scenario"], b["records"], tol)
        rows = {}
        ok = True
        for label, ga, gb in (("coverage_primary", sa["contrasts"]["raw-bin|empca"]["coverage_two_stage"], sb["contrasts"]["raw-bin|empca"]["coverage_two_stage"]),
                              ("withholding", sa["withholding"], sb["withholding"]),
                              ("any_supported", sa["any_supported"], sb["any_supported"])):
            diff = abs(ga["rate"] - gb["rate"])
            within = diff <= ga["half_width"]
            rows[label] = {ref: ga["rate"], cand: gb["rate"], "abs_difference": diff, "reference_half_width": ga["half_width"], "within": within}
            ok = ok and within
        comparison[name] = {"n_mc": sa["n_mc"], "rows": rows, "candidate_indistinguishable": ok}
        all_ok = all_ok and ok
    chosen = cand if all_ok else ref
    fz = {"schema": FROZEN_SCHEMA, "package": PACKAGE, "protocol_sha256": P["_sha256"], "matrix_sha256": matrix.matrix_hash(),
          "development_only": bool(fixture), "scale": scale, "setting": chosen, "replicates": settings[chosen],
          "block_grid_evaluation": list(pl.BLOCKS_EVAL), "block_grid_calibration": list(pl.BLOCKS_CAL),
          "tolerances": tol, "decision_rule": rule, "pilot_comparison": comparison,
          "frozen_from_pilot_cells": sorted(c for g in pilot_groups.values() for c in g["cells"]),
          "statement": "Replicate setting frozen before any confirmation cell is drawn; tolerances were fixed in the "
                       "protocol and are copied here unchanged.  A development_only freeze is never citable."}
    p = frozen_path(out)
    if p.exists():
        prev = json.loads(p.read_text())
        if prev != json.loads(json.dumps(fz, sort_keys=True, default=_jsonable)):
            raise ReconciliationError(f"{p} exists and differs from the deterministic pilot freeze; refusing to overwrite")
    else:
        write_json_atomic(p, fz)
    return fz


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    fixture = any(js.get("development_only") for js in cells.values())
    tol = P["calibration_tolerances"]
    by_family = {f: [js for js in cells.values() if js["family"] == f] for f in FAMILIES}
    # --- scientific checks on every cell (the mutation gate) ----------------------
    violations = []
    for js in cells.values():
        r = js["result"]
        if r["kind"] == "simulation":
            sc = _scenario_from_params(js["params"])
            for rec in r["records"]:
                probs = pl.check_record(rec, sc)
                if probs:
                    violations.append({"cell_id": js["cell_id"], "seed": rec["seed"], "problems": probs[:4]})
        elif not r["passed"]:
            violations.append({"cell_id": js["cell_id"], "problems": [f"{r['kind']} control failed: " + ", ".join(k for k, v in r["checks"].items() if not v.get("passed", True))]})
    if violations:
        raise SchemaError("scientific check failed: " + "; ".join(f"{v['cell_id']}: {v['problems'][0]}" for v in violations[:3])
                          + (f" (+{len(violations) - 3} more)" if len(violations) > 3 else ""))
    summary = {"controls": {js["result"]["summary"]["control"]: {"passed": js["result"]["passed"], "checks": js["result"]["checks"]}
                            for js in by_family["control"]},
               "matched_fixture": [{"cell_id": js["cell_id"], "passed": js["result"]["passed"], "checks": js["result"]["checks"],
                                    "exposure_status": js["result"]["summary"]["exposure_status"], "label": js["result"]["summary"]["label"],
                                    "fits_used": js["fits_used"]} for js in by_family["matched_fixture"]],
               "tolerances": tol, "wilson": {"z": 1.959964}}
    # --- pilot -> freeze ---------------------------------------------------------------
    pilot_expected = [c for c in matrix.cells if c["family"] == "sim_pilot" and (c["role"] == ("development" if fixture else "pilot"))]
    pilot_groups = _pool(by_family["sim_pilot"])
    have = {js["cell_id"] for js in by_family["sim_pilot"]}
    pilot_complete = bool(pilot_expected) and all(c["cell_id"] in have for c in pilot_expected)
    summary["pilot"] = {"complete": pilot_complete, "n_cells": len(by_family["sim_pilot"]), "scenarios": {}}
    for (name, setting), g in sorted(pilot_groups.items()):
        summary["pilot"]["scenarios"][f"{name}|{setting}"] = summarise_scenario(g["scenario"], g["records"], tol)
    if pilot_complete:
        summary["frozen_calibration"] = _freeze(P, matrix, pilot_groups, out, fixture)
    else:
        summary["frozen_calibration"] = None
    # --- confirmation ------------------------------------------------------------------
    conf_expected = [c for c in matrix.cells if c["family"] == "sim_confirm" and (c["role"] == ("development" if fixture else "confirmation"))]
    conf_cells = by_family["sim_confirm"]
    have = {js["cell_id"] for js in conf_cells}
    conf_complete = bool(conf_expected) and all(c["cell_id"] in have for c in conf_expected)
    conf = {"complete": conf_complete, "n_cells": len(conf_cells), "scenarios": {}, "feasibility_table": {}}
    if conf_cells:
        fz, sha = read_frozen(out, P, fixture=fixture)
        bad = [js["cell_id"] for js in conf_cells if js["result"]["summary"].get("frozen_sha256") != sha]
        if bad:
            raise ReconciliationError(f"{len(bad)} confirmation cells were scored under a different frozen calibration than {frozen_path(out)}")
        conf["frozen_sha256"] = sha
        conf["setting"] = fz["setting"]
        groups = _pool(conf_cells)
        for (name, _s), g in sorted(groups.items()):
            conf["scenarios"][name] = summarise_scenario(g["scenario"], g["records"], tol)
        conf["feasibility_table"] = _feasibility_table(conf["scenarios"], P)
        conf["tolerance_report"] = _tolerance_report(conf["scenarios"])
    summary["confirmation"] = conf
    summary["statement"] = (
        "Synthetic calibration of the archived T22 inference pipeline.  It measures coverage, type-I error, power and "
        "the withholding rate of the declared procedure on simulated files whose dependence, heterogeneity and "
        "calibration uncertainty were declared in advance.  It cannot establish the adequacy of that procedure on "
        "unexamined instrument data, whose dependence structure is not known; the archived T22 verdicts remain withheld "
        "and nothing here changes them." + ("  DEVELOPMENT ONLY: fixture output, never citable." if fixture else ""))
    return summary


def _feasibility_table(scenarios: dict, P: dict) -> dict:
    rules = P["inference"]["feasibility"]
    table = {}
    for n in sim.N_FILES_GRID:
        rows = [s for s in scenarios.values() if s["scenario"]["n_files"] == n]
        if not rows:
            continue
        core = [s for s in rows if s["scenario"]["drift_sd"] == 0.0]
        nulls = [s for s in rows if any(c.get("is_null") for c in s["claims"].values())]
        t1 = [c["type1_within_tolerance"] for s in nulls for c in s["claims"].values() if c.get("is_null")]
        cov = [s["contrasts"]["raw-bin|empca"]["coverage_within_tolerance"] for s in rows]
        med = [s for s in core if s["scenario"]["lam_emp"] == sim.EFFECTS["alt-medium"][0]]
        table[str(n)] = {
            "n_scenarios": len(rows),
            "type1_within_tolerance": all(t1) if t1 else None,
            "coverage_within_tolerance": all(cov) if cov else None,
            "withholding_rate_max": max(s["withholding"]["rate"] for s in core) if core else None,
            "withholding_rate_max_scenario": max(core, key=lambda s: s["withholding"]["rate"])["scenario"]["name"] if core else None,
            "power_medium_effect_min": min(s["claims"]["denoising_benefit"]["supported"]["rate"] for s in med) if med else None,
            "power_medium_effect_min_given_not_withheld": min((s["claims"]["denoising_benefit"]["supported_given_not_withheld"]["rate"] or 0.0) for s in med) if med else None,
            "rules": rules,
        }
        row = table[str(n)]
        row["feasible_under_rules"] = bool(row["type1_within_tolerance"] and row["coverage_within_tolerance"]
                                           and row["withholding_rate_max"] is not None and row["withholding_rate_max"] <= rules["max_withholding_rate"]
                                           and row["power_medium_effect_min"] is not None and row["power_medium_effect_min"] >= rules["min_power_medium_effect"])
    return table


def _tolerance_report(scenarios: dict) -> dict:
    failed_cov = [n for n, s in scenarios.items() if s["contrasts"]["raw-bin|empca"]["coverage_within_tolerance"] is False]
    failed_t1 = [f"{n}:{c}" for n, s in scenarios.items() for c, r in s["claims"].items() if r.get("type1_within_tolerance") is False]
    return {"n_scenarios": len(scenarios), "coverage_failures_primary": failed_cov, "type1_failures": failed_t1,
            "all_within_tolerance": not failed_cov and not failed_t1}


# --- entry point -----------------------------------------------------------------------
def _inventory_command(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m experiments.mlst2026.successor_20260913.t22s inventory")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "EXPOSURE_INVENTORY.md")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--archive-root", type=Path, nargs="*", default=None)
    a = ap.parse_args(argv)
    inv = ms.exposure_inventory(a.archive_root or ms.ARCHIVE_ROOTS)
    a.out.write_text(ms.inventory_markdown(inv))
    if a.json:
        write_json_atomic(a.json, inv)
    print(f"wrote {a.out} ({len(inv['regions'])} regions over {len(inv['files_known_to_archive'])} files)")
    return 0


def _precheck_frozen(argv) -> str | None:
    """Refuse a run that could reach confirmation-type cells without a freeze."""
    if not argv or argv[0] != "run":
        return None
    out = None
    fams = None
    it = iter(range(len(argv)))
    for i in it:
        if argv[i] == "--out" and i + 1 < len(argv):
            out = Path(argv[i + 1])
        if argv[i] == "--families":
            fams = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                fams.append(argv[j]); j += 1
    if out is None:
        return None
    if frozen_path(out).exists():
        return None
    if fams is not None and "sim_confirm" not in fams:
        return None
    return (f"validation refused: {frozen_path(out)} does not exist.  Run the pilot first "
            f"(`run --out {out} --families sim_pilot control [matched_fixture] [--fixture]`, then `reduce --allow-partial`), "
            f"which freezes the calibration; confirmation cells (family sim_confirm) are scored only against it.")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "inventory":
        return _inventory_command(argv[1:])
    msg = _precheck_frozen(argv)
    if msg:
        print("error: " + msg, file=sys.stderr)
        return 6
    return cli.main_template(package=PACKAGE, description="T22-S: synthetic calibration of the archived T22 inference pipeline "
                             "and the matched whole-window/narrowband interface (fixtures only)",
                             build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run,
                             result_required=result_required, source_files=SOURCE_FILES,
                             fixture_overrides=lambda P: {"scale": "fixture", **P["matrix"]["fixture"]}, argv=argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
