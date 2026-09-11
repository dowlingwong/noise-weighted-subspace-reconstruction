"""T10/T11 - TIDMAD stage runner (characterisation, development, validation, test).

Protocol: MS ``plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md`` T10/T11/T12.

Stages
------
``characterize``  Reads only the transfer-calibration windows of both channels
                  and records the declared outcome for each (file, w).  Runs
                  before any method is fitted, on development and pool files
                  alike; it uses no fit or evaluation window.
``development``   Scores the endpoint on the three development files across the
                  window grid, assesses dependence, and proposes the operating
                  point.  Development material only.
``freeze``        Writes ``PREDECLARED_T11.md`` and the frozen configuration
                  from the development stage.  Nothing downstream runs without it.
``validation``    The first three eligible-coupled pool files in index order.
``final-test``    Every remaining pool file, evaluated once each.

The validation and final-test stages refuse to start unless the freeze file
exists, so a method outcome on those files cannot be seen while tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402
from experiments.mlst2026.provenance import RunRecorder, write_json  # noqa: E402

# --- frozen allocation (T11) --------------------------------------------------
N_CAL, N_FIT, N_EVAL = 64, 350, 150
GUARD_SAMPLES = 16384
WINDOW_GRID = (4096, 16384, 65536)
STRETCH_SAMPLES = (N_CAL + N_FIT + N_EVAL) * max(WINDOW_GRID) + 2 * GUARD_SAMPLES
DEV_START = 7_097_152
POOL_START = 0
N_PSD_WINDOWS = 256
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260910
VARIANCE_CURVE_REPLICATES = 400
VARIANCE_CURVE_SEED = 20260911
SMOKE_SEED_BASE = 999_000_001  # declared disjoint from every scientific seed

DEV_FILES = ("abra_validation_0000.h5", "abra_validation_0005.h5",
             "abra_validation_0010.h5")
POOL_FILES = tuple(f"abra_validation_{i:04d}.h5" for i in
                   (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19))

def stable_offset(key: str) -> int:
    """Deterministic per-stretch seed offset.

    ``hash()`` on str is salted by PYTHONHASHSEED and differs between processes,
    so it must never appear in a seed. CRC32 is stable across runs and machines.
    """
    return int(zlib.crc32(key.encode())) % 100_000


SQUID = "timeseries/channel0001/timeseries"
REFERENCE = "timeseries/channel0002/timeseries"

#: Primary contrasts. The two usefulness references come first.
CONTRASTS = (("zero", "empca"), ("calibration-mean", "empca"), ("pca", "empca"),
             ("zero", "pca"), ("calibration-mean", "pca"),
             ("raw-bin", "empca"), ("wiener", "empca"), ("shrinkage-cal", "empca"),
             ("zero", "raw-bin"), ("zero", "wiener"))


def allocation(w: int) -> dict:
    used = (N_CAL + N_FIT + N_EVAL) * w + 2 * GUARD_SAMPLES
    cal0 = 0
    fit0 = N_CAL * w + GUARD_SAMPLES
    eval0 = fit0 + N_FIT * w + GUARD_SAMPLES
    return {"window_samples": w, "used_samples": used,
            "stretch_samples": STRETCH_SAMPLES,
            "leftover_samples": STRETCH_SAMPLES - used,
            "cal_offset": cal0, "fit_offset": fit0, "eval_offset": eval0,
            "guard_samples": GUARD_SAMPLES,
            "n_cal": N_CAL, "n_fit": N_FIT, "n_eval": N_EVAL}


def _read(h, dataset, start, length):
    ds = h[dataset]
    stop = min(int(start) + int(length), ds.shape[0])
    return np.asarray(ds[int(start):stop], dtype=np.float64)


def _windows(x: np.ndarray, offset: int, n: int, w: int) -> np.ndarray:
    seg = x[offset:offset + n * w]
    if seg.size < n * w:
        seg = seg[: (seg.size // w) * w]
    return seg.reshape(-1, w)


def load_stretch(path: Path, start: int, w: int) -> dict:
    """Read one stretch and return the unitary rFFT of each window block."""
    import h5py
    alloc = allocation(w)
    with h5py.File(path, "r") as h:
        x = _read(h, SQUID, start, alloc["used_samples"])
        r = _read(h, REFERENCE, start, alloc["used_samples"])
    out = {"allocation": alloc, "n_samples_read": int(min(x.size, r.size))}
    for role, off, n in (("cal", alloc["cal_offset"], N_CAL),
                         ("fit", alloc["fit_offset"], N_FIT),
                         ("eval", alloc["eval_offset"], N_EVAL)):
        wx = _windows(x, off, n, w)
        out[f"{role}_read"] = te.unitary_rfft(wx)
        if role in ("cal", "eval"):
            out[f"{role}_ref"] = te.unitary_rfft(_windows(r, off, n, w))
        out[f"n_{role}"] = int(wx.shape[0])
    return out


def noise_bin_power(science_paths, w: int, n_windows: int = N_PSD_WINDOWS) -> np.ndarray:
    """Readout noise bin power from science-file windows (no injection present).

    Science files carry only channel0001, and are disjoint from every validation
    file, so this is a genuine noise-only estimate.
    """
    import h5py
    need = n_windows * w
    acc, got = [], 0
    for p in science_paths:
        if got >= n_windows:
            break
        with h5py.File(p, "r") as h:
            x = _read(h, SQUID, 0, need)
        W = _windows(x, 0, min(n_windows - got, x.size // w), w)
        acc.append(te.mean_bin_power(te.unitary_rfft(W)) * W.shape[0])
        got += W.shape[0]
    if got == 0:
        raise RuntimeError("no science windows available for the noise PSD")
    return np.sum(acc, axis=0) / got


def score_stretch(st: dict, w: int, N_bins: np.ndarray, *, seed: int,
                  replicates: int, b_eval: int | None, b_cal: int | None,
                  curve_replicates: int = VARIANCE_CURVE_REPLICATES) -> dict:
    """Characterise, estimate, and score one stretch at one window length."""
    alloc = st["allocation"]
    ch = te.characterise(st["cal_ref"], st["cal_read"], w, N_bins,
                         alloc["used_samples"], st["n_eval"])
    k = ch["k_inj"]
    cal = te.fit_transfer(st["cal_read"][:, k], st["cal_ref"][:, k], float(N_bins[k]))
    est, diag = te.compute_estimates(st["eval_read"], st["fit_read"], k, w, N_bins, cal)
    t_eval = cal.g * st["eval_ref"][:, k]

    # per-window Delta series for every contrast (used for the dependence curves)
    e2 = {name: te.squared_errors(q, t_eval, w) for name, q in est.items()}
    deltas = {(a, b): e2[a] - e2[b] for (a, b) in CONTRASTS}

    curves = {f"{a}|{b}": te.block_variance_curve(
        d, te.BLOCK_CANDIDATES_EVAL, replicates=curve_replicates, seed=seed + 17)
        for (a, b), d in deltas.items()}
    acfs = {f"{a}|{b}": te.autocorrelation(d) for (a, b), d in deltas.items()}
    cal_series = np.abs(st["cal_read"][:, k] - cal.g * st["cal_ref"][:, k]) ** 2
    cal_curve = te.block_variance_curve(cal_series, te.BLOCK_CANDIDATES_CAL,
                                        replicates=curve_replicates, seed=seed + 29)
    cal_sel = te.select_block_length(cal_curve, te.BLOCK_CANDIDATES_CAL)

    use_eval = b_eval if b_eval is not None else te.select_block_length(
        curves[f"{CONTRASTS[0][0]}|{CONTRASTS[0][1]}"], te.BLOCK_CANDIDATES_EVAL)["block_length"]
    use_cal = b_cal if b_cal is not None else cal_sel["block_length"]

    draws = te.joint_bootstrap(est, st["eval_read"][:, k], st["eval_ref"][:, k],
                               st["cal_read"][:, k], st["cal_ref"][:, k],
                               float(N_bins[k]), w, list(CONTRASTS),
                               replicates=replicates, seed=seed,
                               b_eval=use_eval, b_cal=use_cal)
    draws_cal = te.joint_bootstrap(est, st["eval_read"][:, k], st["eval_ref"][:, k],
                                   st["cal_read"][:, k], st["cal_ref"][:, k],
                                   float(N_bins[k]), w, list(CONTRASTS),
                                   replicates=replicates, seed=seed + 1,
                                   b_eval=use_eval, b_cal=use_cal,
                                   calibration_only=True)

    contrasts = {}
    for (a, b) in CONTRASTS:
        obs = te.observed_contrast(est, st["eval_read"][:, k], t_eval, w, a, b)
        ci = te.percentile_interval(draws[(a, b)])
        ci_cal = te.percentile_interval(draws_cal[(a, b)])
        sens = {}
        for bb in sorted({max(1, use_eval // 2), min(16, use_eval * 2)}):
            if bb == use_eval:
                continue
            d2 = te.joint_bootstrap(est, st["eval_read"][:, k], st["eval_ref"][:, k],
                                    st["cal_read"][:, k], st["cal_ref"][:, k],
                                    float(N_bins[k]), w, [(a, b)],
                                    replicates=max(400, replicates // 4),
                                    seed=seed + 3 + bb, b_eval=bb, b_cal=use_cal)
            sens[f"b_eval={bb}"] = te.percentile_interval(d2[(a, b)])
        contrasts[f"{a}|{b}"] = {**obs, **ci, "ci_excludes_zero": ci["excludes_zero"],
                                 "calibration_only_component": ci_cal,
                                 "block_sensitivity": sens}

    return {
        "characterisation": ch,
        "transfer_calibration": cal.as_dict(),
        "estimator_diagnostics": diag,
        "allocation": alloc,
        "block_lengths_used": {"evaluation": int(use_eval), "calibration": int(use_cal)},
        "dependence": {"evaluation_curves": curves,
                       "evaluation_autocorrelation": acfs,
                       "calibration_curve": cal_curve,
                       "calibration_selection": cal_sel},
        "rmse_by_method_amplitude_units": {n: float(np.sqrt(np.mean(v)))
                                           for n, v in e2.items()},
        "contrasts": contrasts,
        "conditional_on": ["350 fit windows and the PCA/EMPCA bases and Wiener "
                           "signal PSD estimated from them",
                           "science-file noise bin powers",
                           "the injection bin k_inj",
                           "the block lengths"],
    }


# --- stages -------------------------------------------------------------------
def _science_paths(data_root: Path):
    return sorted((data_root / "tidmad" / "raw").glob("abra_science_*.h5"))


def _raw(data_root: Path, name: str) -> Path:
    return data_root / "tidmad" / "raw" / name


def stage_characterize(args, rec: RunRecorder) -> dict:
    """Declared outcome per (file, w) from transfer-calibration windows only."""
    data_root = Path(args.data_root)
    files = list(DEV_FILES) + list(POOL_FILES) if args.files == "all" else args.files.split(",")
    out = {}
    for w in WINDOW_GRID:
        N = noise_bin_power(_science_paths(data_root), w)
        alloc = allocation(w)
        for name in files:
            start = DEV_START if name in DEV_FILES else POOL_START
            path = _raw(data_root, name)
            if not path.exists():
                out[f"{name}|w={w}"] = {"error": "file not found", "path": str(path)}
                continue
            import h5py
            with h5py.File(path, "r") as h:
                x = _read(h, SQUID, start, N_CAL * w)
                r = _read(h, REFERENCE, start, N_CAL * w)
                n_total = h[SQUID].shape[0]
            Cx = te.unitary_rfft(_windows(x, 0, N_CAL, w))
            Cr = te.unitary_rfft(_windows(r, 0, N_CAL, w))
            n_eval_avail = min(N_EVAL, max(0, (n_total - start - alloc["eval_offset"]) // w))
            ch = te.characterise(Cr, Cx, w, N, alloc["used_samples"], int(n_eval_avail))
            ch["file"] = name
            ch["role"] = "development" if name in DEV_FILES else "pool"
            ch["start_sample"] = int(start)
            out[f"{name}|w={w}"] = ch
            print(f"  [{name} w={w:>5}] {ch['outcome']:<17} "
                  f"f={ch['f_inj_hz']/1e3:9.3f} kHz  refSNR={ch['reference_line_snr']:11.4g}  "
                  f"readSNR={ch['readout_line_snr']:9.4g}  drift={ch['drift_bins_over_stretch']:7.3f} bins"
                  f"{'  ' + ';'.join(ch['reasons']) if ch['reasons'] else ''}")
    return {"characterisation": out}


def stage_development(args, rec: RunRecorder, characterisation: dict) -> dict:
    """Score the endpoint on development files across the window grid."""
    data_root = Path(args.data_root)
    seed = SMOKE_SEED_BASE if args.smoke else BOOTSTRAP_SEED
    reps = 200 if args.smoke else BOOTSTRAP_REPLICATES
    per = {}
    for w in WINDOW_GRID:
        N = noise_bin_power(_science_paths(data_root), w)
        for name in DEV_FILES:
            key = f"{name}|w={w}"
            ch = characterisation.get(key, {})
            st = load_stretch(_raw(data_root, name), DEV_START, w)
            res = score_stretch(st, w, N, seed=seed + stable_offset(key),
                                replicates=reps, b_eval=None, b_cal=None)
            res["file"], res["window_samples"] = name, w
            per[key] = res
            c = res["contrasts"]["zero|empca"]
            print(f"  [{name} w={w:>5}] outcome={res['characterisation']['outcome']:<17} "
                  f"Delta(zero,EMPCA)={c['mean_delta_squared']:+.6g} "
                  f"CI[{c['ci95_lo']:+.4g},{c['ci95_hi']:+.4g}] b_eval={res['block_lengths_used']['evaluation']}")
    return {"development": per}


def choose_operating_point(dev: dict) -> dict:
    """Declared selection: maximise the precision-weighted mean over eligible
    development stretches of Delta(zero, EMPCA); ties -> smaller window."""
    rows, excluded = {}, []
    for key, res in dev.items():
        w = res["window_samples"]
        outcome = res["characterisation"]["outcome"]
        if outcome != "eligible-coupled":
            excluded.append({"key": key, "outcome": outcome})
            continue
        c = res["contrasts"]["zero|empca"]
        var = ((c["ci95_hi"] - c["ci95_lo"]) / (2 * 1.96)) ** 2
        rows.setdefault(w, []).append({"key": key,
                                       "mean_delta_squared": c["mean_delta_squared"],
                                       "bootstrap_variance": var})
    scored = {}
    for w, rs in rows.items():
        scored[w] = te.combine_files(rs)
        scored[w]["n_eligible_development_stretches"] = len(rs)
    if not scored:
        return {"selected_window": None, "verdict": "no-eligible-development-stretch",
                "per_window": scored, "excluded": excluded,
                "reason": "no development stretch was eligible-coupled at any "
                          "window length; the declared selection rule has no "
                          "input and no operating point can be frozen"}
    best = max(scored.items(), key=lambda kv: (kv[1]["theta_iv_precision_weighted"], -kv[0]))
    return {"selected_window": int(best[0]), "verdict": "selected",
            "criterion": "precision-weighted mean of Delta(zero, EMPCA) over "
                         "eligible development stretches; ties -> smaller window",
            "per_window": scored, "excluded": excluded}


def assign_pool_roles(characterisation: dict, w: int) -> dict:
    """Deterministic validation / final-test assignment, including the
    fewer-than-three-eligible case.

    Declared before running: validation is the first three pool files in index
    order whose outcome at the frozen window is *eligible-coupled*.  If fewer
    than three exist, every eligible file found becomes the validation set and
    the shortfall is recorded as an under-powered validation; if none exists,
    the validation set is empty, no confirmation of the frozen operating point
    is possible, the final test still evaluates every pool file so the failure
    regime is reported, and no "useful" verdict may be issued anywhere.
    """
    eligible, other = [], []
    for name in POOL_FILES:
        ch = characterisation.get(f"{name}|w={w}", {})
        (eligible if ch.get("outcome") == "eligible-coupled" else other).append(name)
    validation = eligible[:3]
    final_test = [f for f in POOL_FILES if f not in validation]
    status = ("complete" if len(validation) == 3
              else "underpowered" if validation else "empty")
    return {"window_samples": int(w), "validation_files": validation,
            "final_test_files": final_test,
            "n_eligible_coupled_pool_files": len(eligible),
            "eligible_pool_files": eligible,
            "non_eligible_pool_files": other,
            "validation_status": status,
            "useful_verdict_permitted": status != "empty",
            "note": {"complete": "three eligible-coupled pool files found",
                     "underpowered": f"only {len(validation)} eligible-coupled "
                                     f"pool file(s); validation is under-powered "
                                     f"and is labelled so",
                     "empty": "no eligible-coupled pool file; the frozen "
                              "operating point cannot be confirmed and no "
                              "'useful' verdict may be issued"}[status]}


def stage_evaluate(args, rec: RunRecorder, frozen: dict, files, label: str) -> dict:
    """Score the frozen operating point on a declared file list, once each."""
    data_root = Path(args.data_root)
    w = int(frozen["selected_window"])
    b_eval = int(frozen["block_length_evaluation"])
    b_cal = int(frozen["block_length_calibration"])
    N = noise_bin_power(_science_paths(data_root), w)
    per = {}
    for name in files:
        key = f"{name}|w={w}"
        st = load_stretch(_raw(data_root, name), POOL_START, w)
        res = score_stretch(st, w, N, seed=BOOTSTRAP_SEED + stable_offset(key),
                            replicates=BOOTSTRAP_REPLICATES,
                            b_eval=b_eval, b_cal=b_cal)
        res["file"], res["window_samples"], res["stage"] = name, w, label
        dep = frozen.get("dependence_verdict", "stabilised")
        vz = res["contrasts"]["zero|empca"]
        vc = res["contrasts"]["calibration-mean|empca"]
        res["verdict"] = te.usefulness_verdict(vz, vc, dep)
        if not res["characterisation"]["readout_unit_target_supported"]:
            res["verdict"] = {"verdict": "diagnostic-only",
                              "reason": "reference line SNR below the declared 100; "
                                        "the readout-unit target is not supported and "
                                        "this file contributes only the historical "
                                        "reference-unit diagnostic"}
        if res["characterisation"]["outcome"] != "eligible-coupled":
            res["in_combined_estimate"] = False
            res["verdict"] = {"verdict": "excluded-" + res["characterisation"]["outcome"],
                              "reason": "declared in advance: only eligible-coupled "
                                        "stretches enter the combined estimate"}
        else:
            res["in_combined_estimate"] = bool(
                res["characterisation"]["readout_unit_target_supported"])
        per[key] = res
        print(f"  [{label} {name}] outcome={res['characterisation']['outcome']:<17} "
              f"verdict={res['verdict']['verdict']:<22} "
              f"Delta(zero,EMPCA)={vz['mean_delta_squared']:+.6g} "
              f"CI[{vz['ci95_lo']:+.4g},{vz['ci95_hi']:+.4g}]")
    combined = {}
    for (a, b) in CONTRASTS:
        rows = [{"mean_delta_squared": r["contrasts"][f"{a}|{b}"]["mean_delta_squared"],
                 "bootstrap_variance": r["contrasts"][f"{a}|{b}"]["bootstrap_sd"] ** 2}
                for r in per.values() if r.get("in_combined_estimate")]
        combined[f"{a}|{b}"] = te.combine_files(rows) if rows else {
            "n_files": 0, "note": "no file entered the combined estimate"}
    return {label: {"per_file": per, "combined": combined,
                    "files_evaluated": list(files)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["characterize", "development", "freeze", "validation",
                             "final-test"])
    ap.add_argument("--data-root", default="/ceph/dwong/paper1_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", default="all")
    ap.add_argument("--characterisation", help="JSON from the characterize stage")
    ap.add_argument("--development", help="JSON from the development stage")
    ap.add_argument("--freeze-file", help="frozen configuration JSON")
    ap.add_argument("--smoke", action="store_true",
                    help="implementation check: few replicates, smoke seeds, "
                         "output marked non-scientific")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory {out_dir}")

    rec = RunRecorder(run_id=f"T10_tidmad_{args.stage}",
                      protocol_refs={"programme": "plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md#T10",
                                     "config": "plan/mlst_protocol_2026-09-10.json#tidmad"})
    rec.add_input("runner_source", Path(__file__).resolve())
    rec.add_input("endpoint_module", Path(te.__file__).resolve())
    rec.add_seed("bootstrap_seed", SMOKE_SEED_BASE if args.smoke else BOOTSTRAP_SEED)
    rec.add_seed("smoke", bool(args.smoke))
    data_root = Path(args.data_root)
    for p in _science_paths(data_root)[:2]:
        rec.add_input("science_noise", p)

    payload: dict = {"stage": args.stage, "smoke": bool(args.smoke),
                     "allocation_by_window": {str(w): allocation(w) for w in WINDOW_GRID}}

    if args.stage == "characterize":
        payload.update(stage_characterize(args, rec))

    elif args.stage == "development":
        chj = json.loads(Path(args.characterisation).read_text())
        rec.add_input("characterisation", Path(args.characterisation))
        payload.update(stage_development(args, rec, chj["characterisation"]))
        payload["operating_point_proposal"] = choose_operating_point(payload["development"])

    elif args.stage == "freeze":
        devj = json.loads(Path(args.development).read_text())
        chj = json.loads(Path(args.characterisation).read_text())
        rec.add_input("development", Path(args.development))
        rec.add_input("characterisation", Path(args.characterisation))
        prop = devj["operating_point_proposal"]
        if prop["selected_window"] is None:
            payload["frozen"] = {"selected_window": None, **prop}
        else:
            w = int(prop["selected_window"])
            curves, cal_curves, verdicts = [], [], []
            for key, res in devj["development"].items():
                if res["window_samples"] != w:
                    continue
                if res["characterisation"]["outcome"] != "eligible-coupled":
                    continue
                for name, c in res["dependence"]["evaluation_curves"].items():
                    curves.append({int(k): v for k, v in c.items()})
                cal_curves.append({int(k): v for k, v in
                                   res["dependence"]["calibration_curve"].items()})
            agg = te.aggregate_block_curves(curves, te.BLOCK_CANDIDATES_EVAL)
            agg_cal = te.aggregate_block_curves(cal_curves, te.BLOCK_CANDIDATES_CAL)
            roles = assign_pool_roles(chj["characterisation"], w)
            payload["frozen"] = {
                "selected_window": w,
                "rank": te.RANK,
                "methods": list(te.METHODS),
                "contrasts": [list(c) for c in CONTRASTS],
                "endpoint": "mean paired squared quadrature error difference in "
                            "readout physical-amplitude units a = 2q/sqrt(w)",
                "block_length_evaluation": int(agg["selection"]["block_length"]),
                "block_length_calibration": int(agg_cal["selection"]["block_length"]),
                "dependence_verdict": agg["selection"]["verdict"],
                "dependence_aggregate": agg,
                "dependence_aggregate_calibration": agg_cal,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "selection": prop,
                **roles,
            }
        payload["frozen"]["input_hashes"] = {
            r["role"] + ":" + Path(r["path"]).name: r.get("sha256")
            for r in rec.inputs if r.get("sha256")}

    elif args.stage in ("validation", "final-test"):
        if not args.freeze_file or not Path(args.freeze_file).exists():
            raise SystemExit("validation/final-test require --freeze-file; the "
                             "operating point must be frozen before any method "
                             "outcome on these files is computed")
        fz = json.loads(Path(args.freeze_file).read_text())["frozen"]
        rec.add_input("freeze", Path(args.freeze_file))
        if fz.get("selected_window") is None:
            raise SystemExit("no operating point was frozen; stage cannot run")
        files = (fz["validation_files"] if args.stage == "validation"
                 else fz["final_test_files"])
        if not files:
            payload[args.stage] = {"per_file": {}, "combined": {},
                                   "files_evaluated": [],
                                   "note": fz.get("note", "empty file list")}
        else:
            payload.update(stage_evaluate(args, rec, fz, files, args.stage))

    out_dir.mkdir(parents=True, exist_ok=True)
    payload["provenance"] = rec.finalize()
    digest = write_json(out_dir / f"t10_{args.stage}.json", payload)
    (out_dir / "RESULTS_SHA256").write_text(f"{digest}  t10_{args.stage}.json\n")
    print(f"[T10/{args.stage}] wrote {out_dir}/t10_{args.stage}.json sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
