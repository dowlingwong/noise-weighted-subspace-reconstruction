"""T22 - narrowband TIDMAD at the long window: stage runner.

Protocol: MS ``plan/mlst_protocol_2026-09-11-t22-stage0.json`` (the single
configuration source; this runner reads every number from it and asserts the
few it also derives) and ``plan/T22_STAGE0_PROTOCOL_2026-09-11.md``.

Stages
------
``characterize``  Reads only the 64 transfer-calibration windows of every
                  development and hold-out stretch and records the declared
                  outcome.  No method is fitted and no evaluation window is
                  touched, so running it on the hold-out stretches does not
                  expose an outcome.
``development``   Scores the endpoint on the development stretches across the
                  band grid, runs the B = 1 identity control, assesses
                  dependence and proposes the band under the declared rule.
``freeze``        Writes the frozen configuration (selected band, block
                  lengths, dependence verdict, file roles).  Nothing downstream
                  runs without it.
``validation``    The first three eligible-coupled hold-out files in index
                  order, at the frozen band, once.
``final-test``    Every remaining hold-out file, at the frozen band, once.

``validation`` and ``final-test`` refuse to start without ``--freeze-file``, so
no hold-out method outcome can be seen while the band is still being chosen.
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

from experiments.mlst2026 import tidmad_band as tb  # noqa: E402
from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402
from experiments.mlst2026.provenance import RunRecorder, write_json  # noqa: E402

SQUID = "timeseries/channel0001/timeseries"
REFERENCE = "timeseries/channel0002/timeseries"


# --- configuration ------------------------------------------------------------
class Protocol:
    """The stage-0 JSON, with the derived quantities asserted against it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.raw = json.loads(self.path.read_text())
        if self.raw.get("schema") != "mlst-t22-stage0/1":
            raise SystemExit(f"{path}: unexpected schema {self.raw.get('schema')!r}")
        a = self.raw["allocation"]
        u = self.raw["unchanged_from_t10_t12"]
        self.w = int(u["window_samples"])
        self.rank = int(u["rank"])
        self.n_cal = int(a["n_calibration_windows"])
        self.n_fit = int(a["n_fit_windows"])
        self.n_eval = int(a["n_evaluation_windows"])
        self.guard = int(a["guard_samples"])
        self.fs = float(a["sampling_rate_hz"])
        # Derived, then checked against the declared numbers: one configuration
        # source, and an automatic equality check where the JSON repeats itself.
        fit_off = self.n_cal * self.w + self.guard
        eval_off = fit_off + self.n_fit * self.w + self.guard
        stretch = eval_off + self.n_eval * self.w
        for name, derived, declared in (("fit_offset", fit_off, a["fit_offset"]),
                                        ("evaluation_offset", eval_off, a["evaluation_offset"]),
                                        ("stretch_samples", stretch, a["stretch_samples"])):
            if derived != int(declared):
                raise SystemExit(f"{path}: {name} declared {declared}, derived {derived}")
        self.fit_offset, self.eval_offset, self.stretch = fit_off, eval_off, stretch

        r = self.raw["representation"]
        self.bands = tuple(int(b) for b in r["band_candidates"])
        self.identity_band = int(r["identity_control_band"])
        self.selection_bands = tuple(b for b in self.bands
                                     if b not in set(r["excluded_from_selection"]))
        self.identity_rtol = float(r["identity_assertion"]["relative_tolerance"])

        s = self.raw["band_selection"]
        self.sel_replicates = int(s["resampling"]["replicates"])
        self.sel_seed = int(s["resampling"]["seed"])
        self.min_band = 3
        self.max_band_dim_ratio = 10

        d = self.raw["dependence"]
        self.blocks_eval = tuple(int(b) for b in d["block_candidates_evaluation"])
        self.blocks_cal = tuple(int(b) for b in d["block_candidates_calibration"])

        i = self.raw["inference"]
        self.boot_replicates = int(i["per_file"]["replicates"])
        self.boot_seed = int(i["per_file"]["seed"])
        self.file_replicates = int(i["primary_combined_estimand"]["interval"]["replicates"])
        self.file_seed = int(i["primary_combined_estimand"]["interval"]["seed"])
        self.min_files_for_combined = 4

        self.contrasts = tuple((a_, b_) for a_, b_ in self.raw["contrasts"])
        self.methods = tuple(u["estimators"])
        self.multi_tone_fraction = float(
            u["characterisation_thresholds"]["multi_tone_power_fraction"])

        dr = self.raw["data_roles"]
        self.dev = [(e["file"], int(e["start_sample"]))
                    for e in dr["development"]["files_in_declared_priority_order"]]
        self.dev_full_grid_max = 4
        self.holdout_files = tuple(dr["holdout"]["files"])
        self.holdout_start = int(dr["holdout"]["start_sample"])

        b = self.raw["budget"]
        self.cap = int(b["hard_cap"])
        self.nominal = int(b["nominal_total"])
        self.smoke_seed_base = int(b["smoke_runs"]["seed_base"])
        self.n_psd_windows = 256

    def sha256(self) -> str:
        import hashlib
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def stable_offset(key: str) -> int:
    """Deterministic per-stretch seed offset (``hash()`` is salted, CRC32 is not)."""
    return int(zlib.crc32(key.encode())) % 100_000


# --- data ---------------------------------------------------------------------
def _read(h, dataset, start, length):
    ds = h[dataset]
    stop = min(int(start) + int(length), ds.shape[0])
    return np.asarray(ds[int(start):stop], dtype=np.float64)


def _windows(x: np.ndarray, offset: int, n: int, w: int) -> np.ndarray:
    seg = x[offset:offset + n * w]
    if seg.size < n * w:
        seg = seg[: (seg.size // w) * w]
    return seg.reshape(-1, w)


def load_stretch(path: Path, start: int, P: Protocol, *, roles=("cal", "fit", "eval")) -> dict:
    """Read one stretch and return the unitary rFFT of each requested block."""
    import h5py
    spec = {"cal": (0, P.n_cal), "fit": (P.fit_offset, P.n_fit),
            "eval": (P.eval_offset, P.n_eval)}
    need = max(spec[r][0] + spec[r][1] * P.w for r in roles)
    with h5py.File(path, "r") as h:
        n_total = int(h[SQUID].shape[0])
        x = _read(h, SQUID, start, need)
        r = _read(h, REFERENCE, start, need)
    out = {"n_total_samples": n_total, "start_sample": int(start),
           "end_sample_exclusive": int(start + P.stretch),
           "n_samples_read": int(min(x.size, r.size)),
           "allocation": {"window_samples": P.w, "n_cal": P.n_cal, "n_fit": P.n_fit,
                          "n_eval": P.n_eval, "guard_samples": P.guard,
                          "cal_offset": 0, "fit_offset": P.fit_offset,
                          "eval_offset": P.eval_offset,
                          "stretch_samples": P.stretch}}
    for role in roles:
        off, n = spec[role]
        wx = _windows(x, off, n, P.w)
        out[f"{role}_read"] = te.unitary_rfft(wx)
        if role in ("cal", "eval"):
            out[f"{role}_ref"] = te.unitary_rfft(_windows(r, off, n, P.w))
        out[f"n_{role}"] = int(wx.shape[0])
    return out


def _science_paths(data_root: Path):
    return sorted((data_root / "tidmad" / "raw").glob("abra_science_*.h5"))


def _raw(data_root: Path, name: str) -> Path:
    return data_root / "tidmad" / "raw" / name


def noise_bin_power(science_paths, P: Protocol) -> np.ndarray:
    """Readout noise bin power from science-file windows (no injection present)."""
    import h5py
    need = P.n_psd_windows * P.w
    acc, got = [], 0
    for p in science_paths:
        if got >= P.n_psd_windows:
            break
        with h5py.File(p, "r") as h:
            x = _read(h, SQUID, 0, need)
        W = _windows(x, 0, min(P.n_psd_windows - got, x.size // P.w), P.w)
        acc.append(te.mean_bin_power(te.unitary_rfft(W)) * W.shape[0])
        got += W.shape[0]
    if got == 0:
        raise RuntimeError("no science windows available for the noise PSD")
    return np.sum(acc, axis=0) / got


# --- band admissibility -------------------------------------------------------
def band_admissibility(st: dict, ch: dict, B: int, P: Protocol) -> dict:
    """The declared admissibility tests for one (stretch, band) pair."""
    m = st["cal_ref"].shape[1]
    k = int(ch["k_inj"])
    reasons = []
    if B < P.min_band:
        reasons.append(f"B = {B} < {P.min_band} (the identity control is not a learned band)")
    if 2 * B > P.n_fit / P.max_band_dim_ratio:
        reasons.append(f"2B = {2 * B} exceeds n_fit/{P.max_band_dim_ratio} = "
                       f"{P.n_fit / P.max_band_dim_ratio:g}")
    try:
        bins = tb.band_bins(k, B, m)
    except ValueError as exc:
        return {"band_width": int(B), "admissible": False,
                "reasons": reasons + [f"band-at-edge: {exc}"], "second_line": None}
    sl = tb.second_line_excess(st["cal_ref"], bins, k, float(ch["subbin_offset_delta"]),
                               P.w, fraction=P.multi_tone_fraction)
    if sl["has_second_line"]:
        for h in sl["second_lines"]:
            reasons.append(
                f"second line inside the band: bin {h['bin']} carries "
                f"{h['fraction_of_line']:.4f} of the line bin power and is a local "
                f"maximum against its inward neighbour "
                f"({h['inward_neighbour_fraction']:.4f}); threshold "
                f"{P.multi_tone_fraction}, guard {sl['guard_bins']} bin")
    return {"band_width": int(B), "admissible": not reasons, "reasons": reasons,
            "second_line": sl}


# --- scoring ------------------------------------------------------------------
def score_cell(st: dict, ch: dict, B: int, N_bins: np.ndarray, P: Protocol, *,
               seed: int, replicates: int, b_eval: int | None, b_cal: int | None,
               curve_replicates: int = 400, keep_draws: bool = True,
               keep_estimates: bool = False) -> dict:
    """Characterise, estimate and score one (stretch, band) cell."""
    k = int(ch["k_inj"])
    cal = te.fit_transfer(st["cal_read"][:, k], st["cal_ref"][:, k], float(N_bins[k]))
    est, diag = tb.compute_band_estimates(st["eval_read"], st["fit_read"], k, B,
                                          P.w, N_bins, cal)
    t_eval = cal.g * st["eval_ref"][:, k]

    e2 = {name: te.squared_errors(q, t_eval, P.w) for name, q in est.items()}
    deltas = {(a, b): e2[a] - e2[b] for (a, b) in P.contrasts}
    curves = {f"{a}|{b}": te.block_variance_curve(
        d, P.blocks_eval, replicates=curve_replicates, seed=seed + 17)
        for (a, b), d in deltas.items()}
    acfs = {f"{a}|{b}": te.autocorrelation(d) for (a, b), d in deltas.items()}
    cal_series = np.abs(st["cal_read"][:, k] - cal.g * st["cal_ref"][:, k]) ** 2
    cal_curve = te.block_variance_curve(cal_series, P.blocks_cal,
                                        replicates=curve_replicates, seed=seed + 29)
    cal_sel = te.select_block_length(cal_curve, P.blocks_cal)

    first = f"{P.contrasts[0][0]}|{P.contrasts[0][1]}"
    use_eval = b_eval if b_eval is not None else te.select_block_length(
        curves[first], P.blocks_eval)["block_length"]
    use_cal = b_cal if b_cal is not None else cal_sel["block_length"]

    draws = te.joint_bootstrap(est, st["eval_read"][:, k], st["eval_ref"][:, k],
                               st["cal_read"][:, k], st["cal_ref"][:, k],
                               float(N_bins[k]), P.w, list(P.contrasts),
                               replicates=replicates, seed=seed,
                               b_eval=use_eval, b_cal=use_cal)
    draws_cal = te.joint_bootstrap(est, st["eval_read"][:, k], st["eval_ref"][:, k],
                                   st["cal_read"][:, k], st["cal_ref"][:, k],
                                   float(N_bins[k]), P.w, list(P.contrasts),
                                   replicates=replicates, seed=seed + 1,
                                   b_eval=use_eval, b_cal=use_cal,
                                   calibration_only=True)

    contrasts = {}
    for (a, b) in P.contrasts:
        obs = te.observed_contrast(est, st["eval_read"][:, k], t_eval, P.w, a, b)
        ci = te.percentile_interval(draws[(a, b)])
        ci_cal = te.percentile_interval(draws_cal[(a, b)])
        contrasts[f"{a}|{b}"] = {**obs, **ci, "ci_excludes_zero": ci["excludes_zero"],
                                 "calibration_only_component": ci_cal}

    out = {
        "band_width": int(B),
        "estimator_diagnostics": diag,
        "transfer_calibration": cal.as_dict(),
        "block_lengths_used": {"evaluation": int(use_eval), "calibration": int(use_cal)},
        "dependence": {"evaluation_curves": curves, "evaluation_autocorrelation": acfs,
                       "calibration_curve": cal_curve, "calibration_selection": cal_sel},
        "rmse_by_method_amplitude_units": {n: float(np.sqrt(np.mean(v)))
                                           for n, v in e2.items()},
        "contrasts": contrasts,
        "n_fits": len(P.methods),
        "conditional_on": [
            f"{P.n_fit} fit windows and the band-PCA/band-EMPCA bases and Wiener "
            "signal PSD estimated from them",
            "science-file noise bin powers", "the injection bin k_inj",
            "the band width B", "the block lengths"],
    }
    if B == P.identity_band:
        out["identity_control"] = tb.assert_single_bin_identity(
            diag, f"B=1 cell", P.identity_rtol)
    if keep_draws:
        out["_draws"] = {f"{a}|{b}": draws[(a, b)] for (a, b) in P.contrasts}
    if keep_estimates:
        out["_estimates"] = {"raw-bin": est["raw-bin"], "empca": est["empca"]}
    return out


# --- band selection -----------------------------------------------------------
def paired_band_bootstrap(cells: dict, stretches: dict, N_bins: np.ndarray,
                          P: Protocol, *, bands, b_eval: int, b_cal: int,
                          replicates: int, seed: int) -> dict:
    """Bootstrap the equal-stretch mean of Delta(bin read, band-EMPCA) with the
    candidates paired inside every replicate.

    Each replicate draws one calibration-block and one evaluation-block resample
    per selection stretch, refits ``g_hat`` once, and reuses those same indices
    for every candidate band -- so differences between candidates are not
    inflated by independent resampling noise.
    """
    keys = sorted(cells)
    rng = np.random.default_rng(seed)
    bands = tuple(bands)
    if not bands:
        return {}
    per_rep = {B: np.empty(replicates) for B in bands}
    prepared = {}
    for key in keys:
        st, k = stretches[key], cells[key]["k_inj"]
        prepared[key] = {
            "c_eval": st["eval_read"][:, k], "rho_eval": st["eval_ref"][:, k],
            "c_cal": st["cal_read"][:, k], "rho_cal": st["cal_ref"][:, k],
            "N": float(N_bins[k]),
            "est": {B: cells[key]["estimates"][B] for B in bands},
        }
    for j in range(replicates):
        acc = {B: [] for B in bands}
        for key in keys:
            d = prepared[key]
            n_e, n_c = d["c_eval"].size, d["c_cal"].size
            ic = te.circular_block_indices(rng, n_c, b_cal)
            ie = te.circular_block_indices(rng, n_e, b_eval)
            cal_b = te.fit_transfer(d["c_cal"][ic], d["rho_cal"][ic], d["N"])
            t_b = cal_b.g * d["rho_eval"][ie]
            base = te.squared_errors(d["est"][bands[0]]["raw-bin"][ie], t_b, P.w)
            for B in bands:
                emp = te.squared_errors(d["est"][B]["empca"][ie], t_b, P.w)
                acc[B].append(float(np.mean(base - emp)))
        for B in bands:
            per_rep[B][j] = float(np.mean(acc[B]))
    return per_rep


def choose_band(dev: dict, admiss: dict, boot: dict, P: Protocol, *, bands) -> dict:
    """The declared one-bootstrap-sd smallest-admissible-band rule."""
    selection_keys = [k for k, v in dev.items() if v.get("enters_selection")]
    if not selection_keys:
        return {"selected_band": None, "disposition": "blocked-no-eligible-development-stretch",
                "reason": "no development stretch was eligible-coupled at the T22 stretch, "
                          "so the band-selection rule has no input",
                "selection_stretches": []}
    admissible = []
    per_band = {}
    for B in bands:
        rows = [admiss[k][str(B)] for k in selection_keys]
        ok = all(r["admissible"] for r in rows)
        m = float(np.mean([dev[k]["cells"][str(B)]["contrasts"]["raw-bin|empca"]
                           ["mean_delta_squared"] for k in selection_keys]))
        sd = float(np.std(boot[B], ddof=1))
        per_band[B] = {"admissible": ok, "equal_stretch_mean": m, "bootstrap_sd": sd,
                       "reasons": sorted({r_ for r in rows for r_ in r["reasons"]}),
                       "bootstrap_mean": float(np.mean(boot[B])),
                       "bootstrap_ci95": [float(np.percentile(boot[B], 2.5)),
                                          float(np.percentile(boot[B], 97.5))]}
        if ok:
            admissible.append(B)
    if not admissible:
        return {"selected_band": None, "disposition": "blocked-no-admissible-band",
                "reason": "no candidate band of width >= 3 passed the declared "
                          "admissibility tests on every selection stretch",
                "per_band": per_band, "selection_stretches": selection_keys}
    best = max(admissible, key=lambda B: (per_band[B]["equal_stretch_mean"], -B))
    threshold = per_band[best]["equal_stretch_mean"] - per_band[best]["bootstrap_sd"]
    selected = min(B for B in admissible if per_band[B]["equal_stretch_mean"] >= threshold)
    return {"selected_band": int(selected), "disposition": "selected",
            "rule": "smallest admissible B whose equal-stretch mean Delta(bin read, "
                    "band-EMPCA) is within one paired-bootstrap sd of the best candidate",
            "best_band": int(best), "threshold": threshold,
            "per_band": per_band, "selection_stretches": selection_keys,
            "admissible_bands": admissible}


# --- cross-file inference -----------------------------------------------------
def hierarchical_file_bootstrap(draws_by_file: list[np.ndarray], *, replicates: int,
                                seed: int, point_estimates=None) -> dict:
    """Two-stage percentile bootstrap of the equal-file mean.

    Each replicate resamples files with replacement and, within each drawn file,
    draws one value from that file's joint-bootstrap replicate array, so
    within-file calibration and block uncertainty enters the file-level interval.

    Two point summaries are reported and they are different quantities:
    ``equal_file_mean_of_point_estimates`` is the equal-file mean of the
    **observed** per-file contrasts and is the estimand; it must be supplied in
    ``point_estimates``.  ``equal_file_mean_of_bootstrap_means`` is the
    equal-file mean of each file's bootstrap mean, which differs from it by the
    bootstrap's own (small) bias and is reported as a diagnostic.
    """
    n = len(draws_by_file)
    if n == 0:
        return {"n_files": 0, "note": "no file entered the combined estimate"}
    rng = np.random.default_rng(seed)
    point = float(np.mean([float(np.mean(d)) for d in draws_by_file]))
    obs = (float(np.mean(np.asarray(point_estimates, dtype=float)))
           if point_estimates is not None else float("nan"))
    means = np.empty(replicates)
    for j in range(replicates):
        idx = rng.integers(0, n, size=n)
        vals = [draws_by_file[i][rng.integers(0, draws_by_file[i].size)] for i in idx]
        means[j] = float(np.mean(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Two-sided bootstrap p with the (r + 1) / (R + 1) continuity correction.
    le = float(np.sum(means <= 0.0)); ge = float(np.sum(means >= 0.0))
    p = min(1.0, 2.0 * min((le + 1.0) / (replicates + 1.0), (ge + 1.0) / (replicates + 1.0)))
    return {"n_files": n, "equal_file_mean_of_bootstrap_means": point,
            "equal_file_mean_of_point_estimates": obs,
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "bootstrap_sd": float(np.std(means, ddof=1)),
            "p_two_sided": float(p), "replicates": int(replicates)}


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down adjusted p-values at familywise level 0.05."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (n - i) * p))
        running = adj
        out[name] = adj
    return out


def combine_stage(per_file: dict, P: Protocol, dependence_verdict: str) -> dict:
    """Primary equal-file estimand, secondary sensitivities and the three verdicts."""
    entering = [r for r in per_file.values() if r.get("in_combined_estimate")]
    combined = {}
    for (a, b) in P.contrasts:
        key = f"{a}|{b}"
        draws = [np.asarray(r["_draws"][key]) for r in entering]
        primary = hierarchical_file_bootstrap(
            draws, replicates=P.file_replicates, seed=P.file_seed + stable_offset(key),
            point_estimates=[r["contrasts"][key]["mean_delta_squared"] for r in entering])
        rows = [{"mean_delta_squared": r["contrasts"][key]["mean_delta_squared"],
                 "bootstrap_variance": r["contrasts"][key]["bootstrap_sd"] ** 2}
                for r in entering]
        iv = te.combine_files(rows) if rows else {"n_files": 0}
        y = np.array([r["contrasts"][key]["mean_delta_squared"] for r in entering])
        sd = np.array([r["contrasts"][key]["bootstrap_sd"] for r in entering])
        re_ = random_effects(y, sd)
        combined[key] = {
            "primary_equal_file": primary,
            "secondary_inverse_variance": iv,
            "secondary_random_effects": re_,
            "secondary_equal_file_t_interval": t_interval(y),
            "per_file": {r["file"]: r["contrasts"][key]["mean_delta_squared"]
                         for r in entering},
            "indicative_only": bool(len(entering) < P.min_files_for_combined),
        }
    claims = {
        "limited_recovery": ("zero|empca", "calibration-mean|empca"),
        "denoising_benefit": ("raw-bin|empca",),
        "metric_contribution": ("pca|empca",),
    }
    raw_p = {}
    for name, keys in claims.items():
        ps = [combined[k]["primary_equal_file"].get("p_two_sided", 1.0) for k in keys]
        raw_p[name] = float(max(ps))  # the claim needs every constituent contrast
    adjusted = holm(raw_p)
    verdicts = {}
    for name, keys in claims.items():
        pe = [combined[k]["primary_equal_file"] for k in keys]
        favours = all(v.get("equal_file_mean_of_point_estimates", 0.0) > 0 for v in pe)
        excludes = all(v.get("excludes_zero", False) for v in pe)
        if not entering:
            verdict, reason = "not-evaluable", "no eligible file entered the combined estimate"
        elif dependence_verdict == "inconclusive":
            verdict, reason = "inconclusive", (
                "the frozen dependence assessment did not stabilise; the declared rule "
                "withholds every benefit verdict, not only limited recovery")
        elif favours and excludes and adjusted[name] < 0.05:
            verdict, reason = "supported", (
                f"equal-file interval excludes zero in EMPCA's favour and the "
                f"Holm-adjusted p = {adjusted[name]:.4g} < 0.05")
        elif favours:
            verdict, reason = "ordering-only", (
                f"point estimate favours EMPCA but the decision rule is not met "
                f"(Holm-adjusted p = {adjusted[name]:.4g}; interval excludes zero: {excludes})")
        else:
            verdict, reason = "not-supported", "the point estimate does not favour EMPCA"
        verdicts[name] = {"verdict": verdict, "reason": reason,
                          "contrasts": list(keys), "p_raw": raw_p[name],
                          "p_holm_adjusted": adjusted[name],
                          "dependence_verdict": dependence_verdict,
                          "indicative_only": bool(len(entering) < P.min_files_for_combined)}
    return {"combined": combined, "verdicts": verdicts,
            "files_entering": [r["file"] for r in entering],
            "multiplicity": {"method": "Holm-Bonferroni step-down",
                             "familywise_level": 0.05, "family": sorted(claims)}}


def random_effects(y: np.ndarray, sd: np.ndarray) -> dict:
    if y.size == 0:
        return {"n_files": 0}
    w = 1.0 / sd ** 2
    fe = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - fe) ** 2))
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (y.size - 1)) / C) if C > 0 else 0.0
    wr = 1.0 / (sd ** 2 + tau2)
    re = float(np.sum(wr * y) / np.sum(wr))
    se = float(np.sqrt(1.0 / np.sum(wr)))
    return {"n_files": int(y.size), "estimate": re, "se": se,
            "ci95": [re - 1.96 * se, re + 1.96 * se], "tau": float(np.sqrt(tau2)),
            "cochran_Q": Q, "Q_dof": int(y.size - 1)}


#: Two-sided Student-t 0.975 quantiles for 1..12 degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def t_interval(y: np.ndarray) -> dict:
    if y.size < 2:
        return {"n_files": int(y.size), "note": "fewer than two files"}
    m = float(np.mean(y)); s = float(np.std(y, ddof=1)); dof = y.size - 1
    t = T975.get(dof, 1.96)
    half = t * s / np.sqrt(y.size)
    return {"n_files": int(y.size), "mean": m, "sd": s, "dof": dof, "t975": t,
            "ci95": [m - half, m + half], "excludes_zero": bool(abs(m) > half)}


# --- stages -------------------------------------------------------------------
def stage_characterize(args, P: Protocol, rec: RunRecorder) -> dict:
    data_root = Path(args.data_root)
    N = noise_bin_power(_science_paths(data_root), P)
    targets = [(f, s, "development") for f, s in P.dev]
    targets += [(f, P.holdout_start, "holdout") for f in P.holdout_files]
    out = {}
    for name, start, role in targets:
        path = _raw(data_root, name)
        if not path.exists():
            out[name] = {"error": "file not found", "path": str(path)}
            continue
        # Hash every instrument file once, here, so the later stages can cite
        # these digests instead of re-reading 31 GB per stage.
        rec.add_input(f"stretch:{role}", path)
        import h5py
        with h5py.File(path, "r") as h:
            n_total = int(h[SQUID].shape[0])
            x = _read(h, SQUID, start, P.n_cal * P.w)
            r = _read(h, REFERENCE, start, P.n_cal * P.w)
        Cx = te.unitary_rfft(_windows(x, 0, P.n_cal, P.w))
        Cr = te.unitary_rfft(_windows(r, 0, P.n_cal, P.w))
        avail = min(P.n_eval, max(0, (n_total - start - P.eval_offset) // P.w))
        ch = te.characterise(Cr, Cx, P.w, N, P.stretch, int(avail))
        ch.update({"file": name, "role": role, "start_sample": int(start),
                   "end_sample_exclusive": int(start + P.stretch),
                   "n_total_samples": n_total,
                   "n_evaluation_windows_required": P.n_eval})
        ch["file_sha256"] = next((r["sha256"] for r in reversed(rec.inputs)
                                  if r["path"] == str(path)), None)
        out[name] = ch
        print(f"  [{name} {role:11s}] {ch['outcome']:<17s} f={ch['f_inj_hz']/1e3:9.3f} kHz "
              f"k={ch['k_inj']:6d} refSNR={ch['reference_line_snr']:10.4g} "
              f"readSNR={ch['readout_line_snr']:9.4g} drift={ch['drift_bins_over_stretch']:7.3f} "
              f"delta={ch['subbin_offset_delta']:+.4f}"
              f"{'  ' + ';'.join(ch['reasons']) if ch['reasons'] else ''}")
    return {"characterisation": out}


def stage_development(args, P: Protocol, rec: RunRecorder, chars: dict) -> dict:
    data_root = Path(args.data_root)
    seed0 = P.smoke_seed_base if args.smoke else P.boot_seed
    reps = 200 if args.smoke else P.boot_replicates
    curve_reps = 100 if args.smoke else 400
    N = noise_bin_power(_science_paths(data_root), P)

    eligible_order = [f for f, _ in P.dev
                      if chars.get(f, {}).get("outcome") == "eligible-coupled"]
    full_grid = set(eligible_order[:P.dev_full_grid_max])
    planned = sum(len(P.bands) if f in full_grid else 1 for f, _ in P.dev) * len(P.methods)
    print(f"  [budget] development plans {planned} fits "
          f"({len(full_grid)} stretch(es) on the full band grid, "
          f"{len(P.dev) - len(full_grid)} on the identity control only)")
    if planned > P.cap:
        raise SystemExit(f"planned {planned} fits exceeds the hard cap {P.cap}")

    dev, admiss, stretches, cell_est, fits = {}, {}, {}, {}, 0
    for name, start in P.dev:
        ch = chars.get(name, {})
        if "k_inj" not in ch:
            dev[name] = {"skipped": True, "reason": ch.get("error", "not characterised")}
            continue
        st = load_stretch(_raw(data_root, name), start, P)
        stretches[name] = st
        bands = P.bands if name in full_grid else (P.identity_band,)
        cells, adm, ests = {}, {}, {}
        for B in bands:
            adm[str(B)] = band_admissibility(st, ch, B, P)
            if adm[str(B)]["second_line"] is None:  # band-at-edge: cannot be scored
                cells[str(B)] = {"band_width": B, "scored": False,
                                 "reason": "; ".join(adm[str(B)]["reasons"])}
                continue
            res = score_cell(st, ch, B, N, P, seed=seed0 + stable_offset(f"{name}|B={B}"),
                             replicates=reps, b_eval=None, b_cal=None,
                             curve_replicates=curve_reps, keep_draws=False,
                             keep_estimates=True)
            fits += res["n_fits"]
            ests[B] = res.pop("_estimates")
            res["scored"] = True
            res["admissible_for_selection"] = bool(adm[str(B)]["admissible"])
            cells[str(B)] = res
            c = res["contrasts"]["raw-bin|empca"]
            print(f"  [{name} B={B:>2}] dim={res['estimator_diagnostics']['band_real_dimension']:>3} "
                  f"admissible={str(adm[str(B)]['admissible']):5s} "
                  f"D(bin read,EMPCA)={c['mean_delta_squared']:+.6g} "
                  f"CI[{c['ci95_lo']:+.4g},{c['ci95_hi']:+.4g}] "
                  f"b_eval={res['block_lengths_used']['evaluation']}")
        dev[name] = {"file": name, "start_sample": int(start),
                     "characterisation_outcome": ch.get("outcome"),
                     "enters_selection": name in full_grid,
                     "bands_scored": [int(b) for b in bands], "cells": cells,
                     "k_inj": int(ch["k_inj"])}
        admiss[name] = adm
        cell_est[name] = {"k_inj": int(ch["k_inj"]), "estimates": ests}

    # Dependence, aggregated over the selection stretches at every scored band.
    curves, cal_curves = [], []
    for name, entry in dev.items():
        if not entry.get("enters_selection"):
            continue
        for B, res in entry["cells"].items():
            if not res.get("scored"):
                continue
            for c in res["dependence"]["evaluation_curves"].values():
                curves.append({int(k): v for k, v in c.items()})
            cal_curves.append({int(k): v for k, v in res["dependence"]["calibration_curve"].items()})
    agg = te.aggregate_block_curves(curves, P.blocks_eval)
    agg_cal = te.aggregate_block_curves(cal_curves, P.blocks_cal)

    boot, selection = {}, None
    sel_keys = [k for k, v in dev.items() if v.get("enters_selection")]
    usable = tuple(B for B in P.selection_bands
                   if all(B in cell_est[k]["estimates"] for k in sel_keys))
    if sel_keys:
        boot = paired_band_bootstrap(
            {k: cell_est[k] for k in sel_keys}, {k: stretches[k] for k in sel_keys}, N, P,
            bands=usable,
            b_eval=int(agg["selection"]["block_length"]),
            b_cal=int(agg_cal["selection"]["block_length"]),
            replicates=200 if args.smoke else P.sel_replicates,
            seed=(P.smoke_seed_base if args.smoke else P.sel_seed))
    selection = choose_band(dev, admiss, boot, P, bands=usable)
    return {"development": dev, "band_admissibility": admiss,
            "selection_bands_usable": [int(b) for b in usable],
            "dependence_aggregate": agg, "dependence_aggregate_calibration": agg_cal,
            "band_selection": selection, "fit_count": fits,
            "planned_fit_count": planned}


def stage_evaluate(args, P: Protocol, rec: RunRecorder, frozen: dict, files, label: str,
                   chars: dict) -> dict:
    data_root = Path(args.data_root)
    B = int(frozen["selected_band"])
    b_eval, b_cal = int(frozen["block_length_evaluation"]), int(frozen["block_length_calibration"])
    dep = frozen.get("dependence_verdict", "inconclusive")
    N = noise_bin_power(_science_paths(data_root), P)
    per, fits = {}, 0
    for name in files:
        ch = chars.get(name, {})
        if "k_inj" not in ch:
            per[name] = {"file": name, "skipped": True,
                         "reason": ch.get("error", "not characterised"),
                         "in_combined_estimate": False}
            continue
        st = load_stretch(_raw(data_root, name), P.holdout_start, P)
        res = score_cell(st, ch, B, N, P, seed=P.boot_seed + stable_offset(f"{name}|B={B}"),
                         replicates=P.boot_replicates, b_eval=b_eval, b_cal=b_cal,
                         keep_draws=True)
        fits += res["n_fits"]
        res.update({"file": name, "stage": label, "start_sample": P.holdout_start,
                    "characterisation": ch,
                    "band_admissibility": band_admissibility(st, ch, B, P)})
        supported = bool(ch["readout_unit_target_supported"])
        if ch["outcome"] != "eligible-coupled":
            res["in_combined_estimate"] = False
            res["file_verdict"] = {"verdict": "excluded-" + ch["outcome"],
                                   "reason": "declared in advance: only eligible-coupled "
                                             "stretches enter the combined estimate"}
        elif not supported:
            res["in_combined_estimate"] = False
            res["file_verdict"] = {"verdict": "diagnostic-only",
                                   "reason": "reference line SNR below the declared 100"}
        else:
            res["in_combined_estimate"] = True
            res["file_verdict"] = te.usefulness_verdict(
                res["contrasts"]["zero|empca"], res["contrasts"]["calibration-mean|empca"], dep)
        per[name] = res
        c = res["contrasts"]["raw-bin|empca"]
        z = res["contrasts"]["zero|empca"]
        print(f"  [{label} {name}] {ch['outcome']:<17s} verdict={res['file_verdict']['verdict']:<22s} "
              f"D(zero,EMPCA)={z['mean_delta_squared']:+.5g} "
              f"D(bin read,EMPCA)={c['mean_delta_squared']:+.5g} "
              f"CI[{c['ci95_lo']:+.4g},{c['ci95_hi']:+.4g}]")
    summary = combine_stage(per, P, dep)
    draws = {}
    for name, r in per.items():
        for key, arr in r.pop("_draws", {}).items():
            draws[f"{name}|{key}"] = np.asarray(arr, dtype=np.float64)
    return {label: {"per_file": per, "files_evaluated": list(files), "fit_count": fits,
                    **summary}}, draws


# --- main ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["characterize", "development", "freeze", "validation",
                             "final-test", "recombine"])
    ap.add_argument("--stage-output", help="recombine: the stage JSON to re-summarise")
    ap.add_argument("--draws", help="recombine: the matching *_draws.npz")
    ap.add_argument("--protocol", required=True, help="stage-0 protocol JSON")
    ap.add_argument("--data-root", default="/ceph/dwong/paper1_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--characterisation", help="JSON from the characterize stage")
    ap.add_argument("--development", help="JSON from the development stage")
    ap.add_argument("--freeze-file", help="frozen configuration JSON")
    ap.add_argument("--smoke", action="store_true",
                    help="implementation check: few replicates, smoke seeds, output "
                         "marked non-scientific")
    args = ap.parse_args(argv)

    P = Protocol(Path(args.protocol))
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory {out_dir}")

    rec = RunRecorder(run_id=f"T22_tidmad_narrowband_{args.stage}",
                      protocol_refs={"stage0": "plan/mlst_protocol_2026-09-11-t22-stage0.json",
                                     "programme": "plan/MLST_EXPERIMENT_PROGRAMME.md#II.T22",
                                     "predeclaration": "reviews/2026-09-11-PREDECLARED_T22.md"})
    rec.add_input("runner_source", Path(__file__).resolve())
    rec.add_input("band_module", Path(tb.__file__).resolve())
    rec.add_input("endpoint_module", Path(te.__file__).resolve())
    rec.add_input("protocol", P.path.resolve())
    rec.add_seed("bootstrap_seed", P.smoke_seed_base if args.smoke else P.boot_seed)
    rec.add_seed("band_selection_seed", P.smoke_seed_base if args.smoke else P.sel_seed)
    rec.add_seed("file_level_seed", P.file_seed)
    rec.add_seed("smoke", bool(args.smoke))
    data_root = Path(args.data_root)
    for p in _science_paths(data_root)[:2]:
        rec.add_input("science_noise", p)

    payload: dict = {"stage": args.stage, "smoke": bool(args.smoke),
                     "protocol_sha256": P.sha256(),
                     "allocation": P.raw["allocation"],
                     "band_candidates": list(P.bands)}
    draws: dict = {}

    if args.stage == "characterize":
        payload.update(stage_characterize(args, P, rec))

    elif args.stage == "development":
        chj = json.loads(Path(args.characterisation).read_text())
        rec.add_input("characterisation", Path(args.characterisation))
        payload.update(stage_development(args, P, rec, chj["characterisation"]))

    elif args.stage == "freeze":
        devj = json.loads(Path(args.development).read_text())
        chj = json.loads(Path(args.characterisation).read_text())
        rec.add_input("development", Path(args.development))
        rec.add_input("characterisation", Path(args.characterisation))
        sel = devj["band_selection"]
        agg, agg_cal = devj["dependence_aggregate"], devj["dependence_aggregate_calibration"]
        chars = chj["characterisation"]
        eligible = [f for f in P.holdout_files
                    if chars.get(f, {}).get("outcome") == "eligible-coupled"]
        validation = eligible[:3]
        final_test = [f for f in P.holdout_files if f not in validation]
        status = "complete" if len(validation) == 3 else ("underpowered" if validation else "empty")
        payload["frozen"] = {
            "selected_band": sel.get("selected_band"),
            "disposition": sel.get("disposition"),
            "band_selection": sel,
            "rank": P.rank, "window_samples": P.w, "methods": list(P.methods),
            "contrasts": [list(c) for c in P.contrasts],
            "endpoint": P.raw["unchanged_from_t10_t12"]["endpoint"],
            "block_length_evaluation": int(agg["selection"]["block_length"]),
            "block_length_calibration": int(agg_cal["selection"]["block_length"]),
            "dependence_verdict": agg["selection"]["verdict"],
            "dependence_aggregate": agg, "dependence_aggregate_calibration": agg_cal,
            "bootstrap_replicates": P.boot_replicates, "bootstrap_seed": P.boot_seed,
            "file_level_replicates": P.file_replicates, "file_level_seed": P.file_seed,
            "holdout_start_sample": P.holdout_start,
            "validation_files": validation, "final_test_files": final_test,
            "n_eligible_coupled_holdout_files": len(eligible),
            "eligible_holdout_files": eligible,
            "validation_status": status,
            "positive_verdict_permitted": status != "empty" and sel.get("selected_band") is not None,
            "development_fit_count": devj.get("fit_count"),
            "stage0_protocol": P.raw,
        }
        payload["frozen"]["input_hashes"] = {
            r["role"] + ":" + Path(r["path"]).name: r.get("sha256")
            for r in rec.inputs if r.get("sha256")}

    elif args.stage == "recombine":
        # Deterministic post-processing of an archived hold-out stage: the
        # per-file results and their bootstrap draw arrays are read back and the
        # combined block is recomputed.  No data file is opened and no estimator
        # is fitted, so this is not a second evaluation of the hold-out; the
        # original stage output is left untouched and both versions are kept.
        src = json.loads(Path(args.stage_output).read_text())
        rec.add_input("stage_output", Path(args.stage_output))
        rec.add_input("draws", Path(args.draws))
        fz = json.loads(Path(args.freeze_file).read_text())["frozen"]
        rec.add_input("freeze", Path(args.freeze_file))
        label = src["stage"]
        arrays = np.load(args.draws)
        per = src[label]["per_file"]
        for name, r in per.items():
            r["_draws"] = {f"{a}|{b}": np.asarray(arrays[f"{name}|{a}|{b}"])
                           for (a, b) in P.contrasts if f"{name}|{a}|{b}" in arrays}
        summary = combine_stage(per, P, fz.get("dependence_verdict", "inconclusive"))
        for r in per.values():
            r.pop("_draws", None)
        payload[label] = {"per_file": per,
                          "files_evaluated": src[label]["files_evaluated"],
                          "fit_count": src[label]["fit_count"],
                          "recombined_from": str(Path(args.stage_output).name),
                          "fits_performed": 0, **summary}
        payload["stage"] = label

    elif args.stage in ("validation", "final-test"):
        if not args.freeze_file or not Path(args.freeze_file).exists():
            raise SystemExit("validation/final-test require --freeze-file; the band must be "
                             "frozen before any hold-out method outcome is computed")
        fz = json.loads(Path(args.freeze_file).read_text())["frozen"]
        rec.add_input("freeze", Path(args.freeze_file))
        chj = json.loads(Path(args.characterisation).read_text())
        rec.add_input("characterisation", Path(args.characterisation))
        if fz.get("selected_band") is None:
            raise SystemExit(f"no band was frozen ({fz.get('disposition')}); stage cannot run")
        files = fz["validation_files"] if args.stage == "validation" else fz["final_test_files"]
        planned = len(files) * len(P.methods)
        spent = int(fz.get("development_fit_count") or 0)
        print(f"  [budget] {args.stage} plans {planned} fits; development spent {spent}; "
              f"cap {P.cap}")
        if spent + planned > P.cap:
            raise SystemExit(f"{spent} + {planned} fits would exceed the hard cap {P.cap}")
        if not files:
            payload[args.stage] = {"per_file": {}, "files_evaluated": [], "fit_count": 0,
                                   "note": "empty file list; " + fz.get("validation_status", "")}
        else:
            stage_payload, draws = stage_evaluate(args, P, rec, fz, files, args.stage,
                                                  chj["characterisation"])
            payload.update(stage_payload)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload["provenance"] = rec.finalize()
    name = f"t22_{args.stage}.json"
    digest = write_json(out_dir / name, payload)
    lines = [f"{digest}  {name}"]
    if draws:
        import hashlib
        npz = out_dir / f"t22_{args.stage}_draws.npz"
        np.savez_compressed(npz, **draws)
        lines.append(f"{hashlib.sha256(npz.read_bytes()).hexdigest()}  {npz.name}")
        print(f"[T22/{args.stage}] wrote {npz} ({len(draws)} arrays)")
    (out_dir / "RESULTS_SHA256").write_text("\n".join(lines) + "\n")
    print(f"[T22/{args.stage}] wrote {out_dir}/{name} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
