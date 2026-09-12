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
``recombine``     Deterministic post-processing of an archived hold-out stage
                  output and its bootstrap draws: the combined block is
                  recomputed.  No data file is opened and nothing is fitted.
``audit-budget``  Reconstructs ``FIT_LEDGER.json`` for an existing run tree from
                  its ``T22_*/t22_*.json`` outputs, each verified against its
                  directory's ``RESULTS_SHA256``.  Writes nothing else and never
                  overwrites a ledger.

Binding to the freeze
---------------------
``validation``, ``final-test`` and ``recombine`` refuse to start without
``--freeze-file`` and are bound to it (``bind_to_freeze``) before any instrument
data is read, so no hold-out outcome can be computed under a configuration
other than the frozen one.  The supplied ``--protocol`` must BE the frozen
protocol -- its canonical JSON equal to the freeze's embedded
``stage0_protocol`` and its file digest equal to the freeze's ``protocol:``
input hash -- and the frozen seeds, replicate counts, hold-out start, rank,
window, methods and contrasts must be what it would use.  ``--characterisation``
must be the very file the freeze was built from (digest), and the
eligible-coupled hold-out set and the validation / final-test roles it implies
must be the frozen ones.  ``recombine`` additionally verifies ``--stage-output``
and ``--draws`` against their own ``RESULTS_SHA256`` and that the stage output
was produced under the supplied freeze.  The binding record, with every digest
compared, is written into the stage output.

Amendments
----------
A protocol difference is never admitted by switching a check off.  It is
admitted only for ``recombine`` (post-processing: no data read, no fit) and only
through a registered record passed with ``--amendment`` (schema
``mlst-t22-amendment/1``) that names the frozen and the amended protocol
digests, lists ``recombine`` among the stages it applies to, and lists changed
paths that cover every differing leaf.  Nothing under ``allocation``,
``data_roles``, ``unchanged_from_t10_t12``, ``representation``,
``band_selection``, ``dependence``, ``budget`` or ``contrasts`` can be amended,
and no amendment moves a frozen seed, replicate count or file role.
``validation`` and ``final-test`` accept no amendment at all.

Fit budget and ledger
---------------------
The hard cap is cumulative over the run tree, not per stage.
``FIT_LEDGER.json`` (default: beside the output directory, i.e. in the parent of
``--out``; ``--ledger`` overrides) records every stage output with its digest
and the fits it executed.  Before ``development``, ``validation`` or
``final-test`` opens a data file, the stage's fits are enumerated from the
protocol, characterisation and freeze alone and added to the cumulative executed
count -- the deduplicated UNION of the ledger's entries, a scan of the
``T22_*`` stage outputs beside the ledger and beside the output directory, and
any open, failed or interrupted reservations (``fit_accounting``), so a missing,
stale or relocated ledger cannot lower it and two *disjoint* records sum instead
of collapsing to the larger of the two -- and the stage refuses if the total
would exceed the cap.  Any stage refuses to start where stage outputs exist but no
ledger does (run ``audit-budget`` first).  Smoke runs are appended but counted
separately, as the protocol's ``budget.smoke_runs`` declares.

Every loaded stretch must carry exactly the declared calibration, fit and
evaluation windows.  Characterisation classifies a stretch against T22's own
evaluation-window count, and one classified ``underpowered`` is excluded without
being loaded; a short block reaching the scorer anyway is an integrity error.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.mlst2026 import tidmad_band as tb  # noqa: E402
from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402
from experiments.mlst2026 import fit_accounting as fa  # noqa: E402
from experiments.mlst2026.provenance import (  # noqa: E402
    RunRecorder, git_provenance, sha256_file, write_json)

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
    """Up to ``n`` whole windows from ``offset``.

    A short read is truncated to the whole windows it contains rather than
    failing: characterisation needs exactly that (it reports how many windows a
    stretch can supply).  Scoring must not accept it, so every scored stretch
    passes ``require_complete_stretch`` first.
    """
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


def require_complete_stretch(st: dict, P: Protocol, name: str,
                             roles=("cal", "fit", "eval")) -> None:
    """Refuse a loaded stretch that does not carry exactly the declared windows.

    ``load_stretch`` inherits ``_windows``' truncation, so a file that ends early
    (on either channel) would otherwise be scored on fewer windows than the
    protocol declares, silently changing the estimand and every interval.
    Characterisation classifies such a stretch ``underpowered`` against T22's own
    evaluation-window count and the stages skip it without loading it, so a
    short block arriving here is an integrity error, not an outcome.  Array
    shapes are checked directly (not the ``n_<role>`` counters, which describe
    the readout channel only).
    """
    want = {"cal": P.n_cal, "fit": P.n_fit, "eval": P.n_eval}
    bad = []
    for role in roles:
        for key in [f"{role}_read"] + ([f"{role}_ref"] if role in ("cal", "eval") else []):
            arr = st.get(key)
            got = None if arr is None else int(np.shape(arr)[0])
            if got != want[role]:
                bad.append(f"{key} has {got} windows, the protocol declares {want[role]}")
    if bad:
        raise SystemExit(f"integrity error in stretch {name}: " + "; ".join(bad)
                         + ". Characterisation should already have excluded it; "
                           "refusing to score a truncated stretch")


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
            n_total_ref = int(h[REFERENCE].shape[0])
            x = _read(h, SQUID, start, P.n_cal * P.w)
            r = _read(h, REFERENCE, start, P.n_cal * P.w)
        Cx = te.unitary_rfft(_windows(x, 0, P.n_cal, P.w))
        Cr = te.unitary_rfft(_windows(r, 0, P.n_cal, P.w))
        # Availability is set by the SHORTER channel (a reference that ends early
        # would otherwise pass here and only be caught at scoring), and the
        # outcome is decided against T22's declared evaluation-window count, not
        # the T10-T12 default of 150 that te.characterise applies otherwise.
        avail = min(P.n_eval, max(0, (min(n_total, n_total_ref) - start - P.eval_offset) // P.w))
        ch = te.characterise(Cr, Cx, P.w, N, P.stretch, int(avail), n_eval_required=P.n_eval)
        ch.update({"file": name, "role": role, "start_sample": int(start),
                   "end_sample_exclusive": int(start + P.stretch),
                   "n_total_samples": n_total,
                   "n_total_samples_reference": n_total_ref,
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


def plan_development(P: Protocol, chars: dict) -> dict:
    """The development fit enumeration, from characterisation outcomes alone.

    It needs no instrument data, so the cumulative budget can be checked before
    the first file is opened: the full band grid on the first
    ``dev_full_grid_max`` eligible-coupled stretches in declared priority order,
    the B = 1 identity control on every other stretch, every declared estimator
    per cell.  Stretches that end up skipped still count, so this is an upper
    bound on what the stage executes, never an underestimate.
    """
    eligible_order = [f for f, _ in P.dev
                      if (chars.get(f) or {}).get("outcome") == "eligible-coupled"]
    full_grid = eligible_order[:P.dev_full_grid_max]
    planned = sum(len(P.bands) if f in full_grid else 1 for f, _ in P.dev) * len(P.methods)
    return {"eligible_order": eligible_order, "full_grid": full_grid, "planned": int(planned)}


def stage_development(args, P: Protocol, rec: RunRecorder, chars: dict) -> dict:
    data_root = Path(args.data_root)
    seed0 = P.smoke_seed_base if args.smoke else P.boot_seed
    reps = 200 if args.smoke else P.boot_replicates
    curve_reps = 100 if args.smoke else 400

    plan = plan_development(P, chars)
    full_grid, planned = set(plan["full_grid"]), plan["planned"]
    print(f"  [budget] development plans {planned} fits "
          f"({len(full_grid)} stretch(es) on the full band grid, "
          f"{len(P.dev) - len(full_grid)} on the identity control only)")
    if planned > P.cap:
        raise SystemExit(f"planned {planned} fits exceeds the hard cap {P.cap}")
    N = noise_bin_power(_science_paths(data_root), P)

    dev, admiss, stretches, cell_est, fits = {}, {}, {}, {}, 0
    for name, start in P.dev:
        ch = chars.get(name, {})
        if "k_inj" not in ch:
            dev[name] = {"skipped": True, "reason": ch.get("error", "not characterised")}
            continue
        if ch.get("outcome") == "underpowered":
            # Declared outcome, not an error: the stretch cannot supply the
            # declared windows, so it is recorded and never loaded or scored.
            dev[name] = {"file": name, "skipped": True,
                         "characterisation_outcome": "underpowered",
                         "reason": "characterised underpowered ("
                                   + "; ".join(ch.get("reasons", [])) + "); not loaded"}
            continue
        st = load_stretch(_raw(data_root, name), start, P)
        require_complete_stretch(st, P, name)
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
        if ch.get("outcome") == "underpowered":
            # The declared failed-characterisation rule, applied without loading:
            # the stretch cannot supply the declared evaluation windows.
            per[name] = {"file": name, "skipped": True, "stage": label,
                         "characterisation": ch, "in_combined_estimate": False,
                         "reason": "characterised underpowered ("
                                   + "; ".join(ch.get("reasons", [])) + "); not loaded",
                         "file_verdict": {"verdict": "excluded-underpowered",
                                          "reason": "declared in advance: only eligible-coupled "
                                                    "stretches enter the combined estimate"}}
            continue
        st = load_stretch(_raw(data_root, name), P.holdout_start, P)
        require_complete_stretch(st, P, name)
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


# --- freeze binding and registered amendments ---------------------------------
#: Top-level protocol sections that define what was frozen: the data and its
#: allocation, the estimators and representation, the selection and dependence
#: rules, the budget and the contrast set.  No amendment reaches below them.
NEVER_AMENDABLE = ("allocation", "data_roles", "unchanged_from_t10_t12", "representation",
                   "band_selection", "dependence", "budget", "contrasts")
AMENDMENT_SCHEMA = "mlst-t22-amendment/1"
#: Only post-processing may run under an amendment: it reads no instrument data
#: and fits nothing, so an amendment can never buy a second look at the hold-out.
AMENDABLE_STAGES = ("recombine",)
HOLDOUT_STAGES = ("validation", "final-test")
FROZEN_FILE_KEY = {"validation": "validation_files", "final-test": "final_test_files"}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json(obj) -> str:
    """The comparison form: key order and whitespace are not scientific content,
    every value is (so ``1``, ``1.0`` and ``true`` all differ)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _read_json_with_digest(path, what: str):
    """Read a JSON file once and return ``(object, sha256)`` of those same bytes,
    so the digest that is compared is the digest of exactly what is used."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"{what} {p} does not exist")
    data = p.read_bytes()
    try:
        obj = json.loads(data)
    except ValueError as exc:
        raise SystemExit(f"{what} {p} is not valid JSON: {exc}")
    return obj, hashlib.sha256(data).hexdigest()


def _pointer(prefix: str, key) -> str:
    seg = str(key).replace("~", "~0").replace("/", "~1")
    return f"{prefix}/{seg}" if prefix else seg


def json_leaf_differences(a, b, prefix: str = "") -> list:
    """Every ``a/b/c`` path at which two JSON values differ.

    Objects are compared key by key (a key on one side only differs at that
    key), arrays index by index (surplus indices differ), anything else by its
    canonical text.  The list is empty exactly when the canonical forms agree,
    and it is what an amendment's ``changed_paths`` must cover.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(_pointer(prefix, k))
            else:
                out.extend(json_leaf_differences(a[k], b[k], _pointer(prefix, k)))
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = []
        for i in range(max(len(a), len(b))):
            if i >= len(a) or i >= len(b):
                out.append(_pointer(prefix, i))
            else:
                out.extend(json_leaf_differences(a[i], b[i], _pointer(prefix, i)))
        return out
    return [] if canonical_json(a) == canonical_json(b) else [prefix]


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _clip(items, n: int = 12) -> str:
    items = [str(i) for i in items]
    return ", ".join(items[:n]) + (f" (+{len(items) - n} more)" if len(items) > n else "")


def _manifest(directory) -> dict:
    """``RESULTS_SHA256`` in ``directory`` as ``{file name: recorded sha256}``."""
    manifest = Path(directory) / "RESULTS_SHA256"
    if not manifest.is_file():
        raise SystemExit(f"{directory}: no RESULTS_SHA256, so its outputs cannot be verified")
    out = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(f"{manifest}: malformed line {line!r}")
        out[parts[1].lstrip("*")] = parts[0]
    return out


def results_manifest_digest(path) -> str:
    """The digest the ``RESULTS_SHA256`` beside ``path`` records for it."""
    p = Path(path)
    recorded = _manifest(p.parent).get(p.name)
    if recorded is None:
        raise SystemExit(f"{p.parent / 'RESULTS_SHA256'} records no digest for {p.name}")
    return recorded


def verify_results_manifest(directory) -> dict:
    """Check every file a ``RESULTS_SHA256`` lists; return the verified digests."""
    out = {}
    for name, recorded in _manifest(directory).items():
        f = Path(directory) / name
        if not f.is_file():
            raise SystemExit(f"{Path(directory) / 'RESULTS_SHA256'} lists {name}, which does "
                             "not exist")
        got = sha256_file(f)
        if got != recorded:
            raise SystemExit(f"{f}: sha256 {got} != RESULTS_SHA256 {recorded}")
        out[name] = got
    return out


def _frozen_input_hash(frozen: dict, role: str, freeze_path) -> str:
    """The digest the freeze recorded for its ``role:<file name>`` input."""
    hits = [v for k, v in (frozen.get("input_hashes") or {}).items()
            if k.split(":", 1)[0] == role]
    if len(hits) != 1:
        raise SystemExit(f"freeze {freeze_path}: expected exactly one '{role}:' entry in "
                         f"input_hashes, found {len(hits)}; the freeze cannot be bound")
    return hits[0]


def load_amendments(paths, stage: str, frozen_sha: str, current_sha: str) -> list:
    """Validate registered amendment records; refuse anything malformed.

    An amendment is written before the amended stage runs.  It names the exact
    protocol it amends and the one it produces (by digest), the stages it applies
    to and the protocol paths it changes.  It is the only way a protocol
    difference from the freeze is admitted, it applies only to post-processing,
    and it can never reach below a ``NEVER_AMENDABLE`` section.
    """
    out, seen = [], set()
    for path in paths or ():
        am, digest = _read_json_with_digest(path, "amendment")
        where = f"amendment {path}"
        if not isinstance(am, dict) or am.get("schema") != AMENDMENT_SCHEMA:
            raise SystemExit(f"{where}: schema must be {AMENDMENT_SCHEMA!r}")
        for key in ("id", "reason", "recorded_utc"):
            if not isinstance(am.get(key), str) or not am[key].strip():
                raise SystemExit(f"{where}: '{key}' must be a non-empty string")
        if am["id"] in seen:
            raise SystemExit(f"{where}: amendment id {am['id']!r} supplied twice")
        seen.add(am["id"])
        stages = am.get("applies_to_stages")
        if not isinstance(stages, list) or not stages:
            raise SystemExit(f"{where}: 'applies_to_stages' must be a non-empty list")
        outside = [s for s in stages if s not in AMENDABLE_STAGES]
        if outside:
            raise SystemExit(f"{where}: stages {outside} cannot be amended; only "
                             f"{list(AMENDABLE_STAGES)} (no data read, no fit) accept amendments")
        if stage not in stages:
            raise SystemExit(f"{where}: does not apply to stage {stage!r} (applies to {stages})")
        if am.get("frozen_protocol_sha256") != frozen_sha:
            raise SystemExit(f"{where}: frozen_protocol_sha256 {am.get('frozen_protocol_sha256')} "
                             f"is not the freeze's protocol digest {frozen_sha}")
        if am.get("amended_protocol_sha256") != current_sha:
            raise SystemExit(f"{where}: amended_protocol_sha256 "
                             f"{am.get('amended_protocol_sha256')} is not the supplied "
                             f"protocol's digest {current_sha}")
        cps = am.get("changed_paths")
        if (not isinstance(cps, list) or not cps
                or not all(isinstance(c, str) and c.strip("/") for c in cps)):
            raise SystemExit(f"{where}: 'changed_paths' must be a non-empty list of 'a/b/c' paths")
        cps = [c.strip("/") for c in cps]
        forbidden = sorted({c for c in cps for p in NEVER_AMENDABLE if _under(c, p)})
        if forbidden:
            raise SystemExit(f"{where}: changed_paths {forbidden} lie under a never-amendable "
                             f"section ({', '.join(NEVER_AMENDABLE)})")
        out.append({"id": am["id"], "file": str(path), "sha256": digest,
                    "reason": am["reason"], "recorded_utc": am["recorded_utc"],
                    "changed_paths": cps})
    return out


def bind_to_freeze(P: Protocol, freeze_path, characterisation_path, stage: str,
                   amendments=(), *, stage_output=None, draws=None):
    """Bind a hold-out or post-processing stage to its freeze, before any data is read.

    A hold-out stage takes its allocation, seeds, replicate counts, contrasts and
    file roles from ``--protocol`` and its outcomes from ``--characterisation``,
    so both must be exactly what was frozen, or a changed seed or allocation
    would silently produce a different hold-out evaluation under the frozen
    band's name.  The checks:

    * protocol: canonical JSON equal to ``frozen["stage0_protocol"]`` and file
      sha256 equal to the freeze's ``protocol:`` input hash -- unless registered
      amendments (``recombine`` only, both digests matching) cover every
      differing leaf and none lies under a ``NEVER_AMENDABLE`` section;
    * the frozen seeds, replicate counts, hold-out start, rank, window, methods
      and contrasts equal what the protocol gives (never amendable);
    * characterisation sha256 equal to the freeze's ``characterisation:`` hash
      (the file is required for validation / final-test) and the
      eligible-coupled hold-out set it implies equal to the frozen one;
    * the frozen validation / final-test lists are the declared assignment of
      the frozen eligible set, and the development fit count lies within the
      enumeration;
    * recombine: ``--stage-output`` and ``--draws`` match their own
      ``RESULTS_SHA256``, the stage output's provenance ``freeze`` input is the
      supplied freeze (its ``characterisation`` input, if recorded, the frozen
      one; its ``draws`` input, if recorded, the supplied draws), and the files
      it evaluated are the frozen ones for its label.

    Every failing check is collected and reported in one refusal.  Returns
    ``(record, loaded)``: ``record`` is JSON-ready and goes into the stage
    output; ``loaded`` holds the freeze, characterisation and stage output
    parsed from exactly the bytes whose digests were compared.
    """
    if stage not in HOLDOUT_STAGES + AMENDABLE_STAGES:
        raise SystemExit(f"stage {stage!r} is not bound to a freeze")
    amendments = list(amendments or ())
    if amendments and stage not in AMENDABLE_STAGES:
        raise SystemExit(f"{stage} accepts no amendments: it computes hold-out outcomes and "
                         "runs under the frozen protocol byte for byte")
    if not freeze_path:
        raise SystemExit(f"{stage} requires --freeze-file; the band must be frozen before any "
                         "hold-out method outcome is computed")
    if stage in HOLDOUT_STAGES and not characterisation_path:
        raise SystemExit(f"{stage} requires --characterisation: the file the freeze was built "
                         "from, verified by digest")
    fz, freeze_sha = _read_json_with_digest(freeze_path, "freeze file")
    frozen = fz.get("frozen") if isinstance(fz, dict) else None
    if not isinstance(frozen, dict) or not isinstance(frozen.get("stage0_protocol"), dict):
        raise SystemExit(f"freeze file {freeze_path} has no frozen block embedding a "
                         "stage0_protocol; there is nothing to bind to")
    problems: list = []

    # 1. The protocol itself.
    frozen_proto = frozen["stage0_protocol"]
    frozen_sha = _frozen_input_hash(frozen, "protocol", freeze_path)
    current_sha = P.sha256()
    identical = canonical_json(frozen_proto) == canonical_json(P.raw)
    diffs = json_leaf_differences(frozen_proto, P.raw)
    applied: list = []
    if identical and current_sha == frozen_sha:
        if amendments:
            problems.append("amendments were supplied but the protocol is identical to the "
                            "frozen one; there is nothing to amend")
    elif not amendments:
        if not identical:
            problems.append(f"protocol differs from the frozen stage0_protocol at {len(diffs)} "
                            f"path(s): {_clip(diffs)}")
        if current_sha != frozen_sha:
            problems.append(f"protocol sha256 {current_sha} != the freeze's protocol input "
                            f"hash {frozen_sha}")
    else:
        applied = load_amendments(amendments, stage, frozen_sha, current_sha)
        blocked = [d for d in diffs if any(_under(d, p) for p in NEVER_AMENDABLE)]
        covered = [c for a in applied for c in a["changed_paths"]]
        uncovered = [d for d in diffs
                     if d not in blocked and not any(_under(d, c) for c in covered)]
        if blocked:
            problems.append(f"protocol differs under never-amendable sections, which no "
                            f"amendment can admit: {_clip(blocked)}")
        if uncovered:
            problems.append(f"protocol differences not covered by any registered amendment: "
                            f"{_clip(uncovered)}")

    # 2. What the stage takes from the protocol must be what was frozen.  Compared
    #    even under an amendment: no amendment moves a seed, a count or a contrast.
    expected = {"bootstrap_seed": P.boot_seed, "bootstrap_replicates": P.boot_replicates,
                "file_level_seed": P.file_seed, "file_level_replicates": P.file_replicates,
                "holdout_start_sample": P.holdout_start, "rank": P.rank,
                "window_samples": P.w, "methods": list(P.methods),
                "contrasts": [list(c) for c in P.contrasts]}
    for key, want in expected.items():
        if key not in frozen:
            problems.append(f"the freeze records no {key}")
        elif canonical_json(frozen[key]) != canonical_json(want):
            problems.append(f"{key}: frozen {canonical_json(frozen[key])}, protocol gives "
                            f"{canonical_json(want)}")

    # 3. The characterisation the stage reads outcomes from.
    frozen_ch_sha = _frozen_input_hash(frozen, "characterisation", freeze_path)
    chars, ch_record = None, None
    if characterisation_path:
        chj, ch_sha = _read_json_with_digest(characterisation_path, "characterisation")
        if ch_sha != frozen_ch_sha:
            problems.append(f"characterisation sha256 {ch_sha} != the freeze's "
                            f"characterisation input hash {frozen_ch_sha}")
        chars = chj.get("characterisation") if isinstance(chj, dict) else None
        if not isinstance(chars, dict):
            problems.append(f"characterisation {characterisation_path} has no "
                            "characterisation block")
            chars = None
        ch_record = {"file": str(characterisation_path), "sha256": ch_sha,
                     "frozen_sha256": frozen_ch_sha}

    # 4. File roles: the declared assignment of the frozen eligible set, and that
    #    set re-derived from the characterisation.
    fe, fv, ft = (frozen.get(k) for k in ("eligible_holdout_files", "validation_files",
                                          "final_test_files"))
    if not all(isinstance(x, list) for x in (fe, fv, ft)):
        problems.append("the freeze lacks eligible_holdout_files / validation_files / "
                        "final_test_files lists")
    else:
        if [f for f in P.holdout_files if f in set(fe)] != fe:
            problems.append(f"eligible_holdout_files {fe} is not an in-order subset of the "
                            "protocol's hold-out files")
        if fv != fe[:3]:
            problems.append(f"validation_files {fv} are not the first three eligible hold-out "
                            f"files {fe[:3]}")
        want_ft = [f for f in P.holdout_files if f not in set(fv)]
        if ft != want_ft:
            problems.append(f"final_test_files {ft} are not the hold-out files outside "
                            f"validation {want_ft}")
        if frozen.get("n_eligible_coupled_holdout_files") != len(fe):
            problems.append(f"n_eligible_coupled_holdout_files "
                            f"{frozen.get('n_eligible_coupled_holdout_files')!r} != the "
                            f"{len(fe)} files listed")
        if chars is not None:
            eligible = [f for f in P.holdout_files
                        if (chars.get(f) or {}).get("outcome") == "eligible-coupled"]
            if eligible != fe:
                problems.append(
                    f"eligible-coupled hold-out files in the characterisation {eligible} differ "
                    f"from the frozen {fe} (only in the characterisation: "
                    f"{sorted(set(eligible) - set(fe))}; only in the freeze: "
                    f"{sorted(set(fe) - set(eligible))})")
    dfc = frozen.get("development_fit_count")
    dev_plan = plan_development(P, chars)["planned"] if chars is not None else None
    if isinstance(dfc, bool) or not isinstance(dfc, int) or dfc < 0:
        problems.append(f"development_fit_count {dfc!r} is not a non-negative integer")
    elif dev_plan is not None and dfc > dev_plan:
        problems.append(f"development_fit_count {dfc} exceeds the development enumeration "
                        f"{dev_plan} that this protocol and characterisation give")

    record = {
        "stage": stage, "verdict": "bound", "checked_utc": _utc_now(),
        "freeze": {"file": str(freeze_path), "sha256": freeze_sha},
        "protocol": {"file": str(P.path), "sha256": current_sha,
                     "frozen_input_sha256": frozen_sha,
                     "canonical_sha256": hashlib.sha256(canonical_json(P.raw).encode()).hexdigest(),
                     "frozen_canonical_sha256":
                         hashlib.sha256(canonical_json(frozen_proto).encode()).hexdigest(),
                     "canonical_identical": identical, "differing_paths": diffs},
        "frozen_fields_compared": {k: frozen.get(k) for k in expected},
        "characterisation": ch_record,
        "file_roles": {"eligible_holdout_files": fe, "validation_files": fv,
                       "final_test_files": ft, "development_fit_count": dfc,
                       "development_enumeration": dev_plan},
        "amendments_applied": applied,
    }

    # 5. recombine: the archived output, its draws, and the freeze it was made under.
    src = None
    if stage == "recombine":
        if not stage_output or not draws:
            raise SystemExit("recombine requires --stage-output and --draws")
        src, src_sha = _read_json_with_digest(stage_output, "stage output")
        want = results_manifest_digest(stage_output)
        if src_sha != want:
            problems.append(f"stage output sha256 {src_sha} != RESULTS_SHA256 {want}")
        if not Path(draws).is_file():
            raise SystemExit(f"draws {draws} does not exist")
        draws_sha, want_d = sha256_file(draws), results_manifest_digest(draws)
        if draws_sha != want_d:
            problems.append(f"draws sha256 {draws_sha} != RESULTS_SHA256 {want_d}")
        src = src if isinstance(src, dict) else {}
        by_role: dict = {}
        for r in (src.get("provenance") or {}).get("inputs") or []:
            by_role.setdefault(r.get("role"), []).append(r.get("sha256"))
        if not by_role.get("freeze"):
            problems.append("the stage output records no provenance input with role 'freeze'")
        elif set(by_role["freeze"]) != {freeze_sha}:
            problems.append(f"the stage output was produced under freeze sha256 "
                            f"{sorted(set(by_role['freeze']), key=str)}, not the supplied "
                            f"{freeze_sha}")
        if by_role.get("characterisation") and set(by_role["characterisation"]) != {frozen_ch_sha}:
            problems.append(f"the stage output read characterisation sha256 "
                            f"{sorted(set(by_role['characterisation']), key=str)}, not the "
                            f"frozen {frozen_ch_sha}")
        if by_role.get("draws"):
            if set(by_role["draws"]) != {draws_sha}:
                problems.append(f"the stage output records draws sha256 "
                                f"{sorted(set(by_role['draws']), key=str)}, not the supplied "
                                f"{draws_sha}")
        elif Path(draws).resolve().parent != Path(stage_output).resolve().parent:
            problems.append("the draws are not beside the stage output and the stage output "
                            "records no draws input, so their pairing cannot be verified")
        label = src.get("stage")
        if label not in HOLDOUT_STAGES or not isinstance(src.get(label), dict):
            problems.append(f"the stage output is labelled {label!r}; only validation / "
                            "final-test outputs can be recombined")
        elif src[label].get("files_evaluated") != frozen.get(FROZEN_FILE_KEY[label]):
            problems.append(f"the stage output evaluated {src[label].get('files_evaluated')}, "
                            f"not the frozen {FROZEN_FILE_KEY[label]} "
                            f"{frozen.get(FROZEN_FILE_KEY[label])}")
        record["stage_output"] = {"file": str(stage_output), "sha256": src_sha,
                                  "results_sha256": want, "label": label}
        record["draws"] = {"file": str(draws), "sha256": draws_sha, "results_sha256": want_d}
    else:
        record["files"] = frozen.get(FROZEN_FILE_KEY[stage])

    if problems:
        raise SystemExit(f"refusing to bind stage {stage} to freeze {freeze_path}:\n  - "
                         + "\n  - ".join(problems))
    print(f"  [bind] {stage}: freeze {freeze_sha[:12]}, protocol {current_sha[:12]}"
          + (f" (amended: {', '.join(a['id'] for a in applied)})" if applied else "")
          + (f", characterisation {ch_record['sha256'][:12]}" if ch_record else ""))
    return record, {"frozen": frozen, "characterisation": chars, "stage_output": src}


# --- cumulative fit ledger ----------------------------------------------------
LEDGER_NAME = "FIT_LEDGER.json"
LEDGER_SCHEMA = "mlst-t22-fit-ledger/1"
FIT_STAGES = ("development",) + HOLDOUT_STAGES
#: The per-cell numbers that must agree exactly for a development output to be a
#: deterministic re-score of an earlier one rather than a new evaluation.
RESCORE_FIELDS = ("mean_delta_squared", "ci95_lo", "ci95_hi", "bootstrap_sd")


def fits_in_output(js: dict) -> int:
    """Estimator fits a stage output records as executed.

    ``fits_performed`` wins over ``fit_count`` where both exist: a recombine
    output repeats the original stage's ``fit_count`` for reference but performs
    none.  characterize and freeze outputs fit nothing.
    """
    label = js.get("stage")
    if label == "development":
        block = js
    elif label in HOLDOUT_STAGES and isinstance(js.get(label), dict):
        block = js[label]
    else:
        return 0
    return int(block.get("fits_performed", block.get("fit_count")) or 0)


def stage_outputs_under(dirs, stages=None) -> list:
    """Archived ``T22_*/t22_<stage>.json`` outputs under each directory, de-duplicated."""
    seen, out = set(), []
    for d in dirs:
        for sub in sorted(Path(d).glob("T22_*")):
            if not sub.is_dir():
                continue
            for p in sorted(sub.glob("t22_*.json")):
                if stages is not None and p.stem[len("t22_"):] not in stages:
                    continue
                if p.resolve() not in seen:
                    seen.add(p.resolve())
                    out.append(p)
    return out


def scan_executed_fits(dirs) -> dict:
    """Fits executed according to the stage outputs on disk, independent of any ledger.

    A deleted, stale, hand-edited or relocated ledger cannot lower this count.
    Only development / validation / final-test outputs are read (a recombine
    output is ``t22_recombine.json`` and fits nothing); smoke outputs are
    totalled separately.
    """
    total, smoke, rows = 0, 0, []
    for p in stage_outputs_under(dirs, FIT_STAGES):
        try:
            js = json.loads(p.read_text())
        except ValueError as exc:
            raise SystemExit(f"{p}: unreadable stage output ({exc}); the cumulative fit count "
                             "cannot be established")
        if not isinstance(js, dict):
            raise SystemExit(f"{p}: not a stage output; the cumulative fit count cannot be "
                             "established")
        n = fits_in_output(js)
        if js.get("smoke"):
            smoke += n
        else:
            total += n
        rows.append({"output": str(p), "fits": n, "smoke": bool(js.get("smoke"))})
    return {"executed_total": total, "smoke_total": smoke, "outputs": rows}


def _ledger_totals(ledger: dict) -> dict:
    """Recompute the totals from the entries: the entries are the record."""
    real = [e for e in ledger["entries"] if not e.get("smoke")]
    ledger["executed_total"] = int(sum(e["fits_executed"] for e in real))
    ledger["unique_scientific_total"] = int(sum(e["fits_executed"] for e in real
                                                if not e.get("deterministic_rescore_of")))
    ledger["smoke_executed_total"] = int(sum(e["fits_executed"] for e in ledger["entries"]
                                             if e.get("smoke")))
    ledger["executed_total_exceeds_cap"] = bool(ledger["executed_total"] > ledger["hard_cap"])
    return ledger


def load_ledger(path, P: Protocol):
    """The ledger at ``path`` (None if absent); refuse one that cannot be trusted."""
    path = Path(path)
    if not path.exists():
        return None
    ledger, _ = _read_json_with_digest(path, "fit ledger")
    if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
        raise SystemExit(f"fit ledger {path}: schema must be {LEDGER_SCHEMA!r}")
    if ledger.get("hard_cap") != P.cap:
        raise SystemExit(f"fit ledger {path} records hard cap {ledger.get('hard_cap')}, the "
                         f"protocol declares {P.cap}")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not all(
            isinstance(e, dict) and isinstance(e.get("fits_executed"), int)
            and not isinstance(e["fits_executed"], bool) and e["fits_executed"] >= 0
            for e in entries):
        raise SystemExit(f"fit ledger {path}: every entry needs a non-negative integer "
                         "fits_executed")
    stored = {k: ledger.get(k) for k in ("executed_total", "unique_scientific_total")}
    _ledger_totals(ledger)
    if any(stored[k] != ledger[k] for k in stored):
        raise SystemExit(f"fit ledger {path}: stored totals {stored} disagree with its entries "
                         f"(executed {ledger['executed_total']}, unique "
                         f"{ledger['unique_scientific_total']}); refusing to trust it")
    return ledger


def _write_ledger(path, ledger: dict, *, exclusive: bool = False) -> None:
    path = Path(path)
    data = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with open(path, "x") as fh:          # never overwrite, not even under a race
            fh.write(data)
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


def reservations_path_for(ledger_path, explicit=None) -> Path:
    """The reservation store: ``--reservations`` if given, else beside the ledger."""
    return Path(explicit) if explicit else Path(ledger_path).parent / fa.RESERVATION_NAME


def stage_exposure(P: Protocol, stage: str, files=None) -> list:
    """The instrument files and sample regions the stage's allocation permits it.

    Recorded on the reservation *before* the stage runs, so what a run was
    allowed to touch is auditable independently of what it did touch.  See
    :func:`fit_accounting.attach_exposure` for what this record does and does not
    do.
    """
    if stage == "development":
        targets = [(f, int(s)) for f, s in P.dev]
    elif stage in HOLDOUT_STAGES:
        targets = [(f, int(P.holdout_start)) for f in (files or [])]
    else:
        targets = []
    return [{"dataset_id": f, "region": {"start_sample": s, "n_samples": int(P.stretch),
                                         "end_sample_exclusive": s + int(P.stretch)},
             "note": f"{stage}: declared allocation"} for f, s in targets]


def reserve_stage_fits(P: Protocol, ledger_path, stage: str, planned: int, *, scan_dirs,
                       reservations_path=None, smoke=False, files=None, meta=None) -> dict:
    """Mint the reservation that pays for this stage, before it reads anything.

    Called after the freeze binding and the budget pre-flight and *before* the
    first instrument byte -- before the science files are even hashed.  Three
    things happen here that the pre-flight alone cannot do:

    * a ``running`` reservation whose owning process is gone is swept to
      ``interrupted`` first.  It keeps its charge: the compute it spent before
      dying is spent, so a retry adds to the total instead of resetting it.
    * the cap is re-tested **under the store's exclusive lock**, against the
      deduplicated union of completed executions plus every open charge.  Two
      runners launched at once therefore cannot both see the same room: whoever
      takes the lock second sees the first one's reservation already charged.
    * the stage's planned fits become visible to every other process the moment
      they are reserved, so a crash between here and the ledger append leaves the
      compute charged rather than invisible.

    Returns the reservation; raises ``SystemExit`` if it does not fit.
    """
    res_path = reservations_path_for(ledger_path, reservations_path)
    if res_path.is_file():
        swept = fa.sweep_interrupted(res_path)
        if swept["swept"]:
            print(f"  [reservation] swept {len(swept['swept'])} abandoned reservation(s) to "
                  f"interrupted; their charge is kept: {_clip(swept['swept'], 4)}")
        if swept["unverifiable"]:
            print(f"  [reservation] {len(swept['unverifiable'])} running reservation(s) are "
                  f"owned by another host and cannot be probed from here; still charged")
    ledger = load_ledger(ledger_path, P)
    scan = scan_executed_fits(scan_dirs)
    reservations = fa.load_reservations(res_path, cap=P.cap)
    union = fa.union_executed(ledger, scan, ledger_path=ledger_path, reservations=reservations)
    spent = max(union["executed"], ledger["executed_total"] if ledger else 0,
                scan["executed_total"])
    row = fa.reserve(res_path, P.cap, planned, meta or {}, stage=stage, spent=spent,
                     smoke=bool(smoke), config_id=f"t22:{stage}",
                     exposure=stage_exposure(P, stage, files))
    print(f"  [reservation] {stage} reserved {planned} fit(s) as {row['uuid']} in {res_path} "
          f"(completed {spent} charged before this one: {row['charged_before']}, cap {P.cap}"
          f"{'; smoke, exempt' if smoke else ''})")
    return row


def check_cumulative_budget(P: Protocol, ledger_path, planned: int, stage: str, *,
                            scan_dirs, reservations_path=None) -> dict:
    """Refuse a stage whose planned fits would take the cumulative total past the cap.

    Cumulative expenditure is the deduplicated UNION of what the ledger recorded,
    what a ledger-independent scan of the stage outputs finds, and what the
    reservation store still holds open (``fit_accounting.union_executed``): an
    execution both sources saw counts once, and two *disjoint* records sum rather
    than collapsing to the larger of the two, which is what ``max`` got wrong.
    Failed and interrupted reservations keep their charge, so a crash and a retry
    accumulate.  The old ``max(ledger, scan)`` is kept as a floor: the union may
    raise a refusal but is never allowed to lower one.  Nothing here writes: a
    reservation store is read only where one already exists.
    """
    ledger = load_ledger(ledger_path, P)
    ledger_total = ledger["executed_total"] if ledger is not None else None
    scan = scan_executed_fits(scan_dirs)
    res_path = reservations_path_for(ledger_path, reservations_path)
    reservations = fa.load_reservations(res_path, cap=P.cap)
    union = fa.union_executed(ledger, scan, ledger_path=ledger_path,
                              reservations=reservations)
    floor = max(ledger_total or 0, scan["executed_total"])
    spent = max(union["charged"], floor)
    shown = "absent" if ledger_total is None else ledger_total
    print(f"  [budget] {stage} plans {planned} fits; cumulative executed {spent} "
          f"(ledger {shown}, archive scan {scan['executed_total']}); cap {P.cap}")
    print(f"  [budget] union: {union['executed']} executed, "
          f"{union['unique_scientific']} unique scientific, "
          f"{union['inference_only_fits']} inference-only, "
          f"{union['reserved_open_fits']} reserved open/failed/interrupted "
          f"({union['duplicates_merged']} duplicate record(s) merged)")
    if spent + planned > P.cap:
        raise SystemExit(
            f"{stage} refused: cumulative executed fits {spent} + planned {planned} = "
            f"{spent + planned} would exceed the hard cap {P.cap} (ledger {ledger_path}: "
            f"{shown}; archive scan of {_clip(scan_dirs)}: {scan['executed_total']}; "
            f"union: {union['executed']} executed, {union['unique_scientific']} unique "
            f"scientific, {union['reserved_open_fits']} held by reservations)")
    return {"planned": int(planned), "cumulative_executed_before": int(spent),
            "hard_cap": P.cap, "ledger": str(ledger_path), "ledger_executed_total": ledger_total,
            "archive_scan_executed_total": scan["executed_total"],
            "archive_scan_outputs": scan["outputs"],
            "union": {k: union[k] for k in (
                "executed", "unique_scientific", "fitted_models", "inference_only_fits",
                "restart_fits", "scientific_configurations", "reserved_open_fits",
                "charged", "duplicates_merged", "ledger_only", "scan_only",
                "seen_by_both")},
            "reservation_store": str(res_path) if reservations is not None else None,
            "completed_executed_before": int(max(union["executed"], floor)),
            "legacy_max_floor": int(floor)}


def require_ledger_beside_outputs(ledger_path, scan_dirs, stage: str) -> None:
    """Refuse to start a ledger from nothing where expenditure already exists.

    A ledger created now would open with this stage and omit everything before
    it; ``audit-budget`` reconstructs that history from the verified outputs.
    """
    if Path(ledger_path).exists():
        return
    existing = stage_outputs_under(scan_dirs)
    if existing:
        raise SystemExit(
            f"{stage} refused: fit ledger {ledger_path} does not exist although "
            f"{len(existing)} T22 stage output(s) do ({_clip(existing, 4)}). Reconstruct it "
            "first with --stage audit-budget --results-root DIR, so that no earlier "
            "expenditure goes unrecorded")


def _relative_to(path, base) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def ledger_entry(stage: str, out_dir, output_file: str, output_sha256: str, js: dict,
                 ledger_dir, *, recorded_utc=None) -> dict:
    prov = js.get("provenance") or {}
    git = prov.get("git") or {}
    entry = {"stage": stage, "output_dir": _relative_to(out_dir, ledger_dir),
             "output_file": output_file, "output_sha256": output_sha256,
             "fits_executed": 0 if stage == "recombine" else fits_in_output(js),
             "smoke": bool(js.get("smoke")), "git_commit": git.get("commit"),
             "git_tree_dirty": git.get("tree_dirty"),
             "output_finished_utc": prov.get("finished_utc"),
             "recorded_utc": recorded_utc or _utc_now()}
    if stage == "recombine":
        entry["recombined_stage"] = js.get("stage")
    uid = fa.uuid_in_output(js)
    if uid:
        # the reservation this output was executed under: the one identity that
        # survives the output being moved, rewritten or produced in duplicate
        entry["reservation_uuid"] = uid
    return entry


def record_in_ledger(ledger_path, P: Protocol, entry: dict) -> dict:
    """Append one stage's entry and rewrite the totals, under an exclusive lock.

    The append is a read-modify-write: load the ledger, add an entry, replace the
    file.  Unlocked, two stages completing at the same moment both read the same
    ledger, both append their own entry and the second replace wins -- the first
    entry, and every fit it recorded, is gone from the cap.  ``os.replace`` makes
    each write atomic but does nothing about the interleaving.  The whole cycle
    now runs inside ``fit_accounting.locked``, the same advisory lock the
    reservation store takes, so completions serialise against each other.

    The lock binds processes that take it -- this runner -- and nothing else: a
    hand-edit of the ledger while a stage is completing is still a lost update.
    """
    with fa.locked(ledger_path):
        ledger = load_ledger(ledger_path, P)
        if ledger is None:
            ledger = {"schema": LEDGER_SCHEMA, "hard_cap": P.cap,
                      "counting_convention": P.raw["budget"]["fit_definition"], "entries": []}
        ledger["entries"].append(entry)
        _write_ledger(ledger_path, _ledger_totals(ledger))
    return ledger


def development_contrast_signature(js: dict) -> dict:
    """Every per-cell contrast number a development output reports, keyed by
    (stretch, band, contrast, field); an unscored cell is keyed with None."""
    sig = {}
    for fname, entry in (js.get("development") or {}).items():
        for B, cell in ((entry or {}).get("cells") or {}).items():
            contrasts = (cell or {}).get("contrasts")
            if not isinstance(contrasts, dict):
                sig[f"{fname}|B={B}"] = None
                continue
            for key, c in contrasts.items():
                for fld in RESCORE_FIELDS:
                    sig[f"{fname}|B={B}|{key}|{fld}"] = (c or {}).get(fld)
    return sig


def audit_budget(args, P: Protocol) -> int:
    """Reconstruct ``FIT_LEDGER.json`` for an existing run tree; nothing is re-run.

    Every ``T22_*/t22_*.json`` under ``--results-root`` is verified against its
    directory's ``RESULTS_SHA256`` (every file that manifest lists is checked)
    and entered, in the order the outputs finished, with the fits it records.
    A development output whose every per-cell contrast number
    (``RESCORE_FIELDS``) equals an earlier development output's is marked
    ``deterministic_rescore_of`` it: its fits were executed and count in
    ``executed_total``, but it is no new scientific evaluation and is left out
    of ``unique_scientific_total``.  An existing ledger is never overwritten.
    """
    if not args.results_root:
        raise SystemExit("audit-budget requires --results-root")
    if args.out:
        raise SystemExit("audit-budget writes only the ledger; --out is not used")
    root = Path(args.results_root)
    if not root.is_dir():
        raise SystemExit(f"--results-root {root} is not a directory")
    ledger_path = Path(args.ledger) if args.ledger else root / LEDGER_NAME
    if ledger_path.exists():
        raise SystemExit(f"refusing to overwrite existing fit ledger {ledger_path}")
    outputs = stage_outputs_under([root])
    if not outputs:
        raise SystemExit(f"no T22_*/t22_*.json stage outputs under {root}")

    now, verified, rows = _utc_now(), {}, []
    for p in outputs:
        if p.parent not in verified:
            verified[p.parent] = verify_results_manifest(p.parent)
        js, digest = _read_json_with_digest(p, "stage output")
        recorded = verified[p.parent].get(p.name)
        if recorded is None:
            raise SystemExit(f"{p.parent / 'RESULTS_SHA256'} records no digest for {p.name}")
        if digest != recorded:
            raise SystemExit(f"{p}: sha256 {digest} != RESULTS_SHA256 {recorded}")
        stage = p.stem[len("t22_"):]
        label = js.get("stage") if isinstance(js, dict) else None
        bad_label = (label not in HOLDOUT_STAGES) if stage == "recombine" else (label != stage)
        if bad_label:
            raise SystemExit(f"{p}: the file name says stage {stage!r} but the output is "
                             f"labelled {label!r}")
        entry = ledger_entry(stage, p.parent, p.name, digest, js, ledger_path.parent,
                             recorded_utc=now)
        entry["results_sha256_verified"] = True
        rows.append((entry["output_finished_utc"] or "", p.parent.name, entry, js))
    rows.sort(key=lambda r: (r[0], r[1]))

    checks, originals = [], []
    for _, _, entry, js in rows:
        if entry["stage"] != "development" or entry["smoke"]:
            continue
        sig = development_contrast_signature(js)
        n_values = sum(v is not None for v in sig.values())
        ref = f"{entry['output_dir']}/{entry['output_file']}"
        for earlier_ref, earlier_sig in originals:
            same = n_values > 0 and canonical_json(sig) == canonical_json(earlier_sig)
            checks.append({"earlier": earlier_ref, "later": ref,
                           "values_compared": n_values, "identical": same})
            if same:
                entry["deterministic_rescore_of"] = earlier_ref
                entry["rescore_comparison"] = {"values_compared": n_values,
                                               "fields": list(RESCORE_FIELDS),
                                               "identical": True}
                break
        else:
            originals.append((ref, sig))

    git = git_provenance()
    ledger = {"schema": LEDGER_SCHEMA, "hard_cap": P.cap,
              "counting_convention": P.raw["budget"]["fit_definition"],
              "entries": [r[2] for r in rows],
              "audit": {"built_by": "t22_tidmad_narrowband --stage audit-budget",
                        "results_root": str(root), "recorded_utc": now,
                        "protocol_file": str(P.path), "protocol_sha256": P.sha256(),
                        "runner_git_commit": git.get("commit"),
                        "runner_git_tree_dirty": git.get("tree_dirty"),
                        "outputs_verified_against_results_sha256": len(rows),
                        "development_rescore_checks": checks,
                        "note": "reconstructed after the fact from the archived stage outputs; "
                                "nothing was re-run. executed_total counts every fit executed, "
                                "deterministic re-scores included; unique_scientific_total "
                                "leaves them out."}}
    _ledger_totals(ledger)
    _write_ledger(ledger_path, ledger, exclusive=True)
    for e in ledger["entries"]:
        print(f"  [audit] {e['output_dir']}/{e['output_file']:<24s} {e['stage']:<12s} "
              f"fits={e['fits_executed']:>4d}"
              + (f"  deterministic re-score of {e['deterministic_rescore_of']}"
                 if e.get("deterministic_rescore_of") else ""))
    print(f"[T22/audit-budget] wrote {ledger_path}: executed {ledger['executed_total']}, "
          f"unique scientific {ledger['unique_scientific_total']}, hard cap {P.cap}, "
          f"exceeds cap: {ledger['executed_total_exceeds_cap']}")
    return 0


# --- main ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["characterize", "development", "freeze", "validation",
                             "final-test", "recombine", "audit-budget"])
    ap.add_argument("--stage-output", help="recombine: the stage JSON to re-summarise")
    ap.add_argument("--draws", help="recombine: the matching *_draws.npz")
    ap.add_argument("--protocol", required=True,
                    help="stage-0 protocol JSON (every stage; audit-budget takes the hard cap "
                         "and the fit-counting convention from it)")
    ap.add_argument("--data-root", default="/ceph/dwong/paper1_dataset")
    ap.add_argument("--out", help="output directory (required for every stage except "
                                  "audit-budget)")
    ap.add_argument("--characterisation", help="JSON from the characterize stage")
    ap.add_argument("--development", help="JSON from the development stage")
    ap.add_argument("--freeze-file", help="frozen configuration JSON")
    ap.add_argument("--amendment", action="append", default=[], metavar="PATH",
                    help=f"recombine only, repeatable: a registered {AMENDMENT_SCHEMA} record "
                         "admitting a protocol difference from the freeze")
    ap.add_argument("--ledger", metavar="PATH",
                    help=f"cumulative fit ledger (default: {LEDGER_NAME} in the parent of "
                         "--out; for audit-budget, in --results-root)")
    ap.add_argument("--reservations", metavar="PATH",
                    help=f"fit reservation store (default: {fa.RESERVATION_NAME} beside the "
                         "ledger). Pending, running, failed and interrupted reservations are "
                         "charged against the cap; it is read where it exists, never created "
                         "here")
    ap.add_argument("--results-root", metavar="DIR",
                    help="audit-budget only: the run tree whose T22_*/t22_*.json outputs are "
                         "verified and entered in a new ledger")
    ap.add_argument("--smoke", action="store_true",
                    help="implementation check: few replicates, smoke seeds, output "
                         "marked non-scientific")
    args = ap.parse_args(argv)

    P = Protocol(Path(args.protocol))
    if args.stage == "audit-budget":
        return audit_budget(args, P)
    if not args.out:
        raise SystemExit(f"--out is required for stage {args.stage}")
    if args.results_root:
        raise SystemExit("--results-root is used only by --stage audit-budget")
    if args.amendment and args.stage not in AMENDABLE_STAGES:
        raise SystemExit(f"--amendment is accepted only for {', '.join(AMENDABLE_STAGES)}; "
                         f"{args.stage} accepts no amendments")
    if args.smoke and args.stage in HOLDOUT_STAGES:
        # A smoke run is exempt from the cap, but stage_evaluate would still score
        # the frozen hold-out stretches with the frozen seeds and replicate counts:
        # hold-out exposure outside the budget.  Hold-out code paths are exercised
        # on synthetic fixtures instead.
        raise SystemExit(f"{args.stage} refuses --smoke: a smoke run would score the hold-out "
                         "stretches and expose their outcomes outside the fit ledger; test "
                         "the hold-out path on synthetic fixtures")
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory {out_dir}")
    ledger_path = Path(args.ledger) if args.ledger else out_dir.parent / LEDGER_NAME
    # The expenditure scan covers the ledger's directory AND the output's parent,
    # so pointing --ledger somewhere empty cannot hide the outputs beside --out.
    scan_dirs = list(dict.fromkeys([ledger_path.parent.resolve(), out_dir.parent.resolve()]))

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

    payload: dict = {"stage": args.stage, "smoke": bool(args.smoke),
                     "protocol_sha256": P.sha256(),
                     "allocation": P.raw["allocation"],
                     "band_candidates": list(P.bands)}
    draws: dict = {}

    # --- pre-flight: no instrument data (not even a science-file digest) is read
    # until the freeze binding and the cumulative budget have both passed.
    loaded: dict = {}
    if args.stage in HOLDOUT_STAGES + AMENDABLE_STAGES:
        payload["freeze_binding"], loaded = bind_to_freeze(
            P, args.freeze_file, args.characterisation, args.stage, args.amendment,
            stage_output=args.stage_output, draws=args.draws)
    chars, files, planned = None, None, None
    if args.stage == "development":
        if not args.characterisation:
            raise SystemExit("development requires --characterisation")
        chars = _read_json_with_digest(args.characterisation, "characterisation")[0]["characterisation"]
        planned = plan_development(P, chars)["planned"]
    elif args.stage in HOLDOUT_STAGES:
        fz = loaded["frozen"]
        if fz.get("selected_band") is None:
            raise SystemExit(f"no band was frozen ({fz.get('disposition')}); stage cannot run")
        files = fz[FROZEN_FILE_KEY[args.stage]]
        planned = len(files) * len(P.methods)
    if planned is not None and not args.smoke:
        payload["budget_preflight"] = check_cumulative_budget(
            P, ledger_path, planned, args.stage, scan_dirs=scan_dirs,
            reservations_path=args.reservations)
    elif planned is not None:
        print(f"  [budget] smoke run: its {planned} planned fits are recorded separately and "
              "not counted against the cap")
    require_ledger_beside_outputs(ledger_path, scan_dirs, args.stage)

    # --- the reservation: minted before the first instrument byte -------------
    # Nothing above this line has opened a data file; the science-file digests,
    # noise_bin_power, load_stretch and every estimator call are below it.  A
    # stage that fits nothing (characterize, freeze, recombine) has no planned
    # count and reserves nothing.
    res_path = reservations_path_for(ledger_path, args.reservations)
    reservation = None
    if planned is not None:
        reservation = reserve_stage_fits(
            P, ledger_path, args.stage, planned, scan_dirs=scan_dirs,
            reservations_path=args.reservations, smoke=args.smoke, files=files,
            meta={"out": str(out_dir), "argv": sys.argv[1:]})

    # pending -> running for the whole execution; an exception of any kind (a
    # refusal, a crash, a signal that unwinds) marks it failed, and failed is
    # never refunded.  A death that unwinds nothing leaves it running, and the
    # next launch sweeps it to interrupted -- still charged either way.
    scope = (fa.running(res_path, reservation["uuid"]) if reservation is not None
             else contextlib.nullcontext())
    with scope:
        data_root = Path(args.data_root)
        # Recombination uses archived summaries and draws only, including provenance.
        if args.stage != "recombine":
            for p in _science_paths(data_root)[:2]:
                rec.add_input("science_noise", p)

        if args.stage == "characterize":
            payload.update(stage_characterize(args, P, rec))

        elif args.stage == "development":
            rec.add_input("characterisation", Path(args.characterisation))
            payload.update(stage_development(args, P, rec, chars))

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
            # Everything read here was bound above: the stage output and draws match
            # their RESULTS_SHA256, the output was produced under this freeze, and the
            # protocol is the frozen one or differs only by registered amendments.
            src, fz = loaded["stage_output"], loaded["frozen"]
            rec.add_input("stage_output", Path(args.stage_output))
            rec.add_input("draws", Path(args.draws))
            rec.add_input("freeze", Path(args.freeze_file))
            if args.characterisation:
                rec.add_input("characterisation", Path(args.characterisation))
            for a in payload["freeze_binding"]["amendments_applied"]:
                rec.add_input("amendment", Path(a["file"]))
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

        elif args.stage in HOLDOUT_STAGES:
            # Bound in the pre-flight: the protocol, the freeze and the characterisation
            # are the frozen ones, and the cumulative budget has room for ``files``.
            fz, chars = loaded["frozen"], loaded["characterisation"]
            rec.add_input("freeze", Path(args.freeze_file))
            rec.add_input("characterisation", Path(args.characterisation))
            if not files:
                payload[args.stage] = {"per_file": {}, "files_evaluated": [], "fit_count": 0,
                                       "note": "empty file list; " + fz.get("validation_status", "")}
            else:
                stage_payload, draws = stage_evaluate(args, P, rec, fz, files, args.stage, chars)
                payload.update(stage_payload)

        out_dir.mkdir(parents=True, exist_ok=True)
        # the reservation uuid travels inside the output's own provenance: the
        # one identity that survives the file being moved or rewritten
        payload["provenance"] = rec.finalize(
            {fa.PROVENANCE_KEY: fa.reservation_provenance(reservation)}
            if reservation is not None else None)
        name = f"t22_{args.stage}.json"
        digest = write_json(out_dir / name, payload)
        lines = [f"{digest}  {name}"]
        if draws:
            npz = out_dir / f"t22_{args.stage}_draws.npz"
            np.savez_compressed(npz, **draws)
            lines.append(f"{hashlib.sha256(npz.read_bytes()).hexdigest()}  {npz.name}")
            print(f"[T22/{args.stage}] wrote {npz} ({len(draws)} arrays)")
        (out_dir / "RESULTS_SHA256").write_text("\n".join(lines) + "\n")
        print(f"[T22/{args.stage}] wrote {out_dir}/{name} sha256={digest}")

        # Every stage output enters the ledger (smoke runs flagged and totalled apart),
        # and the totals are recomputed from the entries.
        entry = ledger_entry(args.stage, out_dir, name, digest, payload, ledger_path.parent)
        ledger = record_in_ledger(ledger_path, P, entry)
        print(f"[T22/{args.stage}] ledger {ledger_path}: +{entry['fits_executed']} fit(s)"
              f"{' (smoke, counted separately)' if entry['smoke'] else ''}; executed total "
              f"{ledger['executed_total']} against the hard cap {P.cap}")

        if reservation is not None:
            # running -> completed, carrying what actually ran and the digest of
            # the output it paid for
            fa.mark_completed(res_path, reservation["uuid"], entry["fits_executed"],
                              out_dir / name, output_sha256=digest, inference_only=False)
            print(f"[T22/{args.stage}] reservation {reservation['uuid']} completed: "
                  f"{entry['fits_executed']} fit(s) executed of {planned} planned")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
