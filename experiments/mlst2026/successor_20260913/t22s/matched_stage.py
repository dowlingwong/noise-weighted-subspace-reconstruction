"""T22-S part B: the matched real-data interface (exercised on fixtures only).

Three functions, none of which opens instrument data on its own:

:func:`exposure_inventory`
    Reads the archived run trees (``results/t22_2026-09-11`` and
    ``results/mlst_2026-09-10``: stage JSONs and the freezes) and lists every
    ``(file, start sample, length, stage, role)`` region already exposed, the
    files the archive knows about, and -- per file -- the sample regions the
    archive never read.  It says explicitly that the release file list is not
    in the archive.

:func:`feasibility`
    Given the reduced calibration output and the number of independent files
    available, returns ``feasible`` or a feasibility failure with its reason.
    The rules are the protocol's; nothing is decided from data.

:func:`matched_scores`
    For ONE loaded stretch (the dict ``t22.load_stretch`` returns), scores the
    whole-window estimators (``te.compute_estimates``) and the band estimators
    (``tb.compute_band_estimates``) on IDENTICAL evaluation windows, with one
    transfer calibration, one label array, one allocation and one scorer, and
    returns paired contrasts including whole-window versus band for the same
    method.  The band ``B`` is an input: selecting it on development data is a
    separate step and is not part of matched scoring.  Every output is
    labelled with the exposure status the caller declares; a matched rerun on
    already exposed data must be labelled accordingly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026 import t22_tidmad_narrowband as t22  # noqa: E402
from experiments.mlst2026 import tidmad_band as tb  # noqa: E402
from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402

ARCHIVE_ROOTS = (_ROOT / "results" / "t22_2026-09-11", _ROOT / "results" / "mlst_2026-09-10")
EXPOSURE_LABELS = ("matched-rerun-on-already-exposed-data", "unexposed-hold-out", "synthetic-fixture")
N_PSD_WINDOWS = 256   # archived noise-PSD read: the first 256 windows of each science file

REPRESENTATIONS = ("whole-window", "band")
ARCHIVED_CONTRASTS = (("zero", "empca"), ("calibration-mean", "empca"), ("raw-bin", "empca"), ("pca", "empca"),
                      ("zero", "pca"), ("calibration-mean", "pca"), ("wiener", "empca"), ("shrinkage-cal", "empca"),
                      ("zero", "raw-bin"), ("zero", "wiener"), ("quadrature-OF", "raw-bin"))
LEARNED = ("pca", "empca")
SHARED_METHODS = tuple(m for m in te.METHODS if m not in LEARNED)


class FeasibilityFailure(dict):
    """A dict result whose truth value is False (feasible results are plain dicts)."""

    def __bool__(self):
        return False


# --- (i) exposure inventory -----------------------------------------------------------
def _load(path: Path):
    return json.loads(Path(path).read_text())


def _region(file, start, length, stage, role, tree, *, source, note=None):
    r = {"file": str(file), "start_sample": int(start), "n_samples": int(length),
         "end_sample_exclusive": int(start) + int(length), "stage": stage, "role": role,
         "run_tree": tree, "source": str(source)}
    if note:
        r["note"] = note
    return r


def _t22_regions(root: Path) -> tuple[list, dict, list]:
    regions, totals, notes = [], {}, []
    ch_path = root / "T22_characterize_v1" / "t22_characterize.json"
    if not ch_path.exists():
        notes.append(f"{ch_path} missing: T22 exposure not derivable")
        return regions, totals, notes
    ch = _load(ch_path)
    alloc = ch["allocation"]
    w, n_cal, stretch = int(alloc["window_samples"] if "window_samples" in alloc else 65536), int(alloc["n_calibration_windows"]), int(alloc["stretch_samples"])
    chars = ch["characterisation"]
    for name, c in chars.items():
        if "start_sample" not in c:
            continue
        totals[name] = int(c.get("n_total_samples", 0))
        regions.append(_region(name, c["start_sample"], n_cal * w, "characterize", c.get("role", "?"), root.name,
                               source=ch_path.relative_to(_ROOT), note="calibration windows only"))
    for inp in ch.get("provenance", {}).get("inputs", []):
        if inp.get("role") == "science_noise":
            regions.append(_region(Path(inp["path"]).name, 0, N_PSD_WINDOWS * w, "noise-psd", "science", root.name,
                                   source=ch_path.relative_to(_ROOT), note="first 256 windows, readout channel"))
    for stage_dir, stage in (("T22_development_v1", "development"), ("T22_development_v2", "development")):
        p = root / stage_dir / "t22_development.json"
        if not p.exists():
            continue
        dev = _load(p)
        for name, e in dev["development"].items():
            if e.get("skipped") or "start_sample" not in e:
                continue
            regions.append(_region(name, e["start_sample"], stretch, stage, "development", root.name,
                                   source=p.relative_to(_ROOT), note=f"full stretch; bands scored {e.get('bands_scored')}"))
    for stage_dir, fname, stage in (("T22_validation_v1", "t22_validation.json", "validation"),
                                    ("T22_final_test_v1", "t22_final-test.json", "final-test")):
        p = root / stage_dir / fname
        if not p.exists():
            continue
        js = _load(p)
        for name, e in js[stage]["per_file"].items():
            if e.get("skipped"):
                continue
            regions.append(_region(name, e["start_sample"], stretch, stage, "holdout", root.name,
                                   source=p.relative_to(_ROOT), note=f"full stretch at B={e.get('band_width')}"))
    fz = root / "T22_freeze_v1" / "t22_freeze.json"
    if fz.exists():
        notes.append(f"T22 freeze read: {fz.relative_to(_ROOT)} (roles: validation {len(_load(fz)['frozen'].get('validation_files', []))}, "
                     f"final-test {len(_load(fz)['frozen'].get('final_test_files', []))})")
    return regions, totals, notes


def _t10_regions(root: Path) -> tuple[list, dict, list]:
    regions, totals, notes = [], {}, []
    ch_path = root / "T10_tidmad_characterize_v1" / "t10_characterize.json"
    if not ch_path.exists():
        notes.append(f"{ch_path} missing: T10-T12 exposure not derivable")
        return regions, totals, notes
    ch = _load(ch_path)
    by_w = ch["allocation_by_window"]
    start_by_file = {}
    for key, c in ch["characterisation"].items():
        name, wtag = key.split("|w=")
        a = by_w[wtag]
        totals[name] = int(c.get("n_total_samples", totals.get(name, 0)) or 0)
        start_by_file[name] = int(c["start_sample"])
        regions.append(_region(name, c["start_sample"], int(a["n_cal"]) * int(wtag), "characterize", c.get("role", "?"), root.name,
                               source=ch_path.relative_to(_ROOT), note=f"calibration windows only, w={wtag}"))
    for inp in ch.get("provenance", {}).get("inputs", []):
        if inp.get("role") == "science_noise":
            regions.append(_region(Path(inp["path"]).name, 0, N_PSD_WINDOWS * 65536, "noise-psd", "science", root.name,
                                   source=ch_path.relative_to(_ROOT), note="first 256 windows at the largest window"))
    p = root / "T10_tidmad_development_v1" / "t10_development.json"
    if p.exists():
        dev = _load(p)
        for key in dev["development"]:
            name, wtag = key.split("|w=")
            a = by_w[wtag]
            regions.append(_region(name, start_by_file.get(name, 0), int(a.get("used_samples", a["stretch_samples"])), "development", "development", root.name,
                                   source=p.relative_to(_ROOT), note=f"w={wtag}; samples the allocation used at this window (stretch reserved: {a['stretch_samples']})"))
    for stage_dir, fname, stage in (("T10_tidmad_validation_v1", "t10_validation.json", "validation"),
                                    ("T10_tidmad_final_test_v1", "t10_final-test.json", "final-test")):
        p = root / stage_dir / fname
        if not p.exists():
            continue
        js = _load(p)
        for key, e in js[stage]["per_file"].items():
            if e.get("skipped"):
                continue
            name, wtag = key.split("|w=")
            a = by_w[wtag]
            regions.append(_region(name, start_by_file.get(name, 0), int(a.get("used_samples", a["stretch_samples"])), stage, "pool", root.name,
                                   source=p.relative_to(_ROOT), note=f"w={wtag}; samples the allocation used at this window (stretch reserved: {a['stretch_samples']})"))
    return regions, totals, notes


def _merge_intervals(iv):
    iv = sorted((int(a), int(b)) for a, b in iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def exposure_inventory(archive_roots=ARCHIVE_ROOTS) -> dict:
    regions, totals, notes = [], {}, []
    for root in archive_roots:
        root = Path(root)
        if not root.exists():
            notes.append(f"archive root {root} not present")
            continue
        r, t, n = (_t22_regions(root) if root.name.startswith("t22") else _t10_regions(root))
        regions.extend(r)
        for k, v in t.items():
            totals[k] = max(totals.get(k, 0), v)
        notes.extend(n)
    files = sorted({r["file"] for r in regions})
    per_file = {}
    for f in files:
        rs = [r for r in regions if r["file"] == f]
        merged = _merge_intervals([(r["start_sample"], r["end_sample_exclusive"]) for r in rs])
        total = totals.get(f, 0)
        exposed = sum(b - a for a, b in merged)
        unread = []
        cursor = 0
        for a, b in merged:
            if a > cursor:
                unread.append((cursor, a))
            cursor = max(cursor, b)
        if total and cursor < total:
            unread.append((cursor, total))
        per_file[f] = {"n_total_samples": int(total) if total else None, "exposed_regions_merged": merged,
                       "exposed_samples": int(exposed), "stages": sorted({r["stage"] for r in rs}),
                       "roles": sorted({r["role"] for r in rs}),
                       "never_read_regions": unread if total else "unknown (file length not in archive)"}
    return {
        "schema": "mlst-t22s-exposure-inventory/1",
        "archive_roots": [str(Path(r).relative_to(_ROOT)) if str(r).startswith(str(_ROOT)) else str(r) for r in archive_roots],
        "regions": sorted(regions, key=lambda r: (r["file"], r["start_sample"], r["run_tree"], r["stage"])),
        "files_known_to_archive": files,
        "per_file": per_file,
        "release_file_list": {
            "derivable_from_archive": False,
            "statement": "The archive names the files it opened (the validation files it characterised and two "
                         "science files for the noise PSD); it does not contain the release's file list, so the set "
                         "of release files NOT yet exposed cannot be derived here and is an unresolved input to any "
                         "new-data decision.",
            "files_not_yet_exposed": None,
        },
        "independent_units_note": "A never-read region of an already examined file is not an independent unit: the "
                                  "file's coupling, tone and noise state were inspected when its other regions were "
                                  "scored.  Only files the archive never opened can count as independent units, and "
                                  "which those are is the unresolved input above.",
        "notes": notes,
    }


def inventory_markdown(inv: dict) -> str:
    lines = ["# T22-S exposure inventory (generated from the archived run trees)", "",
             "Generated by `matched_stage.exposure_inventory` from " + ", ".join(f"`{r}`" for r in inv["archive_roots"]) + ".",
             "Nothing here was recomputed from instrument data; every row is read from a stage JSON.", "",
             "## Regions already exposed", "",
             "| file | start sample | n samples | end (excl.) | stage | role | run tree | note |", "|---|---:|---:|---:|---|---|---|---|"]
    for r in inv["regions"]:
        lines.append(f"| {r['file']} | {r['start_sample']} | {r['n_samples']} | {r['end_sample_exclusive']} | {r['stage']} | {r['role']} | {r['run_tree']} | {r.get('note', '')} |")
    lines += ["", "## Per file", "", "| file | total samples | exposed samples | merged exposed regions | stages | never-read regions |", "|---|---:|---:|---|---|---|"]
    for f, e in inv["per_file"].items():
        nr = e["never_read_regions"]
        nr_s = nr if isinstance(nr, str) else "; ".join(f"[{a}, {b})" for a, b in nr)
        lines.append(f"| {f} | {e['n_total_samples']} | {e['exposed_samples']} | " + "; ".join(f"[{a}, {b})" for a, b in e["exposed_regions_merged"]) + f" | {', '.join(e['stages'])} | {nr_s} |")
    lines += ["", "## Release files not yet exposed", "", inv["release_file_list"]["statement"], "",
              "## Independent units", "", inv["independent_units_note"], ""]
    if inv["notes"]:
        lines += ["## Notes", ""] + [f"- {n}" for n in inv["notes"]] + [""]
    return "\n".join(lines)


# --- (ii) feasibility ---------------------------------------------------------------
def feasibility(calibration: dict | None, n_available_files: int, *, rules: dict,
                dependence_verdict: str | None = None) -> dict:
    """Feasibility of a matched instrument comparison under the protocol's rules.

    ``calibration`` is the reduced confirmation block (``REDUCED.json ->
    confirmation``) or ``None`` when no validated calibration exists.
    ``rules`` is ``protocol["inference"]["feasibility"]``.
    """
    reasons = []
    n_min = int(rules["min_independent_files"])
    if n_available_files < n_min:
        reasons.append(f"too few independent units: {n_available_files} < the declared minimum {n_min}")
    if dependence_verdict == "inconclusive":
        reasons.append("dependence assessment unstable on the development data (inconclusive); the declared rule "
                       "withholds every verdict, so a matched run could not decide anything")
    if calibration is None or not calibration.get("complete"):
        reasons.append("no complete confirmation calibration: adequacy of the inference at this file count is not established")
        table = {}
    else:
        table = calibration.get("feasibility_table", {})
        # the largest calibrated n_files not exceeding what is available decides
        usable = [int(n) for n in table if int(n) <= n_available_files]
        if not usable:
            reasons.append(f"no calibrated scenario at n_files <= {n_available_files}")
        else:
            n_use = max(usable)
            row = table[str(n_use)]
            if not row.get("type1_within_tolerance"):
                reasons.append(f"type-I error not within tolerance at n_files={n_use}")
            if not row.get("coverage_within_tolerance"):
                reasons.append(f"coverage not within tolerance at n_files={n_use}")
            if row.get("withholding_rate_max") is not None and row["withholding_rate_max"] > rules["max_withholding_rate"]:
                reasons.append(f"withholding rate {row['withholding_rate_max']:.2f} exceeds the declared maximum "
                               f"{rules['max_withholding_rate']} at n_files={n_use}: a procedure that mostly withholds has no demonstrated useful power")
            if row.get("power_medium_effect_min") is not None and row["power_medium_effect_min"] < rules["min_power_medium_effect"]:
                reasons.append(f"power {row['power_medium_effect_min']:.2f} at the medium effect is below the declared minimum "
                               f"{rules['min_power_medium_effect']} at n_files={n_use}")
    out = {"n_available_files": int(n_available_files), "min_independent_files": n_min,
           "dependence_verdict": dependence_verdict, "rules": rules,
           "label_required": "a matched rerun on already exposed data must be labelled accordingly"}
    if reasons:
        return FeasibilityFailure({**out, "feasible": False, "status": "feasibility-failure", "reasons": reasons,
                                   "consequence": "the correct output is a feasibility failure, not an artificial "
                                                  "four-file significance claim"})
    return {**out, "feasible": True, "status": "feasible", "reasons": []}


# --- (iii) matched scoring ---------------------------------------------------------------
def matched_scores(stretch: dict, *, k_inj: int, B: int, w: int, N_bins: np.ndarray, exposure_status: str,
                   contrasts=ARCHIVED_CONTRASTS) -> dict:
    """Whole-window and band estimates on identical windows, labels and scorer.

    Fits charged: ``len(te.METHODS)`` per representation (the archived counting
    convention: eight estimators per cell), i.e. 16 for the two representations.
    """
    if exposure_status not in EXPOSURE_LABELS:
        raise ValueError(f"exposure_status must be one of {EXPOSURE_LABELS}, got {exposure_status!r}")
    for key in ("cal_read", "cal_ref", "fit_read", "eval_read", "eval_ref"):
        if key not in stretch:
            raise ValueError(f"stretch lacks {key!r}; pass the dict load_stretch returns")
    k = int(k_inj)
    N_bins = np.asarray(N_bins, dtype=float)
    # ONE transfer calibration, ONE label array, ONE evaluation-window set.
    cal = te.fit_transfer(stretch["cal_read"][:, k], stretch["cal_ref"][:, k], float(N_bins[k]))
    t_eval = cal.g * stretch["eval_ref"][:, k]
    eval_index = np.arange(stretch["eval_read"].shape[0])
    est_ww, diag_ww = te.compute_estimates(stretch["eval_read"], stretch["fit_read"], k, w, N_bins, cal)
    est_b, diag_b = tb.compute_band_estimates(stretch["eval_read"], stretch["fit_read"], k, B, w, N_bins, cal)
    reps = {"whole-window": est_ww, "band": est_b}
    e2 = {rep: {m: te.squared_errors(q, t_eval, w) for m, q in est.items()} for rep, est in reps.items()}
    within = {}
    for rep in REPRESENTATIONS:
        within[rep] = {}
        for (a, b) in contrasts:
            d = e2[rep][a] - e2[rep][b]
            within[rep][f"{a}|{b}"] = {"mean_delta_squared": float(np.mean(d)), "sd_per_window": float(np.std(d, ddof=1)),
                                       "n_windows": int(d.size)}
    across = {}
    for m in te.METHODS:
        d = e2["whole-window"][m] - e2["band"][m]
        across[m] = {"mean_delta_squared": float(np.mean(d)), "sd_per_window": float(np.std(d, ddof=1)),
                     "max_abs_per_window": float(np.max(np.abs(d))), "n_windows": int(d.size),
                     "expected_identical": m in SHARED_METHODS}
    shared_identical = all(across[m]["max_abs_per_window"] == 0.0 for m in SHARED_METHODS)
    return {
        "exposure_status": exposure_status,
        "label": "a matched rerun on already exposed data must be labelled accordingly",
        "representation_is_the_only_difference": True,
        "band_width": int(B), "band_selected_here": False,
        "k_inj": k, "window_samples": int(w),
        "shared": {"transfer_calibration": cal.as_dict(), "n_evaluation_windows": int(eval_index.size),
                   "evaluation_window_index": eval_index.tolist(),
                   "label_sha256": _sha_array(t_eval), "n_fit_windows": int(stretch["fit_read"].shape[0]),
                   "n_calibration_windows": int(stretch["cal_read"].shape[0]),
                   "allocation": stretch.get("allocation")},
        "per_representation": {rep: {"evaluation_window_index": eval_index.tolist(), "label_sha256": _sha_array(t_eval),
                                     "rmse_by_method_amplitude_units": {m: float(np.sqrt(np.mean(v))) for m, v in e2[rep].items()},
                                     "contrasts": within[rep]}
                               for rep in REPRESENTATIONS},
        "whole_window_vs_band_same_method": across,
        "shared_methods_identical_across_representations": bool(shared_identical),
        "diagnostics": {"whole-window": diag_ww, "band": diag_b},
        "n_fits": 2 * len(te.METHODS),
        "counting_convention": "eight declared estimators per representation, as in the archived T10-T12/T22 accounting",
    }


def _sha_array(a: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --- fixture support (tests and the development cell) ------------------------------
def write_tidmad_fixture(path: Path, *, seed: int, w: int, n_cal: int, n_fit: int, n_eval: int, guard: int,
                         k_inj: int, extra_samples: int = 4096) -> dict:
    """A synthetic HDF5 file with the TIDMAD dataset layout (float32, two channels)."""
    import h5py
    rng = np.random.default_rng(seed)
    n_total = (n_cal + n_fit + n_eval) * w + 2 * guard + extra_samples
    t = np.arange(n_total, dtype=np.float64)
    phase_drift = 2 * np.pi * (k_inj + 0.2) * t / w          # a slightly off-bin tone, as in the release
    ref = 5.0 * np.cos(phase_drift) + 0.01 * rng.normal(size=n_total)
    # readout: gain 0.3 with a phase, plus coloured noise (an AR(1) in time)
    g_abs, g_ph = 0.3, 0.7
    read = g_abs * 5.0 * np.cos(phase_drift + g_ph)
    noise = np.empty(n_total)
    e = rng.normal(size=n_total)
    noise[0] = e[0]
    for i in range(1, n_total):
        noise[i] = 0.9 * noise[i - 1] + e[i]
    read = read + 0.5 * noise / np.std(noise)
    with h5py.File(path, "w") as h:
        h.create_dataset(t22.SQUID, data=read.astype(np.float32))
        h.create_dataset(t22.REFERENCE, data=ref.astype(np.float32))
    return {"n_total_samples": int(n_total), "k_inj": int(k_inj), "g_abs": g_abs, "g_phase": g_ph}


def fixture_allocation(*, w: int, n_cal: int, n_fit: int, n_eval: int, guard: int):
    """A Protocol-like allocation object for ``t22.load_stretch``."""
    from types import SimpleNamespace
    fit_off = n_cal * w + guard
    eval_off = fit_off + n_fit * w + guard
    return SimpleNamespace(w=w, n_cal=n_cal, n_fit=n_fit, n_eval=n_eval, guard=guard,
                           fit_offset=fit_off, eval_offset=eval_off, stretch=eval_off + n_eval * w)


def load_fixture_stretch(path: Path, start: int, alloc) -> dict:
    """Read the fixture through the archived opener under an exposure allowance
    that names exactly this file and region (the guard is configurable: it
    matches on the file name and the declared sample region)."""
    entry = {"dataset_id": Path(path).name, "region": {"start_sample": int(start), "n_samples": int(alloc.stretch)}}
    with t22.exposure_allowance([entry]):
        st = t22.load_stretch(Path(path), int(start), alloc)
    t22.require_complete_stretch(st, alloc, Path(path).name)
    return st
