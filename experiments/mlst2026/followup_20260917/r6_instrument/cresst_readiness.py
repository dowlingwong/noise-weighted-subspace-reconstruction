"""R6 / CRESST readiness: the archived 350-cell robustness grid and fail-closed gates.

The rank x shrinkage x time-cut robustness grid of the archived CRESST-S package is
enumerated here **from the immutable archived design** (its pairing, schema, seeds,
fit accounting and contrast plan are imported, not reproduced by hand), then marked
unexecutable: every one of the 350 cells needs raw CRESST pulse traces, and no raw
CRESST data is present in this environment.  An adapter from the CRESST release
layout to the declared ``cresst-s-pulse-root/1`` interface is deliberately **not**
written, because there is no raw data here to test it against; the interface is
therefore *implemented and fail-closed*, **not validated on data**.

On-disk interface expected at ``<CRESST_DATA_ROOT>`` (a superset of the archived
``robustness_design`` interface, adding the provenance and split fields this package
gates on)::

    <root>/index.json
        {"format": "cresst-s-pulse-root/1", "synthetic": <bool>, "n_samples": int,
         "sample_rate_hz": float,
         "split": <str>,                         # must equal the declared evaluation split
         "provenance": {"archive_id": <str>, "release": <str>},
         "channels": [{"run": "33", "channel": "36.0", "file": "run33/ch36.0.npz",
                       "n_events": int, "n_noise": int}, ...]}
    <root>/<file>            NPZ with
        pulses  (n_events, n_samples) float64   evaluation-split pulse traces, time ordered
        times   (n_events,)           float64   chronological time, ascending
        noise   (n_noise, n_samples)  float64   noise-only traces for the PSD estimate

A missing root, a missing or malformed index, a missing array, a non-chronological
``times`` array, a declared/actual count mismatch, a channel set that differs from
the protocol's declared channels, an absent provenance block, or a provenance /
split value that differs from the protocol's declaration all raise
:class:`~experiments.mlst2026.successor_20260913.cresst_s.archive_recompute.InputNotAvailable`
with the reason.  A synthetic fixture written by :func:`write_fixture_root` is
labelled non-real and is used for wiring tests only; it is never a CRESST result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.mlst2026.successor_20260913.common.protocol import ProtocolError, load_protocol  # noqa: E402
from experiments.mlst2026.successor_20260913.cresst_s import robustness_design as RD  # noqa: E402
from experiments.mlst2026.successor_20260913.cresst_s.archive_recompute import InputNotAvailable  # noqa: E402

ARCHIVED_PROTOCOL = (_ROOT / "experiments" / "mlst2026" / "successor_20260913" / "cresst_s"
                     / "PROTOCOL_CRESST_S_v1.json")

REQUIRED_ARRAYS = ("pulses", "times", "noise")
ARMS = ("pilot", "confirmation")
EXECUTABLE_HERE = False
REQUIRES = ("<CRESST_DATA_ROOT>: raw CRESST pulse traces in the cresst-s-pulse-root/1 interface. "
            "NO CRESST RAW DATA IS PRESENT IN THIS ENVIRONMENT and none may be sought here; the "
            "release-layout adapter is deliberately unwritten because there is no raw data to test "
            "it against.  The grid is enumerated, seeded, paired and costed only.")
NON_REAL_LABEL = "synthetic-fixture, not real data, not validated on real inputs"


# --- preservation of the archived grid ------------------------------------------------------
def archived_rows() -> dict[str, list[dict]]:
    """The archived grid rows, exactly as the immutable ``robustness_design`` builds them."""
    P = load_protocol(ARCHIVED_PROTOCOL)
    return {arm: RD.grid_rows(P, arm) for arm in ARMS}


def check_grid_matches_archived(P: dict) -> dict:
    """The R6 protocol must pin the archived grid; a drift is a protocol error.

    Compares every grid constant the enumeration depends on (channels and order,
    primary setting, both arms' ranks/shrinkages/time cuts, EM restarts) and then
    compares the R6-enumerated rows with the archived rows field by field.
    """
    A = load_protocol(ARCHIVED_PROTOCOL)
    g = P["cresst"]["grid"]
    checks = {
        "channels": (list(g["channels"]) == list(A["robustness"]["channels"]),
                     list(g["channels"]), list(A["robustness"]["channels"])),
        "primary_setting": (dict(g["primary_setting"]) == dict(A["robustness"]["primary_setting"]),
                            dict(g["primary_setting"]), dict(A["robustness"]["primary_setting"])),
        "arms": (dict(g["arms"]) == dict(A["robustness"]["arms"]),
                 dict(g["arms"]), dict(A["robustness"]["arms"])),
        "empca_restarts": (int(g["empca_restarts"]) == int(A["robustness"]["empca_restarts"]),
                           int(g["empca_restarts"]), int(A["robustness"]["empca_restarts"])),
        "empca_iterations": (int(g["empca_iterations"]) == int(A["robustness"]["empca_iterations"]),
                             int(g["empca_iterations"]), int(A["robustness"]["empca_iterations"])),
        "production_dimensions": (dict(g["production_dimensions"]) == dict(A["robustness"]["production_dimensions"]),
                                  dict(g["production_dimensions"]), dict(A["robustness"]["production_dimensions"])),
    }
    bad = [k for k, (ok, *_rest) in checks.items() if not ok]
    if bad:
        first = bad[0]
        raise ProtocolError(f"R6 CRESST grid differs from the archived design at {first}: "
                            f"R6={checks[first][1]!r} archived={checks[first][2]!r}")
    if int(g["fits_per_cell"]) != int(RD.FITS_PER_CELL):
        raise ProtocolError(f"R6 fits_per_cell {g['fits_per_cell']} != archived {RD.FITS_PER_CELL}")
    archived = archived_rows()
    mine = enumerate_grid(P, cross_check=False)
    a_flat = [r for arm in ARMS for r in archived[arm]]
    if len(mine) != len(a_flat):
        raise ProtocolError(f"R6 enumerates {len(mine)} cells, the archive {len(a_flat)}")
    for m, a in zip(mine, a_flat):
        for key in ("arm", "run", "channel", "channel_label", "rank", "shrinkage", "time_cut",
                    "covariance_model", "contrast_group", "contrast_group_index", "channel_index",
                    "time_cut_index", "is_group_reference", "is_primary_setting", "paired_against",
                    "time_cut_paired_against"):
            if m.get(key) != a.get(key):
                raise ProtocolError(f"R6 row differs from the archive at {key}: {m.get(key)!r} != {a.get(key)!r}")
    return {"matches_archived": True, "checks": {k: v[0] for k, v in checks.items()},
            "n_archived_rows": len(a_flat)}


def enumerate_grid(P: dict, *, cross_check: bool = True) -> list[dict]:
    """Enumerate the 350 archived cells with the archived pairing and schema.

    ``cross_check=True`` (the default) first verifies the R6 protocol pins the
    archived grid.  ``cross_check=False`` is used only *inside* that verification to
    avoid recursion.  Every row is marked unexecutable here, with the required input
    named; no row is ever run or fabricated.
    """
    if cross_check:
        check_grid_matches_archived(P)
    rows: list[dict] = []
    for arm in ARMS:
        for r in archived_rows()[arm]:
            row = dict(r)
            row["instrument"] = "cresst"
            row["executable_here"] = EXECUTABLE_HERE
            row["requires"] = REQUIRES
            rows.append(row)
    return rows


def cost_grid(P: dict, rows: list[dict] | None = None) -> dict:
    """Cost the enumerated grid: cells, fits, and the archived paired-contrast plan."""
    rows = enumerate_grid(P) if rows is None else rows
    archived = archived_rows()
    fits_per_cell = int(P["cresst"]["grid"]["fits_per_cell"])
    per_arm = {}
    n_pairs = n_cut_pairs = n_groups = 0
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        plan = RD.contrast_plan(archived[arm])
        n_pairs += int(plan["n_paired_contrasts"])
        n_cut_pairs += int(plan["n_time_cut_contrasts"])
        n_groups += int(plan["n_contrast_groups"])
        per_arm[arm] = {"cells": len(arm_rows), "fits": len(arm_rows) * fits_per_cell,
                        "contrast_groups": int(plan["n_contrast_groups"]),
                        "paired_contrasts": int(plan["n_paired_contrasts"]),
                        "time_cut_contrasts": int(plan["n_time_cut_contrasts"]),
                        "groups_without_a_reference_setting": plan["groups_without_a_reference_setting"]}
    return {
        "n_cells": len(rows), "n_fits": len(rows) * fits_per_cell, "fits_per_cell": fits_per_cell,
        "n_contrast_groups": n_groups, "n_paired_contrasts": n_pairs,
        "n_time_cut_contrasts": n_cut_pairs, "n_paired_contrasts_total": n_pairs + n_cut_pairs,
        "per_arm": per_arm,
        "estimator_fits_note": ("one fit per classic-PCA invocation plus one per EMPCA random restart: "
                                f"1 + {RD.EMPCA_RESTARTS} = {fits_per_cell} per cell, "
                                f"{len(rows) * fits_per_cell} for the grid"),
        "pairing": "preserved from the archived design (within (channel, time cut) against the group reference)",
        "executable_here": EXECUTABLE_HERE,
        "requires": REQUIRES,
    }


def primary_setting(P: dict) -> dict:
    """The archived primary estimator setting, read from the R6 protocol."""
    return dict(P["cresst"]["grid"]["primary_setting"])


# --- provenance / split gates ---------------------------------------------------------------
def _channel_root(data_root):
    if data_root is None:
        raise InputNotAvailable(
            "input not available: raw CRESST pulse data root (<CRESST_DATA_ROOT>).  The 350-cell "
            "rank/shrinkage/time-cut robustness grid needs the raw CRESST pulse traces; no CRESST raw "
            "data is present in this environment and none may be sought here.  Pass "
            "--cresst-data-root <CRESST_DATA_ROOT> on a host that holds the release; the release-layout "
            "adapter is deliberately unwritten and this interface is NOT validated on real inputs.")
    root = Path(data_root)
    if not root.is_dir():
        raise InputNotAvailable(f"input not available: CRESST pulse data root {root} is not a directory")
    return root


def _check_provenance_and_split(P: dict, js: dict) -> dict:
    contract = P["cresst"]["input_contract"]
    if bool(contract.get("require_provenance", True)):
        prov = js.get("provenance")
        if not isinstance(prov, dict) or not prov.get("archive_id"):
            raise InputNotAvailable(
                "provenance refused: the CRESST index carries no provenance block (an 'archive_id' is "
                "required).  A raw-pulse source without a declared archive identity is not accepted; the "
                "release-layout adapter is deliberately unwritten because there is no raw data here to "
                "validate it against.")
        expected = contract.get("expected_archive_id")
        if expected is not None and str(prov["archive_id"]) != str(expected):
            raise InputNotAvailable(
                f"provenance refused: index archive_id {prov['archive_id']!r} != the protocol's declared "
                f"{expected!r}.  This package refuses a source whose provenance does not match its "
                "declaration rather than guessing.")
    expected_split = contract.get("expected_split")
    split = js.get("split")
    if expected_split is not None:
        if split is None:
            raise InputNotAvailable(
                f"split refused: the CRESST index declares no 'split'; the protocol requires the declared "
                f"evaluation split {expected_split!r}.")
        if str(split) != str(expected_split):
            raise InputNotAvailable(
                f"split refused: index split {split!r} != the protocol's declared {expected_split!r}.  A "
                "source whose split identity does not match its declaration is not accepted.")
    return {"archive_id": (js.get("provenance") or {}).get("archive_id"), "split": split}


def check_inputs(P: dict, data_root) -> dict:
    """Validate a raw CRESST root against the declared contract; fail closed.

    Returns a per-channel inventory for a *synthetic* fixture (labelled non-real) or
    for a source that satisfies the declared provenance and split.  Raises
    :class:`InputNotAvailable` on any missing input, malformed array, non-chronological
    time array, count mismatch, channel-set drift, or provenance/split mismatch.
    """
    root = _channel_root(data_root)
    js = RD.read_index(root)
    prov = _check_provenance_and_split(P, js)
    declared = list(P["cresst"]["grid"]["channels"])
    got = [f"{e['run']}_{e['channel']}" for e in js.get("channels", [])]
    if got != declared:
        raise InputNotAvailable(
            f"split refused: the index declares channels {got}, the protocol declares {declared}.  A "
            "channel set that differs from the archived seven is a split/selection mismatch, not a "
            "robustness grid this package can enumerate against.")
    n_samples = int(js["n_samples"])
    per_channel = {}
    for entry in js.get("channels", []):
        label = f"{entry['run']}_{entry['channel']}"
        p = root / entry["file"]
        if not p.is_file():
            raise InputNotAvailable(f"input not available: {p} (pulses for run {entry['run']} "
                                    f"channel {entry['channel']})")
        with np.load(p, allow_pickle=False) as z:
            missing = [a for a in REQUIRED_ARRAYS if a not in z.files]
            if missing:
                raise InputNotAvailable(f"input not available: {p} lacks the required arrays {missing} "
                                        f"(has {sorted(z.files)})")
            pulses = np.asarray(z["pulses"], dtype=float)
            times = np.asarray(z["times"], dtype=float)
            noise = np.asarray(z["noise"], dtype=float)
        if pulses.ndim != 2 or pulses.shape[1] != n_samples:
            raise InputNotAvailable(f"split refused: {p} pulses shape {pulses.shape} is not "
                                    f"(n_events, {n_samples})")
        if times.shape != (pulses.shape[0],):
            raise InputNotAvailable(f"split refused: {p} times shape {times.shape} is not "
                                    f"({pulses.shape[0]},)")
        if int(times.size) > 1 and np.any(np.diff(times) < 0):
            raise InputNotAvailable(f"split refused: {p} times are not chronological (ascending); the "
                                    "archived time-cut split is undefined on unordered times")
        if noise.ndim != 2 or noise.shape[1] != n_samples:
            raise InputNotAvailable(f"split refused: {p} noise shape {noise.shape} is not "
                                    f"(n_noise, {n_samples})")
        if pulses.shape[0] != int(entry["n_events"]):
            raise InputNotAvailable(f"split refused: {p} carries {pulses.shape[0]} pulses but the index "
                                    f"declares n_events={entry['n_events']}")
        if noise.shape[0] != int(entry["n_noise"]):
            raise InputNotAvailable(f"split refused: {p} carries {noise.shape[0]} noise traces but the "
                                    f"index declares n_noise={entry['n_noise']}")
        if not (np.all(np.isfinite(pulses)) and np.all(np.isfinite(times)) and np.all(np.isfinite(noise))):
            raise InputNotAvailable(f"split refused: {p} carries non-finite values")
        per_channel[label] = {"file": entry["file"], "n_events": int(pulses.shape[0]),
                              "n_noise": int(noise.shape[0]),
                              "times_monotone": bool(times.size <= 1 or np.all(np.diff(times) >= 0))}
    synthetic = bool(js.get("synthetic", False))
    return {
        "instrument": "cresst",
        "data_root": str(root),
        "synthetic": synthetic,
        "real_data": (not synthetic),
        "archive_id": prov["archive_id"],
        "split": prov["split"],
        "n_samples": n_samples,
        "channels": per_channel,
        "n_channels": len(per_channel),
        "label": (NON_REAL_LABEL if synthetic else
                  "declared CRESST source satisfying provenance and split; STILL NOT VALIDATED as a result"),
        "note": ("interface implemented and fail-closed; a passing check is an input gate, not a scientific "
                 "result and not evidence that the adapter is correct on the real release"),
    }


def write_fixture_root(root, *, seed: int = 7, P: dict | None = None) -> Path:
    """Write a small synthetic CRESST-interface root for WIRING TESTS ONLY.

    DEVELOPMENT ONLY: the traces are a synthetic pulse/noise family; the index is
    labelled ``synthetic: true`` and carries the declared provenance/split so the
    gates can be exercised in both directions.  No number derived from it is a
    scientific result and it is never CRESST data.
    """
    P = P or load_protocol(R6_PROTOCOL)
    root = Path(root)
    RD.write_fixture_data_root(root, seed=int(seed), channels=RD.ARCHIVE_CHANNELS,
                               n_events=24, n_noise=32, n_samples=48)
    idx = root / RD.INDEX_NAME
    js = json.loads(idx.read_text())
    js["synthetic"] = True
    js["split"] = P["cresst"]["input_contract"]["expected_split"]
    js["provenance"] = {"archive_id": P["cresst"]["input_contract"]["expected_archive_id"],
                        "release": "DEVELOPMENT ONLY synthetic CRESST-interface fixture"}
    js["label"] = NON_REAL_LABEL
    idx.write_text(json.dumps(js, indent=1, sort_keys=True))
    return root


R6_PROTOCOL = _HERE.parent / "PROTOCOL_R6_INSTRUMENT_v1.json"
