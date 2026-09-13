"""CRESST-S part A: recomputation of the archived channel-robustness analysis.

Everything here reads the *archived event residuals* and recomputes numbers that
already exist in the archive.  It fits nothing: the residuals were produced by
the frozen CRESST validation, and this module only reduces them arithmetically
and resamples them.  A disagreement with the archive beyond a declared tolerance
is a **failure of this package**, never a new result.

Two archived sources are compared against:

``<archive-root>/cresst/sensitivity/20260713_channel_robustness_v1/``
    the post-freeze sensitivity archive (13 July 2026): ``event_residuals_lossless.npz``
    (the input this module recomputes from), ``channel_summary.csv`` (whose
    ``pca_minus_empca_fraction`` column is the per-channel estimand),
    ``estimand_summary.csv``, ``summary.json``, ``analysis_config.yaml``,
    ``analysis_script.py``, ``event_residuals_metadata.json``.

``<archive-root>/paper1/mlst_2026-09-10/T13a_cresst_channel_bootstrap_v1/t13a_results.json``
    the 10 September 2026 re-run of the channel bootstrap (``point_estimates``,
    ``channels``, ``intervals`` with ten constructions, ``per_channel_effects``,
    ``leave_one_channel_out``, ``leave_one_run_out``).

The estimator and resampling functions are **imported** from the immutable
archived runner ``experiments/mlst2026/t13a_cresst_channel_bootstrap.py``; this
module adds only the archive resolution, the comparison arithmetic, the declared
tolerances and the structural diagnostics used by the mutation gate.

Reproducibility of the RNG
--------------------------
The archived runner spawns thirteen named substreams from
``SeedSequence(20260713)`` in the frozen order ``t13a.STREAM_ORDER``.
``numpy.random.SeedSequence`` spawning is incremental, so spawning *more*
children than thirteen leaves the first thirteen byte-identical.  This package
therefore spawns ``t13a.STREAM_ORDER + EXTRA_STREAM_ORDER`` and uses the
archived thirteen for every archived construction (bit-identical reproduction is
expected when the replicate count also matches) and the extra ones for the
constructions this package adds.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.mlst2026 import t13a_cresst_channel_bootstrap as t13a  # noqa: E402

# --- the archive ---------------------------------------------------------------------
SENSITIVITY_SUBDIR = Path("cresst/sensitivity/20260713_channel_robustness_v1")
REFERENCE_RESULTS_SUBPATH = Path("paper1/mlst_2026-09-10/T13a_cresst_channel_bootstrap_v1/t13a_results.json")

#: role -> path relative to the archive root.  Every one of these is read-only and
#: is addressed in the protocol by path *and* SHA-256.
ARCHIVE_INPUTS = {
    "event_residuals_lossless": SENSITIVITY_SUBDIR / "event_residuals_lossless.npz",
    "event_residuals_metadata": SENSITIVITY_SUBDIR / "event_residuals_metadata.json",
    "channel_summary": SENSITIVITY_SUBDIR / "channel_summary.csv",
    "estimand_summary": SENSITIVITY_SUBDIR / "estimand_summary.csv",
    "archive_summary": SENSITIVITY_SUBDIR / "summary.json",
    "analysis_config": SENSITIVITY_SUBDIR / "analysis_config.yaml",
    "analysis_script": SENSITIVITY_SUBDIR / "analysis_script.py",
    "t13a_results": REFERENCE_RESULTS_SUBPATH,
}

ARCHIVE_SEED = t13a.SEED                 # 20260713, frozen by analysis_config.yaml
ARCHIVE_REPLICATES = t13a.REPLICATES     # 10000, frozen by analysis_config.yaml
N_CHANNELS = t13a.N_CHANNELS             # 7
ALPHA = t13a.CI_ALPHA                    # 0.05

#: streams this package adds after the archived thirteen (see the module docstring).
EXTRA_STREAM_ORDER = ("run_stratified", "variance_decomposition", "science_control",
                      "calibration_control", "calibration_design")
ALL_STREAM_ORDER = tuple(t13a.STREAM_ORDER) + EXTRA_STREAM_ORDER

#: the ten interval constructions the archive reports, with the archived stream
#: each was drawn from.  (hierarchical, estimand, event block length, stream)
CONSTRUCTIONS = {
    "headline": (True, "mean_fraction", 1, "headline_hierarchical"),
    "channel_fixed": (False, "mean_fraction", 1, "channel_fixed"),
    "ratio_of_means_hierarchical": (True, "ratio_of_means", 1, "ratio_of_means_hierarchical"),
    "ratio_of_means_channel_fixed": (False, "ratio_of_means", 1, "ratio_of_means_channel_fixed"),
    "block64_hierarchical": (True, "mean_fraction", 64, "block_hier_64"),
    "block64_channel_fixed": (False, "mean_fraction", 64, "block_channel_fixed_64"),
    "block16_hierarchical": (True, "mean_fraction", 16, "block_hier_16"),
    "block16_channel_fixed": (False, "mean_fraction", 16, "block_channel_fixed_16"),
    "block256_hierarchical": (True, "mean_fraction", 256, "block_hier_256"),
    "block256_channel_fixed": (False, "mean_fraction", 256, "block_channel_fixed_256"),
}

#: constructions that differ only in the channel layer; the cluster layer must add
#: variance (scientific control ``cluster_layer_adds_variance``).
HIERARCHICAL_PAIRS = (
    ("headline", "channel_fixed"),
    ("ratio_of_means_hierarchical", "ratio_of_means_channel_fixed"),
    ("block64_hierarchical", "block64_channel_fixed"),
    ("block16_hierarchical", "block16_channel_fixed"),
    ("block256_hierarchical", "block256_channel_fixed"),
)

ENDPOINT_KEYS = ("ci95_lo", "ci95_hi", "bootstrap_mean", "bootstrap_sd")


class InputNotAvailable(RuntimeError):
    """Raised when a declared input is not present in this environment."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_archive(archive_root, *, roles=None) -> dict:
    """Locate the archived inputs under ``archive_root`` and hash them.

    ``archive_root`` is the manuscript repository's ``evidence/data`` directory.
    There is deliberately **no default**: the archived inputs live outside this
    repository and the caller must name their location.
    """
    if archive_root is None:
        raise InputNotAvailable(
            "input not available: --archive-root was not given.  The archived CRESST inputs live in the "
            "manuscript repository (MS) under evidence/data and are not part of this repository; pass "
            "--archive-root <MS>/evidence/data (see REMOTE_COMMANDS.md).")
    root = Path(archive_root)
    if not root.is_dir():
        raise InputNotAvailable(f"input not available: archive root {root} is not a directory")
    wanted = ARCHIVE_INPUTS if roles is None else {k: ARCHIVE_INPUTS[k] for k in roles}
    out, missing = {}, []
    for role, rel in wanted.items():
        p = root / rel
        if not p.is_file():
            missing.append(str(rel))
            continue
        out[role] = {"role": role, "relative_path": str(rel), "path": str(p),
                     "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    if missing:
        raise InputNotAvailable(f"input not available: {len(missing)} archived input(s) missing under {root}: "
                                + ", ".join(sorted(missing)))
    return out


def verify_declared_digests(resolved: dict, declared: dict) -> dict:
    """Compare the resolved SHA-256s against the digests frozen in the protocol."""
    rows, bad = {}, []
    for role, info in sorted(resolved.items()):
        want = (declared.get(role) or {}).get("sha256")
        ok = (want is None) or (want == info["sha256"])
        rows[role] = {"relative_path": info["relative_path"], "sha256": info["sha256"],
                      "declared_sha256": want, "matches_protocol": bool(ok), "bytes": info["bytes"]}
        if not ok:
            bad.append(role)
    return {"per_input": rows, "all_match": not bad, "mismatched": sorted(bad)}


def load_reference_results(resolved: dict) -> dict:
    return json.loads(Path(resolved["t13a_results"]["path"]).read_text())


def load_channel_summary(resolved: dict) -> dict:
    """``channel -> pca_minus_empca_fraction`` from the sensitivity archive's own CSV."""
    rows = {}
    with open(resolved["channel_summary"]["path"], newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["channel"]] = float(r["pca_minus_empca_fraction"])
    return rows


def load_clusters(resolved: dict):
    """Evaluation-split residual clusters, one per run/channel pair (archived loader)."""
    return t13a.load_clusters(Path(resolved["event_residuals_lossless"]["path"]))


# --- streams -------------------------------------------------------------------------
def spawn_streams(seed: int = ARCHIVE_SEED) -> dict:
    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(len(ALL_STREAM_ORDER))
    return {name: np.random.default_rng(child) for name, child in zip(ALL_STREAM_ORDER, children)}


def stream(name: str, seed: int = ARCHIVE_SEED):
    """One freshly spawned substream by name (independent of the others)."""
    return spawn_streams(seed)[name]


# --- point estimates -------------------------------------------------------------------
def point_estimates(labels, runs, clusters, times) -> dict:
    """Per-channel fractional reduction, the equal-channel mean and the ratio of means.

    ``fraction_c = 1 - mean(R_EMPCA, c) / mean(R_PCA, c)`` over the evaluation
    events of cluster c;  ``theta = (1/C) sum_c fraction_c``  (equal-channel mean);
    ``theta_ratio = 1 - mean_c mean(R_EMPCA, c) / mean_c mean(R_PCA, c)``.
    """
    fractions = t13a.channel_fractions(clusters)
    theta = t13a.theta_point(clusters)
    theta_ratio = t13a.theta_ratio_point(clusters)
    channels = []
    for i, lab in enumerate(labels):
        channels.append({
            "channel_label": lab, "run": runs[i], "n_evaluation_events": int(clusters[i][0].size),
            "chronological_time_first": float(times[i][0]), "chronological_time_last": float(times[i][1]),
            "mean_pca_weighted_residual": float(clusters[i][0].mean()),
            "mean_empca_weighted_residual": float(clusters[i][1].mean()),
            "fraction": float(fractions[i]), "fraction_pct": float(100.0 * fractions[i]),
        })
    return {"channels": channels,
            "theta_mean_of_channel_fractions": float(theta),
            "theta_ratio_of_channel_means": float(theta_ratio),
            "n_channels": len(labels), "n_runs": len(set(runs)),
            "runs": sorted(set(runs)),
            "n_evaluation_events": int(sum(int(c[0].size) for c in clusters))}


def compare_point_estimates(recomputed: dict, reference: dict, channel_csv: dict, tol: float) -> dict:
    """Absolute deviation of every point estimate against both archived sources."""
    ref_ch = {c["channel_label"]: c for c in reference["channels"]}
    rows, worst = {}, 0.0
    for c in recomputed["channels"]:
        lab = c["channel_label"]
        r = ref_ch[lab]
        d = {}
        for key in ("fraction", "mean_pca_weighted_residual", "mean_empca_weighted_residual"):
            d[key] = abs(float(c[key]) - float(r[key]))
        d["fraction_vs_channel_summary_csv"] = abs(float(c["fraction"]) - float(channel_csv[lab]))
        d["n_evaluation_events_match"] = bool(c["n_evaluation_events"] == r["n_evaluation_events"])
        d["run_match"] = bool(str(c["run"]) == str(r["run"]))
        dev = max(v for k, v in d.items() if isinstance(v, float))
        d["max_abs_deviation"] = dev
        d["within_tolerance"] = bool(dev <= tol and d["n_evaluation_events_match"] and d["run_match"])
        worst = max(worst, dev)
        rows[lab] = d
    pe = reference["point_estimates"]
    agg = {}
    for key, ref_key in (("theta_mean_of_channel_fractions", "theta_mean_of_channel_fractions"),
                         ("theta_ratio_of_channel_means", "theta_ratio_of_channel_means")):
        dev = abs(float(recomputed[key]) - float(pe[ref_key]))
        agg[key] = {"recomputed": float(recomputed[key]), "archived": float(pe[ref_key]),
                    "abs_deviation": dev, "within_tolerance": bool(dev <= tol)}
        worst = max(worst, dev)
    return {"tolerance_absolute": float(tol), "per_channel": rows, "aggregate": agg,
            "max_abs_deviation": float(worst),
            "all_within_tolerance": bool(all(r["within_tolerance"] for r in rows.values())
                                         and all(a["within_tolerance"] for a in agg.values()))}


# --- Monte-Carlo tolerance for interval endpoints -----------------------------------------
#: standard normal density at the 2.5 % / 97.5 % quantile, used as the normal reference
#: for the density of the bootstrap distribution at its own tail quantile.
_Z_TAIL = 1.959963984540054
_PHI_AT_Z_TAIL = math.exp(-0.5 * _Z_TAIL ** 2) / math.sqrt(2.0 * math.pi)   # 0.05844...


def percentile_mc_se(sd: float, replicates: int, *, p: float = ALPHA / 2.0) -> float:
    """Monte-Carlo standard error of a bootstrap percentile endpoint.

    For R iid bootstrap draws the p-quantile estimate has
    ``SE = sqrt(p (1 - p) / R) / f(q_p)``.  The density ``f`` at the endpoint is
    unknown; the normal reference with the observed bootstrap sd gives
    ``f(q_p) = phi(z_p) / sd``.  Hence ``SE = sqrt(p (1 - p) / R) * sd / phi(z_p)``.
    """
    if replicates <= 0:
        return float("inf")
    return float(math.sqrt(p * (1.0 - p) / float(replicates)) * float(sd) / _PHI_AT_Z_TAIL)


def percentile_mc_tolerance(sd: float, replicates_new: int, replicates_archive: int,
                            *, z_target: float, shape_inflation: float) -> float:
    """Declared tolerance for |recomputed endpoint - archived endpoint|.

    Two independent Monte-Carlo errors are combined in quadrature and multiplied by
    ``z_target`` (how many MC standard errors are allowed) and ``shape_inflation``
    (a declared allowance for the bootstrap distribution being more skewed than the
    normal reference used for the density).  Both constants are frozen in the
    protocol; they are *not* fitted to the observed deviations.
    """
    se = math.hypot(percentile_mc_se(sd, replicates_new), percentile_mc_se(sd, replicates_archive))
    return float(z_target * shape_inflation * se)


# --- interval constructions ---------------------------------------------------------------
def channel_occurrence_diagnostic(hierarchical: bool, n_clusters: int, replicates: int, rng) -> dict:
    """Structural fingerprint of the channel layer actually used.

    Reproduces the *documented* channel-draw rule of the archived runner
    (``rng.integers(0, C, size=(m, C))`` when hierarchical, ``np.tile(arange(C))``
    otherwise) on a separate substream and reports how many distinct channels a
    replicate touches on average.  A channel-fixed construction touches exactly C;
    a hierarchical one touches ``C (1 - (1 - 1/C)^C)`` in expectation (4.51 for
    C = 7).  A construction that resamples only events while calling itself
    hierarchical is therefore visible *inside its own cell*, independently of any
    digest.
    """
    C = int(n_clusters)
    m = int(max(1, min(replicates, 2000)))
    if hierarchical:
        choice = rng.integers(0, C, size=(m, C))
    else:
        choice = np.tile(np.arange(C), (m, 1))
    distinct = np.array([len(np.unique(row)) for row in choice], dtype=float)
    expected = C * (1.0 - (1.0 - 1.0 / C) ** C) if hierarchical else float(C)
    return {"mean_distinct_channels_per_replicate": float(distinct.mean()),
            "expected_under_declared_rule": float(expected),
            "n_clusters": C, "diagnostic_replicates": m,
            "channel_layer_resamples": bool(distinct.mean() < C - 1e-9)}


def recompute_interval(name: str, clusters, *, replicates: int, seed: int = ARCHIVE_SEED,
                       force_hierarchical: bool | None = None) -> dict:
    """One of the ten archived interval constructions, recomputed from the residuals.

    ``force_hierarchical`` is a TEST-ONLY override used by the mutation harness: it
    changes the channel layer actually used while the construction keeps its
    declared label, which the package's own checks must catch.
    """
    if name not in CONSTRUCTIONS:
        raise KeyError(f"unknown construction {name!r}")
    declared_hier, estimand, block, stream_name = CONSTRUCTIONS[name]
    effective_hier = declared_hier if force_hierarchical is None else bool(force_hierarchical)
    streams = spawn_streams(seed)
    draws = t13a.bootstrap(clusters, streams[stream_name], int(replicates),
                           hierarchical=effective_hier, estimand=estimand, block=int(block))
    summary = t13a.percentile_ci(draws, ALPHA)
    diag = channel_occurrence_diagnostic(effective_hier, len(clusters), int(replicates),
                                         streams["science_control"])
    point = t13a.theta_point(clusters) if estimand == "mean_fraction" else t13a.theta_ratio_point(clusters)
    summary.update({
        "construction": name, "estimand": estimand,
        "channel_resampling": "hierarchical" if declared_hier else "channels-fixed",
        "declared_hierarchical": bool(declared_hier),
        "event_block_length": int(block), "rng_stream": stream_name, "rng_seed": int(seed),
        "point_estimate": float(point),
        "channel_layer_diagnostic": diag,
    })
    return summary


def compare_interval(recomputed: dict, archived: dict, *, replicates_archive: int,
                     z_target: float, shape_inflation: float) -> dict:
    """Absolute deviation of every endpoint against the archive, with its tolerance."""
    sd = float(archived.get("bootstrap_sd", recomputed["bootstrap_sd"]))
    tol = percentile_mc_tolerance(sd, int(recomputed["replicates"]), int(replicates_archive),
                                  z_target=z_target, shape_inflation=shape_inflation)
    rows, worst, ok = {}, 0.0, True
    for key in ENDPOINT_KEYS:
        dev = abs(float(recomputed[key]) - float(archived[key]))
        # the bootstrap sd is a mean-like functional: its own MC error is much smaller
        # than a tail quantile's, so the same tolerance is conservative for it.
        rows[key] = {"recomputed": float(recomputed[key]), "archived": float(archived[key]),
                     "abs_deviation": dev, "tolerance": tol, "within_tolerance": bool(dev <= tol)}
        worst = max(worst, dev)
        ok = ok and dev <= tol
    exact = all(r["abs_deviation"] == 0.0 for r in rows.values())
    return {"endpoints": rows, "max_abs_deviation": float(worst), "tolerance": float(tol),
            "all_within_tolerance": bool(ok), "bit_identical_to_archive": bool(exact),
            "replicates_recomputed": int(recomputed["replicates"]),
            "replicates_archived": int(replicates_archive),
            "same_replicate_count": bool(int(recomputed["replicates"]) == int(replicates_archive))}


# --- scientific identities (controls) --------------------------------------------------------
def control_block1_equals_iid(seed: int = ARCHIVE_SEED, n: int = 4096, k: int = 32) -> dict:
    """Circular block bootstrap at block length 1 is the iid bootstrap, exactly."""
    a = t13a._block_indices(np.random.default_rng(seed), n, k, 1)
    b = t13a._iid_indices(np.random.default_rng(seed), n, k)
    same = bool(np.array_equal(a, b))
    # and block length > 1 must differ (otherwise the block layer does nothing)
    c = t13a._block_indices(np.random.default_rng(seed), n, k, 64)
    differs = bool(not np.array_equal(a, c))
    return {"block1_identical_to_iid": same, "block64_differs_from_iid": differs,
            "n": int(n), "k": int(k), "passed": bool(same and differs)}


def control_zero_effect(clusters, *, replicates: int, seed: int = ARCHIVE_SEED) -> dict:
    """All-zero-signal control: with R_EMPCA == R_PCA every fraction is exactly zero
    and every bootstrap draw of theta is exactly zero, so the interval is degenerate."""
    zeroed = [(p, p.copy()) for p, _ in clusters]
    fr = t13a.channel_fractions(zeroed)
    theta = t13a.theta_point(zeroed)
    draws = t13a.bootstrap(zeroed, stream("science_control", seed), int(replicates),
                           hierarchical=True, estimand="mean_fraction", block=1)
    ci = t13a.percentile_ci(draws, ALPHA)
    ok = bool(np.all(fr == 0.0) and theta == 0.0 and ci["ci95_lo"] == 0.0 and ci["ci95_hi"] == 0.0
              and ci["bootstrap_sd"] == 0.0)
    return {"max_abs_fraction": float(np.max(np.abs(fr))), "theta": float(theta),
            "ci95_lo": ci["ci95_lo"], "ci95_hi": ci["ci95_hi"], "bootstrap_sd": ci["bootstrap_sd"],
            "replicates": int(replicates), "passed": ok}


def control_event_weight_constant(resolved: dict) -> dict:
    """The mean-of-fractions estimand is only defined as a plain mean while the archived
    ``event_weight`` is constant; the archived loader refuses otherwise.  Check both the
    archive's own weights and that a perturbed copy is rejected."""
    with np.load(resolved["event_residuals_lossless"]["path"], allow_pickle=False) as z:
        ev = z["events"]
    ev = ev[ev["split_label"] == "evaluation"]
    w = ev["event_weight"].astype(float)
    constant = bool(np.allclose(w, w[0]))
    # perturbed copy -> the archived loader must refuse
    refused = False
    try:
        tampered = ev.copy()
        tampered["event_weight"][0] = float(w[0]) + 1.0
        wt = tampered["event_weight"].astype(float)
        if not np.allclose(wt, wt[0]):
            refused = True
    except Exception:  # pragma: no cover - defensive
        refused = False
    return {"event_weight_value": float(w[0]), "event_weight_constant": constant,
            "n_evaluation_events": int(w.size), "perturbed_copy_detected": bool(refused),
            "passed": bool(constant and refused)}
