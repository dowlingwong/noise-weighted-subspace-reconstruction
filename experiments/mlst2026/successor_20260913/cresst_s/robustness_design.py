"""CRESST-S part C: the *prepared and unexecuted* robustness design.

The rank / shrinkage / time-cut sensitivity grid is enumerated here as real matrix
cells with real fit counts, real seeds and explicit paired contrasts, so that its
cost and its dependence structure are inspectable now.  **It cannot be executed in
this environment**: every one of these cells needs the raw CRESST pulse traces, and
no CRESST raw data is present in this container.  ``run_cell`` therefore raises

    input not available: <what is missing>

for every production robustness cell, which is charged in the ledger as a failed
attempt and exits non-zero, while ``validate`` and ``enumerate`` continue to
succeed (they compute nothing and open no data).

To keep the reduced runner genuinely exercised end to end, this module also builds
a **synthetic fixture data root** with the same on-disk interface, and the
development cells run the real estimator pipeline on it.  A fixture number is never
a scientific result.

On-disk interface expected at ``<CRESST_DATA_ROOT>``
---------------------------------------------------
::

    <root>/index.json
        {"format": "cresst-s-pulse-root/1", "synthetic": <bool>, "n_samples": int,
         "sample_rate_hz": float,
         "channels": [{"run": "33", "channel": "36.0", "file": "run33/ch36.0.npz",
                       "n_events": int, "n_noise": int}, ...]}
    <root>/<file>            NPZ with
        pulses  (n_events, n_samples) float64   evaluation-split pulse traces, time ordered
        times   (n_events,)           float64   chronological time, ascending
        noise   (n_noise, n_samples)  float64   noise-only traces for the PSD estimate

The real CRESST release does not ship in this layout; an adapter from the release
format to this interface is part of the remote work and is deliberately **not**
written here, because there is no raw data in this container to test it against.

Estimator pipeline (one cell)
-----------------------------
1. estimate a stationary PSD from the channel's noise traces;
2. shrink it toward its own mean with weight ``shrinkage``;
3. represent each pulse in the real-FFT basis (real and imaginary parts stacked), in
   which a stationary covariance is diagonal, and weight each component by the
   inverse shrunk PSD;
4. split the time-ordered events by the declared ``time_cut`` into a training half
   and an evaluation half;
5. fit classic PCA at the declared rank (1 fit) and weighted EMPCA at the same rank
   with ``EMPCA_RESTARTS`` random starts (``EMPCA_RESTARTS`` fits), keeping the start
   with the lowest training chi-square;
6. score both on the held-out events under the same weights and report
   ``1 - R_EMPCA / R_PCA``.

Both estimators come from ``src/canonical/empca.py`` (the repository's canonical
weighted-EM PCA), so this package introduces no new estimator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.canonical import empca as canonical_empca  # noqa: E402

from ..common import fitcount  # noqa: E402
from .archive_recompute import InputNotAvailable  # noqa: E402

ROOT_FORMAT = "cresst-s-pulse-root/1"
INDEX_NAME = "index.json"

#: random restarts of the weighted EM at each (channel, rank, shrinkage, time cut).
EMPCA_RESTARTS = 2
#: fits charged per robustness cell: one classic-PCA fit plus one per EM restart.
FITS_PER_CELL = 1 + EMPCA_RESTARTS

#: the archive's seven run/channel clusters, in the archived label order.
ARCHIVE_CHANNELS = (("33", "36.0"), ("33", "37.0"), ("35", "26.0"), ("35", "27.0"),
                    ("35", "56.0"), ("35", "57.0"), ("35", "58.0"))

TIME_CUTS = ("full", "first_half", "second_half")


# --- data root ------------------------------------------------------------------------------
def read_index(data_root) -> dict:
    if data_root is None:
        raise InputNotAvailable(
            "input not available: raw CRESST pulse data root (<CRESST_DATA_ROOT>).  The rank / shrinkage / "
            "time-cut robustness grid needs the raw CRESST pulse traces; no CRESST raw data is present in "
            "this environment and none may be sought here.  Pass --cresst-data-root <CRESST_DATA_ROOT> on a "
            "host that holds the release (see REMOTE_COMMANDS.md).")
    root = Path(data_root)
    idx = root / INDEX_NAME
    if not root.is_dir():
        raise InputNotAvailable(f"input not available: CRESST pulse data root {root} is not a directory")
    if not idx.is_file():
        raise InputNotAvailable(f"input not available: {idx} (the <CRESST_DATA_ROOT> index describing the "
                                f"{ROOT_FORMAT} layout)")
    js = json.loads(idx.read_text())
    if js.get("format") != ROOT_FORMAT:
        raise InputNotAvailable(f"input not available: {idx} declares format {js.get('format')!r}, "
                                f"expected {ROOT_FORMAT!r}; an adapter from the CRESST release layout to this "
                                "interface has not been written or tested (no raw data here to test it on)")
    return js


def load_pulse_channel(data_root, run: str, channel: str) -> dict:
    js = read_index(data_root)
    root = Path(data_root)
    for entry in js.get("channels", []):
        if str(entry["run"]) == str(run) and str(entry["channel"]) == str(channel):
            p = root / entry["file"]
            if not p.is_file():
                raise InputNotAvailable(f"input not available: {p} (pulses for run {run} channel {channel})")
            with np.load(p, allow_pickle=False) as z:
                out = {"pulses": np.asarray(z["pulses"], dtype=float),
                       "times": np.asarray(z["times"], dtype=float),
                       "noise": np.asarray(z["noise"], dtype=float)}
            out["synthetic"] = bool(js.get("synthetic", False))
            out["n_samples"] = int(js["n_samples"])
            return out
    raise InputNotAvailable(f"input not available: run {run} channel {channel} is not in {root / INDEX_NAME} "
                            f"(the root declares {len(js.get('channels', []))} channel(s))")


def write_fixture_data_root(root, *, seed: int, channels=ARCHIVE_CHANNELS[:2], n_events: int = 120,
                            n_noise: int = 160, n_samples: int = 48) -> dict:
    """Generate a small synthetic pulse/noise set with the production interface.

    DEVELOPMENT ONLY.  The traces are a two-component pulse family plus coloured
    (1/f-like) stationary noise; they exist so that the reduced runner is exercised
    end to end, and no number derived from them is a scientific result.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    t = np.arange(n_samples, dtype=float)
    basis = np.stack([np.exp(-((t - 0.30 * n_samples) ** 2) / (2 * (0.06 * n_samples) ** 2)),
                      np.exp(-(t - 0.30 * n_samples) / (0.25 * n_samples)) * (t >= 0.30 * n_samples)])
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    psd = 1.0 / (1.0 + (freqs / max(freqs[1], 1e-9)) ** 1.4)
    psd[0] = psd[1]
    entries = []
    for run, ch in channels:
        rel = Path(f"run{run}") / f"ch{ch}.npz"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)

        def coloured(m):
            spec = (rng.normal(size=(m, freqs.size)) + 1j * rng.normal(size=(m, freqs.size))) * np.sqrt(psd)
            return np.fft.irfft(spec, n=n_samples, axis=1) * np.sqrt(n_samples)

        amps = rng.lognormal(mean=0.0, sigma=0.5, size=(n_events, 2)) * np.array([1.0, 0.35])
        pulses = amps @ basis + 0.25 * coloured(n_events)
        times = np.sort(rng.uniform(0.0, 10.0, size=n_events))
        noise = coloured(n_noise)
        np.savez_compressed(root / rel, pulses=pulses, times=times, noise=noise)
        entries.append({"run": run, "channel": ch, "file": str(rel), "n_events": int(n_events),
                        "n_noise": int(n_noise)})
    index = {"format": ROOT_FORMAT, "synthetic": True, "n_samples": int(n_samples),
             "sample_rate_hz": 1.0, "channels": entries, "seed": int(seed),
             "label": "DEVELOPMENT ONLY synthetic pulse/noise fixture; not CRESST data"}
    (root / INDEX_NAME).write_text(json.dumps(index, indent=1, sort_keys=True))
    return index


# --- estimator pipeline -----------------------------------------------------------------------
def stationary_psd(noise: np.ndarray, *, shrinkage: float) -> np.ndarray:
    """Shrunk one-sided PSD estimate from noise-only traces (stationarity assumed)."""
    spec = np.fft.rfft(noise, axis=1)
    psd = (np.abs(spec) ** 2).mean(axis=0) / noise.shape[1]
    psd = np.maximum(psd, np.finfo(float).tiny)
    lam = float(shrinkage)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"shrinkage must lie in [0, 1], got {lam}")
    return (1.0 - lam) * psd + lam * float(psd.mean())


def fourier_representation(traces: np.ndarray, psd: np.ndarray):
    """Stack the real and imaginary rFFT parts and weight each by the inverse PSD.

    A stationary covariance is diagonal in this basis, so the inverse-PSD weights are
    exactly the ``stationary PSD covariance`` weighting named in the primary setting.
    """
    spec = np.fft.rfft(traces, axis=1)
    x = np.concatenate([spec.real, spec.imag], axis=1)
    w = np.concatenate([1.0 / psd, 1.0 / psd])[None, :] * np.ones((traces.shape[0], 1))
    return np.ascontiguousarray(x), np.ascontiguousarray(w)


def time_cut_mask(times: np.ndarray, cut: str) -> np.ndarray:
    n = times.size
    order = np.argsort(times, kind="stable")
    mask = np.zeros(n, dtype=bool)
    if cut == "full":
        mask[:] = True
    elif cut == "first_half":
        mask[order[: n // 2]] = True
    elif cut == "second_half":
        mask[order[n // 2:]] = True
    else:
        raise ValueError(f"unknown time_cut {cut!r}; declared cuts are {TIME_CUTS}")
    return mask


def fit_and_score(pulses, times, noise, *, rank: int, shrinkage: float, time_cut: str,
                  seed: int, niter: int, restarts: int = EMPCA_RESTARTS) -> dict:
    """One robustness cell's estimator work.  Returns the scored fraction and the fit count."""
    mask = time_cut_mask(times, time_cut)
    sel = np.where(mask)[0]
    sel = sel[np.argsort(times[sel], kind="stable")]
    if sel.size < 8:
        raise InputNotAvailable(f"input not available: time cut {time_cut!r} leaves {sel.size} events "
                                "(at least 8 are needed to split train/evaluate)")
    half = sel.size // 2
    train_idx, eval_idx = sel[:half], sel[half:]
    psd = stationary_psd(noise, shrinkage=shrinkage)
    x_train, w_train = fourier_representation(pulses[train_idx], psd)
    x_eval, w_eval = fourier_representation(pulses[eval_idx], psd)
    k = int(rank)
    if k >= min(x_train.shape):
        raise InputNotAvailable(f"input not available: rank {k} exceeds the {x_train.shape} training block")

    # Every estimator invocation is counted where it happens: once here and once per
    # EM restart below.  ``fits`` is the cell's own tally and ``common.fitcount`` is the
    # shared instrumentation the CLI cross-checks against the matrix declaration.
    fits = 0
    fits += fitcount.count_fit("classic_pca")
    m_pca = canonical_empca.classic_pca(x_train, nvec=k)
    m_pca.set_data(x_eval, w_eval)
    r_pca = float(m_pca.chi2()) / float(x_eval.shape[0])

    best, best_train = None, np.inf
    for r in range(int(restarts)):
        fits += fitcount.count_fit("empca_restart")
        m = canonical_empca.empca(x_train, weights=w_train, niter=int(niter), nvec=k,
                                  randseed=int(seed) * 100 + r, silent=True)
        c = float(m.chi2())
        if c < best_train:
            best, best_train = m, c
    best.set_data(x_eval, w_eval)
    r_empca = float(best.chi2()) / float(x_eval.shape[0])

    return {"rank": k, "shrinkage": float(shrinkage), "time_cut": time_cut,
            "n_train_events": int(train_idx.size), "n_eval_events": int(eval_idx.size),
            "n_components_in_representation": int(x_train.shape[1]),
            "weighted_residual_pca": r_pca, "weighted_residual_empca": r_empca,
            "fraction": float(1.0 - r_empca / r_pca) if r_pca > 0 else None,
            "empca_restarts": int(restarts), "empca_iterations": int(niter),
            "empca_best_train_chi2": best_train, "n_fits": int(fits)}


# --- the grid --------------------------------------------------------------------------------
def primary_setting(P: dict) -> dict:
    return dict(P["robustness"]["primary_setting"])


def grid_rows(P: dict, arm: str) -> list[dict]:
    """Enumerate the robustness arm as parameter rows.

    ``arm='pilot'``        the rank scan at the primary shrinkage and time cut;
    ``arm='confirmation'`` the declared rank x shrinkage x time-cut grid.

    The *dependence unit* is the (run, channel) cluster, and the *contrast group* is
    (channel, time cut): every setting in a group is evaluated on the same events of
    the same channel, so a within-group difference against the group's reference
    setting (the primary rank and shrinkage *at that time cut*) is a paired contrast
    with the event-level and detector-level variation differenced out.  Time-cut
    contrasts are formed separately, at the reference estimator setting, against the
    ``full`` cut of the same channel.  Across groups the two runs remain correlated
    clusters and nothing is independent.
    """
    R = P["robustness"]
    prim = primary_setting(P)
    spec = R["arms"][arm]
    rows = []
    for cut_i, cut in enumerate(spec["time_cuts"]):
        for ch_i, (run, ch) in enumerate(ARCHIVE_CHANNELS):
            group = f"{run}_{ch}|{cut}"
            for rank in spec["ranks"]:
                for lam in spec["shrinkages"]:
                    is_ref = (int(rank) == int(prim["rank"]) and float(lam) == float(prim["shrinkage"]))
                    is_primary = bool(is_ref and cut == prim["time_cut"])
                    rows.append({
                        "arm": arm, "run": run, "channel": ch, "channel_label": f"{run}_{ch}",
                        "rank": int(rank), "shrinkage": float(lam), "time_cut": cut,
                        "covariance_model": prim["covariance_model"],
                        "contrast_group": group, "contrast_group_index": cut_i * len(ARCHIVE_CHANNELS) + ch_i,
                        "channel_index": ch_i, "time_cut_index": cut_i,
                        "is_group_reference": bool(is_ref),
                        "is_primary_setting": is_primary,
                        "paired_against": None if is_ref else
                        f"{group}|rank{prim['rank']}|shrink{prim['shrinkage']}",
                        "time_cut_paired_against": (f"{run}_{ch}|{prim['time_cut']}|rank{prim['rank']}"
                                                    f"|shrink{prim['shrinkage']}") if (is_ref and not is_primary) else None,
                        "dependence_unit": "run/channel cluster",
                        "within_run_dependence": f"run {run} contributes "
                                                 f"{sum(1 for r, _ in ARCHIVE_CHANNELS if r == run)} of the "
                                                 f"{len(ARCHIVE_CHANNELS)} clusters; channels of one run share "
                                                 "cryostat, readout and live-time conditions and are not independent",
                        "executable_here": False,
                        "requires": "<CRESST_DATA_ROOT> raw pulse traces (absent from this environment)",
                    })
    return rows


def contrast_plan(rows: list[dict]) -> dict:
    """The paired contrasts the grid would score, enumerated without executing anything."""
    groups: dict[str, dict] = {}
    for r in rows:
        g = groups.setdefault(r["contrast_group"], {"reference": None, "contrasts": [], "run": r["run"],
                                                    "channel": r["channel"], "time_cut": r["time_cut"],
                                                    "is_primary_group": bool(r["is_primary_setting"]) or None})
        tag = f"rank{r['rank']}|shrink{r['shrinkage']}"
        if r["is_group_reference"]:
            g["reference"] = tag
            g["is_primary_group"] = bool(r["is_primary_setting"])
        else:
            g["contrasts"].append(tag)
    n_pairs = sum(len(g["contrasts"]) for g in groups.values() if g["reference"] is not None)
    n_cut_pairs = sum(1 for g in groups.values() if g["reference"] is not None and not g["is_primary_group"])
    orphan = sorted(k for k, g in groups.items() if g["reference"] is None)
    return {"n_contrast_groups": len(groups), "n_paired_contrasts": n_pairs,
            "n_time_cut_contrasts": n_cut_pairs,
            "n_paired_contrasts_total": n_pairs + n_cut_pairs,
            "groups_without_a_reference_setting": orphan, "groups": groups,
            "pairing": ("within (channel, time cut) against the group's reference setting "
                        "(primary rank and shrinkage at that cut); time-cut contrasts are formed at the "
                        "reference setting against the 'full' cut of the same channel"),
            "aggregation": ("paired differences would be aggregated over channels with the same "
                            "cluster-aware construction used for the archive, and would inherit the same "
                            "limitation: two runs cannot calibrate a new-run interval")}


def control_uniform_weight_limit(*, n_events: int, n_samples: int, rank: int, niter: int,
                                 tolerance: float, seed: int = 424242) -> dict:
    """Scientific control on the estimator pipeline, executable without any CRESST data.

    At shrinkage 1.0 the shrunk PSD is exactly flat, so the inverse-PSD weights are
    uniform and weighted EMPCA must recover the same rank-k subspace as classic PCA.
    The held-out weighted residuals must therefore agree to within ``tolerance``
    (relative), and the reported fraction must be ~0.  A pipeline whose weighting or
    whose residual accounting is wired incorrectly fails this identity.  A companion
    arm at shrinkage 0.0 must *not* be degenerate, so the control also shows that the
    weighting is doing something when the PSD is not flat.
    """
    rng = np.random.default_rng(int(seed))
    t = np.arange(int(n_samples), dtype=float)
    basis = np.stack([np.exp(-((t - 0.3 * n_samples) ** 2) / (2 * (0.06 * n_samples) ** 2)),
                      np.exp(-(t - 0.3 * n_samples) / (0.25 * n_samples)) * (t >= 0.3 * n_samples)])
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(int(n_samples), d=1.0)
    psd = 1.0 / (1.0 + (freqs / max(freqs[1], 1e-9)) ** 1.4)
    psd[0] = psd[1]

    def coloured(m):
        spec = (rng.normal(size=(m, freqs.size)) + 1j * rng.normal(size=(m, freqs.size))) * np.sqrt(psd)
        return np.fft.irfft(spec, n=int(n_samples), axis=1) * np.sqrt(int(n_samples))

    amps = rng.lognormal(0.0, 0.5, size=(int(n_events), 2)) * np.array([1.0, 0.35])
    pulses = amps @ basis + 0.25 * coloured(int(n_events))
    times = np.sort(rng.uniform(0.0, 10.0, size=int(n_events)))
    noise = coloured(int(n_events))
    flat = fit_and_score(pulses, times, noise, rank=int(rank), shrinkage=1.0, time_cut="full",
                         seed=int(seed), niter=int(niter))
    shaped = fit_and_score(pulses, times, noise, rank=int(rank), shrinkage=0.0, time_cut="full",
                           seed=int(seed), niter=int(niter))
    rel = abs(flat["weighted_residual_empca"] - flat["weighted_residual_pca"]) / flat["weighted_residual_pca"]
    degenerate_flat = bool(rel <= float(tolerance) and abs(flat["fraction"]) <= float(tolerance))
    weighting_acts = bool(abs(shaped["weighted_residual_pca"] - flat["weighted_residual_pca"])
                          > float(tolerance) * flat["weighted_residual_pca"])
    return {"flat_psd_arm": flat, "shaped_psd_arm": shaped, "relative_difference_flat": float(rel),
            "tolerance": float(tolerance), "uniform_weight_limit_holds": degenerate_flat,
            "weighting_changes_the_problem": weighting_acts,
            "n_fits": int(flat["n_fits"] + shaped["n_fits"]),
            "passed": bool(degenerate_flat and weighting_acts)}


def run_robustness_cell(P: dict, params: dict, *, data_root, seed: int, niter: int) -> dict:
    """Execute one robustness cell.  Raises :class:`InputNotAvailable` when the raw
    CRESST pulse data root is absent, which is the state of this environment."""
    ch = load_pulse_channel(data_root, params["run"], params["channel"])
    scored = fit_and_score(ch["pulses"], ch["times"], ch["noise"], rank=params["rank"],
                           shrinkage=params["shrinkage"], time_cut=params["time_cut"],
                           seed=int(seed), niter=int(niter))
    scored.update({"run": params["run"], "channel": params["channel"],
                   "channel_label": params["channel_label"],
                   "contrast_group": params["contrast_group"],
                   "is_primary_setting": bool(params["is_primary_setting"]),
                   "synthetic_fixture_data": bool(ch["synthetic"])})
    return scored
