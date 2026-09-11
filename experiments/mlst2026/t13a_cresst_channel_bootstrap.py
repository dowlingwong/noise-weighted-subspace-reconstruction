"""T13-a - CRESST channel-robustness bootstrap on the archived residuals.

Protocol: MS ``plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md`` section T13-a and
``plan/mlst_protocol_2026-09-10.json`` key ``t13a``.

Headline estimand (mean of channel fractions)::

    theta = (1/C) * sum_c [ 1 - Rbar_EMPCA,c / Rbar_PCA,c ],    C = 7

One headline replicate draws 7 channel *occurrences* with replacement from the
7 run/channel clusters; for each occurrence it draws n_c evaluation events with
replacement from that channel -- the *same* drawn events for both methods, so
the PCA/EMPCA pairing is preserved -- forms that occurrence's fraction, and
averages over the 7 occurrences.  A channel drawn twice contributes two
*independent* event draws, i.e. sampled-channel multiplicities are preserved
rather than collapsed.  10 000 replicates, seed 20260713, percentile interval.

This is a *sensitivity* analysis of an archived result.  Seven channels drawn
from two runs do not constitute seven independent detector replications, and no
output of this script may be labelled an independent replication.

The script is read-only with respect to the archive: it never writes into
``transfer_paper/data/cresst/sensitivity/20260713_channel_robustness_v1``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.mlst2026.provenance import RunRecorder, write_json  # noqa: E402

# --- frozen protocol constants ------------------------------------------------
REPLICATES = 10_000
SEED = 20260713
N_CHANNELS = 7
CI_ALPHA = 0.05
BLOCK_MAIN = 64
BLOCK_SENSITIVITY = (16, 256)
PCA_COL = "pca_weighted_residual"
EMPCA_COL = "empca_weighted_residual"

#: Fixed, declared order of the independent RNG substreams spawned from SEED.
#: The headline uses substream 0.  Changing this list changes every number, so
#: it is part of the freeze.
STREAM_ORDER = (
    "headline_hierarchical",
    "channel_fixed",
    "ratio_of_means_hierarchical",
    "ratio_of_means_channel_fixed",
    "block_hier_64",
    "block_hier_16",
    "block_hier_256",
    "block_channel_fixed_64",
    "block_channel_fixed_16",
    "block_channel_fixed_256",
    "per_channel_effects",
    "leave_one_channel_out",
    "leave_one_run_out",
)


# --- data loading -------------------------------------------------------------
def load_clusters(npz_path: Path):
    """Return (labels, runs, list_of_(pca, empca) arrays sorted by time).

    Only ``split_label == 'evaluation'`` rows are used.  Within each cluster the
    events are sorted by ``chronological_time`` so that the block bootstrap acts
    on the true temporal order.  ``event_weight`` is required to be constant
    (it is 1 throughout the archive); the estimand is a plain mean under that
    condition and the script refuses to guess otherwise.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        ev = z["events"]
    ev = ev[ev["split_label"] == "evaluation"]
    weights = ev["event_weight"].astype(float)
    if not np.allclose(weights, weights[0]):
        raise ValueError(
            "event_weight is not constant in the archive; the mean-of-fractions "
            "estimand would need an explicit weighted definition."
        )
    labels = sorted(np.unique(ev["channel_label"]).tolist())
    if len(labels) != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} channels, found {len(labels)}")
    clusters, runs, times = [], [], []
    for lab in labels:
        sub = ev[ev["channel_label"] == lab]
        order = np.argsort(sub["chronological_time"], kind="stable")
        sub = sub[order]
        clusters.append((
            np.ascontiguousarray(sub[PCA_COL].astype(np.float64)),
            np.ascontiguousarray(sub[EMPCA_COL].astype(np.float64)),
        ))
        runs.append(str(np.unique(sub["run"])[0]))
        times.append((float(sub["chronological_time"][0]),
                      float(sub["chronological_time"][-1])))
    return labels, runs, clusters, times


# --- estimands ----------------------------------------------------------------
def channel_fractions(clusters) -> np.ndarray:
    """1 - mean(EMPCA)/mean(PCA) for each cluster, on the observed events."""
    out = np.empty(len(clusters))
    for i, (p, e) in enumerate(clusters):
        out[i] = 1.0 - e.mean() / p.mean()
    return out


def theta_point(clusters) -> float:
    return float(channel_fractions(clusters).mean())


def theta_ratio_point(clusters) -> float:
    """Ratio of channel means -- a baseline-scale-weighted sensitivity, not the
    headline (larger channels' PCA scale dominates the denominator)."""
    p = np.array([c[0].mean() for c in clusters])
    e = np.array([c[1].mean() for c in clusters])
    return float(1.0 - e.mean() / p.mean())


# --- resampling ---------------------------------------------------------------
def _iid_indices(rng, n: int, k: int) -> np.ndarray:
    """(k, n) iid-with-replacement event indices."""
    return rng.integers(0, n, size=(k, n))


def _block_indices(rng, n: int, k: int, block: int) -> np.ndarray:
    """(k, n) circular-block-bootstrap indices with block length ``block``.

    Blocks wrap around the end of the (time-ordered) cluster so that every event
    appears in exactly ``block`` candidate blocks; the concatenation is truncated
    back to n.  For block == 1 this reduces exactly to the iid bootstrap.
    """
    if block <= 1:
        return _iid_indices(rng, n, k)
    b = min(block, n)
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(k, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    return idx.reshape(k, nb * b)[:, :n]


def _cluster_means(rng, cluster, k: int, block: int, chunk_events: int = 4_000_000):
    """k paired bootstrap means (mean_pca, mean_empca) for one cluster.

    Chunked so that peak memory stays bounded regardless of cluster size.
    """
    p, e = cluster
    n = p.size
    mp = np.empty(k)
    me = np.empty(k)
    if k == 0:
        return mp, me
    per = max(1, min(k, chunk_events // max(n, 1)))
    done = 0
    while done < k:
        m = min(per, k - done)
        idx = _block_indices(rng, n, m, block)
        mp[done:done + m] = p[idx].mean(axis=1)
        me[done:done + m] = e[idx].mean(axis=1)
        done += m
    return mp, me


def bootstrap(clusters, rng, replicates: int, *, hierarchical: bool,
              estimand: str, block: int = 1,
              chunk_replicates: int = 1000) -> np.ndarray:
    """Bootstrap draws of theta (``estimand='mean_fraction'``) or theta_ratio
    (``estimand='ratio_of_means'``).

    ``hierarchical=True`` resamples the 7 channel occurrences with replacement
    (cluster-level uncertainty included); ``False`` holds the 7 observed
    channels fixed and resamples only events, giving the channel-fixed
    uncertainty.
    """
    if estimand not in ("mean_fraction", "ratio_of_means"):
        raise ValueError(estimand)
    C = len(clusters)
    out = np.empty(replicates)
    done = 0
    while done < replicates:
        m = min(chunk_replicates, replicates - done)
        if hierarchical:
            choice = rng.integers(0, C, size=(m, N_CHANNELS))
        else:
            choice = np.tile(np.arange(C), (m, 1))
        mp = np.empty(choice.shape)
        me = np.empty(choice.shape)
        for c in range(C):
            mask = choice == c
            k = int(mask.sum())
            if k == 0:
                continue
            a, b = _cluster_means(rng, clusters[c], k, block)
            mp[mask] = a
            me[mask] = b
        if estimand == "mean_fraction":
            out[done:done + m] = (1.0 - me / mp).mean(axis=1)
        else:
            out[done:done + m] = 1.0 - me.mean(axis=1) / mp.mean(axis=1)
        done += m
    return out


def percentile_ci(draws: np.ndarray, alpha: float = CI_ALPHA) -> dict:
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "bootstrap_mean": float(draws.mean()),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "replicates": int(draws.size),
    }


def _pct(d: dict) -> dict:
    """Same summary expressed in percent, for the reporting tables."""
    return {k: (100.0 * v if k in ("ci95_lo", "ci95_hi", "bootstrap_mean",
                                   "bootstrap_sd") else v)
            for k, v in d.items()}


# --- main ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True,
                    help="MS 20260713_channel_robustness_v1 directory")
    ap.add_argument("--out", required=True, help="new versioned output directory")
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="implementation check only: few replicates, output "
                         "marked non-scientific")
    args = ap.parse_args(argv)

    archive = Path(args.archive)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory {out_dir}")

    rec = RunRecorder(
        run_id="T13a_cresst_channel_bootstrap",
        protocol_refs={
            "programme": "plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md#T13-a",
            "config": "plan/mlst_protocol_2026-09-10.json#t13a",
        },
    )
    npz = rec.add_input("archived_residuals_lossless", archive / "event_residuals_lossless.npz")
    rec.add_input("archive_manifest", archive / "manifest.json")
    rec.add_input("archive_summary", archive / "summary.json")
    rec.add_input("archive_config", archive / "analysis_config.yaml")
    rec.add_input("runner_source", Path(__file__).resolve())
    rec.add_seed("base_seed", args.seed)
    rec.add_seed("stream_order", list(STREAM_ORDER))
    rec.add_seed("replicates", args.replicates)

    labels, runs, clusters, times = load_clusters(npz)
    counts = [int(c[0].size) for c in clusters]

    ss = np.random.SeedSequence(args.seed)
    children = ss.spawn(len(STREAM_ORDER))
    streams = {name: np.random.default_rng(child)
               for name, child in zip(STREAM_ORDER, children)}

    R = args.replicates
    fractions = channel_fractions(clusters)
    theta = theta_point(clusters)
    theta_ratio = theta_ratio_point(clusters)

    results: dict = {
        "label": "SENSITIVITY ANALYSIS of an archived result - not an "
                 "independent replication. Seven channels from two runs are "
                 "seven correlated clusters, not seven independent detectors.",
        "smoke": bool(args.smoke),
        "point_estimates": {
            "theta_mean_of_channel_fractions": theta,
            "theta_mean_of_channel_fractions_pct": 100 * theta,
            "theta_ratio_of_channel_means": theta_ratio,
            "theta_ratio_of_channel_means_pct": 100 * theta_ratio,
        },
        "channels": [
            {
                "channel_label": labels[i],
                "run": runs[i],
                "n_evaluation_events": counts[i],
                "chronological_time_first": times[i][0],
                "chronological_time_last": times[i][1],
                "mean_pca_weighted_residual": float(clusters[i][0].mean()),
                "mean_empca_weighted_residual": float(clusters[i][1].mean()),
                "fraction": float(fractions[i]),
                "fraction_pct": float(100 * fractions[i]),
            }
            for i in range(len(labels))
        ],
    }

    # --- headline and the named sensitivities ---------------------------------
    intervals: dict = {}

    def run(name, *, hierarchical, estimand, block, stream):
        draws = bootstrap(clusters, streams[stream], R,
                          hierarchical=hierarchical, estimand=estimand, block=block)
        summary = percentile_ci(draws)
        summary.update({
            "estimand": estimand,
            "channel_resampling": "hierarchical" if hierarchical else "channels-fixed",
            "event_block_length": block,
            "rng_stream": stream,
            "point_estimate": theta if estimand == "mean_fraction" else theta_ratio,
        })
        summary["pct"] = _pct({k: summary[k] for k in
                               ("ci95_lo", "ci95_hi", "bootstrap_mean",
                                "bootstrap_sd", "replicates")})
        intervals[name] = summary
        return draws

    headline = run("headline", hierarchical=True, estimand="mean_fraction",
                   block=1, stream="headline_hierarchical")
    run("channel_fixed", hierarchical=False, estimand="mean_fraction",
        block=1, stream="channel_fixed")
    run("ratio_of_means_hierarchical", hierarchical=True,
        estimand="ratio_of_means", block=1, stream="ratio_of_means_hierarchical")
    run("ratio_of_means_channel_fixed", hierarchical=False,
        estimand="ratio_of_means", block=1, stream="ratio_of_means_channel_fixed")
    for b in (BLOCK_MAIN, *BLOCK_SENSITIVITY):
        run(f"block{b}_hierarchical", hierarchical=True, estimand="mean_fraction",
            block=b, stream=f"block_hier_{b}")
        run(f"block{b}_channel_fixed", hierarchical=False, estimand="mean_fraction",
            block=b, stream=f"block_channel_fixed_{b}")
    results["intervals"] = intervals

    # --- per-channel effects with their own paired event bootstrap ------------
    rng_ch = streams["per_channel_effects"]
    per_channel = []
    for i, lab in enumerate(labels):
        row = {"channel_label": lab, "run": runs[i],
               "n_evaluation_events": counts[i],
               "fraction": float(fractions[i]),
               "fraction_pct": float(100 * fractions[i])}
        for b in (1, BLOCK_MAIN):
            mp, me = _cluster_means(rng_ch, clusters[i], R, b)
            d = percentile_ci(1.0 - me / mp)
            key = "event_iid" if b == 1 else f"event_block{b}"
            row[key] = {k: v for k, v in d.items()}
            row[key + "_pct"] = _pct(d)
        per_channel.append(row)
    results["per_channel_effects"] = per_channel

    # --- leave-one-channel-out and leave-one-run-out ---------------------------
    rng_loco = streams["leave_one_channel_out"]
    loco = []
    for i, lab in enumerate(labels):
        keep = [clusters[j] for j in range(len(clusters)) if j != i]
        pt = theta_point(keep)
        draws = bootstrap(keep, rng_loco, R, hierarchical=True,
                          estimand="mean_fraction", block=1)
        loco.append({"dropped_channel": lab, "n_channels_retained": len(keep),
                     "theta": pt, "theta_pct": 100 * pt,
                     "shift_from_headline_pct": 100 * (pt - theta),
                     **_pct(percentile_ci(draws))})
    results["leave_one_channel_out"] = loco

    rng_loro = streams["leave_one_run_out"]
    loro = []
    for r in sorted(set(runs)):
        keep = [clusters[j] for j in range(len(clusters)) if runs[j] != r]
        pt = theta_point(keep)
        draws = bootstrap(keep, rng_loro, R, hierarchical=True,
                          estimand="mean_fraction", block=1)
        loro.append({"dropped_run": r, "n_channels_retained": len(keep),
                     "retained_channels": [labels[j] for j in range(len(labels))
                                           if runs[j] != r],
                     "theta": pt, "theta_pct": 100 * pt,
                     "shift_from_headline_pct": 100 * (pt - theta),
                     **_pct(percentile_ci(draws))})
    results["leave_one_run_out"] = loro

    # --- write ----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "headline_bootstrap_draws.npz",
                        headline_theta=headline)
    provenance = rec.finalize({"n_evaluation_events": int(sum(counts))})
    results["provenance"] = provenance
    digest = write_json(out_dir / "t13a_results.json", results)
    (out_dir / "RESULTS_SHA256").write_text(f"{digest}  t13a_results.json\n")

    print(f"[T13-a] theta = {100*theta:.4f} %  headline 95% CI "
          f"[{intervals['headline']['pct']['ci95_lo']:.4f}, "
          f"{intervals['headline']['pct']['ci95_hi']:.4f}] %")
    print(f"[T13-a] channel-fixed 95% CI "
          f"[{intervals['channel_fixed']['pct']['ci95_lo']:.4f}, "
          f"{intervals['channel_fixed']['pct']['ci95_hi']:.4f}] %")
    print(f"[T13-a] wrote {out_dir}/t13a_results.json  sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
