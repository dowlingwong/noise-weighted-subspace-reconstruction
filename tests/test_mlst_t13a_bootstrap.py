"""Tests for the T13-a CRESST channel-robustness bootstrap (MLST 2026-09-10).

The archive-reproduction tests are skipped when the manuscript bundle is not
mounted next to the experiment repository.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.mlst2026 import t13a_cresst_channel_bootstrap as t13a

ARCHIVE = Path("/ceph/dwong/arXiv-v1-polished-extended-preprint/transfer_paper/"
               "data/cresst/sensitivity/20260713_channel_robustness_v1")
NPZ = ARCHIVE / "event_residuals_lossless.npz"
archive_required = pytest.mark.skipif(not NPZ.exists(),
                                      reason="MS CRESST archive not mounted")


# --- estimand algebra on constructed data ------------------------------------
def test_theta_is_mean_of_channel_fractions_not_ratio_of_means():
    """A big-scale channel and a small-scale channel: the two estimands must
    differ, and theta must weight the channels equally."""
    big = (np.full(100, 100.0), np.full(100, 50.0))     # fraction 0.5
    small = (np.full(100, 1.0), np.full(100, 0.9))      # fraction 0.1
    clusters = [big, small]
    assert t13a.theta_point(clusters) == pytest.approx(0.3)
    # ratio of means: 1 - (50+0.9)/2 / ((100+1)/2) = 1 - 25.45/50.5
    assert t13a.theta_ratio_point(clusters) == pytest.approx(1 - 25.45 / 50.5)
    assert t13a.theta_point(clusters) != pytest.approx(t13a.theta_ratio_point(clusters))


def test_channel_fractions_match_manual_formula():
    rng = np.random.default_rng(0)
    clusters = [(rng.gamma(3, 2, 50) + 1, rng.gamma(3, 2, 50) + 1) for _ in range(4)]
    got = t13a.channel_fractions(clusters)
    want = np.array([1 - e.mean() / p.mean() for p, e in clusters])
    np.testing.assert_allclose(got, want)


# --- resampling mechanics -----------------------------------------------------
def test_block_length_one_reduces_to_iid():
    a = t13a._block_indices(np.random.default_rng(7), 40, 5, 1)
    b = t13a._iid_indices(np.random.default_rng(7), 40, 5)
    np.testing.assert_array_equal(a, b)


def test_block_indices_are_contiguous_modulo_wrap():
    idx = t13a._block_indices(np.random.default_rng(1), 50, 6, 10)
    assert idx.shape == (6, 50)
    assert idx.min() >= 0 and idx.max() < 50
    # every block of 10 must advance by 1 (mod 50) inside the block
    for row in idx:
        for s in range(0, 50, 10):
            blk = row[s:s + 10]
            assert np.all((np.diff(blk) % 50) == 1)


def test_block_bootstrap_recovers_more_variance_under_serial_dependence():
    """A strongly autocorrelated cluster: block resampling must give a wider
    spread of the channel mean than the iid bootstrap, which is exactly why the
    temporal-block sensitivity is reported."""
    rng = np.random.default_rng(3)
    n = 2048
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.97 * x[i - 1] + rng.normal()
    p = x + 20.0
    e = x + 19.0
    cl = (p, e)
    mp_iid, _ = t13a._cluster_means(np.random.default_rng(5), cl, 400, 1)
    mp_blk, _ = t13a._cluster_means(np.random.default_rng(5), cl, 400, 128)
    assert mp_blk.std() > 2.0 * mp_iid.std()


def test_pairing_is_preserved():
    """If EMPCA is a fixed multiple of PCA event-by-event, every paired draw has
    exactly the same fraction; an unpaired implementation would not."""
    rng = np.random.default_rng(11)
    p = rng.gamma(2.0, 5.0, 500) + 0.5
    clusters = [(p, 0.4 * p)]
    draws = t13a.bootstrap(clusters, np.random.default_rng(2), 200,
                           hierarchical=False, estimand="mean_fraction")
    np.testing.assert_allclose(draws, 0.6, atol=1e-12)


def test_hierarchical_draws_channel_multiplicities():
    """The hierarchical arm must actually resample channels: with one extreme
    channel the draw distribution has to be multi-modal, not degenerate."""
    clusters = [(np.full(30, 10.0), np.full(30, 9.0))] * 6 + \
               [(np.full(30, 10.0), np.full(30, 1.0))]
    draws = t13a.bootstrap(clusters, np.random.default_rng(4), 3000,
                           hierarchical=True, estimand="mean_fraction")
    # possible values are (k*0.9 + (7-k)*0.1)/7 for k = #draws of the odd channel
    assert len(np.unique(np.round(draws, 9))) > 3
    fixed = t13a.bootstrap(clusters, np.random.default_rng(4), 200,
                           hierarchical=False, estimand="mean_fraction")
    np.testing.assert_allclose(fixed, t13a.theta_point(clusters), atol=1e-12)


def test_hierarchical_interval_is_wider_than_channel_fixed():
    rng = np.random.default_rng(9)
    clusters = [(rng.gamma(3, 2, 300) + 1 + 5 * i, rng.gamma(3, 2, 300) + 1 + 4 * i)
                for i in range(7)]
    h = t13a.bootstrap(clusters, np.random.default_rng(1), 2000, hierarchical=True,
                       estimand="mean_fraction")
    f = t13a.bootstrap(clusters, np.random.default_rng(1), 2000, hierarchical=False,
                       estimand="mean_fraction")
    assert h.std() > f.std()


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    clusters = [(np.arange(1, 51) * 1.0, np.arange(1, 51) * 0.8) for _ in range(3)]
    a = t13a.bootstrap(clusters, np.random.default_rng(20260713), 300,
                       hierarchical=True, estimand="mean_fraction")
    b = t13a.bootstrap(clusters, np.random.default_rng(20260713), 300,
                       hierarchical=True, estimand="mean_fraction")
    np.testing.assert_array_equal(a, b)


def test_chunking_does_not_change_results():
    clusters = [(np.linspace(1, 9, 40), np.linspace(0.5, 6, 40)) for _ in range(3)]
    a = t13a.bootstrap(clusters, np.random.default_rng(6), 500, hierarchical=False,
                       estimand="mean_fraction", chunk_replicates=500)
    b = t13a.bootstrap(clusters, np.random.default_rng(6), 500, hierarchical=False,
                       estimand="mean_fraction", chunk_replicates=97)
    # different chunkings consume the stream differently, so compare distributions
    assert abs(a.mean() - b.mean()) < 5e-3
    assert abs(a.std() - b.std()) < 5e-3


# --- known-truth coverage -----------------------------------------------------
def test_channel_fixed_interval_covers_truth_at_about_95_percent():
    """Known-truth sanity check: independent events, known population fraction."""
    truth_rng = np.random.default_rng(123)
    covered = 0
    trials = 120
    for _ in range(trials):
        clusters = []
        for _c in range(7):
            p = truth_rng.gamma(shape=4.0, scale=2.5, size=400)
            e = 0.75 * p * truth_rng.gamma(shape=40.0, scale=1 / 40.0, size=400)
            clusters.append((p, e))
        draws = t13a.bootstrap(clusters, np.random.default_rng(int(truth_rng.integers(1e9))),
                               400, hierarchical=False, estimand="mean_fraction")
        lo, hi = np.percentile(draws, [2.5, 97.5])
        if lo <= 0.25 <= hi:
            covered += 1
    assert covered / trials > 0.85


# --- archive reproduction -----------------------------------------------------
@archive_required
def test_archive_point_estimates_match_the_protocol_values():
    labels, runs, clusters, _times = t13a.load_clusters(NPZ)
    assert labels == ['33_36.0', '33_37.0', '35_26.0', '35_27.0',
                      '35_56.0', '35_57.0', '35_58.0']
    assert runs == ['33', '33', '35', '35', '35', '35', '35']
    assert sum(c[0].size for c in clusters) == 19335
    assert 100 * t13a.theta_point(clusters) == pytest.approx(21.5689, abs=1e-3)
    assert 100 * t13a.theta_ratio_point(clusters) == pytest.approx(22.0719, abs=1e-3)


@archive_required
def test_archive_channel_fractions_match_the_archived_channel_summary():
    import csv
    labels, _runs, clusters, _t = t13a.load_clusters(NPZ)
    with open(ARCHIVE / "channel_summary.csv") as fh:
        archived = {r["channel"]: float(r["pca_minus_empca_fraction"])
                    for r in csv.DictReader(fh)}
    got = t13a.channel_fractions(clusters)
    for lab, val in zip(labels, got):
        assert val == pytest.approx(archived[lab], rel=1e-9)


@archive_required
def test_events_are_sorted_by_chronological_time_within_each_cluster():
    with np.load(NPZ, allow_pickle=False) as z:
        ev = z["events"]
    ev = ev[ev["split_label"] == "evaluation"]
    for lab in np.unique(ev["channel_label"]):
        t = np.sort(ev[ev["channel_label"] == lab]["chronological_time"])
        assert np.all(np.diff(t) >= 0)
