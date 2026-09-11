"""T22 implementation tests: conventions, controls, selection rule and inference.

These are meaningful checks of the things that could silently invalidate the
experiment -- split overlap, calibration-only inputs, the Fourier and
physical-amplitude conventions, identical PCA/EMPCA preprocessing, the baseline
identities, the paired bootstrap and the declared decision rules -- run on
synthetic fixtures only.  No instrument file is opened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.mlst2026 import tidmad_band as tb  # noqa: E402
from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402
from experiments.mlst2026 import t22_tidmad_narrowband as t22  # noqa: E402

#: The stage-0 protocol lives in the manuscript repository; tests locate it via
#: the environment or the conventional sibling checkout, and skip if absent.
PROTOCOL_CANDIDATES = [
    Path(p) for p in (
        __import__("os").environ.get("T22_PROTOCOL", ""),
    ) if p
] + [
    ROOT.parent / "arXiv-v1-polished-extended-preprint"
    / "plan" / "mlst_protocol_2026-09-11-t22-stage0.json",
]


def protocol_path() -> Path:
    for p in PROTOCOL_CANDIDATES:
        if p.is_file():
            return p
    pytest.skip("T22 stage-0 protocol JSON not found (set $T22_PROTOCOL)")


@pytest.fixture(scope="module")
def P():
    return t22.Protocol(protocol_path())


# --- fixtures -----------------------------------------------------------------
W = 1024          # small window with the same parity/structure as 65536
M = W // 2 + 1


def coloured_noise(rng, n, w=W, tilt=2.0):
    """Rows of coloured Gaussian noise with a smooth 1/f^tilt bin profile."""
    k = np.arange(w // 2 + 1)
    prof = 1.0 / (1.0 + k / 8.0) ** tilt + 0.05
    c = (rng.normal(size=(n, w // 2 + 1)) + 1j * rng.normal(size=(n, w // 2 + 1)))
    c *= np.sqrt(prof / 2.0)
    c[:, 0] = c[:, 0].real
    c[:, -1] = c[:, -1].real
    return c, prof


def tone_rows(n, k0, amp, phases, w=W):
    t = np.arange(w, dtype=float)
    return np.array([amp * np.cos(2 * np.pi * k0 * t / w + ph) for ph in phases])


# --- 1. splits and exposure ---------------------------------------------------
def test_allocation_segments_are_disjoint_and_abutting(P):
    w, g = P.w, P.guard
    cal = (0, P.n_cal * w)
    fit = (P.fit_offset, P.fit_offset + P.n_fit * w)
    ev = (P.eval_offset, P.eval_offset + P.n_eval * w)
    assert fit[0] - cal[1] == g
    assert ev[0] - fit[1] == g
    assert ev[1] == P.stretch
    assert cal[1] <= fit[0] < fit[1] <= ev[0] < ev[1]


def test_holdout_stretch_does_not_overlap_the_t10_exposure(P):
    t10_end = (64 + 350 + 150) * P.w + 2 * P.guard      # 36 995 072
    assert t10_end == 36_995_072
    assert P.holdout_start == t10_end + 2 * P.guard
    assert P.holdout_start >= t10_end
    assert P.holdout_start + P.stretch == 83_853_312


def test_holdout_and_development_file_sets_are_disjoint(P):
    dev_files = {f for f, _ in P.dev}
    assert dev_files.isdisjoint(set(P.holdout_files))
    assert len(P.holdout_files) == 8
    assert all(f.startswith("abra_validation_00") for f in P.holdout_files)


def test_validation_and_final_test_partition_the_holdout(P):
    eligible = list(P.holdout_files[:5])                 # any eligible subset
    validation = eligible[:3]
    final_test = [f for f in P.holdout_files if f not in validation]
    assert set(validation).isdisjoint(final_test)
    assert set(validation) | set(final_test) == set(P.holdout_files)


# --- 2. Fourier and amplitude conventions -------------------------------------
def test_unitary_rfft_bin_of_an_aligned_tone():
    k0, amp, ph = 37, 0.75, 0.4
    x = tone_rows(1, k0, amp, [ph])
    c = te.unitary_rfft(x)[0]
    assert np.isclose(abs(c[k0]), amp * np.sqrt(W) / 2, rtol=1e-12)
    assert np.isclose(np.angle(c[k0]), ph, atol=1e-12)


def test_physical_amplitude_is_window_length_independent():
    k0, amp = 11, 0.3
    for w in (256, 1024, 4096):
        t = np.arange(w, dtype=float)
        x = amp * np.cos(2 * np.pi * k0 * t / w)[None, :]
        c = te.unitary_rfft(x)[0, k0]
        assert np.isclose(abs(te.to_physical_amplitude(c, w)), amp, rtol=1e-12)


def test_band_view_is_an_isometry_on_the_band():
    rng = np.random.default_rng(0)
    C, _ = coloured_noise(rng, 5)
    bins = tb.band_bins(20, 5, M)
    V = tb.to_band_view(C, bins)
    assert np.allclose(np.sum(V ** 2, axis=1),
                       2.0 * np.sum(np.abs(C[:, bins]) ** 2, axis=1), rtol=1e-12)
    assert tb.band_view_matches_full_view(C, bins)


def test_band_weights_are_the_whitening_metric():
    rng = np.random.default_rng(1)
    C, prof = coloured_noise(rng, 7)
    bins = tb.band_bins(15, 3, M)
    V = tb.to_band_view(C, bins)
    wts = tb.band_view_weights(prof, bins)
    lhs = np.sum(wts * V ** 2, axis=1)
    rhs = 2.0 * np.sum(np.abs(C[:, bins]) ** 2 / prof[bins], axis=1)
    assert np.allclose(lhs, rhs, rtol=1e-12)


def test_band_bins_reject_dc_and_nyquist():
    assert tb.band_bins(3, 5, M).tolist() == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        tb.band_bins(2, 5, M)            # would reach bin 0
    with pytest.raises(ValueError):
        tb.band_bins(M - 2, 5, M)        # would reach Nyquist
    with pytest.raises(ValueError):
        tb.band_bins(20, 4, M)           # even width


# --- 3. estimators: identities and identical preprocessing --------------------
def _cell(rng, B, k0=20, n_fit=120, n_eval=80, amp=0.6, tilt=2.0):
    C_fit, prof = coloured_noise(rng, n_fit, tilt=tilt)
    C_eval, _ = coloured_noise(rng, n_eval, tilt=tilt)
    ph = rng.uniform(0, 2 * np.pi, size=n_fit)
    C_fit = C_fit + te.unitary_rfft(tone_rows(n_fit, k0, amp, ph))
    ph = rng.uniform(0, 2 * np.pi, size=n_eval)
    C_eval = C_eval + te.unitary_rfft(tone_rows(n_eval, k0, amp, ph))
    cal = te.fit_transfer(C_eval[:, k0], C_eval[:, k0] * 0.5, float(prof[k0]))
    est, diag = tb.compute_band_estimates(C_eval, C_fit, k0, B, W, prof, cal)
    return est, diag, prof, C_fit, C_eval


def test_single_bin_identity_control_holds():
    est, diag, *_ = _cell(np.random.default_rng(2), B=1)
    rec = tb.assert_single_bin_identity(diag, "synthetic", 1e-12)
    assert rec["passed"] and rec["max_relative_difference"] <= 1e-12
    assert np.allclose(est["pca"], est["raw-bin"], rtol=0, atol=1e-12 * np.abs(est["raw-bin"]).max())
    assert np.allclose(est["empca"], est["raw-bin"], rtol=0, atol=1e-12 * np.abs(est["raw-bin"]).max())


def test_single_bin_identity_control_fails_loudly_when_broken():
    _, diag, *_ = _cell(np.random.default_rng(3), B=1)
    broken = dict(diag, pca_vs_rawbin_max_rel_diff=1e-6)
    with pytest.raises(AssertionError):
        tb.assert_single_bin_identity(broken, "broken", 1e-12)


def test_quadrature_of_reproduces_the_bin_read():
    _, diag, *_ = _cell(np.random.default_rng(4), B=5)
    assert diag["of_vs_rawbin_max_rel_diff"] < 1e-10


def test_pca_and_empca_receive_identical_inputs_and_differ_only_by_metric():
    rng = np.random.default_rng(5)
    # White noise across the band: the metric is proportional to the identity,
    # so the two fits must coincide.
    n_fit = 120
    C_fit = (rng.normal(size=(n_fit, M)) + 1j * rng.normal(size=(n_fit, M))) / np.sqrt(2)
    C_fit[:, 0] = C_fit[:, 0].real; C_fit[:, -1] = C_fit[:, -1].real
    flat = np.ones(M)
    bins = tb.band_bins(20, 5, M)
    fits_flat, _ = tb.fit_band_subspaces(C_fit, bins, flat)
    # fit_weighted_pca returns a basis orthonormal in its own metric, so compare
    # the SUBSPACES: orthonormalise both row spaces and take principal angles.
    def ortho(comp):
        q, _ = np.linalg.qr(np.asarray(comp).T)
        return q.T
    overlap = np.linalg.svd(ortho(fits_flat["pca"].components)
                            @ ortho(fits_flat["empca"].components).T, compute_uv=False)
    assert np.allclose(overlap, 1.0, atol=1e-8), "flat metric must give the same subspace"

    # A strongly coloured band separates them.
    prof = np.linspace(1.0, 200.0, M)
    fits_col, _ = tb.fit_band_subspaces(C_fit, bins, prof)
    overlap_col = np.linalg.svd(ortho(fits_col["pca"].components)
                                @ ortho(fits_col["empca"].components).T, compute_uv=False)
    assert overlap_col.max() <= 1.0 + 1e-8
    assert overlap_col.min() < 0.999, "coloured metric must move the subspace"

    # The two fits are given the same windows, the same bins and the same rank.
    assert fits_flat["pca"].components.shape == fits_flat["empca"].components.shape
    assert fits_flat["pca"].components.shape == (te.RANK, 2 * bins.size)


def test_band_dimension_shrinks_with_b():
    for B in (1, 3, 5, 9, 17):
        _, diag, *_ = _cell(np.random.default_rng(6), B=B)
        assert diag["band_real_dimension"] == 2 * B
        assert diag["fit_windows"] == 120


# --- 4. calibration-only inputs ----------------------------------------------
def test_injection_bin_comes_from_the_calibration_reference_only():
    """A much stronger line in the evaluation block must not move ``k_inj``."""
    rng = np.random.default_rng(7)
    n_cal = 64
    C_cal_ref, prof = coloured_noise(rng, n_cal)
    C_cal_ref = C_cal_ref + te.unitary_rfft(
        tone_rows(n_cal, 40, 50.0, rng.uniform(0, 2 * np.pi, n_cal)))
    C_cal_read = 0.3 * C_cal_ref + coloured_noise(rng, n_cal)[0] * 1e-3
    ch = te.characterise(C_cal_ref, C_cal_read, W, prof, 46_825_472, 300,
                         search_lo_hz=0.0)
    assert ch["k_inj"] == 40
    # The same calibration block with an even stronger, different evaluation
    # line is irrelevant: characterise() is never given evaluation windows.
    assert set(te.characterise.__code__.co_varnames[:6]) == {
        "C_cal_ref", "C_cal_read", "w", "N_read_bins", "stretch_samples",
        "n_eval_available"}


def test_second_line_rule_passes_a_leakage_skirt_and_catches_a_real_line():
    """Amendment A4: a decaying skirt is not a second line; a local maximum is."""
    rng = np.random.default_rng(8)
    n, k0 = 64, 30
    t = np.arange(W, dtype=float)
    # (1) Pure single tone, deliberately off-grid, plus frequency jitter across
    # windows so the measured skirt sits well above the ideal boxcar one -- the
    # situation that defeated the pre-A4 rule on real TIDMAD stretches.
    x = np.array([np.cos(2 * np.pi * (k0 + 0.37 + 0.01 * rng.normal()) * t / W + p)
                  for p in rng.uniform(0, 2 * np.pi, n)])
    C = te.unitary_rfft(x)
    bins = tb.band_bins(k0, 9, M)
    sl = tb.second_line_excess(C, bins, k0, 0.37, W)
    assert not sl["has_second_line"], sl["second_lines"]
    assert sl["profile_decays_outward_left"] and sl["profile_decays_outward_right"]
    # The skirt does exceed the idealised Dirichlet prediction: that is exactly
    # why the pre-A4 test rejected real bands.
    assert sl["max_dirichlet_excess_off_centre"] > 0.0

    # (2) A genuine second tone three bins away at ~30 % of the main line.
    x2 = x + np.array([0.55 * np.cos(2 * np.pi * (k0 + 3) * t / W + p)
                       for p in rng.uniform(0, 2 * np.pi, n)])
    sl2 = tb.second_line_excess(te.unitary_rfft(x2), bins, k0, 0.37, W)
    assert sl2["has_second_line"]
    assert [h["bin"] for h in sl2["second_lines"]] == [k0 + 3]

    # (3) The guard: a strong immediate neighbour is never a second line.
    sl3 = tb.second_line_excess(C, tb.band_bins(k0, 3, M), k0, 0.37, W)
    assert not sl3["has_second_line"]


# --- 5. bootstrap logic -------------------------------------------------------
def test_joint_bootstrap_refits_the_gain_and_holds_evaluation_fixed_on_demand():
    rng = np.random.default_rng(9)
    n_e, n_c = 80, 64
    rho_e = rng.normal(size=n_e) + 1j * rng.normal(size=n_e)
    rho_c = rng.normal(size=n_c) + 1j * rng.normal(size=n_c)
    g = 0.4 + 0.2j
    c_e = g * rho_e + 0.01 * (rng.normal(size=n_e) + 1j * rng.normal(size=n_e))
    c_c = g * rho_c + 0.01 * (rng.normal(size=n_c) + 1j * rng.normal(size=n_c))
    est = {"zero": np.zeros(n_e, dtype=complex), "raw-bin": c_e.copy(),
           "calibration-mean": np.full(n_e, complex(np.mean(c_c))),
           "shrinkage-cal": c_e.copy()}
    kw = dict(replicates=200, seed=1234, b_eval=4, b_cal=2)
    full = te.joint_bootstrap(est, c_e, rho_e, c_c, rho_c, 1e-4, W,
                              [("zero", "raw-bin")], **kw)[("zero", "raw-bin")]
    calo = te.joint_bootstrap(est, c_e, rho_e, c_c, rho_c, 1e-4, W,
                              [("zero", "raw-bin")], calibration_only=True,
                              **kw)[("zero", "raw-bin")]
    assert full.std() > calo.std(), "holding evaluation fixed must shrink the spread"
    again = te.joint_bootstrap(est, c_e, rho_e, c_c, rho_c, 1e-4, W,
                               [("zero", "raw-bin")], **kw)[("zero", "raw-bin")]
    assert np.array_equal(full, again), "the bootstrap must be seed-deterministic"


def test_circular_block_indices_respect_the_block_length():
    rng = np.random.default_rng(11)
    idx = te.circular_block_indices(rng, 64, 8)
    assert idx.size == 64
    runs = np.split(idx, 8)
    for r in runs:
        assert np.array_equal(r, (r[0] + np.arange(8)) % 64)


def test_paired_band_bootstrap_pairs_candidates(P):
    """Identical estimates across candidates must give identical replicate values."""
    rng = np.random.default_rng(12)
    n_e, n_c = 60, 32
    st = {"eval_read": np.zeros((n_e, M), dtype=complex),
          "eval_ref": np.zeros((n_e, M), dtype=complex),
          "cal_read": np.zeros((n_c, M), dtype=complex),
          "cal_ref": np.zeros((n_c, M), dtype=complex)}
    k = 20
    st["eval_ref"][:, k] = rng.normal(size=n_e) + 1j * rng.normal(size=n_e)
    st["cal_ref"][:, k] = rng.normal(size=n_c) + 1j * rng.normal(size=n_c)
    st["eval_read"][:, k] = 0.5 * st["eval_ref"][:, k]
    st["cal_read"][:, k] = 0.5 * st["cal_ref"][:, k]
    shared = {"raw-bin": st["eval_read"][:, k].copy(),
              "empca": 0.9 * st["eval_read"][:, k]}
    cells = {"f": {"k_inj": k, "estimates": {3: shared, 5: shared, 9: dict(shared)}}}
    N = np.ones(M)
    out = t22.paired_band_bootstrap(cells, {"f": st}, N, P, bands=(3, 5, 9),
                                    b_eval=4, b_cal=2, replicates=50, seed=7)
    assert np.array_equal(out[3], out[5])
    assert np.array_equal(out[3], out[9])


# --- 6. decision rules --------------------------------------------------------
class _P:
    """Minimal stand-in carrying only what choose_band reads."""
    min_band = 3


def _dev(vals):
    return {f"s{i}": {"enters_selection": True,
                      "cells": {str(B): {"contrasts": {"raw-bin|empca":
                                                       {"mean_delta_squared": v}}}
                                for B, v in row.items()}}
            for i, row in enumerate(vals)}


def _adm(bands, ok=True):
    return {"s0": {str(B): {"admissible": ok, "reasons": [] if ok else ["blocked"],
                            "second_line": {}} for B in bands}}


def test_band_selection_takes_the_smallest_within_one_sd(P):
    dev = _dev([{3: 1.0, 5: 1.8, 9: 2.0, 17: 1.2}])
    boot = {B: np.full(400, 0.0) for B in (3, 5, 9, 17)}
    boot[9] = np.random.default_rng(0).normal(2.0, 0.3, 400)
    sel = t22.choose_band(dev, _adm((3, 5, 9, 17)), boot, P, bands=(3, 5, 9, 17))
    # best is 9; sd(9) ~ 0.3, so the threshold is ~1.7 and the smallest candidate
    # above it is 5.
    assert sel["best_band"] == 9
    assert sel["selected_band"] == 5


def test_band_selection_falls_back_to_the_best_when_the_sd_is_tiny(P):
    dev = _dev([{3: 1.0, 5: 1.8, 9: 2.0, 17: 1.2}])
    boot = {B: np.random.default_rng(1).normal(0.0, 1e-6, 400) for B in (3, 5, 9, 17)}
    sel = t22.choose_band(dev, _adm((3, 5, 9, 17)), boot, P, bands=(3, 5, 9, 17))
    assert sel["selected_band"] == 9


def test_band_selection_reports_blocked_when_nothing_is_admissible(P):
    dev = _dev([{3: 1.0, 5: 1.0}])
    boot = {B: np.zeros(10) for B in (3, 5)}
    sel = t22.choose_band(dev, _adm((3, 5), ok=False), boot, P, bands=(3, 5))
    assert sel["selected_band"] is None
    assert sel["disposition"] == "blocked-no-admissible-band"


def test_band_selection_reports_blocked_without_an_eligible_stretch(P):
    sel = t22.choose_band({"s0": {"enters_selection": False}}, {}, {}, P, bands=(3,))
    assert sel["disposition"] == "blocked-no-eligible-development-stretch"


def test_holm_step_down():
    adj = t22.holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert np.isclose(adj["a"], 0.03)
    assert np.isclose(adj["b"], 0.04)
    assert np.isclose(adj["c"], 0.04)
    assert adj["a"] <= adj["b"] <= adj["c"], "Holm output must be monotone"


def test_hierarchical_bootstrap_carries_within_file_uncertainty():
    rng = np.random.default_rng(13)
    tight = [rng.normal(0.01, 1e-4, 2000) for _ in range(6)]
    loose = [rng.normal(0.01, 5e-2, 2000) for _ in range(6)]
    a = t22.hierarchical_file_bootstrap(tight, replicates=2000, seed=3)
    b = t22.hierarchical_file_bootstrap(loose, replicates=2000, seed=3)
    assert b["bootstrap_sd"] > a["bootstrap_sd"]
    assert a["excludes_zero"] and not b["excludes_zero"]
    assert 0.0 <= a["p_two_sided"] <= 1.0


def test_hierarchical_bootstrap_is_seed_deterministic():
    d = [np.linspace(-1, 1, 100) for _ in range(4)]
    x = t22.hierarchical_file_bootstrap(d, replicates=500, seed=5)
    y = t22.hierarchical_file_bootstrap(d, replicates=500, seed=5)
    assert x == y


def test_empty_combination_is_reported_not_crashed():
    out = t22.hierarchical_file_bootstrap([], replicates=100, seed=1)
    assert out["n_files"] == 0


# --- 7. protocol and budget ---------------------------------------------------
def test_protocol_is_internally_consistent(P):
    assert P.rank == 2 and P.w == 65536
    assert all(B % 2 == 1 for B in P.bands)
    assert P.identity_band == 1 and 1 not in P.selection_bands
    declared = set(P.methods)
    for a, b in P.contrasts:
        assert a in declared and b in declared
    assert {"zero", "calibration-mean", "raw-bin", "pca", "empca"} <= declared


def test_declared_budget_matches_the_enumeration(P):
    b = P.raw["budget"]
    total = (b["development_full_grid"]["fits"] + b["development_identity_only"]["fits"]
             + b["validation"]["fits"] + b["final_test"]["fits"])
    assert total == b["nominal_total"] == 240
    assert total <= P.cap == 250
    assert b["development_full_grid"]["fits"] == 4 * len(P.bands) * len(P.methods)
    assert b["development_identity_only"]["fits"] == 2 * 1 * len(P.methods)


def test_worst_case_development_enumeration_stays_inside_the_cap(P):
    worst = (P.dev_full_grid_max * len(P.bands)
             + (len(P.dev) - P.dev_full_grid_max) * 1) * len(P.methods)
    holdout = len(P.holdout_files) * len(P.methods)
    assert worst + holdout <= P.cap


def test_protocol_json_round_trips(P):
    assert json.loads(P.path.read_text())["schema"] == "mlst-t22-stage0/1"
    assert len(P.sha256()) == 64
