"""T22 implementation tests: conventions, controls, selection rule and inference.

These are meaningful checks of the things that could silently invalidate the
experiment -- split overlap, calibration-only inputs, the Fourier and
physical-amplitude conventions, identical PCA/EMPCA preprocessing, the baseline
identities, the paired bootstrap and the declared decision rules -- run on
synthetic fixtures only.  No instrument file is opened.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import types
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
    kw = dict(replicates=500, seed=5, point_estimates=[0.1, 0.2, 0.3, 0.4])
    x = t22.hierarchical_file_bootstrap(d, **kw)
    y = t22.hierarchical_file_bootstrap(d, **kw)
    assert x == y


def test_hierarchical_bootstrap_reports_the_observed_estimand_not_the_bootstrap_mean():
    """The two point summaries are different quantities and must not be conflated."""
    rng = np.random.default_rng(21)
    # Each file's draw array is deliberately offset from its observed value.
    draws = [rng.normal(0.5, 0.01, 400), rng.normal(0.7, 0.01, 400)]
    observed = [0.10, 0.20]
    out = t22.hierarchical_file_bootstrap(draws, replicates=400, seed=2,
                                          point_estimates=observed)
    assert np.isclose(out["equal_file_mean_of_point_estimates"], 0.15)
    assert np.isclose(out["equal_file_mean_of_bootstrap_means"], 0.6, atol=0.01)
    assert out["equal_file_mean_of_point_estimates"] != out["equal_file_mean_of_bootstrap_means"]
    # Without point estimates the estimand field is explicitly NaN, never a
    # silently substituted bootstrap quantity.
    bare = t22.hierarchical_file_bootstrap(draws, replicates=50, seed=2)
    assert np.isnan(bare["equal_file_mean_of_point_estimates"])


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


# --- 8. enforcement: freeze binding, window counts, cumulative budget ---------
# Every test below is synthetic.  Entry-point tests replace every route to
# instrument data with one that raises AssertionError("data read"), so a
# SystemExit proves the refusal happened before any data file was touched.
#: The real stage function, kept before ``no_data`` replaces the module attribute,
#: for the tests that exercise its own gates with every data route still raising.
_STAGE_EVALUATE = t22.stage_evaluate


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _patch_h5py(monkeypatch, file_fn):
    try:
        import h5py
        monkeypatch.setattr(h5py, "File", file_fn)
    except ImportError:  # pragma: no cover - h5py is a declared dependency
        monkeypatch.setitem(sys.modules, "h5py", types.SimpleNamespace(File=file_fn))


def _write_chars(path: Path, P, *, holdout=None) -> Path:
    """Synthetic characterisation: every stretch eligible-coupled except hold-out
    0016 (chirped), as in the archived run; ``holdout`` overrides outcomes."""
    holdout = holdout or {}
    chars = {}
    for f, s in P.dev:
        chars[f] = {"file": f, "role": "development", "start_sample": s, "k_inj": 1000,
                    "outcome": "eligible-coupled", "reasons": [],
                    "readout_unit_target_supported": True}
    for f in P.holdout_files:
        chars[f] = {"file": f, "role": "holdout", "start_sample": P.holdout_start,
                    "k_inj": 1000, "reasons": [], "readout_unit_target_supported": True,
                    "outcome": holdout.get(f, "chirped" if f.endswith("0016.h5")
                                           else "eligible-coupled")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": "characterize", "characterisation": chars}, indent=2))
    return path


def _write_freeze(path: Path, P, chars_path: Path) -> Path:
    """A freeze assembled the way the freeze stage assembles one."""
    chars = json.loads(chars_path.read_text())["characterisation"]
    eligible = [f for f in P.holdout_files if chars[f]["outcome"] == "eligible-coupled"]
    validation = eligible[:3]
    frozen = {
        "selected_band": 9, "disposition": "selected", "rank": P.rank, "window_samples": P.w,
        "methods": list(P.methods), "contrasts": [list(c) for c in P.contrasts],
        "block_length_evaluation": 32, "block_length_calibration": 8,
        "dependence_verdict": "inconclusive",
        "bootstrap_replicates": P.boot_replicates, "bootstrap_seed": P.boot_seed,
        "file_level_replicates": P.file_replicates, "file_level_seed": P.file_seed,
        "holdout_start_sample": P.holdout_start, "validation_files": validation,
        "final_test_files": [f for f in P.holdout_files if f not in validation],
        "n_eligible_coupled_holdout_files": len(eligible), "eligible_holdout_files": eligible,
        "validation_status": "complete", "positive_verdict_permitted": True,
        "development_fit_count": t22.plan_development(P, chars)["planned"],
        "stage0_protocol": P.raw,
        "input_hashes": {f"protocol:{P.path.name}": P.sha256(),
                         f"characterisation:{chars_path.name}": _sha(chars_path)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": "freeze", "frozen": frozen}, indent=2))
    return path


def _protocol_variant(tmp_path: Path, P, name: str, edit) -> Path:
    raw = json.loads(P.path.read_text())
    edit(raw)
    out = tmp_path / name
    out.write_text(json.dumps(raw, indent=2))
    return out


def _write_stage_output(directory: Path, name: str, payload: dict) -> Path:
    """Write a stage JSON and a RESULTS_SHA256 covering every file in its directory."""
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (directory / "RESULTS_SHA256").write_text("".join(
        f"{_sha(f)}  {f.name}\n" for f in sorted(directory.iterdir())
        if f.name != "RESULTS_SHA256"))
    return p


def _write_ledger(path: Path, P, rows) -> None:
    entries = [{"stage": s, "output_dir": f"T22_{s}_v{i}", "output_file": f"t22_{s}.json",
                "output_sha256": "0" * 64, "fits_executed": n, "smoke": smoke,
                "git_commit": "test", "recorded_utc": "2026-09-11T00:00:00Z"}
               for i, (s, n, smoke) in enumerate(rows)]
    ledger = t22._ledger_totals({"schema": t22.LEDGER_SCHEMA, "hard_cap": P.cap,
                                 "counting_convention": P.raw["budget"]["fit_definition"],
                                 "entries": entries})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2))


@pytest.fixture
def no_data(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("data read")
    for name in ("noise_bin_power", "load_stretch", "stage_evaluate", "score_cell"):
        monkeypatch.setattr(t22, name, boom)
    _patch_h5py(monkeypatch, boom)
    return boom


@pytest.fixture
def bound(tmp_path, P):
    """A matching characterisation and freeze, kept outside the run tree."""
    chars = _write_chars(tmp_path / "inputs" / "chars.json", P)
    freeze = _write_freeze(tmp_path / "inputs" / "freeze.json", P, chars)
    return types.SimpleNamespace(chars=chars, freeze=freeze, runs=tmp_path / "runs",
                                 data=tmp_path / "no-data")


def _run(stage, protocol, b, out_name, *extra):
    return t22.main(["--stage", stage, "--protocol", str(protocol),
                     "--data-root", str(b.data), "--out", str(b.runs / out_name),
                     "--freeze-file", str(b.freeze), "--characterisation", str(b.chars),
                     *extra])


def _frozen_for_scoring():
    return {"selected_band": 9, "block_length_evaluation": 32,
            "block_length_calibration": 8, "dependence_verdict": "inconclusive"}


def _stretch_arrays(P, *, n_cal=None, n_fit=None, n_eval=None, m=4):
    z = lambda n: np.zeros((n, m), dtype=complex)  # noqa: E731
    n_cal = P.n_cal if n_cal is None else n_cal
    n_fit = P.n_fit if n_fit is None else n_fit
    n_eval = P.n_eval if n_eval is None else n_eval
    return {"cal_read": z(n_cal), "cal_ref": z(n_cal), "fit_read": z(n_fit),
            "eval_read": z(n_eval), "eval_ref": z(n_eval)}


# (a) the review's demonstration: a changed file-level bootstrap seed
def test_holdout_stage_refuses_a_changed_file_level_seed_before_reading_data(
        tmp_path, P, bound, no_data):
    def edit(raw):
        raw["inference"]["primary_combined_estimand"]["interval"]["seed"] = 20260912
    proto = _protocol_variant(tmp_path, P, "file_seed.json", edit)
    with pytest.raises(SystemExit) as exc:
        _run("validation", proto, bound, "T22_validation_v1")
    msg = str(exc.value)
    assert "inference/primary_combined_estimand/interval/seed" in msg
    assert "file_level_seed: frozen 20260911, protocol gives 20260912" in msg
    assert "protocol sha256" in msg
    assert not (bound.runs / "T22_validation_v1").exists()


# (b) per-file bootstrap seed or replicate count
@pytest.mark.parametrize("field,value,frozen_key", [("seed", 20260912, "bootstrap_seed"),
                                                     ("replicates", 1999, "bootstrap_replicates")])
def test_holdout_stage_refuses_a_changed_per_file_bootstrap(tmp_path, P, bound, no_data,
                                                            field, value, frozen_key):
    def edit(raw):
        raw["inference"]["per_file"][field] = value
    proto = _protocol_variant(tmp_path, P, f"per_file_{field}.json", edit)
    with pytest.raises(SystemExit) as exc:
        _run("final-test", proto, bound, "T22_final_test_v1")
    assert f"inference/per_file/{field}" in str(exc.value)
    assert f"{frozen_key}: frozen" in str(exc.value)


# (c) a changed allocation that is still a well-formed protocol
def test_holdout_stage_refuses_a_changed_allocation(tmp_path, P, bound, no_data):
    def edit(raw):
        a = raw["allocation"]
        a["n_evaluation_windows"] = 299
        a["stretch_samples"] = (a["evaluation_offset"]
                                + 299 * raw["unchanged_from_t10_t12"]["window_samples"])
    proto = _protocol_variant(tmp_path, P, "allocation.json", edit)
    assert t22.Protocol(proto).n_eval == 299
    with pytest.raises(SystemExit) as exc:
        _run("validation", proto, bound, "T22_validation_v1")
    assert "allocation/n_evaluation_windows" in str(exc.value)
    assert "allocation/stretch_samples" in str(exc.value)


# (d) a characterisation other than the frozen one
def test_holdout_stage_refuses_a_characterisation_other_than_the_frozen_one(P, bound, no_data):
    bound.chars.write_text(bound.chars.read_text() + "\n")   # same content, other bytes
    with pytest.raises(SystemExit) as exc:
        _run("validation", P.path, bound, "T22_validation_v1")
    assert "characterisation sha256" in str(exc.value)


def test_binding_refuses_an_eligible_set_that_differs_from_the_freeze(P, bound):
    # One file cannot both match the digest and differ in content, so emulate a
    # freeze whose recorded digest points at a characterisation with 0013 chirped.
    other = _write_chars(bound.chars.with_name("other.json"), P,
                         holdout={"abra_validation_0013.h5": "chirped"})
    fz = json.loads(bound.freeze.read_text())
    fz["frozen"]["input_hashes"] = {k: v for k, v in fz["frozen"]["input_hashes"].items()
                                    if not k.startswith("characterisation:")}
    fz["frozen"]["input_hashes"]["characterisation:other.json"] = _sha(other)
    bound.freeze.write_text(json.dumps(fz))
    with pytest.raises(SystemExit) as exc:
        t22.bind_to_freeze(P, bound.freeze, other, "validation", [])
    assert "eligible-coupled hold-out files in the characterisation" in str(exc.value)
    assert "abra_validation_0013.h5" in str(exc.value)


# (e) a matching triple binds; scoring is mocked so no data is touched
def test_matching_freeze_binds_and_the_stage_runs_without_touching_data(P, bound, no_data,
                                                                        monkeypatch):
    calls = []

    def fake_evaluate(args, P_, rec, frozen, files, label, chars):
        calls.append((label, list(files)))
        return {label: {"per_file": {}, "files_evaluated": list(files),
                        "fit_count": len(files) * len(P_.methods)}}, {}
    monkeypatch.setattr(t22, "stage_evaluate", fake_evaluate)
    assert _run("validation", P.path, bound, "T22_validation_v1") == 0

    out_file = bound.runs / "T22_validation_v1" / "t22_validation.json"
    out = json.loads(out_file.read_text())
    b = out["freeze_binding"]
    assert b["verdict"] == "bound" and b["amendments_applied"] == []
    assert b["protocol"]["sha256"] == b["protocol"]["frozen_input_sha256"] == P.sha256()
    assert b["protocol"]["canonical_identical"] and b["protocol"]["differing_paths"] == []
    assert b["characterisation"]["sha256"] == b["characterisation"]["frozen_sha256"]
    frozen = json.loads(bound.freeze.read_text())["frozen"]
    assert calls == [("validation", frozen["validation_files"])]
    assert out["budget_preflight"]["planned"] == 24

    ledger = json.loads((bound.runs / t22.LEDGER_NAME).read_text())
    assert ledger["schema"] == t22.LEDGER_SCHEMA and ledger["hard_cap"] == P.cap
    assert ledger["executed_total"] == ledger["unique_scientific_total"] == 24
    (entry,) = ledger["entries"]
    assert entry["stage"] == "validation" and entry["output_dir"] == "T22_validation_v1"
    assert entry["output_sha256"] == _sha(out_file) and entry["smoke"] is False


# (f) the evaluation-window requirement is executable
def test_characterise_uses_the_declared_evaluation_window_requirement(P):
    w, k, n_cal = 4096, 300, 64
    rng = np.random.default_rng(31)
    t = np.arange(w)
    ref = np.array([50.0 * np.cos(2 * np.pi * k * (t + j * w) / w) + 1e-3 * rng.normal(size=w)
                    for j in range(n_cal)])
    read = np.array([50.0 * np.cos(2 * np.pi * k * (t + j * w) / w) + rng.normal(size=w)
                     for j in range(n_cal)])
    Cr, Cx = te.unitary_rfft(ref), te.unitary_rfft(read)
    N = np.ones(w // 2 + 1)
    assert P.n_eval == 300
    strict = te.characterise(Cr, Cx, w, N, n_cal * w, 200, n_eval_required=P.n_eval)
    assert strict["outcome"] == "underpowered"
    assert strict["reasons"] == ["200 < 300 evaluation windows"]
    assert strict["n_evaluation_windows_required"] == 300
    default = te.characterise(Cr, Cx, w, N, n_cal * w, 200)
    assert default["outcome"] == "eligible-coupled"
    assert default["n_evaluation_windows_required"] == te.N_EVAL_REQUIRED == 150


def test_stage_characterize_passes_the_t22_requirement(tmp_path, P, monkeypatch):
    name = P.holdout_files[0]
    raw = tmp_path / "tidmad" / "raw"
    raw.mkdir(parents=True)
    (raw / name).write_bytes(b"placeholder: never parsed, the h5py layer is faked")
    n_total = P.holdout_start + P.eval_offset + 200 * P.w    # 200 evaluation windows exist

    class _Dataset:
        shape = (n_total,)

        def __getitem__(self, s):
            return np.zeros(max(0, min(s.stop, n_total) - s.start))

    _patch_h5py(monkeypatch, lambda path, mode="r": contextlib.nullcontext(
        {t22.SQUID: _Dataset(), t22.REFERENCE: _Dataset()}))
    monkeypatch.setattr(t22, "noise_bin_power", lambda paths, P_: np.ones(P_.w // 2 + 1))
    seen = {}

    def fake_characterise(C_ref, C_read, w, N, stretch, n_avail, **kw):
        seen.update(kw, n_eval_available=n_avail)
        required = kw.get("n_eval_required", te.N_EVAL_REQUIRED)
        return {"outcome": "underpowered" if n_avail < required else "eligible-coupled",
                "reasons": [], "f_inj_hz": 0.0, "k_inj": 1, "reference_line_snr": 0.0,
                "readout_line_snr": 0.0, "drift_bins_over_stretch": 0.0,
                "subbin_offset_delta": 0.0}
    monkeypatch.setattr(te, "characterise", fake_characterise)
    rec = t22.RunRecorder(run_id="test", protocol_refs={})
    out = t22.stage_characterize(types.SimpleNamespace(data_root=str(tmp_path)), P, rec)
    assert seen["n_eval_required"] == P.n_eval == 300
    assert seen["n_eval_available"] == 200
    assert out["characterisation"][name]["outcome"] == "underpowered"


# (g) a short stretch never reaches the scorer
def test_a_short_stretch_is_refused_before_scoring(tmp_path, P, no_data, monkeypatch):
    chars = json.loads(_write_chars(tmp_path / "chars.json", P).read_text())["characterisation"]
    monkeypatch.setattr(t22, "noise_bin_power", lambda paths, P_: np.ones(4))
    monkeypatch.setattr(t22, "band_admissibility", no_data)
    args = types.SimpleNamespace(data_root=str(tmp_path), smoke=False)
    rec = t22.RunRecorder(run_id="test", protocol_refs={})

    name = P.holdout_files[0]
    monkeypatch.setattr(t22, "load_stretch",
                        lambda path, start, P_, **kw: _stretch_arrays(P_, n_eval=P_.n_eval - 1))
    with pytest.raises(SystemExit) as exc:
        _STAGE_EVALUATE(args, P, rec, _frozen_for_scoring(), [name], "validation", chars)
    assert name in str(exc.value) and "eval_read has 299 windows" in str(exc.value)

    dev_name = P.dev[0][0]
    monkeypatch.setattr(t22, "load_stretch",
                        lambda path, start, P_, **kw: _stretch_arrays(P_, n_fit=P_.n_fit - 1))
    with pytest.raises(SystemExit) as exc:
        t22.stage_development(args, P, rec, chars)
    assert dev_name in str(exc.value) and "fit_read has 349 windows" in str(exc.value)


def test_an_underpowered_stretch_is_excluded_without_being_loaded(tmp_path, P, no_data,
                                                                   monkeypatch):
    files = list(P.holdout_files[:3])
    chars = json.loads(_write_chars(tmp_path / "chars.json", P,
                                    holdout={f: "underpowered" for f in files}
                                    ).read_text())["characterisation"]
    monkeypatch.setattr(t22, "noise_bin_power", lambda paths, P_: np.ones(4))
    args = types.SimpleNamespace(data_root=str(tmp_path), smoke=False)
    rec = t22.RunRecorder(run_id="test", protocol_refs={})
    payload, draws = _STAGE_EVALUATE(args, P, rec, _frozen_for_scoring(), files,
                                     "validation", chars)
    per = payload["validation"]["per_file"]
    assert payload["validation"]["fit_count"] == 0 and draws == {}
    assert {r["file_verdict"]["verdict"] for r in per.values()} == {"excluded-underpowered"}
    assert not any(r["in_combined_estimate"] for r in per.values())


# (h) the cap is cumulative
def test_cumulative_ledger_refuses_a_stage_that_would_cross_the_cap(P, bound, no_data,
                                                                    monkeypatch):
    ledger = bound.runs / t22.LEDGER_NAME
    _write_ledger(ledger, P, [("development", 176, False), ("validation", 54, False)])
    with pytest.raises(SystemExit) as exc:
        _run("validation", P.path, bound, "T22_validation_v2")
    msg = str(exc.value)
    assert "cumulative executed fits 230" in msg and "planned 24" in msg
    assert "hard cap 250" in msg

    # Smoke runs are counted separately and do not consume the cap.
    ledger.unlink()
    _write_ledger(ledger, P, [("development", 176, False), ("development", 500, True)])
    monkeypatch.setattr(t22, "stage_evaluate", lambda args, P_, rec, fz, files, label, ch: (
        {label: {"per_file": {}, "files_evaluated": list(files), "fit_count": 24}}, {}))
    assert _run("validation", P.path, bound, "T22_validation_v2") == 0
    after = json.loads(ledger.read_text())
    assert after["executed_total"] == 200 and after["smoke_executed_total"] == 500


def test_archive_scan_refuses_when_the_ledger_is_missing(tmp_path, P, bound, no_data):
    _write_stage_output(bound.runs / "T22_development_v1", "t22_development.json",
                        {"stage": "development", "smoke": False, "fit_count": 176})
    _write_stage_output(bound.runs / "T22_validation_v1", "t22_validation.json",
                        {"stage": "validation", "smoke": False, "validation": {"fit_count": 60}})
    assert not (bound.runs / t22.LEDGER_NAME).exists()
    with pytest.raises(SystemExit) as exc:
        _run("final-test", P.path, bound, "T22_final_test_v1")
    assert "cumulative executed fits 236" in str(exc.value) and "planned 40" in str(exc.value)

    # Pointing --ledger somewhere empty does not hide the outputs beside --out.
    with pytest.raises(SystemExit) as exc:
        _run("final-test", P.path, bound, "T22_final_test_v1",
             "--ledger", str(tmp_path / "elsewhere" / t22.LEDGER_NAME))
    assert "cumulative executed fits 236" in str(exc.value)

    # Under the cap, a missing ledger beside existing outputs is still refused.
    (bound.runs / "T22_validation_v1" / "t22_validation.json").unlink()
    with pytest.raises(SystemExit) as exc:
        _run("validation", P.path, bound, "T22_validation_v2")
    assert "audit-budget" in str(exc.value)


# (i) recombine: registered amendments instead of disabled checks
def _amendment(path: Path, frozen_sha: str, amended: Path, changed_paths) -> Path:
    path.write_text(json.dumps({
        "schema": "mlst-t22-amendment/1", "id": "A5", "reason": "synthetic test",
        "recorded_utc": "2026-09-12T00:00:00Z", "applies_to_stages": ["recombine"],
        "frozen_protocol_sha256": frozen_sha, "amended_protocol_sha256": _sha(amended),
        "changed_paths": list(changed_paths)}))
    return path


@pytest.fixture
def archived_validation(P, bound):
    """A validation output produced under ``bound``'s freeze, its draws, its
    manifest, and the ledger that audit-budget reconstructs for it."""
    frozen = json.loads(bound.freeze.read_text())["frozen"]
    d = bound.runs / "T22_validation_v1"
    d.mkdir(parents=True)
    np.savez_compressed(d / "t22_validation_draws.npz", placeholder=np.zeros(3))
    files = frozen["validation_files"]
    payload = {"stage": "validation", "smoke": False,
               "validation": {"per_file": {f: {"file": f, "in_combined_estimate": False}
                                           for f in files},
                              "files_evaluated": files, "fit_count": 24},
               "provenance": {"inputs": [
                   {"role": "freeze", "path": str(bound.freeze), "sha256": _sha(bound.freeze)},
                   {"role": "characterisation", "path": str(bound.chars),
                    "sha256": _sha(bound.chars)}],
                   "git": {"commit": "test", "tree_dirty": False},
                   "finished_utc": "2026-09-11T16:24:15Z"}}
    out = _write_stage_output(d, "t22_validation.json", payload)
    assert t22.main(["--stage", "audit-budget", "--protocol", str(P.path),
                     "--results-root", str(bound.runs)]) == 0
    return types.SimpleNamespace(json=out, draws=d / "t22_validation_draws.npz")


def _recombine(protocol, b, av, out_name, *extra):
    return t22.main(["--stage", "recombine", "--protocol", str(protocol),
                     "--data-root", str(b.data), "--out", str(b.runs / out_name),
                     "--freeze-file", str(b.freeze), "--stage-output", str(av.json),
                     "--draws", str(av.draws), *extra])


def test_recombine_admits_a_protocol_difference_only_through_a_registered_amendment(
        tmp_path, P, bound, archived_validation, no_data):
    av = archived_validation
    n = len(P.raw["amendments"])

    def add_amendment(raw):
        raw["amendments"].append({"id": "A5", "date": "2026-09-12", "what": "post-processing"})
    proto = _protocol_variant(tmp_path, P, "with_a5.json", add_amendment)

    with pytest.raises(SystemExit) as exc:                      # unregistered
        _recombine(proto, bound, av, "T22_validation_v2")
    assert f"amendments/{n}" in str(exc.value)

    am = _amendment(tmp_path / "a5.json", P.sha256(), proto, [f"amendments/{n}"])
    assert _recombine(proto, bound, av, "T22_validation_v2", "--amendment", str(am)) == 0
    out = json.loads((bound.runs / "T22_validation_v2" / "t22_recombine.json").read_text())
    b = out["freeze_binding"]
    assert [a["id"] for a in b["amendments_applied"]] == ["A5"]
    assert b["amendments_applied"][0]["changed_paths"] == [f"amendments/{n}"]
    assert b["protocol"]["differing_paths"] == [f"amendments/{n}"]
    ledger = json.loads((bound.runs / t22.LEDGER_NAME).read_text())
    assert ledger["entries"][-1]["stage"] == "recombine"
    assert ledger["entries"][-1]["fits_executed"] == 0 and ledger["executed_total"] == 24

    def restate_note(raw):
        raw["allocation"]["note"] = "restated"
    proto2 = _protocol_variant(tmp_path, P, "allocation_note.json", restate_note)
    am2 = _amendment(tmp_path / "allocation_amendment.json", P.sha256(), proto2,
                     ["allocation/note"])
    with pytest.raises(SystemExit) as exc:                      # never-amendable prefix
        _recombine(proto2, bound, av, "T22_validation_v3", "--amendment", str(am2))
    assert "never-amendable" in str(exc.value)

    with pytest.raises(SystemExit) as exc:                      # hold-out stages: none at all
        _run("validation", proto, bound, "T22_validation_v4", "--amendment", str(am))
    assert "accepts no amendments" in str(exc.value)


def test_recombine_verifies_its_draws_and_the_freeze_it_was_produced_under(
        P, bound, archived_validation, no_data):
    av = archived_validation
    original = av.draws.read_bytes()
    np.savez_compressed(av.draws, placeholder=np.ones(3))
    with pytest.raises(SystemExit) as exc:
        _recombine(P.path, bound, av, "T22_validation_v2")
    assert "draws sha256" in str(exc.value)
    av.draws.write_bytes(original)

    other = bound.freeze.with_name("freeze_copy.json")
    other.write_text(bound.freeze.read_text() + "\n")
    with pytest.raises(SystemExit) as exc:
        t22.bind_to_freeze(P, other, None, "recombine", [],
                           stage_output=av.json, draws=av.draws)
    assert "produced under freeze" in str(exc.value)


# (j) audit-budget on a synthetic archive
def _dev_output(values: dict, finished: str) -> dict:
    return {"stage": "development", "smoke": False, "fit_count": 176, "planned_fit_count": 176,
            "development": {"abra_validation_0010.h5": {"cells": {
                B: {"scored": True, "contrasts": {"raw-bin|empca": dict(zip(t22.RESCORE_FIELDS, v))}}
                for B, v in values.items()}}},
            "provenance": {"git": {"commit": "c", "tree_dirty": True}, "finished_utc": finished}}


def test_audit_budget_marks_a_deterministic_rescore_and_never_overwrites(tmp_path, P):
    root = tmp_path / "t22"
    vals = {"3": [1.0, 0.5, 1.5, 0.25], "9": [2.0, 1.0, 3.0, 0.5]}
    _write_stage_output(root / "T22_characterize_v1", "t22_characterize.json",
                        {"stage": "characterize", "smoke": False,
                         "provenance": {"finished_utc": "2026-09-11T16:12:41Z"}})
    _write_stage_output(root / "T22_development_v1", "t22_development.json",
                        _dev_output(vals, "2026-09-11T16:15:13Z"))
    v2 = _dev_output(vals, "2026-09-11T16:20:22Z")
    v2["band_selection"] = {"selected_band": 9}            # other content may differ
    _write_stage_output(root / "T22_development_v2", "t22_development.json", v2)
    _write_stage_output(root / "T22_validation_v1", "t22_validation.json",
                        {"stage": "validation", "smoke": False, "validation": {"fit_count": 24},
                         "provenance": {"finished_utc": "2026-09-11T16:24:15Z"}})
    _write_stage_output(root / "T22_validation_v2", "t22_recombine.json",
                        {"stage": "validation", "smoke": False,
                         "validation": {"fit_count": 24, "fits_performed": 0},
                         "provenance": {"finished_utc": "2026-09-11T16:35:11Z"}})
    argv = ["--stage", "audit-budget", "--protocol", str(P.path), "--results-root", str(root)]
    assert t22.main(argv) == 0

    ledger = json.loads((root / t22.LEDGER_NAME).read_text())
    assert [e["output_dir"] for e in ledger["entries"]] == [
        "T22_characterize_v1", "T22_development_v1", "T22_development_v2",
        "T22_validation_v1", "T22_validation_v2"]
    assert [e["fits_executed"] for e in ledger["entries"]] == [0, 176, 176, 24, 0]
    by_dir = {e["output_dir"]: e for e in ledger["entries"]}
    assert by_dir["T22_development_v2"]["deterministic_rescore_of"] == \
        "T22_development_v1/t22_development.json"
    assert "deterministic_rescore_of" not in by_dir["T22_development_v1"]
    assert ledger["executed_total"] == 376 and ledger["unique_scientific_total"] == 200
    assert ledger["executed_total_exceeds_cap"] is True
    assert all(e["results_sha256_verified"] for e in ledger["entries"])

    before = (root / t22.LEDGER_NAME).read_bytes()
    with pytest.raises(SystemExit) as exc:
        t22.main(argv)
    assert "refusing to overwrite" in str(exc.value)
    assert (root / t22.LEDGER_NAME).read_bytes() == before


def test_audit_budget_counts_a_differing_rescore_and_refuses_a_broken_manifest(tmp_path, P):
    root = tmp_path / "t22"
    _write_stage_output(root / "T22_development_v1", "t22_development.json",
                        _dev_output({"3": [1.0, 0.5, 1.5, 0.25]}, "2026-09-11T16:15:13Z"))
    _write_stage_output(root / "T22_development_v2", "t22_development.json",
                        _dev_output({"3": [1.0, 0.5, 1.5, 0.2500001]}, "2026-09-11T16:20:22Z"))
    assert t22.main(["--stage", "audit-budget", "--protocol", str(P.path),
                     "--results-root", str(root)]) == 0
    ledger = json.loads((root / t22.LEDGER_NAME).read_text())
    assert ledger["unique_scientific_total"] == ledger["executed_total"] == 352
    assert not any("deterministic_rescore_of" in e for e in ledger["entries"])

    broken = tmp_path / "broken"
    p = _write_stage_output(broken / "T22_validation_v1", "t22_validation.json",
                            {"stage": "validation", "validation": {"fit_count": 24}})
    p.write_text(p.read_text() + " ")
    with pytest.raises(SystemExit) as exc:
        t22.main(["--stage", "audit-budget", "--protocol", str(P.path),
                  "--results-root", str(broken)])
    assert "RESULTS_SHA256" in str(exc.value)
    assert not (broken / t22.LEDGER_NAME).exists()


# Smoke mode is exempt from the cap, but on a hold-out stage it would still score
# the frozen hold-out stretches with the frozen seeds: exposure outside the ledger.
def test_holdout_stages_refuse_smoke_mode_before_reading_data(P, bound, no_data):
    for stage in ("validation", "final-test"):
        out = f"T22_{stage}_smoke"
        with pytest.raises(SystemExit) as exc:
            _run(stage, P.path, bound, out, "--smoke")
        assert "refuses --smoke" in str(exc.value)
        assert not (bound.runs / out).exists()
