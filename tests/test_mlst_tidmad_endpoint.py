"""Tests for the T10-T12 TIDMAD quadrature endpoint (MLST 2026-09-10).

These cover the five items the instruction required to be resolved before an
operating point may be selected: window-length normalisation, the joint
calibration/evaluation bootstrap with shared-gain uncertainty, line SNR in
bin-power units, the plateau/dependence rule, and the no-denoising quadrature
identity with its analytical baselines.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.mlst2026 import tidmad_endpoint as te

FS = te.FS_HZ


def _tone(w, k, amp, phase, fs=FS):
    n = np.arange(w)
    return amp * np.cos(2 * np.pi * k * n / w + phase)


# --- item 1: representation and window-length normalisation -------------------
def test_real_view_roundtrip_and_isometry():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 256))
    C = te.unitary_rfft(x)
    V = te.to_real_view(C)
    np.testing.assert_allclose(te.from_real_view(V, C.shape[1]), C, atol=1e-12)
    # isometry: the real view has the same norm as the time-domain signal
    np.testing.assert_allclose(np.sum(V ** 2, axis=1), np.sum(x ** 2, axis=1), rtol=1e-12)


def test_unitary_coefficient_scales_as_sqrt_window():
    """The premise of item 1: raw |c| grows like sqrt(w) for the same tone."""
    amp, f = 3.0, FS * 8 / 1024
    mags = {}
    for w in (1024, 4096, 16384):
        k = int(round(f * w / FS))
        c = te.unitary_rfft(_tone(w, k, amp, 0.7))
        mags[w] = abs(c[k])
    assert mags[4096] / mags[1024] == pytest.approx(2.0, rel=1e-6)
    assert mags[16384] / mags[1024] == pytest.approx(4.0, rel=1e-6)


def test_physical_amplitude_is_window_length_invariant():
    """After 2q/sqrt(w) the same tone gives the same number at every w."""
    amp, phase, f = 3.0, 0.7, FS * 8 / 1024
    for w in (1024, 4096, 16384):
        k = int(round(f * w / FS))
        c = te.unitary_rfft(_tone(w, k, amp, phase))
        a = te.to_physical_amplitude(c[k], w)
        assert abs(a) == pytest.approx(amp, rel=1e-9)
        assert np.angle(a) == pytest.approx(phase, abs=1e-9)


def test_squared_error_endpoint_is_in_amplitude_units():
    w = 4096
    q = np.array([2.0 + 0.0j]) * np.sqrt(w) / 2
    t = np.array([1.5 + 0.0j]) * np.sqrt(w) / 2
    np.testing.assert_allclose(te.squared_errors(q, t, w), [0.25], rtol=1e-12)


def test_dirichlet_attenuation_matches_direct_measurement():
    w, k, amp = 4096, 300, 1.0
    for delta in (0.0, 0.1, 0.25, 0.5):
        n = np.arange(w)
        x = amp * np.cos(2 * np.pi * (k + delta) * n / w)
        c = te.unitary_rfft(x)
        measured = abs(te.to_physical_amplitude(c[k], w)) / amp
        assert measured == pytest.approx(abs(te.dirichlet_attenuation(delta, w)), rel=0.02)


# --- item 5: no-denoising quadrature identity and baselines -------------------
def test_quadrature_of_equals_raw_bin_under_diagonal_weighting():
    """Bin-aligned tone + diagonal Fourier weights => weighted LS returns the
    raw bin coefficient exactly, whatever the weights are."""
    rng = np.random.default_rng(3)
    w, k = 2048, 137
    m = w // 2 + 1
    X = np.array([_tone(w, k, 2.0, 0.3) + rng.normal(scale=0.5, size=w) for _ in range(20)])
    C = te.unitary_rfft(X)
    for N in (np.ones(m), np.linspace(1.0, 9.0, m), rng.uniform(0.5, 5.0, m)):
        got = te.quadrature_of_bin(C, k, m, w, N)
        np.testing.assert_allclose(got, C[:, k], atol=1e-9, rtol=1e-9)


def test_wiener_and_shrinkage_are_between_zero_and_raw_bin():
    """Both analytical baselines shrink the raw coefficient rather than
    replacing it with zero or the mean."""
    rng = np.random.default_rng(4)
    w, k = 1024, 61
    m = w // 2 + 1
    Xf = np.array([_tone(w, k, 2.0, 0.2) + rng.normal(scale=1.0, size=w) for _ in range(60)])
    Xe = np.array([_tone(w, k, 2.0, 0.2) + rng.normal(scale=1.0, size=w) for _ in range(30)])
    N = np.full(m, 1.0)
    rho = rng.normal(size=32) + 1j * rng.normal(size=32)
    cal = te.fit_transfer(2.0 * rho + 0.01 * rng.normal(size=32), rho, 1.0)
    est, diag = te.compute_estimates(te.unitary_rfft(Xe), te.unitary_rfft(Xf), k, w, N, cal)
    assert 0.0 < diag["wiener_bin_gain"] < 1.0
    assert 0.0 <= diag["shrinkage_lambda"] <= 1.0
    assert diag["of_vs_rawbin_max_rel_diff"] < 1e-9
    for name in te.METHODS:
        assert name in est and est[name].shape == (30,)


# --- item 3: line SNR in bin-power units --------------------------------------
def test_line_snr_recovers_a_constructed_bin_power_ratio():
    rng = np.random.default_rng(5)
    w, k, sigma, amp = 8192, 500, 1.0, 0.4
    X = np.array([_tone(w, k, amp, rng.uniform(0, 2 * np.pi)) + rng.normal(scale=sigma, size=w)
                  for _ in range(64)])
    C = te.unitary_rfft(X)
    # expected: line bin power (amp*sqrt(w)/2)^2 / w ... in unitary units
    # |c_line| = amp*sqrt(w)/2 -> power = amp^2 w / 4 ; noise bin power = sigma^2
    expected = (amp ** 2 * w / 4.0) / sigma ** 2
    got = te.line_snr(C, k)
    assert got == pytest.approx(expected, rel=0.05)


def test_bin_power_from_psd_matches_white_noise_bin_power():
    rng = np.random.default_rng(6)
    w, sigma = 16384, 2.0
    C = te.unitary_rfft(rng.normal(scale=sigma, size=(200, w)))
    measured = te.mean_bin_power(C)[100:2000].mean()
    psd_onesided = 2.0 * sigma ** 2 / FS
    assert measured == pytest.approx(te.bin_power_from_psd(psd_onesided), rel=0.05)


# --- item 2: transfer calibration and shared-gain uncertainty ----------------
def test_fit_transfer_recovers_a_known_complex_gain():
    rng = np.random.default_rng(7)
    g = 1.7 * np.exp(1j * 0.9)
    rho = rng.normal(size=200) + 1j * rng.normal(size=200)
    cal = te.fit_transfer(g * rho, rho, 1.0)
    assert cal.g == pytest.approx(g, rel=1e-10)
    assert cal.residual_bin_power == pytest.approx(0.0, abs=1e-18)


def test_shared_gain_bootstrap_covers_the_truth_on_known_transfer_simulations():
    """Item 2's validation: on simulations with a known transfer the joint
    bootstrap interval for the gain must cover it at about the nominal rate."""
    truth = np.random.default_rng(8)
    g_true = 2.0 * np.exp(1j * 0.4)
    covered = 0
    trials = 200
    for _ in range(trials):
        rho = (truth.normal(size=64) + 1j * truth.normal(size=64))
        c = g_true * rho + 0.25 * (truth.normal(size=64) + 1j * truth.normal(size=64))
        rng = np.random.default_rng(int(truth.integers(1e9)))
        draws = np.empty(400, dtype=complex)
        for j in range(400):
            i = te.circular_block_indices(rng, 64, 1)
            draws[j] = te.fit_transfer(c[i], rho[i], 1.0).g
        lo, hi = np.percentile(draws.real, [2.5, 97.5])
        if lo <= g_true.real <= hi:
            covered += 1
    assert 0.88 < covered / trials < 1.0


def test_joint_bootstrap_is_wider_than_calibration_only():
    rng = np.random.default_rng(9)
    n_e, n_c, w, k = 150, 64, 1024, 40
    rho_c = rng.normal(size=n_c) + 1j * rng.normal(size=n_c)
    c_c = 2.0 * rho_c + 0.1 * (rng.normal(size=n_c) + 1j * rng.normal(size=n_c))
    rho_e = rng.normal(size=n_e) + 1j * rng.normal(size=n_e)
    c_e = 2.0 * rho_e + 0.1 * (rng.normal(size=n_e) + 1j * rng.normal(size=n_e))
    q = {"zero": np.zeros(n_e, dtype=complex), "raw-bin": c_e.copy(),
         "calibration-mean": np.full(n_e, c_c.mean()), "shrinkage-cal": 0.9 * c_e}
    contrasts = [("zero", "raw-bin")]
    both = te.joint_bootstrap(q, c_e, rho_e, c_c, rho_c, 0.01, w, contrasts,
                              replicates=300, seed=1, b_eval=1, b_cal=1)
    conly = te.joint_bootstrap(q, c_e, rho_e, c_c, rho_c, 0.01, w, contrasts,
                               replicates=300, seed=1, b_eval=1, b_cal=1,
                               calibration_only=True)
    assert both[("zero", "raw-bin")].std() > conly[("zero", "raw-bin")].std()


def test_calibration_dependent_methods_are_refit_inside_each_replicate():
    """If the calibration-mean baseline were held fixed, its contrast would have
    no calibration-driven spread at all when the evaluation set is frozen."""
    rng = np.random.default_rng(10)
    n_e, n_c, w = 150, 64, 1024
    rho_c = rng.normal(size=n_c) + 1j * rng.normal(size=n_c)
    c_c = 2.0 * rho_c + 0.5 * (rng.normal(size=n_c) + 1j * rng.normal(size=n_c))
    rho_e = rng.normal(size=n_e) + 1j * rng.normal(size=n_e)
    c_e = 2.0 * rho_e
    q = {"zero": np.zeros(n_e, dtype=complex),
         "calibration-mean": np.full(n_e, c_c.mean())}
    d = te.joint_bootstrap(q, c_e, rho_e, c_c, rho_c, 0.25, w,
                           [("zero", "calibration-mean")], replicates=300, seed=2,
                           b_eval=1, b_cal=1, calibration_only=True)
    assert d[("zero", "calibration-mean")].std() > 0


# --- item 4: plateau / dependence rule ---------------------------------------
def test_plateau_rule_picks_the_first_genuinely_flat_block_length():
    curve = {1: dict(variance=1.0, usable=True), 2: dict(variance=2.0, usable=True),
             4: dict(variance=3.0, usable=True), 8: dict(variance=3.1, usable=True),
             16: dict(variance=3.15, usable=True)}
    sel = te.select_block_length(curve, te.BLOCK_CANDIDATES_EVAL)
    assert sel["block_length"] == 4 and sel["verdict"] == "stabilised"


def test_plateau_rule_rejects_a_single_flat_adjacent_pair():
    """A curve that flattens at 2 but resumes rising must NOT select 2 -- this is
    exactly the failure of the single adjacent variance-ratio rule."""
    curve = {1: dict(variance=1.0, usable=True), 2: dict(variance=1.05, usable=True),
             4: dict(variance=2.0, usable=True), 8: dict(variance=2.05, usable=True),
             16: dict(variance=2.08, usable=True)}
    sel = te.select_block_length(curve, te.BLOCK_CANDIDATES_EVAL)
    assert sel["block_length"] == 4 and sel["verdict"] == "stabilised"


def test_largest_candidate_is_never_accepted_as_a_plateau_vacuously():
    """The endpoint ambiguity: b_max trivially satisfies "no larger candidate
    rises", so it must never be returned as `stabilised`."""
    curve = {b: dict(variance=float(b), usable=True) for b in te.BLOCK_CANDIDATES_EVAL}
    sel = te.select_block_length(curve, te.BLOCK_CANDIDATES_EVAL)
    assert sel["verdict"] != "stabilised"
    assert 16 not in sel["plateau_candidates"]


def test_endpoint_flat_pair_is_flagged_not_silently_accepted():
    """Both topmost ratios flat but their product exceeds the threshold, so no
    plateau candidate qualifies: the declared endpoint branch fires and flags."""
    curve = {1: dict(variance=1.0, usable=True), 2: dict(variance=1.5, usable=True),
             4: dict(variance=2.0, usable=True), 8: dict(variance=2.28, usable=True),
             16: dict(variance=2.598, usable=True)}
    sel = te.select_block_length(curve, te.BLOCK_CANDIDATES_EVAL)
    assert sel["verdict"] == "stabilised-at-endpoint"
    assert sel["block_length"] == 8
    assert "flagged" in sel["reason"]


def test_a_genuinely_flat_top_selects_the_plateau_candidate_not_the_endpoint():
    curve = {1: dict(variance=1.0, usable=True), 2: dict(variance=1.5, usable=True),
             4: dict(variance=2.0, usable=True), 8: dict(variance=2.05, usable=True),
             16: dict(variance=2.1, usable=True)}
    sel = te.select_block_length(curve, te.BLOCK_CANDIDATES_EVAL)
    assert sel["verdict"] == "stabilised" and sel["block_length"] == 4


def test_still_rising_curve_is_inconclusive_and_withholds_useful():
    curve = {b: dict(variance=float(b), usable=True) for b in te.BLOCK_CANDIDATES_EVAL}
    sel = te.select_block_length(curve, te.BLOCK_CANDIDATES_EVAL)
    assert sel["verdict"] == "inconclusive" and sel["block_length"] == 16
    v = te.usefulness_verdict(
        {"mean_delta_squared": 5.0, "ci_excludes_zero": True},
        {"mean_delta_squared": 4.0, "ci_excludes_zero": True}, sel["verdict"])
    assert v["verdict"] == "inconclusive"


def test_usefulness_requires_both_references():
    ok = {"mean_delta_squared": 1.0, "ci_excludes_zero": True}
    bad = {"mean_delta_squared": -1.0, "ci_excludes_zero": True}
    incl = {"mean_delta_squared": 1.0, "ci_excludes_zero": False}
    assert te.usefulness_verdict(ok, ok, "stabilised")["verdict"] == "useful"
    assert te.usefulness_verdict(ok, bad, "stabilised")["verdict"] == "not-useful"
    assert te.usefulness_verdict(ok, incl, "stabilised")["verdict"] == "ordering"


def test_insufficient_blocks_are_marked_unusable():
    x = np.random.default_rng(0).normal(size=30)
    curve = te.block_variance_curve(x, te.BLOCK_CANDIDATES_EVAL, replicates=50, seed=1)
    assert curve[16]["usable"] is False and curve[16]["n_blocks"] == 1
    assert curve[4]["usable"] is True


def test_block_variance_curve_rises_under_positive_dependence():
    rng = np.random.default_rng(11)
    n = 150
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal()
    curve = te.block_variance_curve(x, te.BLOCK_CANDIDATES_EVAL, replicates=400, seed=3)
    assert curve[16]["variance"] > 2.0 * curve[1]["variance"]


def test_aggregation_takes_the_most_dependent_curve():
    flat = {b: dict(variance=1.0, usable=True) for b in te.BLOCK_CANDIDATES_EVAL}
    rising = {b: dict(variance=float(b), usable=True) for b in te.BLOCK_CANDIDATES_EVAL}
    agg = te.aggregate_block_curves([flat, rising], te.BLOCK_CANDIDATES_EVAL)
    assert agg["curve"][16]["variance"] == pytest.approx(16.0)
    assert agg["selection"]["verdict"] == "inconclusive"


def test_calibration_grid_stops_where_five_blocks_stop():
    x = np.random.default_rng(0).normal(size=64)
    curve = te.block_variance_curve(x, te.BLOCK_CANDIDATES_CAL, replicates=50, seed=1)
    assert all(curve[b]["usable"] for b in te.BLOCK_CANDIDATES_CAL)
    assert curve[8]["n_blocks"] == 8 >= te.MIN_BLOCKS


# --- characterisation ---------------------------------------------------------
def _stretch(w, n_cal, k, amp, coupling, noise=1.0, second=None, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ref, read = [], []
    for j in range(n_cal):
        n = np.arange(w) + j * w
        kk = k + drift * j
        s = np.cos(2 * np.pi * kk * n / w)
        r = amp * s + 0.001 * rng.normal(size=w)
        if second is not None:
            r = r + second[1] * amp * np.cos(2 * np.pi * (k + second[0]) * n / w)
        ref.append(r)
        read.append(coupling * amp * s + noise * rng.normal(size=w))
    return te.unitary_rfft(np.array(ref)), te.unitary_rfft(np.array(read))


def test_characterisation_flags_a_clean_coupled_stretch_eligible():
    w, k = 4096, 300
    Cr, Cx = _stretch(w, 64, k, 50.0, 1.0, noise=1.0, seed=1)
    N = np.full(w // 2 + 1, 1.0)
    out = te.characterise(Cr, Cx, w, N, 64 * w, 150)
    assert out["outcome"] == "eligible-coupled"
    assert out["k_inj"] == k
    assert out["readout_line_snr"] > 1.0


def test_characterisation_flags_multi_tone_where_the_band_is_resolvable():
    """At w = 65536 the bin width is 152.6 Hz, so +-500 Hz spans 3 bins and a
    companion 2 bins away is inside the declared band."""
    w, k = 65536, 4800
    Cr, Cx = _stretch(w, 32, k, 50.0, 1.0, second=(2, 0.8), seed=2)
    N = np.full(w // 2 + 1, 1.0)
    out = te.characterise(Cr, Cx, w, N, 32 * w, 150)
    assert out["outcome"] == "multi-tone"
    assert out["multi_tone_band_resolvable_at_declared_500hz"] is True
    assert out["second_line_power_fraction"] >= te.MULTI_TONE_POWER_FRACTION


def test_multi_tone_band_is_recorded_as_bin_limited_at_coarse_windows():
    """At w = 4096 one bin spans 2441 Hz, so the declared +-500 Hz rule cannot
    resolve anything; that must be recorded, not silently passed."""
    w, k = 4096, 300
    Cr, Cx = _stretch(w, 64, k, 50.0, 1.0, seed=2)
    N = np.full(w // 2 + 1, 1.0)
    out = te.characterise(Cr, Cx, w, N, 64 * w, 150)
    assert out["multi_tone_band_resolvable_at_declared_500hz"] is False
    assert out["multi_tone_band_hz_effective"] > te.MULTI_TONE_HZ


def test_subbin_two_tone_diagnostics_separate_one_tone_from_two():
    """The known file-0000 case is two tones ~120 Hz apart -- sub-bin at every
    window length used. The reported beat diagnostics must still see it."""
    w, k = 16384, 800
    one, _ = _stretch(w, 64, k, 50.0, 1.0, noise=1e-6, seed=7)
    two, _ = _stretch(w, 64, k, 50.0, 1.0, noise=1e-6, second=(0.2, 1.0), seed=7)
    N = np.full(w // 2 + 1, 1.0)
    a = te.characterise(one, one, w, N, 64 * w, 150)
    b = te.characterise(two, two, w, N, 64 * w, 150)
    assert b["line_power_coefficient_of_variation"] > 5 * a["line_power_coefficient_of_variation"]


def test_characterisation_flags_uncoupled_when_readout_snr_below_one():
    w, k = 4096, 300
    Cr, Cx = _stretch(w, 64, k, 50.0, 1e-4, noise=1.0, seed=3)
    N = np.full(w // 2 + 1, 1.0)
    out = te.characterise(Cr, Cx, w, N, 64 * w, 150)
    assert out["outcome"] == "uncoupled"
    assert out["readout_line_snr"] < 1.0


def test_characterisation_flags_underpowered():
    w, k = 4096, 300
    Cr, Cx = _stretch(w, 64, k, 50.0, 1.0, seed=4)
    N = np.full(w // 2 + 1, 1.0)
    out = te.characterise(Cr, Cx, w, N, 64 * w, 12)
    assert out["outcome"] == "underpowered"


def test_reference_snr_gate_is_separate_from_the_outcome():
    w, k = 4096, 300
    Cr, Cx = _stretch(w, 64, k, 0.02, 1.0, noise=1e-6, seed=5)
    N = np.full(w // 2 + 1, 1e-12)
    out = te.characterise(Cr, Cx, w, N, 64 * w, 150)
    assert "readout_unit_target_supported" in out
    assert out["errors_in_variables"]["claim"].startswith("necessary declared condition")


# --- combination --------------------------------------------------------------
def test_combine_files_reports_weighted_and_unweighted_with_Q():
    per = [{"mean_delta_squared": 1.0, "bootstrap_variance": 0.01},
           {"mean_delta_squared": 1.2, "bootstrap_variance": 0.04},
           {"mean_delta_squared": 0.9, "bootstrap_variance": 0.01}]
    out = te.combine_files(per)
    assert out["n_files"] == 3
    assert out["unweighted_mean"] == pytest.approx((1.0 + 1.2 + 0.9) / 3)
    assert out["theta_iv_precision_weighted"] != out["unweighted_mean"]
    assert out["Q_dof"] == 2 and out["cochran_Q"] >= 0
