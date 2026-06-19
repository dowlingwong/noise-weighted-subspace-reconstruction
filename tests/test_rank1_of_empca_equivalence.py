"""Numerical check of the paper's core claim (Experiment A):

the OF amplitude is the rank-1 weighted-ML (EMPCA) coefficient, when
weighting, preprocessing, and gauge conventions are matched.
"""

import numpy as np

from src.canonical.OptimumFilter import OptimumFilter
from src.canonical.make_weights import build_of_one_sided_weights
from src.canonical.empca_equivalence_utils import (
    fit_empca_no_smoothing,
    phase_align_basis,
    project_gls as gls_amplitude,  # of.py removed; rank-1 GLS lives in equivalence utils
    weighted_cosine,
)


def test_gls_amplitude_matches_optimum_filter(sim_data):
    """src.of.gls_amplitude must reproduce OptimumFilter.fit amplitudes."""
    s = sim_data["template"]
    J = sim_data["psd"]
    traces = sim_data["traces"]
    n = sim_data["n"]
    fs = sim_data["fs"]

    of = OptimumFilter(s, J, fs)
    w = build_of_one_sided_weights(J, n)
    s_f = np.fft.rfft(s)
    X_f = np.fft.rfft(traces, axis=1)

    amp_of = np.array([of.fit(tr)[0] for tr in traces])
    amp_gls = gls_amplitude(X_f, s_f, w)

    np.testing.assert_allclose(amp_gls, amp_of, rtol=1e-8, atol=1e-12)


def test_rank1_empca_recovers_of_direction_and_amplitude(sim_data):
    s = sim_data["template"]
    J = sim_data["psd"]
    traces = sim_data["traces"]
    n = sim_data["n"]

    w = build_of_one_sided_weights(J, n)
    s_f = np.fft.rfft(s)
    X_f = np.fft.rfft(traces, axis=1)

    eigvec, coeff, chi2s = fit_empca_no_smoothing(X_f, w, n_comp=1, n_iter=100)
    assert chi2s[-1] <= chi2s[0]  # objective decreased

    u = phase_align_basis(eigvec[0], s_f, w)

    # Direction: leading weighted component aligns with the template.
    cos = weighted_cosine(u, s_f, w)
    assert cos > 0.99, f"weighted cosine {cos}"

    # Amplitude: EMPCA coefficients are an affine match to OF amplitudes.
    amp_of = gls_amplitude(X_f, s_f, w)
    c = np.real(coeff[:, 0] * np.exp(1j * np.angle(np.sum(np.conj(eigvec[0]) * s_f * w))))
    r = np.corrcoef(np.abs(coeff[:, 0]), amp_of)[0, 1]
    r2 = np.corrcoef(c, amp_of)[0, 1]
    assert max(r, abs(r2)) > 0.999, f"corr(EMPCA coeff, OF amp) = {max(r, abs(r2))}"
