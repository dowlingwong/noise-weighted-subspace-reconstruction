"""S5 metric-reversal regression on the likelihood-preserving representation.

After routing S5 through the central rFFT<->real primitive (Re and Im retained,
not real-part-only), the colored-noise reversal must still hold: ordinary PCA
wins raw MSE, while weighted PCA wins the weighted residual, the NLL, and the
recovery of the true (clean) signal.
"""

import numpy as np

from src.noise_geometry.experiments.synthetic import run_s5_metric_reversal


def test_s5_uses_complex_representation():
    r = run_s5_metric_reversal({})
    assert r["representation"] == "complex_rfft_real_imag_stacked"


def test_s5_metric_reversal_direction():
    r = run_s5_metric_reversal({})

    # PCA optimises raw MSE; weighted PCA does not.
    assert r["pca_raw_residual_to_observed"] < r["weighted_pca_raw_residual_to_observed"]

    # Weighted PCA optimises the likelihood metric; PCA does not.
    assert r["weighted_pca_weighted_residual_to_observed"] < r["pca_weighted_residual_to_observed"]
    assert r["weighted_pca_nll_mean"] < r["pca_nll_mean"]

    # The point of the experiment: weighted PCA actually recovers the signal.
    assert r["weighted_pca_clean_mse_diagnostic"] < r["pca_clean_mse_diagnostic"]

    # Under steep colored noise the two subspaces genuinely differ.
    assert r["subspace_angle_deg_max"] > 1.0
