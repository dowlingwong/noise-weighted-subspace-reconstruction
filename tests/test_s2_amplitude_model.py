"""S2 amplitude-model behaviour and the finite-sample (sqrt N) acceptance.

Findings encoded here:
  * The amplitude_model flag selects which metric is the headline, but the
    rank-1 OF/EMPCA equivalence is *robust to coefficient phase*: the complex
    EMPCA covariance is rank-1 (it depends only on |c|^2), so the template is
    recovered whether coefficients are real or complex.
  * The residual rank-1 EMPCA-vs-OF angle is therefore finite-sample: it shrinks
    at the 1/sqrt(N) rate, which is S2's honest acceptance (not a single-seed
    threshold).
"""

import numpy as np
import pytest

from src.noise_geometry.experiments.synthetic import run_s2_of_empca
from src.noise_geometry.validation import run_seed_sweep, summarize


def test_headline_metric_follows_amplitude_model():
    real = run_s2_of_empca({"amplitude_model": "real", "n_samples": 512, "n_traces": 512})
    assert real["acceptance_metric"] == "weighted_angle_deg"
    assert real["acceptance_angle_deg"] == real["weighted_angle_deg"]

    cplx = run_s2_of_empca({"amplitude_model": "complex", "n_samples": 512, "n_traces": 512})
    assert cplx["acceptance_metric"] == "template_in_span_angle_deg"
    assert cplx["acceptance_angle_deg"] == cplx["template_in_span_angle_deg"]


def test_equivalence_robust_to_coefficient_phase():
    """Complex-phase coefficients do not break the rank-1 recovery."""
    real = run_s2_of_empca({"amplitude_model": "real", "n_samples": 512, "n_traces": 1024})
    cplx = run_s2_of_empca({"amplitude_model": "complex", "n_samples": 512, "n_traces": 1024})
    assert cplx["weighted_cosine"] > 0.99
    assert cplx["template_in_span_angle_deg"] < 6.0
    # the two models agree on the equivalence quality
    assert abs(real["weighted_cosine"] - cplx["weighted_cosine"]) < 0.02


def test_angle_follows_sqrt_n_scaling():
    """Mean rank-1 EMPCA-vs-OF angle halves for every 4x increase in n_traces."""
    seeds = [0, 1, 2]
    means = {}
    for n in (256, 1024, 4096):
        rows = run_seed_sweep(
            run_s2_of_empca, {"n_samples": 512, "n_traces": n, "amplitude_model": "real"}, seeds
        )
        means[n] = summarize(rows, keys=["weighted_angle_deg"])["weighted_angle_deg"]["mean"]

    # angle ~ C / sqrt(n): each 4x in n should roughly halve the angle.
    for n_lo, n_hi in ((256, 1024), (1024, 4096)):
        ratio = means[n_lo] / means[n_hi]
        assert 1.6 < ratio < 2.5, f"n {n_lo}->{n_hi} ratio {ratio:.2f} not ~2 (sqrt-N)"

    # monotone decrease toward zero
    assert means[256] > means[1024] > means[4096]


def test_sqrt_n_extrapolated_limit_is_consistent_with_zero():
    """Fit angle = C/sqrt(n); the implied large-N limit is ~0 (cos -> 1)."""
    seeds = [0, 1, 2, 3]
    ns = np.array([256, 1024, 4096], dtype=float)
    angles = np.array(
        [
            summarize(
                run_seed_sweep(
                    run_s2_of_empca, {"n_samples": 512, "n_traces": int(n), "amplitude_model": "real"}, seeds
                ),
                keys=["weighted_angle_deg"],
            )["weighted_angle_deg"]["mean"]
            for n in ns
        ]
    )
    # log-log slope of angle vs n should be near -0.5
    slope = np.polyfit(np.log(ns), np.log(angles), 1)[0]
    assert -0.6 < slope < -0.4, f"log-log slope {slope:.3f} not ~ -0.5"
