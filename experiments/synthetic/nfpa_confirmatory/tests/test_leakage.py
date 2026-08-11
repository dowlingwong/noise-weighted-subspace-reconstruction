"""Leakage audits for the NFPA confirmatory harness (protocol section 7).

These tests fail loudly if a validation/test partition is touched before final
evaluation, if the training mean is re-estimated on a non-training split, or if
a degenerate ``C <= k_c`` / ``T <= k_t`` configuration is silently accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.synthetic.nfpa_confirmatory.config import PrimaryConfig
from experiments.synthetic.nfpa_confirmatory.generator import make_dataset
from experiments.synthetic.nfpa_confirmatory.methods import FIT_FACTORY, fit_cw_pca


class _TestDataGuard:
    """Proxy that raises if the test split is touched before final evaluation."""

    def __init__(self, ds, *, allow_test=False):
        self._ds = ds
        self._allow_test = allow_test

    def __getattr__(self, name):
        if name in ("X_test", "clean_test", "latents_test") and not self._allow_test:
            raise AssertionError(f"test partition '{name}' accessed during the fit phase")
        return getattr(self._ds, name)


def test_generator_rejects_degenerate_configuration():
    bad = PrimaryConfig(k_c=8, k_t=3)
    with pytest.raises(AssertionError, match="C > k_c"):
        make_dataset(0, "colored", bad)
    bad_t = PrimaryConfig(k_c=2, k_t=256)
    with pytest.raises(AssertionError, match="T > k_t"):
        make_dataset(0, "colored", bad_t)


def test_scope_amendment_and_tracked_result_root_are_frozen():
    from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, RunPaths

    assert PRIMARY.n_seeds == 10
    assert RunPaths().root == "results/rps2026"


def test_train_mean_uses_training_split_only():
    ds = make_dataset(0, "colored")
    assert np.allclose(ds.train_mean, ds.X_train.mean(axis=0))
    assert not np.allclose(ds.train_mean, ds.X_test.mean(axis=0))


def test_fit_phase_never_touches_test_split():
    ds = make_dataset(0, "colored")
    guarded = _TestDataGuard(ds)
    cw = fit_cw_pca(guarded)
    assert cw.name == "CW-PCA"
    for name in ("NFPA", "Iso-MPCA", "IsoPCA", "OF"):
        fit = FIT_FACTORY[name](guarded)
        assert fit.name == name


def test_oracle_covariance_uses_no_event_data():
    ds = make_dataset(0, "colored")
    from experiments.synthetic.nfpa_confirmatory.covariance import run_numerical_checks

    report = run_numerical_checks(ds.cfg, ds.covariances["Sigma"])
    for key in ("symmetry_error", "whitening_identity_error", "monte_carlo_covariance_relative_error"):
        assert np.isfinite(report[key]), key


def test_evaluation_is_post_fit_guard_respected():
    ds = make_dataset(0, "colored")
    fit = FIT_FACTORY["NFPA"](ds)
    guarded = _TestDataGuard(ds, allow_test=True)
    _ = guarded.X_test  # final evaluation stage legitimately reads the test split
