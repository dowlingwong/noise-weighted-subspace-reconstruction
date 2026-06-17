import numpy as np

from src.noise_geometry.autoencoders import tied_linear_ae_closed_form
from src.noise_geometry.subspace import fit_weighted_pca, principal_angles, project_onto_basis


def test_exact_weighted_linear_ae_matches_empca():
    rng = np.random.default_rng(31)
    basis = np.linalg.qr(rng.normal(size=(12, 3)))[0].T
    samples = rng.normal(size=(400, 3)) @ basis + rng.normal(scale=0.05, size=(400, 12))
    weights = np.linspace(0.5, 3.0, samples.shape[1])
    empca = fit_weighted_pca(samples, weights, rank=3)
    ae = tied_linear_ae_closed_form(samples, 3, weights=weights)

    angles = principal_angles(empca.components, ae.components, weights=weights)
    assert np.max(angles) < 1e-5
    empca_recon = project_onto_basis(samples, empca.components, weights=weights, mean=empca.mean)
    ae_recon = project_onto_basis(samples, ae.components, weights=weights, mean=ae.mean)
    np.testing.assert_allclose(empca_recon, ae_recon, atol=1e-10)


def test_full_covariance_weighted_basis_has_bridge_gauge():
    rng = np.random.default_rng(32)
    samples = rng.normal(size=(300, 8))
    mixing = rng.normal(size=(8, 8))
    covariance = mixing @ mixing.T + 0.5 * np.eye(8)
    metric = np.linalg.inv(covariance)
    fit = fit_weighted_pca(samples, metric, rank=3)
    gram = fit.components @ metric @ fit.components.T
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)
