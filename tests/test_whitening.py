import numpy as np

from src.noise_geometry.metrics import weighted_inner
from src.noise_geometry.noise import inverse_covariance, whiten_with_covariance


def test_covariance_whitening_is_approximately_white():
    rng = np.random.default_rng(11)
    cov = np.array([[2.0, 0.7, 0.2], [0.7, 1.5, 0.4], [0.2, 0.4, 0.8]])
    samples = rng.multivariate_normal(np.zeros(3), cov, size=12000)
    whitened = whiten_with_covariance(samples, cov)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(3), atol=0.05)


def test_weighted_inner_matches_whitened_euclidean_inner():
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    metric = inverse_covariance(cov)
    u = np.array([1.2, -0.4])
    v = np.array([0.3, 2.1])
    uw = whiten_with_covariance(u[None, :], cov)[0]
    vw = whiten_with_covariance(v[None, :], cov)[0]
    np.testing.assert_allclose(weighted_inner(u, v, metric), np.dot(uw, vw), atol=1e-12)
