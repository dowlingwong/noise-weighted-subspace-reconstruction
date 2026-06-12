"""Rank-k regression tests for the exact ('full') M-step.

Background: `solve_eigvec_fast` ignores the off-diagonal coupling of
C^dagger C between components; the old `solve_eigvec_full` crashed with the
1D weights actually used in the pipelines. These tests pin the fix and
quantify fast-vs-full agreement (relevant to the paper's Experiment E).
"""

import numpy as np
import pytest

from src.EMPCA.empca_TCY_optimized import empca_solver, w_orthonormalize
from src.make_weights import build_of_one_sided_weights
from src.EMPCA.empca_equivalence_utils import fit_empca_no_smoothing


def _rank2_data(sim_data, jitter_frac=0.05, seed=11):
    """Pulses with shape variation: template + random timing-derivative mix."""
    rng = np.random.default_rng(seed)
    s = sim_data["template"]
    ds = np.gradient(s)
    ds = ds / np.linalg.norm(ds) * np.linalg.norm(s)
    amps = sim_data["amps"]
    b = rng.normal(0.0, jitter_frac, size=amps.shape)
    clean = amps[:, None] * s[None, :] + (amps * b)[:, None] * ds[None, :]
    noise = sim_data["traces"] - amps[:, None] * s[None, :]
    return clean + noise


def test_full_solver_accepts_diagonal_weights(sim_data):
    """Regression: 1D weights must not crash solve_eigvec_full."""
    J = sim_data["psd"]
    n = sim_data["n"]
    w = build_of_one_sided_weights(J, n)
    X_f = np.fft.rfft(_rank2_data(sim_data), axis=1)

    solver = empca_solver(2, X_f, w)
    eig = solver.solve_eigvec_full()
    assert eig.shape == (2, X_f.shape[1])
    assert np.all(np.isfinite(eig.real)) and np.all(np.isfinite(eig.imag))
    # DC bin has zero weight -> must be zeroed by convention
    assert np.allclose(eig[:, w == 0], 0.0)


def test_full_mode_chi2_not_worse_than_fast(sim_data):
    J = sim_data["psd"]
    n = sim_data["n"]
    w = build_of_one_sided_weights(J, n)
    X_f = np.fft.rfft(_rank2_data(sim_data), axis=1)

    _, _, chi2_fast = fit_empca_no_smoothing(X_f, w, n_comp=2, n_iter=150, patience=20, mode="fast")
    _, _, chi2_full = fit_empca_no_smoothing(X_f, w, n_comp=2, n_iter=150, patience=20, mode="full")

    assert min(chi2_full) <= min(chi2_fast) * (1 + 1e-6), (
        f"full-mode chi2 {min(chi2_full):.6g} worse than fast {min(chi2_fast):.6g}"
    )


def test_w_orthonormalize_enforces_bridge_gauge(sim_data):
    """Rows of w_orthonormalize(A, w) satisfy Phi W Phi^dagger = I (P†Σ⁻¹P=I)."""
    rng = np.random.default_rng(3)
    J = sim_data["psd"]
    n = sim_data["n"]
    w = build_of_one_sided_weights(J, n)

    A = rng.standard_normal((3, w.shape[0])) + 1j * rng.standard_normal((3, w.shape[0]))
    Q = w_orthonormalize(A, w)
    gram = (Q * w[None, :]) @ Q.conj().T
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-8)


def test_fast_vs_full_subspace_agreement_rank2(sim_data):
    """Quantify how far the fast approximation is from the exact M-step."""
    J = sim_data["psd"]
    n = sim_data["n"]
    w = build_of_one_sided_weights(J, n)
    X_f = np.fft.rfft(_rank2_data(sim_data), axis=1)

    eig_fast, _, _ = fit_empca_no_smoothing(X_f, w, n_comp=2, n_iter=150, patience=20, mode="fast")
    eig_full, _, _ = fit_empca_no_smoothing(X_f, w, n_comp=2, n_iter=150, patience=20, mode="full")

    Qa = w_orthonormalize(eig_fast, w)
    Qb = w_orthonormalize(eig_full, w)
    # Weighted principal angles between the two rank-2 subspaces
    M = (Qa * w[None, :]) @ Qb.conj().T
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    angles_deg = np.degrees(np.arccos(sv))
    # They should span nearly the same subspace on well-conditioned sim data;
    # if this fails, fast-mode bias is large enough to matter for Exp E.
    assert angles_deg.max() < 5.0, f"principal angles (deg): {angles_deg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
