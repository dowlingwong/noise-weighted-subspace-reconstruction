"""Run SYN-C: a compact MSE-vs-Mahalanobis metric-reversal demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.metrics import mse, weighted_residual
from noise_geometry.subspace import fit_pca, fit_weighted_pca, principal_angles, project_onto_basis


def make_dataset(seed: int, n_traces: int, n_features: int):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_features, endpoint=False)
    signal_basis = np.vstack(
        [
            np.sin(2.0 * np.pi * 9.0 * t),
            np.cos(2.0 * np.pi * 13.0 * t),
        ]
    )
    signal_basis /= np.linalg.norm(signal_basis, axis=1, keepdims=True)
    coeff = rng.normal(size=(n_traces, 2))
    clean = coeff @ signal_basis

    # Low-frequency directions have high variance but low inverse-noise weight.
    freqs = np.fft.rfftfreq(n_features)
    psd = np.ones_like(freqs)
    psd[1:] = 1.0 / freqs[1:] ** 2
    psd[0] = psd[1]
    weights_f = np.zeros_like(psd)
    weights_f[1:-1] = 2.0 / psd[1:-1]
    weights_f[-1] = 1.0 / (2.0 * psd[-1])

    Xf_noise = (rng.normal(size=(n_traces, psd.size)) + 1j * rng.normal(size=(n_traces, psd.size))) * np.sqrt(psd)[None, :]
    Xf_noise[:, 0] = rng.normal(size=n_traces) * np.sqrt(psd[0])
    noise = 0.25 * np.fft.irfft(Xf_noise, n=n_features, axis=1)
    return clean + noise, clean, weights_f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--n-traces", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=256)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/synthetic/metric_reversal/summary.json")
    args = parser.parse_args()

    X, clean, w_f = make_dataset(args.seed, args.n_traces, args.n_features)
    X_f = np.fft.rfft(X, axis=1).real
    clean_f = np.fft.rfft(clean, axis=1).real

    pca = fit_pca(X_f, args.rank)
    wpca = fit_weighted_pca(X_f, w_f, args.rank)
    pca_recon = project_onto_basis(X_f, pca.components, mean=pca.mean)
    wpca_recon = project_onto_basis(X_f, wpca.components, weights=w_f, mean=wpca.mean)

    summary = {
        "seed": args.seed,
        "rank": args.rank,
        "pca_raw_residual_to_observed": float(mse(X_f, pca_recon)),
        "weighted_pca_raw_residual_to_observed": float(mse(X_f, wpca_recon)),
        "pca_weighted_residual_to_observed": float(np.mean(weighted_residual(X_f, pca_recon, w_f))),
        "weighted_pca_weighted_residual_to_observed": float(np.mean(weighted_residual(X_f, wpca_recon, w_f))),
        "pca_clean_mse_diagnostic": float(mse(clean_f, pca_recon)),
        "weighted_pca_clean_mse_diagnostic": float(mse(clean_f, wpca_recon)),
        "subspace_angle_deg_max": float(np.max(principal_angles(pca.components, wpca.components, weights=w_f))),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
