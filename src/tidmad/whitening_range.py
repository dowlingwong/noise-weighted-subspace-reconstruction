"""Measure how far the two rung-4 objectives actually differ.

Why
---
Round 6 found no metric reversal at rung 4: the ``Sigma^-1``-trained arm was
better on *both* held-out criteria (-1.15e-4 on the normalized likelihood
residual, -57.2 on raw MSE, both intervals excluding zero, the second opposite
to prediction). The paper offers an explanation -- that whitening the
band-limited spectrogram is close to a constant rescaling, leaving the two
losses nearly collinear -- but that explanation was *inferred from the effect
signature, not measured*. This module measures it.

The argument
------------
The two objectives are

    MSE  = mean( r^2 )
    chi2 = mean( r^2 / J )

over bands. They are proportional -- the same objective up to a scale, with
identical minimisers -- exactly when ``J`` is constant across the bands that
carry the residual. The degree of non-proportionality is therefore a property of
the *spread* of the weights ``1/J``, not of their magnitude.

Reported statistics, over the ``n_bands_used`` kept bands:

* ``ratio_max_min``          ``J.max()/J.min()`` -- the crude dynamic range;
* ``ratio_p99_p01``          the same at the 1st/99th percentiles, robust to a
                             single degenerate bin;
* ``cv_inverse_J``           ``std(1/J)/mean(1/J)`` -- **the decisive number.**
                             At 0 the objectives coincide; the larger it is, the
                             more the weighting can reorder candidate
                             reconstructions.
* ``effective_n_bands``      participation ratio of ``1/J``, i.e. how many bands
                             the weighting effectively attends to. Close to
                             ``n_bands_used`` means near-uniform weighting.

Band 0 is DC and is degenerate for zero-mean calibration noise, so every
statistic is reported both with and without it.

Verdict
-------
``cv_inverse_J`` below roughly 0.1 supports the collinearity account: the paper's
explanation becomes a measurement. Substantially above it, the account is wrong
and the absence of reversal at rung 4 needs a different explanation -- which is
worth knowing before submission, not after.

Usage
-----
    PYTHONPATH=src python -m tidmad.whitening_range \\
        --data-dir <resolved_tidmad_raw_dir> \\
        --out results/rps2026/E6_whitening_range.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import FROZEN


def _stats(J: np.ndarray, label: str) -> dict:
    """Dispersion statistics for one weight vector."""
    J = np.asarray(J, dtype=np.float64)
    inv = 1.0 / J
    participation = float(inv.sum() ** 2 / np.square(inv).sum())
    return {
        "label": label,
        "n_bands": int(J.size),
        "J_min": float(J.min()),
        "J_max": float(J.max()),
        "J_median": float(np.median(J)),
        "ratio_max_min": float(J.max() / J.min()),
        "ratio_p99_p01": float(np.percentile(J, 99) / np.percentile(J, 1)),
        "cv_inverse_J": float(inv.std() / inv.mean()),
        "effective_n_bands": participation,
        "effective_fraction": participation / J.size,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="directory holding abra_science_*.h5 (the noise-only files J is fitted on)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = FROZEN
    from .psd import estimate_band_psd

    # Exactly the estimator, science files and configuration training used.
    from .train import _noise_only_windows

    J, _ = estimate_band_psd(
        _noise_only_windows(args.data_dir, run),
        run.stft,
        sampling_frequency=run.stft.sampling_frequency_hz,
        window=run.data.psd_window,
        average=run.data.psd_average,
    )
    J = np.asarray(J, dtype=np.float64)

    bin_hz = run.stft.sampling_frequency_hz / run.stft.n_fft
    record = {
        "n_bands_used": run.stft.n_bands_used,
        "bin_width_hz": bin_hz,
        "band_edge_hz": run.stft.n_bands_used * bin_hz,
        "psd_window": run.data.psd_window,
        "psd_average": run.data.psd_average,
        "n_calibration_windows": run.data.n_calibration_windows,
        "all_bands": _stats(J, "bands 0..N-1 (includes DC)"),
        "excluding_dc": _stats(J[1:], "bands 1..N-1 (DC dropped)"),
    }

    cv = record["excluding_dc"]["cv_inverse_J"]
    record["collinearity_supported"] = bool(cv < 0.1)
    record["verdict"] = (
        "Weights are near-uniform: chi2 and MSE are close to the same objective over "
        "the modelled band, which supports the paper's collinearity explanation for the "
        "absence of metric reversal at rung 4."
        if cv < 0.1 else
        "Weights vary substantially: chi2 and MSE are genuinely different objectives "
        "here, so near-collinearity does NOT explain the absence of reversal at rung 4. "
        "The paper's explanation must be revised before submission."
    )

    for key in ("all_bands", "excluding_dc"):
        s = record[key]
        print(f"  {s['label']}")
        print(f"    J range        {s['J_min']:.4g} .. {s['J_max']:.4g}   (median {s['J_median']:.4g})")
        print(f"    max/min        {s['ratio_max_min']:.3g}      p99/p01  {s['ratio_p99_p01']:.3g}")
        print(f"    CV(1/J)        {s['cv_inverse_J']:.4f}   <- decisive")
        print(f"    effective bands {s['effective_n_bands']:.1f} of {s['n_bands']}"
              f"  ({100*s['effective_fraction']:.1f}%)")
        print()
    print(f"  collinearity supported: {record['collinearity_supported']}")
    print(f"  {record['verdict']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
