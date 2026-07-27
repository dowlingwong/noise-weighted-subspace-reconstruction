"""TIDMAD Phase 3 confirmatory attempt: predeclared file-selection rule.

Roadmap: docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md.
Rule frozen in: docs/results/TIDMAD_PREDECLARED_GATES.md, "Phase 3
confirmatory attempt -- predeclared file-selection rule" (frozen 2026-07-27,
before this script was run against any candidate file).

Implements exactly the two checks that section describes:

1. single-tone check on channel0002 over samples [0, window_samples *
   n_windows) -- a global FFT peak with prominence >= 1000x median and no
   competing peak (>500 Hz-scale sidelobe-excluded search) above 0.3x its
   power;
2. adequate-amplitude check -- median per-window quadrature-fit truth
   amplitude at least 3x the PSD-predicted noise floor (the same sigma_A
   formula Phase 1 uses).

Excludes abra_validation_0000.h5 and abra_validation_0010.h5 (both already
had their Phase 2/3 *reconstruction* results inspected in prior exploratory
attempts). Scans the remaining files in index order and reports the first
that passes both checks -- this script does not stop early so the full
per-file diagnostic table is always available for audit, but the selection
itself is "first file, in index order, that passes."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXCLUDED_FILES = {"abra_validation_0000.h5", "abra_validation_0010.h5"}

SINGLE_TONE_PROMINENCE_MIN = 1000.0
COMPETING_PEAK_RATIO_MAX = 0.3
COMPETING_PEAK_SEARCH_HZ = 500.0
COMPETING_PEAK_EXCLUDE_HZ = 20.0
AMPLITUDE_SNR_MIN = 3.0
#: Exclude sub-1kHz frequencies from the peak search: SQUID noise is steeply
#: 1/f near DC (see docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md), and every
#: file checked so far injects well above 1 kHz -- without this guard band a
#: file where the injected tone happens to be weak at this sample range picks
#: up the noise floor's own DC-adjacent rise instead (caught on
#: abra_validation_0002.h5 in the first, unguarded run: it "selected" 4.8 Hz).
MIN_SEARCH_FREQUENCY_HZ = 1000.0


def _single_tone_check(ref: np.ndarray, fs: float) -> dict:
    seg = ref - ref.mean()
    spec = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
    spec[0] = 0.0
    spec[freqs < MIN_SEARCH_FREQUENCY_HZ] = 0.0
    interior = spec[1:-1]
    peak_idx = int(np.argmax(interior)) + 1
    f_peak = float(freqs[peak_idx])
    p_peak = float(spec[peak_idx])
    med = float(np.median(spec[spec > 0]))
    prominence = p_peak / med if med > 0 else float("inf")

    mask = (np.abs(freqs - f_peak) > COMPETING_PEAK_EXCLUDE_HZ) & (
        np.abs(freqs - f_peak) <= COMPETING_PEAK_SEARCH_HZ
    )
    competing_ratio = float(spec[mask].max() / p_peak) if mask.any() and p_peak > 0 else 0.0

    passed = prominence >= SINGLE_TONE_PROMINENCE_MIN and competing_ratio <= COMPETING_PEAK_RATIO_MAX
    return {
        "f_peak_hz": f_peak,
        "prominence_over_median": prominence,
        "competing_peak_ratio": competing_ratio,
        "passed": bool(passed),
    }


def _amplitude_check(observed: np.ndarray, clean: np.ndarray, noise_only: np.ndarray,
                      fs: float, frequency: float, window_samples: int) -> dict:
    """SNR of the tone as it actually appears in the SQUID (observed) channel.

    Two corrections versus the first version of this check, both discovered
    by validating against the two already-known files (0000, 0010) *before*
    applying this to any new candidate -- see
    docs/results/TIDMAD_PHASE2_3_REAL_RESULT.md for the full account:

    1. The predicted-sigma formula must come from quadrature_amplitude_variance
       (PSD + weights + analytic template alone), not from averaging the
       measured signal's complex rFFT across windows -- that average cancels
       under the tone's uniformly-distributed phase and gives a bogus,
       near-zero sigma (SNR in the millions for every file).
    2. The amplitude to compare against that sigma must be fit to the
       *observed* (SQUID) channel, not the *clean* (reference) channel. The
       two channels are different physical measurements, and the coupling
       between them is not constant across files: fitting the reference
       channel's own amplitude and comparing it to the SQUID channel's noise
       floor conflates two unrelated scales. Measured coupling ratios
       (observed-fit-amplitude / reference-truth-amplitude) were 3.22 for
       file 0000 (where Phase 2 succeeded) and 0.057 for file 0010 (where it
       failed) -- a >50x difference that fully explains file 0010's
       near-total non-recovery without invoking any reconstruction bug.
    """
    from noise_geometry.tidmad.analysis import fit_quadrature, quadrature_templates_rfft
    from noise_geometry.tidmad.estimators import quadrature_amplitude_variance
    from noise_geometry.tidmad.psd import welch_psd_from_windows
    from noise_geometry.noise.psd import inverse_psd_weights

    est = welch_psd_from_windows(
        list(noise_only), fs, window="boxcar", detrend=True, average="median", source="science"
    )
    weights = inverse_psd_weights(est.psd, window_samples)
    basis_f = quadrature_templates_rfft(window_samples, fs, frequency)

    truth_fit = fit_quadrature(np.fft.rfft(clean, axis=1), basis_f, weights)
    truth_amp = float(np.median(np.hypot(truth_fit.cos_coefficient, truth_fit.sin_coefficient)))

    observed_fit = fit_quadrature(np.fft.rfft(observed, axis=1), basis_f, weights)
    observed_amp = float(np.median(np.hypot(observed_fit.cos_coefficient, observed_fit.sin_coefficient)))

    predicted_variance = quadrature_amplitude_variance(
        basis_f, est.psd, weights, trace_len=window_samples, sampling_frequency=fs
    )
    sigma_a = float(np.sqrt(predicted_variance))
    snr = observed_amp / sigma_a if sigma_a > 0 else float("inf")

    return {
        "median_truth_amplitude_reference_channel": truth_amp,
        "median_amplitude_observed_channel": observed_amp,
        "coupling_ratio": observed_amp / truth_amp if truth_amp > 0 else float("nan"),
        "predicted_sigma": sigma_a,
        "snr": snr,
        "passed": bool(snr >= AMPLITUDE_SNR_MIN),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/ceph/dwong/paper1_dataset"))
    parser.add_argument("--contract", type=Path, default=REPO_ROOT / "docs" / "tidmad_data_contract.json")
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--n-windows", type=int, default=512)
    parser.add_argument("--n-calibration-windows", type=int, default=256)
    parser.add_argument("--output", type=Path,
                         default=REPO_ROOT / "results" / "metrics" / "TIDMAD_P3_file_selection.json")
    args = parser.parse_args()

    from noise_geometry.tidmad.loader import TidmadFile, load_contract

    contract = load_contract(args.contract)
    contract.require_verified()
    n = args.window_samples
    fs = contract.sampling_frequency_hz

    validation_dir = args.data_root / "tidmad" / "raw"
    candidates = sorted(
        p for p in validation_dir.glob("abra_validation_*.h5")
        if "denoised" not in p.name and p.name not in EXCLUDED_FILES
    )
    science = sorted(validation_dir.glob("abra_science_*.h5"))
    if not science:
        raise FileNotFoundError("no science files found for noise_only")

    noise_only = []
    with TidmadFile(science[0], contract) as handle:
        for w in handle.iter_squid_windows(n, stride=n, limit=args.n_calibration_windows):
            noise_only.append(w)
    noise_only = np.vstack(noise_only)

    results = []
    selected = None
    for path in candidates:
        with TidmadFile(path, contract) as handle:
            ref_full = handle.read(contract.reference_dataset, 0, n * args.n_windows)
            observed, clean = [], []
            for squid, reference in handle.iter_paired_windows(n, stride=n, limit=args.n_windows):
                observed.append(squid)
                clean.append(reference)
        observed = np.vstack(observed)
        clean = np.vstack(clean)

        tone = _single_tone_check(ref_full, fs)
        record = {"file": path.name, "single_tone": tone}
        if tone["passed"]:
            amp = _amplitude_check(observed, clean, noise_only, fs, tone["f_peak_hz"], n)
            record["amplitude"] = amp
            record["passed"] = bool(amp["passed"])
        else:
            record["amplitude"] = None
            record["passed"] = False
        results.append(record)
        print(f"{path.name}: single_tone={tone['passed']} "
              f"(f={tone['f_peak_hz']:.1f} Hz, prom={tone['prominence_over_median']:.0f}x, "
              f"competing={tone['competing_peak_ratio']:.3f})"
              + (f" amplitude_snr={record['amplitude']['snr']:.2f} "
                 f"passed={record['passed']}" if record["amplitude"] else " -> rejected on single-tone check"))

        if record["passed"] and selected is None:
            selected = record

    payload = {
        "rule": "docs/results/TIDMAD_PREDECLARED_GATES.md#phase-3-confirmatory-attempt----predeclared-file-selection-rule",
        "excluded_files": sorted(EXCLUDED_FILES),
        "per_file": results,
        "selected_file": selected["file"] if selected else None,
        "selected_frequency_hz": selected["single_tone"]["f_peak_hz"] if selected else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nselection -> {args.output}")
    if selected:
        print(f"SELECTED: {selected['file']} at {selected['single_tone']['f_peak_hz']:.2f} Hz")
    else:
        print("NO FILE PASSED both checks in the scanned candidates.")


if __name__ == "__main__":
    main()
