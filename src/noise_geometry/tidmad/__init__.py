"""TIDMAD (ABRACADABRA) stage: Paper 1's second real-data anchor.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md``. The stage exists to
close two structural gaps in Paper 1's real-data evidence:

* **Ground truth.** CRESST has no clean signal, so it can compare weighted
  residuals but cannot show that the MSE-preferred reconstruction recovers the
  signal *worse*. TIDMAD's injected reference channel can.
* **A detector-facing metric.** CRESST's trigger efficiency was inconclusive.
  TIDMAD ships a published denoising score and a dark-matter limit chain.

It also supplies something neither existing stage can: the unknown-phase
sinusoid manifold is *exactly* two-dimensional, so the rank argument has an
analytically known answer on real data.

Module map
----------
``templates``      quadrature basis and on-grid frequency handling
``estimators``     rank-1 fixed-phase and rank-2 quadrature GLS, predicted variance
``injections``     synthetic-injection harness (the Phase 1 gate G1 instrument)
``psd``            streaming Welch PSD and measured stationarity selection
``splits``         chronological window splits, guard gap, leakage audit
``negative_controls``  phase 6: five falsification controls
``loader``         chunked HDF5 reader driven by a verified data contract
``experiments``    config-driven entry point used by scripts/run_experiment.py
"""

from __future__ import annotations

from .experiments import run_tidmad_phase, run_tidmad_phase1
from .estimators import (
    QuadratureFit,
    fit_fixed_phase,
    fit_quadrature,
    fixed_phase_snr_loss,
    quadrature_amplitude_variance,
)
from .negative_controls import CONTROLS, run_negative_controls
from .injections import InjectionResult, inject_sinusoid, run_injection_check
from .psd import (
    PSDEstimate,
    StationarityReport,
    assess_stationarity,
    block_psd_series,
    select_stationary_blocks,
    welch_psd_from_windows,
)
from .splits import WindowSplit, chronological_window_split, leakage_audit, window_hashes
from .templates import (
    nearest_bin_frequency,
    quadrature_templates,
    quadrature_templates_rfft,
    sinusoid,
)

__all__ = [
    "InjectionResult",
    "PSDEstimate",
    "QuadratureFit",
    "StationarityReport",
    "WindowSplit",
    "assess_stationarity",
    "block_psd_series",
    "chronological_window_split",
    "fit_fixed_phase",
    "fit_quadrature",
    "fixed_phase_snr_loss",
    "inject_sinusoid",
    "leakage_audit",
    "nearest_bin_frequency",
    "quadrature_amplitude_variance",
    "quadrature_templates",
    "quadrature_templates_rfft",
    "CONTROLS",
    "run_injection_check",
    "run_negative_controls",
    "run_tidmad_phase",
    "run_tidmad_phase1",
    "select_stationary_blocks",
    "sinusoid",
    "welch_psd_from_windows",
    "window_hashes",
]
