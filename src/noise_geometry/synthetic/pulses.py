"""Reusable pulse templates for controlled experiments."""

from __future__ import annotations

import numpy as np


def exponential_pulse(n_samples: int, sampling_frequency: float, *, tau_rise: float = 8e-4, tau_decay: float = 8e-3):
    """Return a normalized rise-decay pulse template."""
    t = np.arange(int(n_samples)) / float(sampling_frequency)
    pulse = np.exp(-t / float(tau_decay)) - np.exp(-t / float(tau_rise))
    peak = np.max(np.abs(pulse))
    if peak == 0:
        raise ValueError("degenerate pulse template")
    return pulse / peak
