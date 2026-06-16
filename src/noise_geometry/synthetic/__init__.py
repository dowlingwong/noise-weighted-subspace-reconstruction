"""Synthetic pulse benchmarks used by Paper 1 experiments."""

from .benchmarks import make_rank1_pulse_dataset, run_of_empca_equivalence
from .pulses import exponential_pulse

__all__ = ["exponential_pulse", "make_rank1_pulse_dataset", "run_of_empca_equivalence"]
