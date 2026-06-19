"""GWOSC pipeline helpers.

These functions are intentionally lightweight until public-data dependencies
(`gwosc`, `gwpy`) are installed by the user.
"""

from .smoke import dependency_status
from .analysis import load_cached_event, run_gwosc_experiment
from .reference import (
    gwpy_psd_reference,
    gwpy_whiten_reference,
    run_gwpy_reference_check,
    whiten_time_series_rfft,
)

__all__ = [
    "dependency_status",
    "gwpy_psd_reference",
    "gwpy_whiten_reference",
    "load_cached_event",
    "run_gwosc_experiment",
    "run_gwpy_reference_check",
    "whiten_time_series_rfft",
]
