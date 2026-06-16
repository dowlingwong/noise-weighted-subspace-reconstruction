"""Optimal-filter and matched-filter projections."""

from .optimal import gls_amplitude, matched_filter_score, project_rank1

__all__ = ["gls_amplitude", "matched_filter_score", "project_rank1"]
