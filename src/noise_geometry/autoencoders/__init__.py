"""Linear autoencoder baselines.

The closed-form helpers here provide the tied linear AE solution under MSE or
diagonal inverse-noise weighting. Optional neural training loops can be added
later without changing experiment contracts.
"""

from .linear import tied_linear_ae_closed_form

__all__ = ["tied_linear_ae_closed_form"]
