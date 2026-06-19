"""Linear autoencoder baselines.

The closed-form helpers here provide the tied linear AE solution under MSE or
diagonal inverse-noise weighting. The ``trained`` module adds independently
gradient-trained tied linear AEs (direct weighted loss and whitened MSE) used
to verify the S3 EMPCA bridge by optimisation rather than by construction.
"""

from .linear import tied_linear_ae_closed_form
from .trained import (
    TrainedAEResult,
    empca_optimal_loss,
    load_trained_ae,
    run_s3_bridge,
    save_trained_ae,
    train_weighted_linear_ae,
    train_whitened_linear_ae,
    weighted_reconstruction_loss,
)

__all__ = [
    "tied_linear_ae_closed_form",
    "TrainedAEResult",
    "empca_optimal_loss",
    "load_trained_ae",
    "run_s3_bridge",
    "save_trained_ae",
    "train_weighted_linear_ae",
    "train_whitened_linear_ae",
    "weighted_reconstruction_loss",
]
