import numpy as np
import pytest
from dataclasses import replace

from experiments.paper1.common import smoke_config
from experiments.oracle.oracle_cov.generator import make_training_set
from experiments.oracle.oracle_cov.subjects import make_subject
from experiments.oracle.oracle_cov.protocol import DEFAULT


@pytest.fixture(scope="session")
def ts():
    return make_training_set(0, smoke_config())


@pytest.fixture(scope="session")
def cfg():
    return replace(DEFAULT, window_size=40, n_clean_windows=9, mlp_epochs=5)


@pytest.fixture(scope="session")
def lin(ts):
    return make_subject("lin_cw", ts)


@pytest.fixture(scope="session")
def mlp(ts):
    return make_subject("mlp_ae", ts, epochs=5)
