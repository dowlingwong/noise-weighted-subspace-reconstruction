"""Executable end-to-end example for the public statistical contracts."""

import numpy as np

from noise_module import (
    ArtifactInjector,
    NoiseGenerator,
    ReferenceDataset,
    TemporalNoiseWrapper,
    ValidationConfig,
    calibrate_dataset,
    validate_stationary_gaussian,
)


config = {"noise_type": "pink", "noise_power": 1.0, "sampling_frequency": 1024.0}
generator = NoiseGenerator(config, seed=1)
trace = generator.generate_noise(4096)
trace = TemporalNoiseWrapper(
    {"variance_modulation": True, "variance_scale_range": [0.8, 1.2]}, seed=2
).apply(trace)
trace, artifact_metadata = ArtifactInjector(
    {
        "sampling_frequency": 1024.0,
        "enable_lines": True,
        "lines": [{"freq": 60.0, "amp": 0.05}],
    },
    seed=3,
).apply(trace, return_components=True)

validation = validate_stationary_gaussian(
    NoiseGenerator(config, seed=4),
    1024,
    ValidationConfig(ensemble_size=64, bootstrap_samples=100, relative_psd_tolerance=0.25),
)
dataset = ReferenceDataset(
    np.vstack([generator.generate_noise(2048) for _ in range(12)]),
    1024.0,
    "example",
    two_dimensional_layout="records_samples",
)
preset = calibrate_dataset(dataset, nperseg=256, bootstrap_samples=50)
regenerated = preset.generate(1024, seed=5)

assert trace.shape == (4096,)
assert artifact_metadata["artifact_only"].shape == trace.shape
assert validation.statistics["variance_mean"] > 0
assert regenerated.shape == (1024,)
print("noise-module end-to-end example passed")
