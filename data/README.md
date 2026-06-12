# Data Layout

Keep only small, non-sensitive examples in this repository.

Expected structure:
- `k_alpha/`: K-alpha HDF5 traces, RQ labels, and template.
- `Noise_PSD/`: one-sided PSD arrays used for whitening and Mahalanobis loss.
- `weight/`: legacy SNR^2 weight arrays.
- `noise_samples/`: detector-noise examples and Study A/B/C exploratory assets.
- large production traces: external storage, referenced via config path overrides
