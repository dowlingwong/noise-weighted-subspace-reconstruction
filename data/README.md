# Data Layout

Keep only small, non-sensitive examples in this repository.

Expected structure:
- `weights/`: PSD and SNR^2 weight arrays (`.npy`)
- `sample/`: tiny smoke-test traces for local checks
- large production traces: external storage, referenced via config paths
