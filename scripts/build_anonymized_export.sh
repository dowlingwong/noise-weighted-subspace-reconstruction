#!/usr/bin/env bash
#
# Build the anonymized code release for the R/PS 2026 submission.
#
# Recipe and rationale: paper_rps2026/notes/anonymized_code_release.md (v2).
# The release is a FRESH, WHITELIST-BUILT tree -- never a cleaned copy of this
# repository, whose git history, plans, notes and logs all carry identity.
#
# Usage, from the repository root:
#     ./scripts/build_anonymized_export.sh [outdir]      # default: export/rps2026_anonymized
#
# The script is the documentation: every included file is listed here, every
# neutralization is a visible sed, and gate 2 (the zero-hit grep) runs at the
# end and FAILS THE BUILD on any hit. Gates 3-6 (manual read, clean-env figure
# reproduction, upload, signed-out check) are on the operator; gate 4 can be run
# with:  cd <outdir> && python scripts/make_rps_figure.py --out /tmp/evidence.pdf
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
OUT="${1:-export/rps2026_anonymized}"

LEAK_REGEX='dwong|wongdowling|dowling|ceph|deepthought|delight|kit\.edu|karlsruhe|imperial|sjostrom|junjie'

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------------------------------------------------------- 1. whitelist
# Source. --parents preserves layout so PYTHONPATH=src works as in the paper.
copy() { for f in "$@"; do install -D "$f" "$OUT/$f"; done }

copy src/tidmad/*.py src/tidmad/configs/*.yaml src/tidmad/backbone/*.py src/tidmad/tests/*.py
copy src/NFPA/nfpa_demo.py                     # the .ipynb is excluded: embedded outputs leak
copy src/noise_geometry/__init__.py \
     src/noise_geometry/filters/__init__.py src/noise_geometry/filters/optimal.py \
     src/noise_geometry/metrics/__init__.py src/noise_geometry/metrics/reconstruction.py \
     src/noise_geometry/noise/__init__.py src/noise_geometry/noise/*.py \
     src/noise_geometry/subspace/__init__.py src/noise_geometry/subspace/*.py
# noise_geometry.tidmad: test_psd needs only splits.py; the package __init__
# pulls in the whole Paper-1 stage, so ship a minimal one instead.
install -D src/noise_geometry/tidmad/splits.py "$OUT/src/noise_geometry/tidmad/splits.py"
install -D src/noise_geometry/tidmad/psd.py "$OUT/src/noise_geometry/tidmad/psd.py"
printf '"""Chronological window splits (subset for the confirmatory tests)."""\n' \
    > "$OUT/src/noise_geometry/tidmad/__init__.py"

copy experiments/__init__.py experiments/synthetic/__init__.py \
     experiments/synthetic/nfpa_confirmatory/*.py \
     experiments/synthetic/nfpa_confirmatory/tests/*.py
copy scripts/make_rps_figure.py scripts/rung3_whitening_range.py scripts/train_tidmad_local.sh

# Exact records the figures and tables read -- individually named, never
# whole directories (logs beside them carry host paths).
copy results/rps2026/E1_nfpa_blockA/block_A_colored_raw.csv \
     results/rps2026/E2_nfpa_blockB/block_B_white_raw.csv \
     results/rps2026/E5_t_mse_eval_series.csv \
     results/rps2026/E5_t_chi2_eval_series.csv \
     results/rps2026/E5_t_mse_inband_unrounded.json \
     results/rps2026/E5_t_chi2_inband_unrounded.json \
     results/rps2026/E5_benchmark_contrast_interval.json \
     results/rps2026/E6b_rung3_whitening_range.json
# When synced from the server, add:
#   results/rps2026/E6_whitening_range.json
#   results/rps2026/E7_* (validated seed-replication artifacts)
for f in results/rps2026/E6_whitening_range.json results/rps2026/E7_*; do
    [ -e "$f" ] && install -D "$f" "$OUT/$f" && echo "  included (synced): $f"
done || true

find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

# ------------------------------------------------------- 2. neutralizations
# Each sed targets a known leak; the final grep proves none survive.

# paths.py: host-specific candidate lists become placeholders.
python3 - "$OUT/src/tidmad/paths.py" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("/ceph/dwong/datasets/tidmad/raw", "/path/to/tidmad/raw")
s = s.replace("/ceph/dwong/paper1_dataset/tidmad/raw", "/alternate/path/to/tidmad/raw")
s = s.replace("/ceph/dwong/datasets/tidmad/denoised", "/path/to/tidmad/denoised")
s = s.replace("/ceph/dwong/paper1_dataset/tidmad/denoised", "/alternate/path/to/tidmad/denoised")
s = s.replace("/ceph/dwong/src/TIDMAD", "/path/to/TIDMAD")
s = re.sub(r'"""Locate the TIDMAD data trees.*?"""',
           '"""Locate the TIDMAD data trees.\n\nCandidates are tried in order; the first directory that actually contains the\nexpected files wins. Set TIDMAD_DATA_ROOT / TIDMAD_DENOISED_DIR /\nTIDMAD_UPSTREAM to override; overrides are still content-checked.\n"""',
           s, flags=re.S)
s = re.sub(r"\n( *)f\"    find /ceph[^\n]*\n", '\n\\1f\"    find / -maxdepth 6 -name {pattern!r} 2>/dev/null\"\n', s)
open(p, "w").write(s)
PY

# train_tidmad_local.sh: same treatment for the launcher.
sed -i \
  -e 's|/ceph/dwong/datasets/tidmad/raw|/path/to/tidmad/raw|g' \
  -e 's|/ceph/dwong/paper1_dataset/tidmad/raw|/alternate/path/to/tidmad/raw|g' \
  -e 's|find /ceph/dwong -maxdepth 6|find / -maxdepth 6|g' \
  -e 's|find /ceph -maxdepth 6|find / -maxdepth 6|g' \
  -e 's|# Companion to train_local_l40s.sh (DELight); same conventions, different task:|# Trains one factored-attention architecture under two residual metrics:|' \
  -e 's|inherited from the DELight launcher whose training loop used|inherited from an earlier launcher whose training loop used|' \
  -e '/EXPERIMENT_RUN_LOG.md/d' -e '/deepthought/d' \
  "$OUT/scripts/train_tidmad_local.sh"

# checkpoints.py: the example storage URL names an institution host.
sed -i 's|root://[^"]*|root://storage.example.org//training_runs/run-name|' \
  "$OUT/src/tidmad/backbone/checkpoints.py"

# model.py: an inherited comment names the upstream experiment.
sed -i 's/the DELight model/the upstream model/' "$OUT/src/tidmad/model.py"

# whitening_range.py docstring cites the run history; keep it generic.
sed -i 's/PYTHONPATH=src python -m tidmad.whitening_range/PYTHONPATH=src python -m tidmad.whitening_range/' \
  "$OUT/src/tidmad/whitening_range.py"

# ------------------------------------------------------- 3. neutral packaging
cat > "$OUT/pyproject.toml" <<'EOF'
[project]
name = "rps2026-anonymized"
version = "0.1.0"
description = "Code and archived records for an R/PS 2026 workshop submission (anonymized for double-blind review)."
requires-python = ">=3.10"
dependencies = ["numpy", "scipy", "matplotlib", "h5py", "pyyaml"]

[project.optional-dependencies]
train = ["torch", "jaxtyping", "einops"]
test = ["pytest"]
EOF

cat > "$OUT/LICENSE" <<'EOF'
MIT License

Copyright (c) 2026 The Authors (identities withheld for double-blind review;
this notice will be completed in the camera-ready release)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

cat > "$OUT/README.md" <<'EOF'
# Anonymized code and records — R/PS 2026 submission

Code and archived experiment records for the workshop submission
*Noise Covariance as Representation Geometry: One Metric, Four Architecture
Classes* (double-blind; author identities withheld).

## Reproduce the paper's figure and statistics

Figure 1 (all three panels, every number recomputed from the archived records):

    pip install numpy scipy matplotlib
    python scripts/make_rps_figure.py --out evidence.pdf

The rung-3 whitening-dispersion comparison (Section 4's 1.10 / 0 values):

    PYTHONPATH=src:experiments/synthetic python scripts/rung3_whitening_range.py

## Layout

- `src/tidmad/` — rung-4 training, PSD estimation, scoring (including the
  unrounded rescore), whitening-dispersion measurement, dataset-path resolver.
- `src/NFPA/`, `experiments/synthetic/nfpa_confirmatory/` — rung-3 NFPA and the
  predeclared confirmatory study (Blocks A/B).
- `src/noise_geometry/` — shared covariance/PSD/subspace utilities.
- `results/rps2026/` — the exact CSV/JSON records the paper's figure, tables and
  quoted intervals are computed from.
- `scripts/train_tidmad_local.sh` — the GPU training launcher (paths are
  placeholders; set `TIDMAD_DATA_ROOT`).

## Data

TIDMAD is public (CC-BY-4.0): Fry et al., *TIDMAD: Time-series Dataset for
Discovering Dark Matter*, NeurIPS 2024 Datasets & Benchmarks. Download per the
TIDMAD repository instructions; the training entry point expects
`abra_training_*.h5`, `abra_validation_*.h5` and `abra_science_*.h5` under one
directory. Gravitational-wave strain: GWOSC public releases.

## Tests

    pip install pytest
    PYTHONPATH=src:. pytest src/tidmad/tests experiments/synthetic/nfpa_confirmatory/tests

Torch-dependent tests skip automatically if `torch` is not installed.
EOF

# ------------------------------------------------------------------ 4. gate 2
echo ""
echo "Gate 2: anonymity grep over the export..."
if hits=$(grep -rniE "$LEAK_REGEX" "$OUT" 2>/dev/null); then
    echo "FAIL: identity/host strings survive in the export:" >&2
    echo "$hits" | head -20 >&2
    exit 1
fi
echo "  zero hits."

n_files=$(find "$OUT" -type f | wc -l)
echo ""
echo "Export built at $OUT ($n_files files)."
echo "Remaining gates: 3 read every file; 4 clean-env figure reproduction +"
echo "tests; 5 upload one immutable commit to anonymous.4open.science;"
echo "6 verify the URL signed-out."
