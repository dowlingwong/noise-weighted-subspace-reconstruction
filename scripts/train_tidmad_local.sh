#!/usr/bin/env bash
#
# Local TIDMAD rung-4 training driver.
#
# Companion to train_local_l40s.sh (DELight); same conventions, different task:
# one factored-attention architecture trained under two residual metrics that
# differ ONLY in the training objective. See:
#   plan/EXPERIMENT_CHECKLIST_2026-08-08.md   (E3, E4)
#   plan/PLAN_03_RUNG4_TIDMAD_2026-08-08.md   (design, gates)
#
# Usage:
#   ./scripts/train_tidmad_local.sh check          # GPU + data preflight only
#   ./scripts/train_tidmad_local.sh smoke          # a few steps, measures throughput
#   ./scripts/train_tidmad_local.sh bench          # 200 steps, prints a budget table
#   ./scripts/train_tidmad_local.sh mse            # arm T-mse
#   ./scripts/train_tidmad_local.sh chi2           # arm T-chi2
#   ./scripts/train_tidmad_local.sh chi2_of        # secondary arm
#
# Hands-off (screen):
#   screen -S rps_t_mse -L -Logfile screenlog_t_mse.txt \
#       ./scripts/train_tidmad_local.sh mse
#   # detach: Ctrl-A then D    reattach: screen -r rps_t_mse
#
# Environment overrides:
#   GPU_INDEX=0                 which single GPU to pin
#   TIDMAD_DATA_ROOT            default /ceph/dwong/datasets/tidmad/raw
#   TIDMAD_STEPS                override num_steps (smoke defaults to 20)
#   TIDMAD_TIMEOUT              wall-clock cap, e.g. 6h (default: none for smoke/check)
#   TIDMAD_RESUME=auto          resume from <out_dir>/latest.pt (or give a path)
#   TIDMAD_WANDB=1              log to Weights & Biases
#   EXPECT_GPU_NAME             substring the GPU name must contain (optional)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-check}"
GPU_INDEX="${GPU_INDEX:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
TIDMAD_DATA_ROOT="${TIDMAD_DATA_ROOT:-/ceph/dwong/datasets/tidmad/raw}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python environment not found at $PYTHON" >&2
    echo "Create it with: uv sync --no-dev" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# Muon's decorator is evaluated at import time and does not honor
# TORCHDYNAMO_DISABLE on every PyTorch release. The RHEL 8 compiler on the
# remote host cannot compile PyTorch 2.12's generated C++20 kernel, so use the
# numerically identical eager Newton--Schulz path for both arms.
export RECONSTRUCTION_DISABLE_TORCH_COMPILE="${RECONSTRUCTION_DISABLE_TORCH_COMPILE:-1}"
export PYTHONUNBUFFERED=1          # so `screen -L` and `tee` see output live
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

case "$PROFILE" in
    check)   CONFIG=""                            ; TAG="check" ;;
    smoke)   CONFIG="src/tidmad/configs/t_mse.yaml"   ; TAG="smoke" ;;
    bench)   CONFIG="src/tidmad/configs/t_mse.yaml"   ; TAG="bench" ;;
    mse)     CONFIG="src/tidmad/configs/t_mse.yaml"   ; TAG="t_mse" ;;
    chi2)    CONFIG="src/tidmad/configs/t_chi2.yaml"  ; TAG="t_chi2" ;;
    chi2_of) CONFIG="src/tidmad/configs/t_chi2_of.yaml"; TAG="t_chi2_of" ;;
    *)
        echo "Usage: $0 [check|smoke|mse|chi2|chi2_of]" >&2
        exit 2
        ;;
esac

OUT_DIR="${TIDMAD_OUT_DIR:-$REPO_ROOT/artifacts/tidmad/$TAG/$RUN_STAMP}"

# ---------------------------------------------------------------- GPU preflight
echo "Checking GPU index $GPU_INDEX..."
EXPECT_GPU_NAME="${EXPECT_GPU_NAME:-}" "$PYTHON" - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Run this on the machine that owns the GPU and has "
        "a working NVIDIA driver."
    )
if torch.cuda.device_count() != 1:
    raise RuntimeError(
        "Expected exactly one visible GPU after CUDA_VISIBLE_DEVICES filtering, "
        f"found {torch.cuda.device_count()}. Set GPU_INDEX to a single index."
    )

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)

expect = os.environ.get("EXPECT_GPU_NAME", "").strip()
if expect and expect.upper() not in name.upper():
    raise RuntimeError(f"Selected GPU is {name!r}, expected it to contain {expect!r}.")

# Minimum capability: 5.2 (Maxwell), not 7.0.
#
# This check previously required 7.0 with the message "BF16/FP16 path
# unsupported", inherited from the DELight launcher whose training loop used
# BF16/FP16 autocast. `tidmad.train` has no autocast, no GradScaler and no
# half-precision anywhere: it is pure FP32. Requiring 7.0 therefore rejected
# hardware that runs this loop correctly — it blocked the GTX TITAN X
# (capability 5.2) on 2026-08-11 for a code path that does not exist.
#
# If a mixed-precision path is ever added, raise this back to (7, 0) *and* say
# so in the same commit.
MIN_CAPABILITY = (5, 2)
if cap < MIN_CAPABILITY:
    raise RuntimeError(
        f"Compute capability {cap} < {MIN_CAPABILITY}; FP32 training unsupported."
    )

# Force allocation, arithmetic and synchronisation: catches driver/ECC faults
# before any data indexing or checkpoint creation begins. FP32 deliberately —
# it must exercise the same precision the training loop uses.
x = torch.randn((2048, 2048), device="cuda", dtype=torch.float32)
y = x @ x
checksum = y.mean().item()
torch.cuda.synchronize()
del x, y
torch.cuda.empty_cache()

free, total = torch.cuda.mem_get_info()
print(
    "TIDMAD_GPU_OK "
    f"name={name!r} capability={cap[0]}.{cap[1]} "
    f"total_mib={props.total_memory // (1024 * 1024)} free_mib={free // (1024 * 1024)} "
    f"checksum={checksum:.6f}"
)
PY

# --------------------------------------------------------------- data preflight
echo "Checking dataset at $TIDMAD_DATA_ROOT..."
if [ ! -d "$TIDMAD_DATA_ROOT" ]; then
    echo "ERROR: dataset root not found: $TIDMAD_DATA_ROOT" >&2
    echo "Datasets live under /ceph/dwong/datasets (see EXPERIMENT_CHECKLIST §0.1)." >&2
    exit 1
fi

TIDMAD_DATA_ROOT="$TIDMAD_DATA_ROOT" "$PYTHON" - <<'PY'
import glob
import os
import sys

import h5py

root = os.environ["TIDMAD_DATA_ROOT"]
groups = {
    "training":   sorted(glob.glob(os.path.join(root, "abra_training_*.h5"))),
    "validation": sorted(glob.glob(os.path.join(root, "abra_validation_*.h5"))),
    "science":    sorted(glob.glob(os.path.join(root, "abra_science_*.h5"))),
}
for name, files in groups.items():
    print(f"  {name}: {len(files)} files")

if not groups["training"]:
    sys.exit("ERROR: no abra_training_*.h5 found; training cannot proceed.")
if not groups["science"]:
    sys.exit("ERROR: no abra_science_*.h5 found; J(f) is estimated from noise-only science files.")

# Channel semantics are the release convention and the whole contract rests on
# them. If these names differ, every downstream number is suspect.
probe = groups["training"][0]
with h5py.File(probe, "r") as fh:
    channels = list(fh["timeseries"])
print(f"  channels in {os.path.basename(probe)}: {channels}")
for required in ("channel0001", "channel0002"):
    if required not in channels:
        sys.exit(f"ERROR: {required} missing from {probe}. Data contract no longer holds — STOP.")
print("TIDMAD_DATA_OK")
PY

if [ "$PROFILE" = "check" ]; then
    echo "Preflight passed. No training requested."
    exit 0
fi

# ------------------------------------------------------------------- run set-up
mkdir -p "$OUT_DIR"

RESUME_ARG=()
if [ "${TIDMAD_RESUME:-}" = "auto" ] || [ -n "${TIDMAD_RESUME:-}" ]; then
    RESUME_ARG=(--resume "${TIDMAD_RESUME}")
    echo "Resume: ${TIDMAD_RESUME}"
fi

WANDB_ARG=()
if [ "${TIDMAD_WANDB:-0}" = "1" ]; then
    WANDB_ARG=(--wandb)
fi

STEPS_ARG=()
if [ "$PROFILE" = "smoke" ]; then
    STEPS_ARG=(--steps "${TIDMAD_STEPS:-20}")
elif [ "$PROFILE" = "bench" ]; then
    # Enough steps that the reported rate is not dominated by warmup.
    STEPS_ARG=(--steps "${TIDMAD_STEPS:-200}")
elif [ -n "${TIDMAD_STEPS:-}" ]; then
    STEPS_ARG=(--steps "$TIDMAD_STEPS")
fi

TIMEOUT_PREFIX=()
if [ -n "${TIDMAD_TIMEOUT:-}" ]; then
    TIMEOUT_PREFIX=(timeout "$TIDMAD_TIMEOUT")
    echo "Wall-clock cap: $TIDMAD_TIMEOUT"
elif [ "$PROFILE" != "smoke" ]; then
    echo "WARNING: no TIDMAD_TIMEOUT set. On a shared machine a runaway run has" >&2
    echo "         nothing to stop it. Set TIDMAD_TIMEOUT (e.g. 6h) from the" >&2
    echo "         throughput measured by the smoke profile." >&2
fi

echo "$$" > "$OUT_DIR/run.pid"

echo "Starting TIDMAD training"
echo "  Profile:  $PROFILE"
echo "  Config:   $CONFIG"
echo "  Data:     $TIDMAD_DATA_ROOT"
echo "  Out:      $OUT_DIR"
echo "  GPU:      index $GPU_INDEX"
echo "  PID:      $$ (recorded at $OUT_DIR/run.pid)"
echo "  Git:      $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"

nice -n 10 "${TIMEOUT_PREFIX[@]}" \
    "$PYTHON" -m tidmad.train \
        --config "$CONFIG" \
        --data-dir "$TIDMAD_DATA_ROOT" \
        --out "$OUT_DIR" \
        "${STEPS_ARG[@]}" "${RESUME_ARG[@]}" "${WANDB_ARG[@]}" \
    2>&1 | tee "$OUT_DIR/training.log"

STATUS=${PIPESTATUS[0]}
rm -f "$OUT_DIR/run.pid"

if [ "$STATUS" -eq 124 ]; then
    echo "Run hit the wall-clock cap ($TIDMAD_TIMEOUT) and was terminated." >&2
    echo "This is a clean stop, not a crash. Resume from the latest checkpoint." >&2
fi

if [ "$PROFILE" = "bench" ] && [ "$STATUS" -eq 0 ]; then
    echo
    echo "=== measured throughput -> budget ==="
    "$PYTHON" "$REPO_ROOT/scripts/budget_from_rate.py" --log "$OUT_DIR/training.log" || \
        echo "  could not parse a rate; read steps/s from $OUT_DIR/training.log manually"
fi

echo "Exit status: $STATUS. Saved files:"
find "$OUT_DIR" -maxdepth 1 -type f -printf '%f\n' | sort

exit "$STATUS"
