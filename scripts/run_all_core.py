"""Run all implemented synthetic Paper 1 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

CORE_CONFIGS = [
    "configs/synthetic/s0_smoke.yaml",
    "configs/synthetic/s1_of_crb.yaml",
    "configs/synthetic/s2_of_empca.yaml",
    "configs/synthetic/s3_ae_bridge.yaml",
    "configs/synthetic/s4_white_control.yaml",
    "configs/synthetic/s5_metric_reversal.yaml",
    "configs/synthetic/s6_timing_jitter_rank_sweep.yaml",
    "configs/synthetic/s7_covariance_robustness.yaml",
    "configs/synthetic/s8_residual_calibration.yaml",
    "configs/synthetic/s9_multichannel_covariance.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()

    for config in CORE_CONFIGS:
        cmd = [sys.executable, str(REPO_ROOT / "scripts/run_experiment.py"), "--config", str(REPO_ROOT / config)]
        if args.data_root is not None:
            cmd.extend(["--data-root", str(args.data_root)])
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
