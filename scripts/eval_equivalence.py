#!/usr/bin/env python3
"""Run legacy OF/EMPCA strict equivalence notebook script if available."""

from pathlib import Path
import runpy


def main():
    script = Path(__file__).resolve().parents[1] / "PCA_dev" / "wk4" / "equivalence" / "verify_gpu_equivalence.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
