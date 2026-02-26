#!/usr/bin/env python3
"""Run legacy trace generation from the new repo entrypoint."""

from pathlib import Path
import runpy


def main():
    script = Path(__file__).resolve().parents[1] / "PCA_dev" / "reusable" / "make_traces_pair.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
