"""Score a TIDMAD arm through the *unmodified* upstream ``benchmark.py``.

Reuses Paper 1's wrapper (``noise_geometry.tidmad.score.run_upstream_benchmark``),
which shells out to the release's script and captures the raw output. There is
deliberately no internal reimplementation of the score: a reimplementation that
disagreed by a normalization factor would not be comparable to the published
scoreboard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from noise_geometry.tidmad.score import run_upstream_benchmark


def score(upstream_dir: Path, data_dir: Path, model_tag: str, *, coarse: bool = True) -> dict:
    """Score ``abra_validation_denoised_<model_tag>_00NN.h5`` under ``data_dir``.

    Returns the captured command/returncode/stdout record. The raw stdout is the
    evidence; parsing the score from it is left to the caller.
    """
    record = run_upstream_benchmark(upstream_dir, data_dir, model_tag, coarse=coarse)
    print(f"command: {' '.join(record['command'])}")
    print(f"returncode: {record['returncode']}")
    print(record["stdout"])
    if record["stderr"]:
        print(record["stderr"], file=__import__("sys").stderr)
    if record["returncode"] != 0:
        raise RuntimeError(f"upstream benchmark.py failed with code {record['returncode']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True, help="directory containing TIDMAD benchmark.py")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--coarse", action="store_true", default=True)
    args = parser.parse_args()
    score(args.upstream, args.data_dir, args.model_tag, coarse=args.coarse)


if __name__ == "__main__":
    main()
