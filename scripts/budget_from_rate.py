"""Turn a measured training rate into the wall-clock budget decision.

Used by ``train_tidmad_local.sh bench``. Safety rules 3 and 7 of the remote
execution prompt both require a *measured* rate before a full arm is launched;
this turns that measurement into the two numbers that actually get chosen:
``TIDMAD_TIMEOUT`` and ``num_steps``.

    python scripts/budget_from_rate.py --rate 12.4
    python scripts/budget_from_rate.py --log artifacts/tidmad/bench/<stamp>/training.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CANDIDATE_STEPS = (5_000, 10_000, 25_000, 50_000, 150_000)
FROZEN_STEPS = 150_000

_RATE = re.compile(r"([0-9]+\.[0-9]+)\s*steps/s")


def rate_from_log(path: Path, tail: int = 3) -> float:
    """Mean of the last ``tail`` reported rates; later ones exclude warmup."""
    rates = [float(m.group(1)) for m in _RATE.finditer(path.read_text())]
    if not rates:
        raise SystemExit(f"no 'N.NNN steps/s' found in {path}")
    use = rates[-tail:]
    return sum(use) / len(use)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rate", type=float, help="measured steps/second")
    src.add_argument("--log", type=Path, help="training.log to parse")
    ap.add_argument("--arms", type=int, default=2, help="number of arms to run")
    args = ap.parse_args()

    rate = args.rate if args.rate is not None else rate_from_log(args.log)
    if rate <= 0:
        raise SystemExit(f"rate must be positive, got {rate}")

    print(f"  measured rate: {rate:.3f} steps/s")
    print()
    print(f"  {'steps':>8}  {'per arm':>10}  {f'{args.arms} arms':>10}  note")
    for n in CANDIDATE_STEPS:
        hours = n / rate / 3600
        note = "frozen budget" if n == FROZEN_STEPS else ""
        if hours > 24:
            note = (note + "  overnight is not enough").strip()
        print(f"  {n:>8}  {hours:>9.1f} h  {args.arms * hours:>9.1f} h  {note}")

    frozen_hours = FROZEN_STEPS / rate / 3600
    print()
    if frozen_hours <= 6:
        print(f"  The frozen {FROZEN_STEPS:,} steps fits in {frozen_hours:.1f} h per arm.")
        print(f"  Set TIDMAD_TIMEOUT above that (e.g. {max(1, int(frozen_hours * 1.5))}h) and change nothing else.")
    else:
        print(f"  The frozen {FROZEN_STEPS:,} steps needs {frozen_hours:.1f} h per arm.")
        print("  Either run resumable segments with TIDMAD_RESUME=auto, or reduce")
        print("  num_steps to a budget above — and RE-FREEZE if you change it, since")
        print("  num_steps is part of the predeclared configuration.")
    print()
    print("  Both arms must use the SAME num_steps. The contrast is only valid at")
    print("  matched budget; that is what the config-equality assertion enforces.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
