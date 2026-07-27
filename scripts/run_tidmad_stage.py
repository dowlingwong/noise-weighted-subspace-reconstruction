"""One-command driver for the whole TIDMAD stage: download, run, archive.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md``.
Predeclaration: ``docs/results/TIDMAD_PREDECLARED_GATES.md``.

    # everything that can run without data (phases 1-4, 6, synthetic)
    uv run python scripts/run_tidmad_stage.py --mode synthetic

    # the real thing, from an empty data root
    uv run python scripts/run_tidmad_stage.py --mode file \
        --data-root /ceph/dwong/paper1_dataset \
        --tidmad-checkout ~/src/TIDMAD --download

    # see what would happen, touch nothing
    uv run python scripts/run_tidmad_stage.py --mode file --dry-run

Design notes
------------
**Resumable and idempotent.** Each step checks whether its output already
exists and skips unless ``--force`` is given. A stage that takes hours on a
2 GB download should never require starting over because one late step failed.

**Ordered by dependency, and it stops at the first hard failure.** Phase 0's
G0 gate is the clearest example: if the published baseline scores cannot be
reproduced, no later number is comparable to anything, so continuing would
manufacture a result rather than measure one. Soft failures -- a gate that
returns a negative *finding* rather than a bug -- are recorded and the run
continues, because a negative finding is evidence.

**Every step writes a run record**, and the driver ends by writing a manifest
with SHA-256 for each, so the archive is auditable rather than merely present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG_DIR = REPO_ROOT / "configs" / "tidmad"
EVIDENCE_DIR = REPO_ROOT / "evidence" / "tidmad"


@dataclass
class StepResult:
    name: str
    status: str  # "ok" | "skipped" | "soft_failure" | "hard_failure" | "would_run"
    detail: str = ""
    seconds: float = 0.0
    outputs: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return self.status == "hard_failure"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 86400) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _config_output(config_path: Path) -> Path | None:
    """Read the ``output:`` field of a config without importing yaml machinery."""
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    out = payload.get("output")
    return (REPO_ROOT / out) if out else None


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------
def step_download(args: argparse.Namespace) -> StepResult:
    """Fetch the small subset Phases 0-4 need (not the ~TB full release)."""
    destination = args.data_root / "tidmad" / "raw"
    existing = sorted(destination.glob("abra_*.h5")) if destination.exists() else []
    if existing and not args.force:
        return StepResult("download", "skipped", f"{len(existing)} files already present")
    if not args.download:
        return StepResult(
            "download",
            "skipped",
            "pass --download to fetch (and --tidmad-checkout for the upstream downloader)",
        )
    if args.tidmad_checkout is None:
        return StepResult(
            "download",
            "hard_failure",
            "--download requires --tidmad-checkout <path to a jessicafry/TIDMAD clone>",
        )
    if args.dry_run:
        return StepResult("download", "would_run", f"-> {destination}")

    start = time.time()
    result = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "download" / "download_tidmad.py"),
            "--data-root",
            str(args.data_root),
            "--upstream",
            str(args.tidmad_checkout),
            "--train-files",
            str(args.train_files),
            "--validation-files",
            str(args.validation_files),
            "--science-files",
            str(args.science_files),
            "--run",
        ]
    )
    if result.returncode != 0:
        return StepResult("download", "hard_failure", result.stderr.strip()[-500:], time.time() - start)
    files = sorted(destination.glob("abra_*.h5"))
    return StepResult(
        "download",
        "ok",
        f"{len(files)} files in {destination}",
        time.time() - start,
        [str(p) for p in files],
    )


def step_contract(args: argparse.Namespace) -> StepResult:
    """Phase 0a: establish channel semantics from the files themselves."""
    contract = args.contract
    if contract.exists() and not args.force:
        payload = json.loads(contract.read_text(encoding="utf-8"))
        state = "verified" if payload.get("verified") else "draft (verified=false)"
        return StepResult("phase0_contract", "skipped", f"{contract.name} exists: {state}")
    if args.dry_run:
        return StepResult("phase0_contract", "would_run", f"-> {contract}")

    start = time.time()
    result = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "tidmad_phase0_contract.py"),
            "--data-root",
            str(args.data_root),
            "--output",
            str(contract),
        ]
    )
    if result.returncode != 0:
        return StepResult(
            "phase0_contract",
            "hard_failure",
            "channel roles could not be established; resolve by hand before continuing. "
            + result.stderr.strip()[-400:],
            time.time() - start,
        )
    return StepResult(
        "phase0_contract", "ok", result.stdout.strip().splitlines()[-1], time.time() - start, [str(contract)]
    )


def _tidmad_python(checkout: Path) -> str:
    """Prefer a venv inside the checkout, if one exists, over our own interpreter.

    The release's code was written against numpy<2 (e.g. ``int8_array + 128``
    relies on numpy's old value-based scalar promotion, removed by NEP 50 in
    numpy 2.0). This project's own environment tracks current numpy, so
    running the unmodified upstream scripts under our interpreter raises
    OverflowError on exactly that line. Set up ``<checkout>/.venv`` with
    numpy<2 (see docs/runbooks/TIDMAD_STAGE_RUNBOOK.md) to avoid it.
    """
    venv_python = Path(checkout) / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def step_baseline_gate(args: argparse.Namespace) -> StepResult:
    """Phase 0b, gate G0: reproduce the published `none` and `savgol` scores.

    This is the hard gate of the whole stage. A denoising score we cannot
    reproduce is a score we cannot compare against, so a failure here stops
    everything downstream rather than producing numbers that merely look
    comparable to the published scoreboard.
    """
    if args.mode != "file":
        return StepResult("phase0_G0_baseline", "skipped", "synthetic mode: no upstream score to reproduce")
    if args.tidmad_checkout is None:
        return StepResult(
            "phase0_G0_baseline",
            "hard_failure",
            "gate G0 needs --tidmad-checkout: the score is defined by the release's benchmark.py",
        )
    output = EVIDENCE_DIR / "phase0" / "G0_baseline_reproduction.json"
    if output.exists() and not args.force:
        return StepResult("phase0_G0_baseline", "skipped", f"{output.name} exists")
    if args.dry_run:
        return StepResult("phase0_G0_baseline", "would_run", "runs upstream inference.py + benchmark.py")

    start = time.time()
    data_dir = args.data_root / "tidmad" / "raw"
    tidmad_python = _tidmad_python(args.tidmad_checkout)
    captured = {"tidmad_python": tidmad_python}
    for model in ("none", "savgol"):
        if model != "none":
            # -p/--parallel: the release's sequential inference.py path leaves
            # `model` unbound for classical (non-neural) denoisers -- its
            # elif chain has an empty `else: pass` for exactly this case --
            # and crashes with UnboundLocalError. The parallel path passes
            # None explicitly instead of `model`, which is correct for
            # non-neural methods, so it is the officially-supported way to
            # avoid the bug without touching upstream code.
            inference = _run(
                [tidmad_python, "inference.py", "-d", str(data_dir), "-m", model, "-p"],
                cwd=args.tidmad_checkout,
            )
            captured[f"{model}_inference_returncode"] = inference.returncode
            if inference.returncode != 0:
                return StepResult(
                    "phase0_G0_baseline",
                    "hard_failure",
                    f"upstream inference.py failed for {model}: {inference.stderr.strip()[-400:]}",
                    time.time() - start,
                )
        benchmark = _run(
            [tidmad_python, "benchmark.py", "-d", str(data_dir), "-m", model, "-c", "-p"],
            cwd=args.tidmad_checkout,
        )
        captured[model] = {
            "returncode": benchmark.returncode,
            "stdout": benchmark.stdout,
            "stderr": benchmark.stderr,
        }
        if benchmark.returncode != 0:
            return StepResult(
                "phase0_G0_baseline",
                "hard_failure",
                f"upstream benchmark.py failed for {model}: {benchmark.stderr.strip()[-400:]}",
                time.time() - start,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "gate": "G0",
                "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "upstream": str(args.tidmad_checkout),
                "data_dir": str(data_dir),
                "captured": captured,
                "note": (
                    "Raw upstream output is the evidence. Compare these scores against "
                    "the published values BY HAND and record the verdict in "
                    "docs/results/TIDMAD_PREDECLARED_GATES.md before setting "
                    "verified=true in the data contract."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return StepResult(
        "phase0_G0_baseline",
        "ok",
        "upstream scores captured; verify against the published values by hand",
        time.time() - start,
        [str(output)],
    )


def step_phase(args: argparse.Namespace, phase: int, config_name: str) -> StepResult:
    """Run one phase through the config-driven entry point."""
    config = CONFIG_DIR / config_name
    if not config.exists():
        return StepResult(f"phase{phase}", "hard_failure", f"missing config {config}")

    output = _config_output(config)
    if output and output.exists() and not args.force:
        return StepResult(f"phase{phase}", "skipped", f"{output.name} exists")
    if args.dry_run:
        return StepResult(f"phase{phase}", "would_run", f"config {config.name}")

    start = time.time()
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_experiment.py"),
        "--config",
        str(config),
        "--data-root",
        str(args.data_root),
    ]
    result = _run(command)
    elapsed = time.time() - start

    if result.returncode != 0:
        return StepResult(
            f"phase{phase}", "hard_failure", result.stderr.strip()[-600:], elapsed
        )
    if output is None or not output.exists():
        return StepResult(f"phase{phase}", "hard_failure", "no run record written", elapsed)

    record = json.loads(output.read_text(encoding="utf-8"))
    passed = bool(record.get("metrics", {}).get("all_gates_passed", False))
    failing = [
        name
        for name, gate in record.get("metrics", {}).get("gates", {}).items()
        if not gate.get("passed", False)
    ]
    # A failed gate here is a *finding*, not a crash: Phase 2 failing would be a
    # real scope limitation on the claim and Phase 4's real-data models are
    # expected to fail. The driver records and continues; only infrastructure
    # errors stop the run.
    status = "ok" if passed else "soft_failure"
    detail = "all gates passed" if passed else f"gates not passed: {', '.join(failing)}"
    return StepResult(f"phase{phase}", status, detail, elapsed, [str(output)])


def step_archive(args: argparse.Namespace, steps: list[StepResult]) -> StepResult:
    """Copy every run record into the evidence tree with a checksum manifest."""
    if args.dry_run:
        return StepResult("archive", "would_run", f"-> {EVIDENCE_DIR / args.run_id}")

    start = time.time()
    destination = EVIDENCE_DIR / args.run_id
    destination.mkdir(parents=True, exist_ok=True)

    files = []
    for step in steps:
        for path_str in step.outputs:
            path = Path(path_str)
            if not path.exists() or path.suffix != ".json":
                continue
            target = destination / path.name
            target.write_bytes(path.read_bytes())
            files.append(
                {
                    "path": target.name,
                    "step": step.name,
                    "sha256": _sha256(target),
                    "size_bytes": target.stat().st_size,
                }
            )

    manifest = {
        "schema": "tidmad-stage-evidence/1.0.0",
        "run_id": args.run_id,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": args.mode,
        "data_root": str(args.data_root),
        "git_commit": _git_commit(),
        "predeclaration": "docs/results/TIDMAD_PREDECLARED_GATES.md",
        "steps": [
            {"name": s.name, "status": s.status, "detail": s.detail, "seconds": round(s.seconds, 2)}
            for s in steps
        ],
        "files": files,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return StepResult(
        "archive", "ok", f"{len(files)} records -> {destination}", time.time() - start
    )


def _git_commit() -> str:
    result = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
SYNTHETIC_PHASES = [
    (1, "phase1_estimator_identity.yaml"),
    (2, "phase2_metric_reversal.yaml"),
    (3, "phase3_rank_manifold.yaml"),
    (4, "phase4_calibration_boundary.yaml"),
    (6, "phase6_negative_controls.yaml"),
    (6, "phase6_negative_controls_high_snr.yaml"),
]

FILE_PHASES = [
    (1, "phase1_estimator_identity_real_noise.yaml"),
    (2, "phase2_metric_reversal_real_noise.yaml"),
    (3, "phase3_rank_manifold_real_noise.yaml"),
    (4, "phase4_calibration_boundary_real_noise.yaml"),
    (6, "phase6_negative_controls_real_noise.yaml"),
    (5, "phase5_denoising_score.yaml"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "file"], default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("/ceph/dwong/paper1_dataset"))
    parser.add_argument(
        "--contract", type=Path, default=REPO_ROOT / "docs" / "tidmad_data_contract.json"
    )
    parser.add_argument("--tidmad-checkout", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-files", type=int, default=1)
    parser.add_argument("--validation-files", type=int, default=2)
    parser.add_argument("--science-files", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="re-run steps whose outputs exist")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()

    print(f"TIDMAD stage driver | mode={args.mode} | run_id={args.run_id}")
    print(f"predeclaration: docs/results/TIDMAD_PREDECLARED_GATES.md")
    if args.dry_run:
        print("DRY RUN: nothing will be written\n")

    steps: list[StepResult] = []

    def record(step: StepResult) -> bool:
        steps.append(step)
        marker = {
            "ok": "  ok  ",
            "skipped": " skip ",
            "soft_failure": " note ",
            "hard_failure": " FAIL ",
            "would_run": " plan ",
        }[step.status]
        timing = f" ({step.seconds:.1f}s)" if step.seconds > 0.05 else ""
        print(f"[{marker}] {step.name}: {step.detail}{timing}")
        return not step.blocking

    if args.mode == "file":
        if not record(step_download(args)):
            return _finish(args, steps, blocked=True)
        if not record(step_contract(args)):
            return _finish(args, steps, blocked=True)
        if not record(step_baseline_gate(args)):
            return _finish(args, steps, blocked=True)

    phases = SYNTHETIC_PHASES if args.mode == "synthetic" else FILE_PHASES
    for phase, config_name in phases:
        if not record(step_phase(args, phase, config_name)):
            return _finish(args, steps, blocked=True)

    if not args.skip_archive:
        record(step_archive(args, steps))

    _finish(args, steps, blocked=False)


def _finish(args: argparse.Namespace, steps: list[StepResult], *, blocked: bool) -> None:
    print("\n--- summary ---")
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.status] = counts.get(step.status, 0) + 1
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")

    findings = [s for s in steps if s.status == "soft_failure"]
    if findings:
        print("\nGates not passed (these are findings to report, not crashes):")
        for step in findings:
            print(f"  - {step.name}: {step.detail}")

    if blocked:
        print("\nSTOPPED at a hard failure. Nothing downstream was run, deliberately:")
        print("  a stage built on an unreproducible baseline would manufacture a result.")
        raise SystemExit(1)

    if findings:
        print("\nCompleted with findings. Record them in the write-up before publishing.")
    else:
        print("\nCompleted; all gates passed.")


if __name__ == "__main__":
    main()
