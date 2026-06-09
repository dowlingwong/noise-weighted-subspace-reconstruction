#!/usr/bin/env python3
"""Run the Paper 2 training suite on a server.

Examples
--------
Check the full plan without training:

    PYTHONPATH=. python scripts/run_paper2_training_suite.py --suite all --dry-run

Run every available Paper 2 experiment on a CUDA server:

    PYTHONPATH=. python scripts/run_paper2_training_suite.py \
      --suite all \
      --require-cuda \
      --num-workers 4 \
      --pin-memory \
      --run-suffix server01

Run only the transformer 2x2 ablation:

    PYTHONPATH=. python scripts/run_paper2_training_suite.py \
      --suite transformer_2x2 \
      --require-cuda

If the data live outside the repository, override the defaults with
`--trace-path`, `--rq-path`, and `--psd-path`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "paper2" / "configs"
RESULTS_DIR = REPO_ROOT / "paper2" / "results"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    phase: str
    kind: str
    config_name: str | None
    model: str
    dataset: str
    purpose: str


def cfg_run(
    config_name: str,
    phase: str,
    model: str,
    purpose: str,
) -> RunSpec:
    return RunSpec(
        run_id=config_name,
        phase=phase,
        kind="paper2_config",
        config_name=config_name,
        model=model,
        dataset=(
            "HDF5 detector traces from data.trace_path plus optional RQ labels "
            "from data.rq_path; PSD from preprocessing.psd_path."
        ),
        purpose=purpose,
    )


NFPA_RUN = RunSpec(
    run_id="nfpa_demo",
    phase="nfpa",
    kind="subprocess",
    config_name=None,
    model="NFPA, EMPCA, raw-MSE factored AE, raw PCA",
    dataset=(
        "Synthetic two-channel Brownian-noise pulse dataset generated inside "
        "src/NFPA/nfpa_demo.py: C=2, T=256, N_train=800, N_test=300."
    ),
    purpose=(
        "Current Paper 2 proof-of-concept: metric reversal, restricted "
        "Kronecker bridge, rank-one NFPA to OF check."
    ),
)


AE_2X2 = [
    cfg_run(
        "ae_raw_mse",
        "ae_2x2",
        "small convolutional reconstruction AE",
        "Nonlinear AE baseline with raw input and misspecified isotropic MSE.",
    ),
    cfg_run(
        "ae_raw_mahalanobis",
        "ae_2x2",
        "small convolutional reconstruction AE",
        "Separates loss geometry from input whitening: raw input, Mahalanobis loss.",
    ),
    cfg_run(
        "ae_prewhite_mse",
        "ae_2x2",
        "small convolutional reconstruction AE",
        "Tests whether prewhitened encoder input alone changes the learned solution.",
    ),
    cfg_run(
        "ae_prewhite_mahalanobis",
        "ae_2x2",
        "small convolutional reconstruction AE",
        "Geometry-first AE reference: prewhitened encoder input and Mahalanobis loss.",
    ),
]

TRANSFORMER_2X2 = [
    cfg_run(
        "transformer_raw_mse",
        "transformer_2x2",
        "token-preserving transformer reconstruction AE",
        "Wrong metric / raw-token transformer baseline.",
    ),
    cfg_run(
        "transformer_raw_mahalanobis",
        "transformer_2x2",
        "token-preserving transformer reconstruction AE",
        "Tests whether Mahalanobis loss helps without prewhitening attention input.",
    ),
    cfg_run(
        "transformer_prewhite_mse",
        "transformer_2x2",
        "token-preserving transformer reconstruction AE",
        "Tests attention on prewhitened tokens with isotropic native-output MSE.",
    ),
    cfg_run(
        "transformer_prewhite_mahalanobis",
        "transformer_2x2",
        "token-preserving transformer reconstruction AE",
        "Strongest current transformer config: prewhitened tokens and Mahalanobis loss.",
    ),
]

ARCHITECTURE = [
    cfg_run(
        "experiment_d_linear_prewhite_mahalanobis",
        "architecture",
        "patch-linear reconstruction AE",
        "Fixed geometry architecture-bias baseline.",
    ),
    cfg_run(
        "experiment_d_cnn_prewhite_mahalanobis",
        "architecture",
        "small convolutional reconstruction AE",
        "Fixed geometry CNN/locality inductive-bias comparison.",
    ),
    cfg_run(
        "experiment_d_transformer_prewhite_mahalanobis",
        "architecture",
        "token-preserving transformer reconstruction AE",
        "Fixed geometry attention inductive-bias comparison.",
    ),
]

SUITES: dict[str, list[RunSpec]] = {
    "nfpa": [NFPA_RUN],
    "ae_2x2": AE_2X2,
    "transformer_2x2": TRANSFORMER_2X2,
    "architecture": ARCHITECTURE,
    "all": [NFPA_RUN, *AE_2X2, *TRANSFORMER_2X2, *ARCHITECTURE],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="all")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated run IDs to execute, e.g. ae_raw_mse,transformer_raw_mse.",
    )
    parser.add_argument("--trace-path", default=None, help="Override data.trace_path for all YAML configs.")
    parser.add_argument(
        "--rq-path",
        default=None,
        help="Override data.rq_path for all YAML configs. Use 'none' to disable RQ loading.",
    )
    parser.add_argument(
        "--psd-path",
        default=None,
        help="Override preprocessing.psd_path for all YAML configs.",
    )
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--run-suffix",
        default=None,
        help="Append this suffix to experiment.name in copied configs to avoid overwriting old runs.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-label",
        default=None,
        help="Folder name under paper2/results/_suite_runs. Defaults to a timestamp.",
    )
    return parser.parse_args()


def require_yaml():
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - server environment check
        raise RuntimeError("PyYAML is required. Install it with `pip install pyyaml`.") from exc
    return yaml


def load_yaml(path: Path) -> dict[str, Any]:
    yaml = require_yaml()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, raw: dict[str, Any]) -> None:
    yaml = require_yaml()
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)


def resolve_repo_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def selected_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs = list(SUITES[args.suite])
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        specs = [spec for spec in specs if spec.run_id in wanted]
        missing = wanted - {spec.run_id for spec in specs}
        if missing:
            raise SystemExit(f"Unknown --only run ID(s): {', '.join(sorted(missing))}")
    return specs


def apply_overrides(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw = json.loads(json.dumps(raw))
    if args.trace_path is not None:
        raw["data"]["trace_path"] = args.trace_path
    if args.rq_path is not None:
        raw["data"]["rq_path"] = None if args.rq_path.lower() in {"none", "null", ""} else args.rq_path
    if args.psd_path is not None:
        raw["preprocessing"]["psd_path"] = args.psd_path
    if args.max_events is not None:
        raw["data"]["max_events"] = args.max_events
    if args.epochs is not None:
        raw["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        raw["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        raw["data"]["num_workers"] = args.num_workers
    if args.pin_memory:
        raw["data"]["pin_memory"] = True
    if args.run_suffix:
        raw["experiment"]["name"] = f"{raw['experiment']['name']}_{args.run_suffix}"
    return raw


def validate_config(raw: dict[str, Any], spec: RunSpec) -> list[str]:
    errors: list[str] = []
    trace_path = resolve_repo_path(raw["data"].get("trace_path"))
    rq_path = resolve_repo_path(raw["data"].get("rq_path"))
    psd_path = resolve_repo_path(raw["preprocessing"].get("psd_path"))
    if trace_path is None or not trace_path.exists():
        errors.append(f"{spec.run_id}: missing trace_path {trace_path}")
    if rq_path is not None and not rq_path.exists():
        errors.append(f"{spec.run_id}: missing rq_path {rq_path}")
    if psd_path is None or not psd_path.exists():
        errors.append(f"{spec.run_id}: missing psd_path {psd_path}")
    if not errors:
        errors.extend(validate_hdf5_schema(raw, spec, trace_path, rq_path))
    return errors


def validate_hdf5_schema(
    raw: dict[str, Any],
    spec: RunSpec,
    trace_path: Path | None,
    rq_path: Path | None,
) -> list[str]:
    errors: list[str] = []
    try:
        import h5py
    except Exception:
        return errors

    if trace_path is not None:
        try:
            with h5py.File(trace_path, "r") as handle:
                if "traces" not in handle:
                    errors.append(f"{spec.run_id}: {trace_path} has no 'traces' dataset")
                else:
                    traces = handle["traces"]
                    expected_len = int(raw["data"]["trace_len"])
                    if len(traces.shape) != 2:
                        errors.append(
                            f"{spec.run_id}: traces must have shape (N, T), got {traces.shape}"
                        )
                    elif int(traces.shape[1]) != expected_len:
                        errors.append(
                            f"{spec.run_id}: trace_len={expected_len}, but traces shape is {traces.shape}"
                        )
        except Exception as exc:
            errors.append(f"{spec.run_id}: failed to inspect {trace_path}: {exc}")

    if rq_path is not None:
        try:
            with h5py.File(rq_path, "r") as handle:
                if "rqs" not in handle:
                    errors.append(f"{spec.run_id}: {rq_path} has no 'rqs' dataset")
                else:
                    rqs = handle["rqs"]
                    names = set(rqs.dtype.names or ())
                    required = {"A", "time_shift", "OF_ampl_0", "OF_time_0", "trace_index"}
                    missing = sorted(required - names)
                    if missing:
                        errors.append(
                            f"{spec.run_id}: rqs dataset is missing required field(s): {missing}"
                        )
        except Exception as exc:
            errors.append(f"{spec.run_id}: failed to inspect {rq_path}: {exc}")

    return errors


def dependency_errors(specs: list[RunSpec], require_cuda: bool) -> list[str]:
    errors: list[str] = []
    need_torch = any(spec.kind == "paper2_config" for spec in specs)
    need_nfpa = any(spec.run_id == "nfpa_demo" for spec in specs)
    for module_name in ["numpy"]:
        if importlib.util.find_spec(module_name) is None:
            errors.append(f"Missing Python module: {module_name}")
    if need_nfpa and importlib.util.find_spec("matplotlib") is None:
        errors.append("Missing Python module for NFPA demo: matplotlib")
    if need_torch:
        for module_name in ["h5py", "yaml"]:
            if importlib.util.find_spec(module_name) is None:
                errors.append(f"Missing Python module for paper2 training: {module_name}")
        try:
            import torch
        except Exception as exc:  # pragma: no cover - server environment check
            errors.append(f"Missing or broken torch install: {exc}")
        else:
            print(
                f"[suite] torch={torch.__version__} cuda_available={torch.cuda.is_available()}",
                flush=True,
            )
            if torch.cuda.is_available():
                print(
                    f"[suite] cuda_device={torch.cuda.get_device_name(0)} "
                    f"cuda_version={torch.version.cuda}",
                    flush=True,
                )
            elif require_cuda:
                errors.append("--require-cuda was set, but torch.cuda.is_available() is false")
    return errors


def prepare_manifest(
    specs: list[RunSpec],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    copied_configs: dict[str, Path] = {}
    validation_errors: list[str] = []

    for spec in specs:
        row = asdict(spec)
        if spec.kind == "paper2_config":
            src = CONFIG_DIR / f"{spec.config_name}.yaml"
            if not src.exists():
                validation_errors.append(f"{spec.run_id}: missing config {src}")
                continue
            raw = apply_overrides(load_yaml(src), args)
            validation_errors.extend(validate_config(raw, spec))
            dst = config_dir / f"{spec.run_id}.yaml"
            write_yaml(dst, raw)
            copied_configs[spec.run_id] = dst
            row.update(
                {
                    "resolved_config": str(dst),
                    "experiment_name": raw["experiment"]["name"],
                    "model_family": raw["model"]["family"],
                    "input_mode": raw["preprocessing"]["input_mode"],
                    "loss_mode": raw["loss"]["mode"],
                    "trace_path": raw["data"].get("trace_path"),
                    "rq_path": raw["data"].get("rq_path"),
                    "psd_path": raw["preprocessing"].get("psd_path"),
                    "max_events": raw["data"].get("max_events"),
                    "epochs": raw["training"].get("epochs"),
                    "batch_size": raw["data"].get("batch_size"),
                }
            )
        else:
            row.update(
                {
                    "resolved_config": None,
                    "experiment_name": "nfpa_demo",
                    "model_family": "nfpa_numpy",
                    "input_mode": "prewhitened",
                    "loss_mode": "chi2_als",
                }
            )
        manifest.append(row)

    if validation_errors:
        raise RuntimeError("\n".join(validation_errors))
    return manifest, copied_configs


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_manifest_csv(path: Path, manifest: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in manifest for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)


def print_plan(manifest: list[dict[str, Any]]) -> None:
    print("\n[suite] Training plan")
    for idx, row in enumerate(manifest, 1):
        print(
            f"  {idx:02d}. {row['run_id']} | phase={row['phase']} | "
            f"model={row['model']} | input={row.get('input_mode')} | "
            f"loss={row.get('loss_mode')}",
            flush=True,
        )
        print(f"      dataset: {row['dataset']}", flush=True)
        print(f"      purpose: {row['purpose']}", flush=True)


def run_nfpa(run_dir: Path) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "nfpa_demo.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "NFPA" / "nfpa_demo.py")],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"[suite] nfpa_demo log={log_path}", flush=True)


def run_paper2_config(config_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from paper2.trainers.train_reconstruction import run_experiment

    old_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        run_experiment(config_path)
    finally:
        os.chdir(old_cwd)


def expected_checkpoint(row: dict[str, Any]) -> Path | None:
    if row["kind"] != "paper2_config":
        return None
    return RESULTS_DIR / row["experiment_name"] / "checkpoint_best.pt"


def main() -> int:
    args = parse_args()
    specs = selected_specs(args)
    run_label = args.run_label or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / "_suite_runs" / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        dep_errors = dependency_errors(specs, args.require_cuda)
        if dep_errors:
            for error in dep_errors:
                print(f"[suite] ERROR: {error}", file=sys.stderr)
            return 2

    try:
        manifest, copied_configs = prepare_manifest(specs, args, run_dir)
    except Exception as exc:
        print(f"[suite] ERROR while preparing configs: {exc}", file=sys.stderr)
        return 2

    write_json(run_dir / "manifest.json", manifest)
    write_manifest_csv(run_dir / "manifest.csv", manifest)
    print_plan(manifest)
    print(f"\n[suite] manifest={run_dir / 'manifest.json'}", flush=True)

    if args.dry_run or args.check_only:
        print("[suite] no training was run", flush=True)
        return 0

    status: list[dict[str, Any]] = []
    for row in manifest:
        started = time.time()
        record = {
            "run_id": row["run_id"],
            "phase": row["phase"],
            "experiment_name": row["experiment_name"],
            "status": "started",
            "seconds": None,
            "error": None,
        }
        write_json(run_dir / "status.json", [*status, record])
        checkpoint = expected_checkpoint(row)
        if args.skip_existing and checkpoint is not None and checkpoint.exists():
            record["status"] = "skipped_existing"
            record["seconds"] = 0.0
            status.append(record)
            write_json(run_dir / "status.json", status)
            print(f"[suite] skipped existing {row['run_id']} checkpoint={checkpoint}", flush=True)
            continue

        print(f"\n[suite] START {row['run_id']}", flush=True)
        try:
            if row["kind"] == "subprocess":
                run_nfpa(run_dir)
            else:
                run_paper2_config(copied_configs[row["run_id"]])
            record["status"] = "completed"
        except Exception as exc:  # pragma: no cover - operational path
            record["status"] = "failed"
            record["error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            print(f"[suite] FAILED {row['run_id']}: {record['error']}", file=sys.stderr)
            if not args.continue_on_error:
                record["seconds"] = round(time.time() - started, 3)
                status.append(record)
                write_json(run_dir / "status.json", status)
                return 1
        finally:
            record["seconds"] = round(time.time() - started, 3)

        status.append(record)
        write_json(run_dir / "status.json", status)
        print(
            f"[suite] END {row['run_id']} status={record['status']} "
            f"seconds={record['seconds']}",
            flush=True,
        )

    failed = [row for row in status if row["status"] == "failed"]
    if failed:
        print(f"[suite] completed with {len(failed)} failed run(s)", file=sys.stderr)
        return 1
    print(f"[suite] complete status={run_dir / 'status.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
