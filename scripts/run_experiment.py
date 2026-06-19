"""Run one config-driven Paper 1 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.experiments import run_synthetic_experiment
from noise_geometry.cresst import run_cresst_experiment
from noise_geometry.gwosc import run_gwosc_experiment
from noise_geometry.utils import RunRecord, git_commit_hash, load_config, resolve_data_root, write_run_record


def _default_output(config: dict) -> Path:
    experiment_id = str(config.get("experiment_id", config.get("id", "experiment"))).lower()
    return REPO_ROOT / "results" / "metrics" / f"{experiment_id}_seed{config.get('seed', 'na')}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = resolve_data_root(args.data_root, config)
    config["resolved_data_root"] = str(data_root)

    dataset = str(config.get("dataset", "synthetic")).lower()
    if dataset == "synthetic":
        metrics = run_synthetic_experiment(config)
        status = "complete" if metrics.get("status") != "planned" else "planned"
    elif dataset == "gwosc":
        metrics = run_gwosc_experiment(config, data_root)
        status = (
            "complete"
            if bool(metrics.get("acceptance", {}).get("passed", False))
            else "failed_acceptance"
        )
    elif dataset == "cresst":
        metrics = run_cresst_experiment(config, data_root)
        status = "complete"
    elif dataset == "tidmad":
        metrics = {
            "experiment": config.get("experiment_id", dataset),
            "status": "planned",
            "message": "TIDMAD remains optional until GWOSC and CRESST are stable.",
        }
        status = "planned"
    else:
        raise ValueError(f"unknown dataset type: {dataset}")

    output = args.output or Path(config.get("output", _default_output(config)))
    record = RunRecord(
        experiment_id=str(config.get("experiment_id", config.get("id", dataset))),
        status=status,
        metrics=metrics,
        config=config,
        dataset_metadata={"dataset": dataset, "data_root": str(data_root)},
        preprocessing_metadata=config.get("preprocessing", {}),
        model_metadata=config.get("models", {}),
        git_commit=git_commit_hash(REPO_ROOT),
    )
    path = write_run_record(record, output)
    config_copy = path.with_suffix(".config.yaml")
    config_copy.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    log_path = path.with_suffix(".log")
    log_path.write_text(
        f"experiment_id={record.experiment_id}\nstatus={record.status}\nmetrics={path}\nconfig={config_copy}\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
