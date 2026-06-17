import json
from pathlib import Path
import subprocess
import sys


def test_config_driven_synthetic_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "s0.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/run_experiment.py"),
            "--config",
            str(repo_root / "configs/synthetic/s0_smoke.yaml"),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "S0"
    assert record["status"] == "complete"
    assert record["metrics"]["of_truth_corr"] > 0.8
