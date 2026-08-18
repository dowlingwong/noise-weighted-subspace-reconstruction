"""Round 9 / Phase 1: E7 gates and the rung-4 seed-replication analysis.

E7 (the second rung-4 seed, 20260818) was already complete on the machine when
Round 9 started: both arms hold ``checkpoint_00025000.pt`` and a 50-evaluation
training log. Nothing is relaunched here. This script applies the four Round 8
gates, extracts both arms' evaluation series, and writes the registered
summary comparing the new seed with the archived round-6 seed.

Sign convention throughout: ``t_chi2 - t_mse``. A negative difference means the
chi-square-trained arm is better on that metric (all four metrics are errors).

Every round-6 number is recomputed from the archived evaluation series, never
transcribed.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "rps2026"
SEED = 20260818
ROUND6_SEED = 20260808
ARMS = ("t_mse", "t_chi2")
EXPECTED_STEPS = 25000
EXPECTED_EVALS = 50
EXPECTED_PARAMS = 569352
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260818

STEP_RE = re.compile(
    r"^step\s+(\d+)\s+train_mse=(\S+)\s+train_chi2=(\S+)\s+val_mse=(\S+)\s+val_chi2=(\S+)\s+"
    r"(\S+)\s+steps/s\s+eta\s+(\S+)\s+h"
)
SERIES_COLUMNS = ["step", "train_mse", "train_chi2", "val_mse", "val_chi2", "steps_per_s", "eta_s"]
METRICS = ("val_chi2", "val_mse", "train_chi2", "train_mse")


def run_dir(arm: str) -> Path:
    base = REPO / "artifacts" / "tidmad" / f"{arm}_s{SEED}"
    stamps = sorted(p for p in base.iterdir() if p.is_dir())
    if not stamps:
        raise SystemExit(f"no run directory under {base}")
    return stamps[-1]


def parse_log(path: Path) -> tuple[list[dict], dict]:
    rows, banner = [], {}
    for line in path.read_text().splitlines():
        m = STEP_RE.match(line)
        if m:
            step, tr_mse, tr_chi2, va_mse, va_chi2, sps, eta_h = m.groups()
            rows.append(
                {
                    "step": int(step),
                    "train_mse": float(tr_mse),
                    "train_chi2": float(tr_chi2),
                    "val_mse": float(va_mse),
                    "val_chi2": float(va_chi2),
                    "steps_per_s": float(sps),
                    "eta_s": float(eta_h) * 3600.0,
                }
            )
        elif line.startswith("[tidmad.train]"):
            if (m2 := re.search(r"seed=(\d+)\s+steps=(\d+)", line)):
                banner["seed"], banner["steps"] = int(m2.group(1)), int(m2.group(2))
            if (m3 := re.search(r"arm loss=(\S+)\s+model_params=(\d+)", line)):
                banner["loss"], banner["model_params"] = m3.group(1), int(m3.group(2))
            if (m4 := re.search(r"windows:\s+(\d+) total,\s+(\d+) fit,\s+(\d+) held out", line)):
                banner["windows"] = [int(m4.group(i)) for i in (1, 2, 3)]
            if (m5 := re.search(r"stopped at step (\d+)/(\d+) after ([\d.]+) h", line)):
                banner["final_step"] = int(m5.group(1))
                banner["target_steps"] = int(m5.group(2))
                banner["wall_hours"] = float(m5.group(3))
    return rows, banner


def read_series(path: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(path.open()):
        out.append({k: (int(r[k]) if k == "step" else float(r[k])) for k in SERIES_COLUMNS})
    return out


def bootstrap(diffs: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, diffs.size, size=(BOOTSTRAP_RESAMPLES, diffs.size))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_evaluations": int(diffs.size),
        "mean_difference": float(diffs.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "fraction_favouring_chi2_arm": float(np.mean(diffs < 0.0)),
    }


def paired(mse_rows, chi2_rows, metric: str) -> dict:
    steps_a = [r["step"] for r in mse_rows]
    steps_b = [r["step"] for r in chi2_rows]
    if steps_a != steps_b:
        raise SystemExit(f"step grids differ for {metric}; the arms are not comparable")
    d = np.array([c[metric] - m[metric] for m, c in zip(mse_rows, chi2_rows)], dtype=float)
    out = bootstrap(d)
    out["metric"] = metric
    out["contrast"] = "t_chi2 - t_mse"
    return out


def at_step(rows, step): return next(r for r in rows if r["step"] == step)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    series, banners, logs = {}, {}, {}
    for arm in ARMS:
        d = run_dir(arm)
        logs[arm] = d / "training.log"
        rows, banner = parse_log(logs[arm])
        series[arm], banners[arm] = rows, banner
        cfg = json.loads((d / "run_config.json").read_text())
        banner["run_config"] = cfg
        banner["run_dir"] = str(d.relative_to(REPO))
        banner["final_checkpoint_exists"] = (d / f"checkpoint_{EXPECTED_STEPS:08d}.pt").exists()

    # ---------------------------------------------------------------- gates
    gates = {}

    g1_seed = all(banners[a].get("seed") == SEED for a in ARMS)
    g1_windows = banners["t_mse"].get("windows") == banners["t_chi2"].get("windows")
    gates["gate_1_same_seed_and_data_root"] = {
        "passed": bool(g1_seed and g1_windows),
        "seeds": {a: banners[a].get("seed") for a in ARMS},
        "window_partition": {a: banners[a].get("windows") for a in ARMS},
        "note": (
            "The seed-specific screen logs were not retained, so the resolved data-root "
            "string is not available for these two arms. Data identity is instead "
            "established by content: both arms report an identical window partition "
            "(10067 total / 7046 fit / 3017 held out), which a different data root or "
            "guard setting could not reproduce."
        ),
    }

    cfgs = {a: banners[a]["run_config"]["run_config"] for a in ARMS}
    diff_fields = []
    def walk(pa, pb, prefix=""):
        for k in sorted(set(pa) | set(pb)):
            va, vb = pa.get(k), pb.get(k)
            if isinstance(va, dict) and isinstance(vb, dict):
                walk(va, vb, f"{prefix}{k}.")
            elif va != vb:
                diff_fields.append({"field": f"{prefix}{k}", "t_mse": va, "t_chi2": vb})
    walk(cfgs["t_mse"], cfgs["t_chi2"])
    params_ok = all(banners[a].get("model_params") == EXPECTED_PARAMS for a in ARMS)
    gates["gate_2_configs_differ_only_in_loss"] = {
        "passed": bool(params_ok and [d["field"] for d in diff_fields] == ["train.loss"]),
        "model_params": {a: banners[a].get("model_params") for a in ARMS},
        "expected_model_params": EXPECTED_PARAMS,
        "differing_fields": diff_fields,
    }

    gates["gate_3_both_reached_target_steps"] = {
        "passed": all(
            banners[a].get("final_step") == EXPECTED_STEPS
            and banners[a].get("final_checkpoint_exists")
            for a in ARMS
        ),
        "final_step": {a: banners[a].get("final_step") for a in ARMS},
        "wall_hours": {a: banners[a].get("wall_hours") for a in ARMS},
        "cap_hours": 8,
        "final_checkpoint_exists": {a: banners[a].get("final_checkpoint_exists") for a in ARMS},
    }

    steps = {a: [r["step"] for r in series[a]] for a in ARMS}
    gates["gate_4_identical_evaluation_grids"] = {
        "passed": bool(
            steps["t_mse"] == steps["t_chi2"] and len(steps["t_mse"]) == EXPECTED_EVALS
        ),
        "n_evaluations": {a: len(steps[a]) for a in ARMS},
        "expected": EXPECTED_EVALS,
        "grids_identical": steps["t_mse"] == steps["t_chi2"],
    }

    all_pass = all(g["passed"] for g in gates.values())

    # ------------------------------------------------- write the eval series
    written = {}
    for arm in ARMS:
        txt = RESULTS / f"E7_{arm}_s{SEED}_eval_series.txt"
        raw = [l for l in logs[arm].read_text().splitlines() if STEP_RE.match(l)]
        txt.write_text("\n".join(raw) + "\n")
        csv_path = RESULTS / f"E7_{arm}_s{SEED}_eval_series.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SERIES_COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(series[arm])
        written[arm] = {"txt": str(txt.relative_to(REPO)), "csv": str(csv_path.relative_to(REPO))}

    if not all_pass:
        payload = {"gates": gates, "all_gates_passed": False,
                   "note": "Analysis withheld: a load-bearing comparability gate failed."}
        (RESULTS / "E7_rung4_seed_replication.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("E7 GATE FAILURE", file=sys.stderr)
        print(json.dumps(gates, indent=2, sort_keys=True), file=sys.stderr)
        return 4

    # ---------------------------------------------------------- new seed
    new = {m: paired(series["t_mse"], series["t_chi2"], m) for m in METRICS}
    final = {
        a: {m: at_step(series[a], EXPECTED_STEPS)[m] for m in METRICS} for a in ARMS
    }
    last10 = {
        a: {m: float(np.mean([r[m] for r in series[a][-10:]])) for m in METRICS} for a in ARMS
    }

    # ------------------------------------------------ archived round-6 seed
    old_series = {a: read_series(RESULTS / f"E5_{a}_eval_series.csv") for a in ARMS}
    old = {m: paired(old_series["t_mse"], old_series["t_chi2"], m) for m in METRICS}

    agreement = {}
    for m in METRICS:
        so, sn = np.sign(old[m]["mean_difference"]), np.sign(new[m]["mean_difference"])
        agreement[m] = {
            "round6_seed": ROUND6_SEED,
            "round6_mean_difference": old[m]["mean_difference"],
            "round9_seed": SEED,
            "round9_mean_difference": new[m]["mean_difference"],
            "sign_agrees": bool(so == sn),
            "both_intervals_exclude_zero": bool(
                old[m]["excludes_zero"] and new[m]["excludes_zero"]
            ),
        }

    payload = {
        "experiment": "E7_rung4_seed_replication",
        "status": "complete_on_arrival_not_relaunched",
        "sign_convention": (
            "t_chi2 minus t_mse on every metric. All four metrics are errors, so a "
            "negative difference favours the chi-square-trained arm."
        ),
        "gates": gates,
        "all_gates_passed": True,
        "seeds": {"round6": ROUND6_SEED, "round9": SEED},
        "eval_series_files": written,
        "round9_seed_paired_differences": new,
        "round6_seed_paired_differences": old,
        "sign_agreement": agreement,
        "held_out_at_final_step": final,
        "mean_over_last_ten_evaluations": last10,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "evaluation (paired at identical step values)",
        },
        "interpretation_boundary": (
            "The resampling unit is the evaluation, not an independent training run, so "
            "these intervals describe the paired difference along one pair of trajectories. "
            "With two seeds the seed-level uncertainty itself remains unquantified."
        ),
        "provenance": {
            "code_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True
            ).stdout.strip(),
            "run_dirs": {a: banners[a]["run_dir"] for a in ARMS},
            "wall_hours": {a: banners[a]["wall_hours"] for a in ARMS},
        },
    }
    out = RESULTS / "E7_rung4_seed_replication.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": {k: v["passed"] for k, v in gates.items()},
                      "sign_agreement": {m: agreement[m]["sign_agrees"] for m in METRICS}},
                     indent=2, sort_keys=True))
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
