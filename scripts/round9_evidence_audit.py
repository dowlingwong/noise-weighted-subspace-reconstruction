"""Round 9 / Phase 4 - independent evidence audit.

Re-derives every check from the artifacts on disk rather than trusting the
runners' own manifests. In particular the summaries are regenerated from the
seed shards with the same deterministic functions and compared byte-for-byte,
so a summary that does not follow from its raw rows is caught here.

Exits nonzero if any check fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from experiments.synthetic.nfpa_confirmatory import r9_common as rc  # noqa: E402
from experiments.synthetic.nfpa_confirmatory import run_estimated_covariance as e10  # noqa: E402
from experiments.synthetic.nfpa_confirmatory import run_replication as e9  # noqa: E402
from experiments.synthetic.nfpa_confirmatory import run_semantics as e8  # noqa: E402
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash  # noqa: E402

R = REPO / "results" / "rps2026"
BASELINE_DIGEST = (R / "Round9_remote" / "E1_baseline_digest.txt").read_text().split()[0]

checks: list[dict] = []


def check(name: str, passed: bool, detail=None) -> bool:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    return bool(passed)


def audit_archive() -> None:
    digest = rc.sha256_of(rc.ARCHIVED_E1_CSV)
    check("archived_E1_digest_unchanged", digest == BASELINE_DIGEST,
          {"preflight": BASELINE_DIGEST, "now": digest})


def audit_shards(label, out_dir: Path, expected_seeds, expected_methods, key_fields):
    if not out_dir.exists():
        check(f"{label}_present", False, {"missing": str(out_dir)})
        return None
    shards = sorted(out_dir.glob("seed_*.csv"))
    rows = rc.concat_shards(shards)
    seeds = sorted({int(r["seed"]) for r in rows})
    check(f"{label}_exact_seed_set", seeds == sorted(expected_seeds),
          {"found": seeds, "expected": sorted(expected_seeds)})
    methods = sorted({r["method"] for r in rows})
    check(f"{label}_method_set", methods == sorted(expected_methods),
          {"found": methods, "expected": sorted(expected_methods)})
    check(f"{label}_row_count", len(rows) == len(expected_seeds) * len(expected_methods),
          {"rows": len(rows), "expected": len(expected_seeds) * len(expected_methods)})
    keys = [tuple(r[f] for f in key_fields) for r in rows]
    check(f"{label}_no_duplicate_keys", len(set(keys)) == len(keys),
          {"n_keys": len(keys), "n_unique": len(set(keys))})
    hashes = sorted({r["config_hash"] for r in rows})
    check(f"{label}_single_config_hash", len(hashes) == 1, {"hashes": hashes})
    return rows


def audit_e8():
    root = R / "E8_semantics"
    conds = {}
    for cond, spec in e8.CONDITIONS.items():
        d = root / spec["label"]
        rows = audit_shards(f"E8_{spec['label']}", d, range(10), rc.REGISTERED_METHODS,
                            ("seed", "method", "condition"))
        if rows is None:
            continue
        conds[cond] = (d, rows, spec)
        statuses = sorted({r["analysis_status"] for r in rows})
        check(f"E8_{spec['label']}_analysis_status",
              statuses == [spec["analysis_status"]], {"found": statuses})
        # regenerate the summary byte-for-byte from the shards
        cfg = replace(PRIMARY, nonseparability_alpha=spec["alpha"])
        regen = e8.summarize(rows, spec, cfg, sorted({int(r["seed"]) for r in rows}))
        want = json.dumps(regen, indent=2, sort_keys=True, allow_nan=True) + "\n"
        have = (d / "summary.json").read_text()
        check(f"E8_{spec['label']}_summary_regenerates_byte_for_byte", want == have)
        summ = json.loads(have)
        for m, block in summ["paired_contrasts_nfpa_minus_iso_mpca"].items():
            if block.get("identified"):
                check(f"E8_{spec['label']}_{m}_bootstrap_unit_is_seed",
                      block.get("resampling_unit") == "seed"
                      and block.get("n_seeds") == 10
                      and block.get("bootstrap_seed") == rc.BOOTSTRAP_SEED,
                      {"unit": block.get("resampling_unit"), "n": block.get("n_seeds")})
    if len(conds) == 2:
        h = {c: {r["config_hash"] for r in rows} for c, (_, rows, _) in conds.items()}
        check("E8_config_hashes_differ_between_alphas",
              h["primary"].isdisjoint(h["separable"]), {k: sorted(v) for k, v in h.items()})
        check("E8_primary_hash_matches_archive",
              h["primary"] == {config_hash(PRIMARY)}, {"primary": sorted(h["primary"])})
    audit = root / "archived_fit_reproduction.csv"
    if audit.exists():
        rows = rc.read_rows_csv(audit)
        check("E8_archived_reproduction_all_pass",
              all(r["passed"] == "True" for r in rows),
              {"n": len(rows),
               "max_rel_error": max((float(r["relative_error"]) for r in rows), default=None)})


def audit_e9():
    d = R / "E9_blockA_replication"
    rows = audit_shards("E9", d, range(10, 30), rc.REGISTERED_METHODS, ("seed", "method"))
    if rows is None:
        return
    check("E9_never_wrote_into_archive", not (rc.ARCHIVED_E1_DIR / "replication_raw.csv").exists())
    regen = e9.summarize_subset(rows, "post_hoc_replication", sorted({int(r["seed"]) for r in rows}))
    have = json.loads((d / "replication_summary.json").read_text())
    check("E9_replication_block_regenerates", have["replication"] == regen)
    check("E9_replication_reported_before_pool", "replication" in have)
    permitted = have.get("pooling_permitted")
    pooled_exists = (d / "pooled_summary.json").exists()
    check("E9_pooling_rule_respected", bool(permitted) == pooled_exists,
          {"pooling_permitted": permitted, "pooled_summary_written": pooled_exists})
    for m, block in have["replication"]["primary_contrasts"].items():
        check(f"E9_{m}_bootstrap_unit_is_seed",
              block.get("resampling_unit") == "seed" and block.get("n_seeds") == 20)


def audit_e10():
    d = R / "E10_estimated_covariance"
    rows = audit_shards("E10", d, range(10), ["NFPA", "Iso-MPCA"], ("seed", "method"))
    if rows is None:
        return
    nfpa = [r for r in rows if r["method"] == "NFPA"]
    iso = [r for r in rows if r["method"] == "Iso-MPCA"]
    check("E10_nfpa_uses_estimated_whitening",
          all(r["whitening"] == "estimated" for r in nfpa))
    check("E10_control_is_covariance_independent",
          all(r["whitening"] == "none_covariance_independent_control" for r in iso))
    check("E10_control_reproduces_archive",
          all(float(r["control_reproduction_rel_error_raw_mse"]) <= rc.REPRODUCTION_RTOL
              and float(r["control_reproduction_rel_error_normalized_chi2"]) <= rc.REPRODUCTION_RTOL
              for r in iso),
          {"max_rel": max(max(float(r["control_reproduction_rel_error_raw_mse"]),
                              float(r["control_reproduction_rel_error_normalized_chi2"]))
                          for r in iso)})
    check("E10_calibration_events_are_the_unit",
          all(int(r["n_calibration_events"]) == PRIMARY.N_calibration for r in rows))
    summ = json.loads((d / "summary.json").read_text())
    regen = e10.summarize(rows, PRIMARY, sorted({int(r["seed"]) for r in rows}))
    check("E10_summary_regenerates_byte_for_byte",
          json.dumps(regen, indent=2, sort_keys=True, allow_nan=True) + "\n"
          == (d / "summary.json").read_text())
    check("E10_replication_unit_documented", "calibration EVENT" in summ["replication_unit"])
    check("E10_scale_nonidentifiability_documented",
          "reciprocal scale" in summ["identifiability_note"])
    check("E10_evaluation_is_oracle", "oracle covariance" in summ["evaluation_note"])
    for block in summ["nfpa_degradation_estimated_minus_oracle"].values():
        if block.get("identified", True):
            check("E10_degradation_bootstrap_unit_is_seed",
                  block.get("resampling_unit") == "seed" and block.get("n_seeds") == 10)


def audit_machine():
    ps = subprocess.run(["ps", "-u", "dwong", "-o", "cmd="], capture_output=True, text=True).stdout
    stale = [l for l in ps.splitlines()
             if any(k in l for k in ("run_semantics", "run_replication", "run_estimated_covariance",
                                     "round9_phase3_run", "tidmad.train"))]
    check("no_round9_processes_remaining", not stale, {"stale": stale})
    sc = subprocess.run(["screen", "-ls"], capture_output=True, text=True).stdout
    check("no_round9_screen_remaining", "rps_round9" not in sc, {"screen_ls": sc.strip()})
    pids = list((REPO / "artifacts").rglob("run.pid"))
    check("no_stale_run_pid", not pids, {"found": [str(p) for p in pids]})


def main() -> int:
    audit_archive()
    audit_e8()
    audit_e9()
    audit_e10()
    audit_machine()
    failed = [c for c in checks if not c["passed"]]
    out = R / "Round9_remote" / "evidence_audit.json"
    out.write_text(json.dumps(
        {"all_passed": not failed, "n_checks": len(checks),
         "n_failed": len(failed), "checks": checks}, indent=2, sort_keys=True, default=str) + "\n")
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
        if not c["passed"]:
            print(f"        {c['detail']}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed -> {out.relative_to(REPO)}")
    return 0 if not failed else 6


if __name__ == "__main__":
    raise SystemExit(main())
