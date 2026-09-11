#!/usr/bin/env python3
"""Verify the T22 enforcement against an archived run, independently of the tests.

This exercises the frozen-protocol binding, the amendment scope, the executable
evaluation-window requirement, the hold-out smoke refusal and the cumulative fit
accounting on the *real* archive and the *real* protocol, rather than on the
fixtures in ``tests/test_mlst_t22_narrowband.py``.  It is the check a reviewer
can run to see that the enforcement holds for the run as archived.

It never opens instrument data: every data path of the runner is patched to
raise, so a probe that reached scoring would fail rather than read anything.  It
writes only into a temporary directory, and it re-runs no T22 stage.

    python3 scripts/verify_t22_enforcement.py \
        --protocol <MS>/plan/mlst_protocol_2026-09-11-t22-stage0.json

Exit status is 0 only when every probe passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True,
                    help="stage-0 protocol JSON (manuscript repository, "
                         "plan/mlst_protocol_2026-09-11-t22-stage0.json)")
    ap.add_argument("--results", default=None,
                    help="archived run tree (default: <repo>/results/t22_2026-09-11)")
    ap.add_argument("--repo", default=None,
                    help="experiment repository root (default: this script's parent)")
    args = ap.parse_args(argv)

    global EXP, PROTO, R, FREEZE, CHARS, VAL, DRAWS, LEDGER, t22, te, h5py, FAILS
    EXP = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(EXP))
    from experiments.mlst2026 import t22_tidmad_narrowband as t22_mod
    from experiments.mlst2026 import tidmad_endpoint as te_mod
    import h5py as h5py_mod
    t22, te, h5py = t22_mod, te_mod, h5py_mod

    PROTO = Path(args.protocol).resolve()
    R = Path(args.results).resolve() if args.results else EXP / "results/t22_2026-09-11"
    FREEZE = R / "T22_freeze_v1/t22_freeze.json"
    CHARS = R / "T22_characterize_v1/t22_characterize.json"
    VAL = R / "T22_validation_v1/t22_validation.json"
    DRAWS = R / "T22_validation_v1/t22_validation_draws.npz"
    LEDGER = R / "FIT_LEDGER.json"
    for p in (PROTO, FREEZE, CHARS, VAL, DRAWS, LEDGER):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")
    FAILS = []
    probes()
    print("\n" + ("ALL PROBES PASSED" if not FAILS
                  else f"{len(FAILS)} PROBE(S) FAILED: {FAILS}"))
    return 1 if FAILS else 0


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def no_data():
    boom = mock.Mock(side_effect=AssertionError("instrument data read"))
    return (mock.patch.multiple(t22, noise_bin_power=boom, load_stretch=boom,
                                stage_evaluate=boom, score_cell=boom),
            mock.patch.object(h5py, "File", boom))


def variant(tmp, name, edit):
    raw = json.loads(PROTO.read_text()); edit(raw)
    p = Path(tmp) / name; p.write_text(json.dumps(raw, indent=2)); return p


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
        return None
    except SystemExit as exc:
        return str(exc)


def probes():
    tmp = Path(tempfile.mkdtemp(prefix="t22-enforcement-"))
    P = t22.Protocol(PROTO)

    # 1. the real protocol + archived characterisation bind to the archived freeze
    d1, d2 = no_data()
    with d1, d2:
        rec, loaded = t22.bind_to_freeze(P, FREEZE, CHARS, "validation")
    check("archived freeze binds the real protocol and characterisation (validation)",
          rec["protocol"]["canonical_identical"]
          and rec["protocol"]["sha256"] == rec["protocol"]["frozen_input_sha256"]
          and rec["characterisation"]["sha256"] == rec["characterisation"]["frozen_sha256"],
          f"protocol sha {rec['protocol']['sha256'][:12]}, chars sha {rec['characterisation']['sha256'][:12]}")

    # 2. frozen seeds / replicates cannot be changed
    for label, edit, needle in [
        ("file-level bootstrap seed", lambda r: r["inference"]["primary_combined_estimand"]["interval"].__setitem__("seed", 20260912), "file_level_seed"),
        ("per-file bootstrap seed", lambda r: r["inference"]["per_file"].__setitem__("seed", 424242), "bootstrap_seed"),
        ("per-file replicates", lambda r: r["inference"]["per_file"].__setitem__("replicates", 500), "bootstrap_replicates"),
    ]:
        p = variant(tmp, f"{label.replace(' ', '_')}.json", edit)
        d1, d2 = no_data()
        with d1, d2:
            msg = refused(t22.bind_to_freeze, t22.Protocol(p), FREEZE, CHARS, "validation")
        check(f"changed {label} is refused", bool(msg) and needle in msg, (msg or "NOT REFUSED").splitlines()[0][:110])

    # 3. changed allocation (300 -> 299 evaluation windows, stretch adjusted so Protocol still builds)
    def edit_alloc(r):
        a = r["allocation"]; a["n_evaluation_windows"] = 299
        a["stretch_samples"] = a["evaluation_offset"] + 299 * r["unchanged_from_t10_t12"]["window_samples"]
    p = variant(tmp, "allocation.json", edit_alloc)
    d1, d2 = no_data()
    with d1, d2:
        msg = refused(t22.bind_to_freeze, t22.Protocol(p), FREEZE, CHARS, "validation")
    check("changed evaluation-window allocation is refused", bool(msg) and "allocation" in msg,
          (msg or "NOT REFUSED").splitlines()[0][:110])

    # 4. characterisation must be the frozen bytes
    ch = json.loads(CHARS.read_text())
    first = sorted(ch["characterisation"])[0]
    ch["characterisation"][first]["reference_line_snr"] = 1.0
    alt = tmp / "chars.json"; alt.write_text(json.dumps(ch, indent=2))
    d1, d2 = no_data()
    with d1, d2:
        msg = refused(t22.bind_to_freeze, P, FREEZE, alt, "validation")
    check("characterisation other than the frozen one is refused", bool(msg) and "characterisation sha256" in msg,
          (msg or "NOT REFUSED").splitlines()[0][:110])

    # 5. amendment scope (recombine only; never over frozen evaluation settings)
    def amendment(path, changed, stages=("recombine",), proto_sha=None):
        fz = json.loads(FREEZE.read_text())["frozen"]
        frozen_sha = [v for k, v in fz["input_hashes"].items() if k.startswith("protocol:")][0]
        rec = {"schema": "mlst-t22-amendment/1", "id": path.stem, "reason": "probe",
               "recorded_utc": "2026-09-11T22:00:00Z", "applies_to_stages": list(stages),
               "frozen_protocol_sha256": frozen_sha, "amended_protocol_sha256": proto_sha,
               "changed_paths": list(changed)}
        path.write_text(json.dumps(rec, indent=2)); return path

    harmless = variant(tmp, "harmless.json", lambda r: r.__setitem__("title", r["title"] + " (probe)"))
    h_sha = hashlib.sha256(harmless.read_bytes()).hexdigest()
    d1, d2 = no_data()
    with d1, d2:
        msg = refused(t22.bind_to_freeze, t22.Protocol(harmless), FREEZE, None, "recombine",
                      stage_output=VAL, draws=DRAWS)
    check("recombine refuses an unregistered protocol difference",
          bool(msg) and "differs from the frozen stage0_protocol" in msg and "protocol sha256" in msg,
          " | ".join(l.strip() for l in (msg or "NOT REFUSED").splitlines()[1:3])[:150])

    am_ok = amendment(tmp / "am_ok.json", ["title"], proto_sha=h_sha)
    d1, d2 = no_data()
    with d1, d2:
        rec2, _ = t22.bind_to_freeze(t22.Protocol(harmless), FREEZE, None, "recombine",
                                     [am_ok], stage_output=VAL, draws=DRAWS)
    check("recombine accepts a registered amendment covering the change",
          [a["id"] for a in rec2["amendments_applied"]] == ["am_ok"])

    am_bad = amendment(tmp / "am_bad.json", ["allocation"], proto_sha=h_sha)
    d1, d2 = no_data()
    with d1, d2:
        msg = refused(t22.bind_to_freeze, t22.Protocol(harmless), FREEZE, None, "recombine",
                      [am_bad], stage_output=VAL, draws=DRAWS)
    check("an amendment over a never-amendable section is refused", bool(msg) and "never-amendable" in msg,
          (msg or "NOT REFUSED").splitlines()[0][:110])

    seedp = variant(tmp, "seed_for_amendment.json", lambda r: r["inference"]["primary_combined_estimand"]["interval"].__setitem__("seed", 20260912))
    am_seed = amendment(tmp / "am_seed.json", ["inference/primary_combined_estimand/interval/seed"],
                        proto_sha=hashlib.sha256(seedp.read_bytes()).hexdigest())
    d1, d2 = no_data()
    with d1, d2:
        msg = refused(t22.bind_to_freeze, t22.Protocol(seedp), FREEZE, None, "recombine",
                      [am_seed], stage_output=VAL, draws=DRAWS)
    check("an amendment cannot move a frozen evaluation seed", bool(msg) and "file_level_seed" in msg,
          (msg or "NOT REFUSED").splitlines()[0][:110])

    am_val = amendment(tmp / "am_val.json", ["title"], stages=("validation",), proto_sha=h_sha)
    d1, d2 = no_data()
    with d1, d2:
        msg = refused(t22.bind_to_freeze, t22.Protocol(harmless), FREEZE, CHARS, "validation", [am_val])
    check("validation accepts no amendment at all", bool(msg) and "no amendments" in msg,
          (msg or "NOT REFUSED").splitlines()[0][:110])

    # 6. the 300-window requirement is executable
    rng = np.random.default_rng(0)
    w, nwin = 1024, 8
    m = w // 2 + 1
    tone = np.zeros((nwin, m), dtype=complex); tone[:, 100] = 50.0
    noise = (rng.normal(size=(nwin, m)) + 1j * rng.normal(size=(nwin, m))) * 0.01
    Cr = tone + noise; Cx = tone * 0.5 + noise
    N = np.ones(m) * 1e-4
    out300 = te.characterise(Cr, Cx, w, N, 64 * w, 200, )
    out300b = te.characterise(Cr, Cx, w, N, 64 * w, 200, n_eval_required=300)
    out150 = te.characterise(Cr, Cx, w, N, 64 * w, 200, n_eval_required=150)
    check("characterise(n_eval_required=300) marks 200 available windows underpowered",
          out300b["outcome"] == "underpowered" and "200 < 300" in " ".join(out300b["reasons"]),
          out300b["outcome"])
    check("the T10-T12 default (150) keeps its behaviour on the same input",
          out150["outcome"] != "underpowered" and out300["outcome"] == out150["outcome"],
          f"default={out300['outcome']}, explicit150={out150['outcome']}")
    check("characterise records the requirement it applied",
          out300b.get("n_evaluation_windows_required") == 300 and out150.get("n_evaluation_windows_required") == 150)

    # 7. --smoke refused on hold-out stages, before any data read
    for stage in ("validation", "final-test"):
        d1, d2 = no_data()
        out = tmp / f"smoke_{stage}"
        with d1, d2:
            msg = refused(t22.main, ["--stage", stage, "--protocol", str(PROTO), "--out", str(out),
                                     "--freeze-file", str(FREEZE), "--characterisation", str(CHARS),
                                     "--data-root", str(tmp / "no-data"), "--ledger", str(tmp / "L.json"), "--smoke"])
        check(f"--smoke refused on {stage} without touching data",
              bool(msg) and "refuses --smoke" in msg and not out.exists(),
              (msg or "NOT REFUSED").splitlines()[0][:110])

    # 8. cumulative accounting against the real archive ledger (416 + 24 > 250)
    d1, d2 = no_data()
    out = tmp / "T22_validation_v3"
    with d1, d2:
        msg = refused(t22.main, ["--stage", "validation", "--protocol", str(PROTO), "--out", str(out),
                                 "--freeze-file", str(FREEZE), "--characterisation", str(CHARS),
                                 "--data-root", str(tmp / "no-data"), "--ledger", str(LEDGER)])
    check("a further hold-out stage is refused on the spent archive (416 + 24 > 250)",
          bool(msg) and "416" in msg and "250" in msg and not out.exists(),
          (msg or "NOT REFUSED").splitlines()[0][:150])

    # 9. archive-derived expenditure when the ledger is absent
    d1, d2 = no_data()
    out2 = R / "T22_validation_v4"          # inside the archive; must never be created
    with d1, d2:
        msg = refused(t22.main, ["--stage", "validation", "--protocol", str(PROTO), "--out", str(out2),
                                 "--freeze-file", str(FREEZE), "--characterisation", str(CHARS),
                                 "--data-root", str(tmp / "no-data"), "--ledger", str(tmp / "absent.json")])
    check("a missing ledger beside spent outputs is refused from the archive scan",
          bool(msg) and ("416" in msg or "audit-budget" in msg) and not out2.exists(),
          (msg or "NOT REFUSED").splitlines()[0][:150])

    # 10. ledger consistency and archive agreement
    led = json.loads(LEDGER.read_text())
    entries = led["entries"]
    exec_total = sum(e["fits_executed"] for e in entries if not e["smoke"])
    uniq = sum(e["fits_executed"] for e in entries if not e["smoke"] and not e.get("deterministic_rescore_of"))
    digests_ok = True
    for e in entries:
        f = LEDGER.parent / e["output_dir"] / e["output_file"]
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != e["output_sha256"]:
            digests_ok = False
    check("ledger totals are the sum of its entries (416 executed, 240 distinct, cap 250)",
          exec_total == led["executed_total"] == 416 and uniq == led["unique_scientific_total"] == 240
          and led["hard_cap"] == 250 and led["executed_total_exceeds_cap"] is True,
          f"executed {exec_total}, distinct {uniq}")
    check("every ledger entry matches the archived stage output it names", digests_ok,
          f"{len(entries)} entries")
    scan = t22.scan_executed_fits([R])
    check("archive scan alone reproduces 416 executed fits", scan["executed_total"] == 416,
          str(scan["executed_total"]))


if __name__ == "__main__":
    raise SystemExit(main())
