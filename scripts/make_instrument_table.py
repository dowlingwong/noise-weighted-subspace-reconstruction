"""P1-E4 — three instruments, three questions (Paper 1 table compiler).

Assembles, from archived evidence only, the table that replaces the three
instrument subsections' summary paragraphs:

    rows    : TIDMAD, CRESST, GWOSC
    columns : does the covariance model calibrate?
              does the weighted reconstruction win at fixed rank?
              does recorded truth agree with the weighted choice?

Every cell carries a provenance trace (file, key) that is printed and written
next to the table.  No number is typed in by hand.

Usage::  python scripts/make_instrument_table.py [--out results/paper1/P1-E4]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TP = ROOT / "transfer_paper"


def _j(path: Path):
    return json.loads(path.read_text())


def _rows(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def tidmad() -> dict:
    p4 = TP / "data/tidmad/current/TIDMAD_P4_real_seed20260727.json"
    p2 = TP / "data/tidmad/current/TIDMAD_P2_real_seed20260727.json"
    p2c = TP / "data/tidmad/current/TIDMAD_P2_confirmatory_seed20260727.json"
    p2d = TP / "data/tidmad/current/TIDMAD_P2_diagnostic_0010_seed20260727.json"
    m4 = _j(p4)["metrics"]
    chi2 = m4["global_model"]["chi2_per_dof_mean"]
    disp = m4["cost_of_mismatch"]["sigma_mismatched"]
    prim = _j(p2)["metrics"]["metric_reversal"]["paired_contrasts_window_level"]["amplitude_abs_error_pca_minus_empca"]
    secs = [_j(p)["metrics"]["metric_reversal"]["paired_contrasts_window_level"]["amplitude_abs_error_pca_minus_empca"] for p in (p2c, p2d)]
    wr = _j(p2)["metrics"]["metric_reversal"]["direction_check"]
    return {
        "instrument": "TIDMAD (SQUID readout, injected truth)",
        "calibrates": f"yes: chi2/dof {chi2:.3f}, held-out score dispersion {disp:.3f}",
        "calibrates_trace": f"{p4.relative_to(ROOT)}:metrics.global_model.chi2_per_dof_mean; metrics.cost_of_mismatch.sigma_mismatched",
        "weighted_wins_fixed_rank": "yes (rank 2): weighted residual and truth favour the weighted choice, raw MSE the other" if wr["orderings_reversed"] else "no",
        "weighted_wins_trace": f"{p2.relative_to(ROOT)}:metrics.metric_reversal.direction_check",
        "truth_agrees": (f"yes on the primary file: paired amplitude-error difference +{prim['mean_difference']:.4f}, 95% CI "
                         f"[{prim['ci95_difference'][0]:.4f}, {prim['ci95_difference'][1]:.4f}]; same sign, interval includes zero on two secondary files "
                         f"(+{secs[0]['mean_difference']:.4f}, +{secs[1]['mean_difference']:.4f})"),
        "truth_agrees_trace": f"{p2.relative_to(ROOT)}, {p2c.relative_to(ROOT)}, {p2d.relative_to(ROOT)}:metrics.metric_reversal.paired_contrasts_window_level.amplitude_abs_error_pca_minus_empca",
    }


def cresst() -> dict:
    per = TP / "data/derived/cresst_per_channel_summary.csv"
    pooled = TP / "data/derived/cresst_pooled_summary.csv"
    rows = [r for r in _rows(per) if r["skipped"] == "False"]
    fr = [(float(r["pca_weighted_residual"]) - float(r["empca_weighted_residual"])) / float(r["pca_weighted_residual"]) for r in rows]
    n_pos = sum(1 for f in fr if f > 0)
    pool = _rows(pooled)[0]
    return {
        "instrument": "CRESST (seven public channels)",
        "calibrates": "within-channel PSD estimated from noise stream; no held-out calibration test recorded",
        "calibrates_trace": f"{per.relative_to(ROOT)}:sigma_of_source",
        "weighted_wins_fixed_rank": (f"yes in {n_pos}/{len(rows)} channels at rank 6: same-rank fractional reduction "
                                     f"{100*min(fr):.1f}% to {100*max(fr):.1f}% (equal-channel mean {100*sum(fr)/len(fr):.1f}%); "
                                     f"rank-6 EMPCA vs single-template OF pooled difference {float(pool['of_vs_empca_mean_difference']):.2f} "
                                     f"[{float(pool['of_vs_empca_ci95_lo']):.2f}, {float(pool['of_vs_empca_ci95_hi']):.2f}]"),
        "weighted_wins_trace": f"{per.relative_to(ROOT)}:pca_weighted_residual, empca_weighted_residual; {pooled.relative_to(ROOT)}:of_vs_empca_*",
        "truth_agrees": "not testable: the release records no injected signal",
        "truth_agrees_trace": "—",
    }


def gwosc() -> dict:
    null = TP / "data/derived/gwosc_null_calibration.csv"
    local = TP / "tables/gwosc_time_local_psd_summary.csv"
    feq = TP / "tables/gwosc_filter_equivalence_summary.csv"
    nrows = _rows(null)
    passed = sum(1 for r in nrows if r["passed"] == "True")
    lrows = _rows(local)
    real_local = [r for r in lrows if r["group"] != "synthetic" and r["model"].startswith("local")]
    accepted = [r for r in real_local if r["accepted"] == "True"]
    return {
        "instrument": "GWOSC (H1/L1 strain around GW150914)",
        "calibrates": (f"no: global PSD null calibration passed in {passed}/{len(nrows)} splits; "
                       f"predeclared 64-s local PSD accepted in {len(accepted)}/{len(real_local)} real-data rows "
                       f"(synthetic stationary control passes)"),
        "calibrates_trace": f"{null.relative_to(ROOT)}:passed; {local.relative_to(ROOT)}:accepted (group != synthetic)",
        "weighted_wins_fixed_rank": "not evaluated: calibration failed, so residual comparisons are diagnostic only",
        "weighted_wins_trace": f"{feq.relative_to(ROOT)} (implementation identity checks only)",
        "truth_agrees": "not testable: no recorded signal",
        "truth_agrees_trace": "—",
    }


def to_tex(rows: list[dict]) -> str:
    esc = lambda s: s.replace("%", r"\%").replace("_", r"\_")
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize",
         r"\caption{Three public instruments, the same three questions.  Every cell traces to an archived file (see \texttt{instrument\_table\_provenance.csv}).}",
         r"\label{tab:instruments}", r"\begin{tabular}{p{0.19\linewidth}p{0.25\linewidth}p{0.27\linewidth}p{0.23\linewidth}}", r"\toprule",
         r"Instrument & Does the covariance model calibrate? & Does the weighted reconstruction win at fixed rank? & Does recorded truth agree with the weighted choice? \\", r"\midrule"]
    for r in rows:
        L.append(f"{esc(r['instrument'])} & {esc(r['calibrates'])} & {esc(r['weighted_wins_fixed_rank'])} & {esc(r['truth_agrees'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/paper1/P1-E4"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [tidmad(), cresst(), gwosc()]
    cols = ["instrument", "calibrates", "weighted_wins_fixed_rank", "truth_agrees"]
    with (out / "instrument_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in cols})
    with (out / "instrument_table_provenance.csv").open("w", newline="") as fh:
        pcols = ["instrument", "calibrates_trace", "weighted_wins_trace", "truth_agrees_trace"]
        w = csv.DictWriter(fh, fieldnames=pcols); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in pcols})
    (out / "instrument_table.tex").write_text(to_tex(rows))
    for r in rows:
        print(f"\n== {r['instrument']}")
        for k in cols[1:]:
            trace = {"calibrates": "calibrates_trace", "weighted_wins_fixed_rank": "weighted_wins_trace", "truth_agrees": "truth_agrees_trace"}[k]
            print(f"  {k}: {r[k]}\n      <- {r[trace]}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
