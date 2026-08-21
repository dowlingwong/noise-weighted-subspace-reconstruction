"""Build the Round 10 appendix table from authenticated E10b and E11 evidence.

Nothing here is transcribed: every archived input is byte-verified against a
pinned SHA-256 digest before parsing, so a table that disagrees with the evidence
cannot be produced. Most quantities come from the two summaries; the relative
percentages are computed from the authenticated E10b raw rows against NFPA's own
archived oracle held-out value, which is the denominator stated in the caption.

E10b is one frozen sensitivity condition, not a robustness sweep. E11 is a
post-hoc diagnostic correction; only ``_v2`` columns carry scientific meaning,
and the legacy columns appear only to quantify how little the defect mattered.

Run from the repository root:

    python scripts/make_rps_round10_table.py \
        --out paper_rps2026/tables/round10_sensitivity.tex
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

RESULTS = Path("results/rps2026")

E10B_DIR = RESULTS / "E10b_calibrated_estimated_covariance"
E11_DIR = RESULTS / "E11_corrected_angles"

# Pinned by the Round 10 evidence commit 64df3c7b5da61f48aaaebbf31c0e8daf963d7c67.
#
# Every archived file this script reads is listed here, not merely the summaries.
# The relative-degradation percentages are computed from E10b's per-seed rows, so
# `raw.csv` is evidence too: authenticating only the summaries would leave the
# denominators unverified, and a modified `raw.csv` could change a printed number
# while both summary digests still matched. The `raw.csv` digests below are the
# `raw_digest` fields recorded in the corresponding run manifests.
EXPECTED_DIGESTS = {
    E10B_DIR / "summary.json":
        "1846242438133ce52c1dd4adf2d78f34cfc34570c4748ac3e71f50f26b0d496b",
    E10B_DIR / "raw.csv":
        "f76c212f60e075a43196cd535f124262ccc6eb7f49bd5ce6734ff509851409ac",
    E11_DIR / "summary.json":
        "d4f9d004430f5bc9cf55a347c33bced1448190d1e1c9cc8678b1841c22bd6170",
}


def read_verified(path: Path) -> bytes:
    """Return a file's bytes only if its SHA-256 matches the pinned value.

    Fails closed on an unpinned path as well: adding a new archived input
    without pinning it is itself the defect this guard exists to prevent.
    """
    if path not in EXPECTED_DIGESTS:
        raise SystemExit(
            f"REFUSING to build the table: {path} is read but not pinned in "
            "EXPECTED_DIGESTS. Pin its digest before using it as evidence."
        )
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    want = EXPECTED_DIGESTS[path]
    if got != want:
        raise SystemExit(
            f"REFUSING to build the table: {path} digest {got} != pinned {want}. "
            "The archived evidence changed; investigate before regenerating."
        )
    return raw


def load_verified(path: Path) -> dict:
    """Read an archived JSON summary only if its digest matches the pinned value."""
    return json.loads(read_verified(path))


def e10b_relative_degradations() -> dict[str, tuple[float, float, float]]:
    """(absolute delta, NFPA oracle denominator, percent) per metric, from raw rows.

    The percentages in the caption are defined against NFPA's own archived
    oracle held-out value, so the denominator is computed here rather than
    assumed.
    """
    text = read_verified(E10B_DIR / "raw.csv").decode()
    rows = [r for r in csv.DictReader(text.splitlines())
            if r["method"] == "NFPA"]
    out = {}
    for metric, dcol, ocol in (
        ("chi2", "delta_normalized_chi2_estimated_minus_oracle",
         "archived_oracle_test_normalized_chi2"),
        ("mse", "delta_raw_mse_estimated_minus_oracle",
         "archived_oracle_test_raw_mse"),
    ):
        delta = statistics.fmean(float(r[dcol]) for r in rows)
        denom = statistics.fmean(float(r[ocol]) for r in rows)
        out[metric] = (delta, denom, 100.0 * delta / denom)
    return out


def _sci(x: float, digits: int = 2) -> str:
    """LaTeX scientific notation, e.g. 7.90\\times10^{-5}."""
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def e10b_block(summary: dict) -> list[str]:
    # The archived summary uses flat dotted keys; index it exactly as written.
    chi2 = summary["contrast_estimated_nfpa_minus_iso.test_normalized_chi2"]
    mse = summary["contrast_estimated_nfpa_minus_iso.test_raw_mse"]
    diag = {name: summary[f"diagnostics.{name}"] for name in
            ("product_scale_ratio_vs_oracle", "channel_shape_error",
             "temporal_shape_error", "kronecker_relative_frobenius_error")}
    rel = e10b_relative_degradations()

    def ci(d: dict, digits: int) -> str:
        return (f"${d['mean_difference']:+.{digits}f}$ "
                f"$[{d['ci95_low']:+.{digits}f},\\,{d['ci95_high']:+.{digits}f}]$")

    lines = [
        r"\multicolumn{2}{l}{\emph{E10b --- estimated separable covariance, "
        r"$N_{\mathrm{cal}}=2048$, %d seeds}} \\" % summary["n_seeds"],
        r"\quad $\Delta\bar\chi^2$, NFPA $-$ isotropic & " + ci(chi2, 6) + r" \\",
        r"\quad $\Delta\mathrm{MSE}$, NFPA $-$ isotropic & " + ci(mse, 7) + r" \\",
        r"\quad NFPA $\bar\chi^2$ degradation vs.\ oracle & $%s$ (%.4f\%%) \\"
        % (_sci(rel["chi2"][0]), rel["chi2"][2]),
        r"\quad NFPA MSE degradation vs.\ oracle & $%s$ (%.4f\%%) \\"
        % (_sci(rel["mse"][0]), rel["mse"][2]),
        r"\quad Product-scale ratio vs.\ oracle & $%.4f$ \\"
        % diag["product_scale_ratio_vs_oracle"]["mean"],
        r"\quad Channel / temporal shape error & $%.4f$ / $%.4f$ \\"
        % (diag["channel_shape_error"]["mean"], diag["temporal_shape_error"]["mean"]),
        r"\quad Kronecker relative Frobenius error & $%.4f$ \\"
        % diag["kronecker_relative_frobenius_error"]["mean"],
    ]
    return lines


def e11_block(summary: dict) -> tuple[list[str], float, float]:
    lines = [r"\multicolumn{2}{l}{\emph{E11 --- corrected principal angles to "
             r"CW-PCA (degrees)}} \\"]
    shifts = {}
    for cohort, label in (("frozen", "Frozen, $n=10$"),
                          ("replication", "Replication, $n=20$")):
        block = summary[cohort]
        nfpa = block["NFPA.angle_to_cw_pca_mean_deg_v2"]["mean_difference"]
        iso = block["Iso-MPCA.angle_to_cw_pca_mean_deg_v2"]["mean_difference"]
        iso_legacy = block["Iso-MPCA.angle_to_cw_pca_mean_deg_legacy"]["mean_difference"]
        contrast = block["contrast_v2_mean_angle_nfpa_minus_iso"]["mean_difference"]
        shifts[cohort] = iso_legacy - iso
        lines.append(
            r"\quad %s & NFPA $%.2f$, isotropic $%.2f$, difference $%+.2f$ \\"
            % (label, nfpa, iso, contrast)
        )
    return lines, shifts["frozen"], shifts["replication"]


def build() -> str:
    e10b = load_verified(E10B_DIR / "summary.json")
    e11 = load_verified(E11_DIR / "summary.json")
    e11_lines, shift_frozen, shift_repl = e11_block(e11)

    body = "\n".join(
        e10b_block(e10b)
        + [r"\addlinespace"]
        + e11_lines
    )

    return r"""%% Generated by scripts/make_rps_round10_table.py -- do not edit by hand.
%% Every archived input is byte-verified against a pinned SHA-256 digest before
%% parsing; see the script for the pinned digests.
%% [H] rather than [t]: the table's subsection precedes it in the source, but a
%% top float would otherwise render above that heading.
\begin{table}[H]
  \centering
  \small
  \caption{Round 10. \textbf{E10b} is a single frozen sensitivity condition:
  the separable covariance is estimated from $2{,}048$ independent noise-only
  calibration events rather than taken as oracle, at the otherwise unchanged
  $C=8>k_c=2$ configuration. The reversal survives in all ten seeds. The
  degradation percentages are relative to NFPA's \emph{own} corresponding
  archived oracle held-out value. This is one condition, not a sweep across
  calibration sizes, covariance families or misspecification.
  \textbf{E11} is a post-hoc diagnostic correction of the whitened-basis
  vectorization used for principal angles; only the \texttt{\_v2} quantities
  carry scientific meaning.}
  \label{tab:round10}
  \begin{tabular}{@{}lr@{}}
    \toprule
    Quantity & Estimate \\
    \midrule
%s
    \bottomrule
  \end{tabular}
  \par\vspace{2pt}
  \begin{minipage}{\linewidth}\footnotesize
  E11 note: the correction leaves NFPA unchanged and shifts the isotropic
  baseline by $%.2f^\circ$ in the frozen cohort and $%.2f^\circ$ in the
  replication cohort, so no archived scientific conclusion depended on the
  defect. Displayed intervals are $95\%%$ bootstrap intervals with seed as the
  resampling unit. Cohorts are
  never pooled for confirmatory reading.
  \end{minipage}
\end{table}
""" % (body, shift_frozen, shift_repl)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=Path("paper_rps2026/tables/round10_sensitivity.tex"))
    args = ap.parse_args()

    tex = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"  written {args.out}")

    rel = e10b_relative_degradations()
    for metric, (delta, denom, pct) in rel.items():
        print(f"  E10b {metric:5s} delta {delta:.6e}  denom {denom:.9f}  {pct:.4f}%")
    e11 = load_verified(E11_DIR / "summary.json")
    for cohort in ("frozen", "replication"):
        b = e11[cohort]
        print(f"  E11 {cohort:12s} NFPA v2 "
              f"{b['NFPA.angle_to_cw_pca_mean_deg_v2']['mean_difference']:.4f}  "
              f"iso v2 {b['Iso-MPCA.angle_to_cw_pca_mean_deg_v2']['mean_difference']:.4f}  "
              f"iso legacy "
              f"{b['Iso-MPCA.angle_to_cw_pca_mean_deg_legacy']['mean_difference']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
