#!/usr/bin/env python3
"""Render the paper-facing CRESST figure from the archived JSON using PGFPlots."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "transfer_paper/data/cresst/current/cresst/p2_cresst_validation_seed22.json"
OUTDIR = ROOT / "transfer_paper/figures"
TEXFILE = OUTDIR / "cresst_validation.tex"


def point_row(index: int, item: dict[str, object]) -> str:
    mean = float(item["mean_difference"])
    lo, hi = map(float, item["ci95_difference"])
    return f"{mean:.12g} {mean-lo:.12g} {hi-mean:.12g} {index}"


def main() -> None:
    archive = json.loads(SOURCE.read_text())
    metrics = archive["metrics"]
    per_channel = metrics["per_channel"]
    pooled = metrics["pooled"]

    channels = list(per_channel)
    channel_labels = ",".join("{" + value.replace("_", "/") + "}" for value in channels)
    of_rows = "\n".join(point_row(i, per_channel[ch]["of_vs_empca"]) for i, ch in enumerate(channels))
    mt_rows = "\n".join(
        point_row(i, per_channel[ch]["empca_competitive_with_multi_template_of"])
        for i, ch in enumerate(channels)
    )

    controls = [
        ("Wrong PSD", "wrong_psd"),
        ("Noise-only train", "noise_only_train"),
        ("Random basis", "random_weighted_basis"),
        ("Mismatched OF", "mismatched_of_template"),
        ("Permuted bins", "permuted_time_bins"),
    ]
    control_labels = ",".join("{" + label + "}" for label, _ in controls)
    control_rows = "\n".join(
        point_row(i, pooled["negative_controls"][key]) for i, (_, key) in enumerate(controls)
    )
    angle_rows = "\n".join(
        f"{i} {float(per_channel[ch]['ae_empca_max_principal_angle_deg']):.12g}"
        for i, ch in enumerate(channels)
    )
    maximum_angle = max(
        float(per_channel[ch]["ae_empca_max_principal_angle_deg"]) for ch in channels
    )

    tex = rf"""\documentclass[tikz,border=3pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}
\definecolor{{oiBlue}}{{HTML}}{{0072B2}}
\definecolor{{oiOrange}}{{HTML}}{{E69F00}}
\definecolor{{oiGreen}}{{HTML}}{{009E73}}
\definecolor{{oiRed}}{{HTML}}{{D55E00}}
\begin{{document}}
\begin{{tikzpicture}}[font=\sffamily\small]
\begin{{axis}}[
  name=top, width=16.2cm, height=6.3cm,
  xmode=log, xmin=8, xmax=5000,
  ymin=-0.5, ymax=6.5, y dir=reverse,
  ytick={{0,1,2,3,4,5,6}}, yticklabels={{{channel_labels}}}, ylabel={{Run/channel}},
  xlabel={{Mean paired weighted-residual difference (95\% event-bootstrap CI)}},
  title={{Held-out within-channel reconstruction effects}},
  xmajorgrids, grid style={{gray!25}},
  legend style={{draw=none, at={{(0.99,0.03)}}, anchor=south east}},
  error bars/x dir=both, error bars/x explicit,
]
\addplot+[only marks, mark=*, mark size=2.1pt, oiBlue, error bars/.cd, x dir=both, x explicit]
table[x index=0, x error minus index=1, x error plus index=2, y index=3] {{
{of_rows}
}};
\addlegendentry{{OF $-$ EMPCA}}
\addplot+[only marks, mark=square*, mark size=2.0pt, oiOrange, error bars/.cd, x dir=both, x explicit]
table[x index=0, x error minus index=1, x error plus index=2, y index=3] {{
{mt_rows}
}};
\addlegendentry{{$1.1\times$ multi-template OF $-$ EMPCA}}
\node[anchor=north west, font=\bfseries\large] at (rel axis cs:0.01,0.98) {{A}};
\end{{axis}}

\begin{{axis}}[
  at={{($(top.south west)+(0,-2.1cm)$)}}, anchor=north west,
  width=7.9cm, height=6.1cm,
  xmode=log, xmin=0.6, xmax=120000,
  ymin=-0.5, ymax=4.5, y dir=reverse,
  ytick={{0,1,2,3,4}}, yticklabels={{{control_labels}}},
  xlabel={{Control $-$ EMPCA (pooled mean, 95\% CI)}},
  title={{Negative controls}}, xmajorgrids, grid style={{gray!25}},
  error bars/x dir=both, error bars/x explicit,
]
\addplot+[only marks, mark=*, mark size=2.1pt, oiGreen, error bars/.cd, x dir=both, x explicit]
table[x index=0, x error minus index=1, x error plus index=2, y index=3] {{
{control_rows}
}};
\node[anchor=north west, font=\bfseries\large] at (rel axis cs:0.02,0.98) {{B}};
\end{{axis}}

\begin{{axis}}[
  at={{($(top.south west)+(9.5cm,-2.1cm)$)}}, anchor=north west,
  width=6.7cm, height=6.1cm,
  ymode=log, ymin=6e-7, ymax=3e-6,
  xmin=-0.5, xmax=6.5, xtick={{0,1,2,3,4,5,6}}, xticklabels={{{channel_labels}}},
  x tick label style={{rotate=45, anchor=east}},
  ylabel={{Maximum principal angle (degrees)}},
  title={{Tied linear-AE / EMPCA agreement}},
  ymajorgrids, grid style={{gray!25}},
]
\addplot+[only marks, mark=diamond*, mark size=2.4pt, oiBlue]
table[x index=0, y index=1] {{
{angle_rows}
}};
\addplot+[gray, dashed, mark=none] coordinates {{(0,{maximum_angle:.12g}) (6,{maximum_angle:.12g})}};
\node[anchor=north east, font=\scriptsize] at (rel axis cs:0.98,0.96)
  {{maximum $= {maximum_angle:.2e}^\circ$}};
\node[anchor=north west, font=\bfseries\large] at (rel axis cs:0.02,0.98) {{C}};
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    TEXFILE.write_text(tex)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TEXFILE.name],
        cwd=OUTDIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pngalpha",
         "-r400", "-sOutputFile=cresst_validation.png", "cresst_validation.pdf"],
        cwd=OUTDIR,
        check=True,
    )


if __name__ == "__main__":
    main()
