"""Execute section.py cell-by-cell and append the result to the tutorial notebook.

`section.py` is written with `# %%` / `# %% [markdown]` markers, so it runs as a
plain script *and* splits into notebook cells. This runner executes each code
cell in one shared namespace, captures stdout and every figure, and writes the
cells (with their real outputs) onto the end of qp_simulator_tutorial.ipynb.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def split_cells(path: Path):
    cells, cur, kind = [], [], "code"
    for line in path.read_text().splitlines():
        if line.startswith("# %%"):
            if cur:
                cells.append((kind, "\n".join(cur).strip("\n")))
            cur = []
            kind = "markdown" if "[markdown]" in line else "code"
        else:
            cur.append(line)
    if cur:
        cells.append((kind, "\n".join(cur).strip("\n")))
    out = []
    for kind, src in cells:
        if not src.strip():
            continue
        if kind == "markdown":
            src = "\n".join(ln[2:] if ln.startswith("# ") else
                            ("" if ln.strip() == "#" else ln)
                            for ln in src.splitlines())
        out.append((kind, src.strip("\n")))
    return out


def run(cells, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    ns = {"__name__": "__main__"}
    executed, counter, fig_n = [], 0, 0
    real_show = plt.show

    for kind, src in cells:
        if kind == "markdown":
            executed.append(dict(cell_type="markdown", metadata={},
                                 source=src.splitlines(keepends=True)))
            continue
        counter += 1
        outs, buf = [], io.StringIO()
        captured = []

        def show(*a, **k):
            for num in plt.get_fignums():
                f = plt.figure(num)
                b = io.BytesIO()
                f.savefig(b, format="png", dpi=130, bbox_inches="tight",
                          facecolor="white")
                captured.append(b.getvalue())
                plt.close(f)

        plt.show = show
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                exec(compile(src, f"<cell {counter}>", "exec"), ns)
        except Exception:
            buf.write("\n" + traceback.format_exc())
            print(f"!! cell {counter} raised:\n{traceback.format_exc()}",
                  file=sys.__stderr__)
        finally:
            plt.show = real_show

        text = buf.getvalue()
        if text.strip():
            outs.append(dict(output_type="stream", name="stdout",
                             text=text.splitlines(keepends=True)))
        for png in captured:
            fig_n += 1
            (fig_dir / f"figure_{fig_n:02d}.png").write_bytes(png)
            outs.append(dict(output_type="display_data",
                             data={"image/png": base64.b64encode(png).decode(),
                                   "text/plain": ["<Figure>"]},
                             metadata={}))
        executed.append(dict(cell_type="code", execution_count=counter,
                             metadata={}, outputs=outs,
                             source=src.splitlines(keepends=True)))
        print(f"  cell {counter:2d}: {len(text.splitlines())} lines out, "
              f"{len(captured)} figure(s)", file=sys.__stderr__)
    return executed


def main():
    here = Path(__file__).parent
    cells = split_cells(here / "nfpa_section.py")
    print(f"{len(cells)} cells "
          f"({sum(k=='code' for k, _ in cells)} code)", file=sys.__stderr__)
    executed = run(cells, here / "figures")

    nb_dir = here.parent / "notebooks" / "qp_simulator"
    base = json.loads((nb_dir / "qp_simulator_tutorial.ipynb").read_text())
    base["cells"] = base["cells"] + executed
    out = nb_dir / "qp_simulator_tutorial_nfpa.ipynb"
    out.write_text(json.dumps(base, indent=1))
    print(f"wrote {out} ({len(base['cells'])} cells)", file=sys.__stderr__)


if __name__ == "__main__":
    main()
