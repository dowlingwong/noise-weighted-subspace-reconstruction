# Notebook execution

Run the notebooks from the repository root or from this directory. They locate
the transfer bundle relative to their own expected directory layout and write
figures under `transfer_paper/figures/`.

The environment currently does not declare Jupyter as a project dependency.
Install a notebook frontend only when interactive execution is needed:

```bash
uv pip install jupyterlab
uv run jupyter lab transfer_paper/notebooks
```

For non-interactive figure regeneration, Jupyter is unnecessary:

```bash
uv run python transfer_paper/scripts/render_available_figures.py
```

The follow-up notebooks are intentionally valid before the remote results
exist. They report a pending state instead of fabricating empty plots.

