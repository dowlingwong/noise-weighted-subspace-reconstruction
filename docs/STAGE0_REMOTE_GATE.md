# Stage 0 Remote Reproducibility Gate

Run this gate on the target Linux server from a clean checkout. It is the
handoff for item 4 in the validation roadmap.

## Preconditions

1. Install `git`, Python 3.10 or newer, and `uv`.
2. Clone or update the repository to the commit being evaluated.
3. Confirm the checkout is clean:

   ```bash
   git status --short
   ```

   The command must print nothing. Do not edit source, configs, or the lockfile
   on the server.

4. If the server requires a non-default external data location, set it before
   the run:

   ```bash
   export PAPER1_DATA_ROOT=/ceph/dwong/paper1_dataset
   ```

   Stage 0 itself runs the synthetic suite, but recording the production data
   root prevents later public-data runs from silently using a different path.

## One-command gate

From the repository root:

```bash
python3 scripts/stage0_remote_gate.py
```

The script runs the exact five acceptance commands from the roadmap:

```bash
uv sync --extra dev --extra gwosc
uv run pytest -q
uv run python scripts/run_all_core.py
uv run python scripts/make_tables.py
uv run python scripts/make_all_figures.py
```

It stops at the first failure. Use `--continue-on-failure` only when collecting
diagnostics; acceptance still requires all five commands to pass.

## Evidence produced

Artifacts are written to:

```text
results/stage0/<UTC timestamp>_<git commit>/
```

The directory contains:

- `summary.json`: authoritative pass/fail result, commit, branch, clean-tree
  checks, command durations, and log paths;
- `environment_before.json`: OS, host, CPU, memory, GPU/driver when available,
  bootstrap Python, `uv`, and git metadata;
- `dependencies.txt`: the resolved environment from `uv pip freeze`;
- one combined stdout/stderr log for each acceptance command.

Stage 0 is closed only when `summary.json` contains:

```json
{
  "accepted": true
}
```

Archive the whole artifact directory with the commit or release record. The
script exits nonzero when any command fails or when the checkout is dirty
before or after the run.

## Failure handling

- Dirty checkout: commit/stash local work and rerun. `--allow-dirty` is
  diagnostic only and can never produce an accepted gate.
- `uv sync` failure: preserve `01_uv_sync.log`; do not modify dependency pins on
  the server without first making the change in the repository.
- Test or experiment failure: preserve the artifact directory and rerun only
  after the fix is committed and the server checkout is clean again.
- GPU not found: this is recorded but is not a failure; Stage 0 does not require
  a GPU.
