# Incident — pinned T22 interpreter environment partially deleted

**Detected:** 2026-09-16 22:18 local, deepthought, during a status check.
**Event time:** ~14:04–14:06:34 local (12:04–12:06 UTC), same day.
**Actor:** unknown. No trace in `~/.bash_history`; likely another agent session
or a script. Not this agent.

## What was damaged

* `~/.cache` cleaned, including the **uv package cache** (no offline rebuild).
* Audit directory `/home/dwong/mlst-successor-audit-20260913` **archived** to
  `/home/dwong/mlst-successor-audit-20260913.tar.gz` (3,853 entries) and removed.
  This looks deliberate and the archive is intact; it is **not** being extracted
  over that decision.
* Pinned T22 venv `/home/dwong/mlst-successor-py312-20260913`: an interrupted
  `rm -rf`. Removed: `bin/`, `pyvenv.cfg`, `include/`, `lib64`, `.lock`,
  `CACHEDIR.TAG`, `.gitignore`, **every `.dist-info`**, `pytest` and its
  dependencies. `numpy`/`scipy` directories modified at 14:06:34 (mid-deletion).
  Only 6 top-level entries remained in `site-packages`.
* T22 monitor `watch_t22.py` (pid 3017682) **died** — its output directory was
  removed. Last heartbeat 12:05:25 UTC, one minute before the deletion. No
  `T22_TERMINAL.json` will be written automatically.

## What was NOT damaged

* Launcher 2933400 and worker 3050746 — both execute the **base** interpreter
  `/home/dwong/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`,
  which is intact, and hold their modules in memory.
* **72 cells completed after the deletion, 72 started, 0 failed or interrupted.**
* Run root `/ceph/dwong/mlst-successor-runs-20260913/t22s`, ledger and all cells.
* T22's receipt `/ceph/dwong/mlst-successor-runs-20260913/preflight/PREFLIGHT.json`.

## Why action was required

When confirmation finishes, the launcher spawns a **fresh** `reduce` process via
`sys.executable` = `/home/dwong/mlst-successor-py312-20260913/bin/python`, which
no longer existed. The final reduction would have failed.

## Decision

User chose, when offered: **rebuild the environment now, before the reduce**
(alternatives offered: let it fail and recover by procedure; stop and
investigate). This modifies T22's pinned environment under a live launcher and
precedes the fresh preflight RESUMPTION.md normally requires after an environment
change. Both facts are disclosed here for review.

---

## Remediation (22:23–22:33 local)

**1. Rebuilt in staging, not in place.** `uv 0.12.5`, `uv venv --relocatable`
from the intact base interpreter
`/home/dwong/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`,
into `/home/dwong/.mlst-successor-py312-20260913.rebuild`. Installed from PyPI
(cache was gone) using the tested pins `numpy==2.4.6 scipy==1.17.1 pytest==9.1.0
h5py==3.16.0`; resolved dependencies `iniconfig 2.3.0, packaging 26.3,
pluggy 1.6.0, pygments 2.21.0`. uv cache directed to `/var/tmp` so `~/.cache`
was not refilled. Log: `rebuild_build.log`.

**2. Verified before the swap.** Every non-path field of `environment()` matched
T22's receipt; the `t22s` runner imported cleanly with `PYTHONDONTWRITEBYTECODE=1`;
the live repository stayed clean. Log: `prestage_verify.log`.

**3. Swapped at 22:24:37**, after confirming a confirmation worker was running
and **zero** reduce processes existed: two back-to-back renames.

* damaged venv → **quarantined, not deleted**:
  `/home/dwong/mlst-successor-py312-20260913.damaged-20260916`
* rebuilt venv → `/home/dwong/mlst-successor-py312-20260913`

**4. Verified after the swap.** `environment()` equals T22's receipt
**exactly**, including `executable`; `sys.prefix` correct. Launcher and worker
still alive. Log: `swap.log`.

**5. Fresh full preflight on the rebuilt environment — PASSED.** Run against a
local-disk copy of T22's exact source (commit `1d29658`, supplied manifest
`3c22548d…`, manifest-verified), into
`/var/tmp/mlst-t22-envrebuild-preflight-20260916/preflight-rebuilt-env`:
**136 tests, 0 failures, 0 errors, 0 skips**, 96.9 s. Its environment dict is
**identical** to T22's original receipt. Copied here as
`PREFLIGHT_rebuilt_env.json`. **T22's own receipt was not replaced**
(`f2d03b8f…`, mtime 2026-09-13 15:33:09).

This satisfies RESUMPTION.md's fresh-preflight requirement after an environment
change. It completed **after** the swap and **before** the final reduction.

## Monitoring replaced

The dead monitor was replaced by `watch_t22_v2.py` (read-only heartbeat only; no
terminal copy or reduction). Its first reading reported
`reduce_process_running: true` — a **false positive**: substring matching hit the
agent's own shell wrappers. Verified directly that the only T22 process was
pid 3050746 running `run`. Detection fixed to parse argv, monitor restarted, and
the bad record annotated in `T22_MONITOR.jsonl`.

Terminal validation will be run deliberately with `validate_t22_terminal.py`,
writing `T22_TERMINAL_MANUAL.json` whose status is set only by passing checks.

## Disclosure for review

The final T22 reduction will run under a **rebuilt** environment, not the
original bytes. Versions, interpreter build, executable path and threading are
identical and a fresh preflight passed, but the installed wheel files were
re-downloaded and are not byte-verified against the originals, whose metadata
was deleted.
