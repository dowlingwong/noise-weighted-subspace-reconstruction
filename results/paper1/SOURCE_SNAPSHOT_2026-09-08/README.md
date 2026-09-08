# Source snapshot for the September 2026 Paper 1 runs

Written 2026-09-08 by the manuscript-revision pass, *not* by the run session.

Every `results/paper1/P1-E*/manifest.json` from the 7 September 2026 runs records
`code_commit = 3026d2593897c07c933386fd5d278515083c6f31` with `git_tree_dirty = true`
and `hostname = vm` (a cloud container, see `RESULTS_REPORT_2026-09.md`, section
"Where these runs were executed"). No commit exists whose tree is the executed one.

This directory records the closest available evidence of that tree, as found in the
laptop checkout on 2026-09-08:

* `tracked_code_changes_vs_3026d25.patch` — `git diff HEAD -- experiments src scripts tests`
  (the modifications to tracked code files relative to commit 3026d25);
* `source_inventory.json` — SHA-256 and byte count of every untracked `.py` file under
  `experiments/`, `src/`, `scripts/`, `tests/`, and of every modified tracked code file.

What this does and does not establish: the run manifests record the *commit*, the
configuration hash, seeds, host and library versions, and the SHA-256 of every output,
but they do not record a hash of the executed source files. The report states that the
code changes were copied back from the container to this checkout after the runs, so the
snapshot here is expected to be that code; this cannot be verified byte-for-byte after the
fact and is recorded as a **provenance gap of the executed source**, separate from the
integrity of the result files themselves (which verify against the manifests). A commit
made now would be a commit of this snapshot, not of the historical execution tree, and is
labelled as such in the suggested commit message.
