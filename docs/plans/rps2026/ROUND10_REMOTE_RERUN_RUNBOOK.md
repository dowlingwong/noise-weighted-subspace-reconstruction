# Round 10 remote rerun runbook — E10b and E11 only

_Server execution of the two frozen corrected runners · 2026-08-19_
_Standing rules: `AGENT_PROMPT_REMOTE_EXPERIMENTS.md` and the shared-server
patterns of `ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md` apply in full._

## Scope — absolute

Run **E10b** (`run_calibrated_estimated_covariance`) and **E11**
(`run_corrected_angles`). Nothing else. **E6–E9 and the legacy E10 must not be
rerun**; their outputs are archived evidence and are read-only inputs to the
E10b/E11 gates. No manuscript, figure, table, bibliography, threshold, or
archived-evidence edit of any kind.

## Identifiers

| Field | Value |
|---|---|
| repository | `https://github.com/dowlingwong/noise-weighted-subspace-reconstruction.git` |
| server handoff branch | `codex/rps-round10-desk-integration` |
| `ROUND10_CONTENT_COMMIT` | `ROUND10_CONTENT_COMMIT_PLACEHOLDER` |
| output branch (create) | `round10/remote-corrections` |
| evidence base (ancestor) | `c9686a0179bae0435d35bce1bbdaafb11839ef86` |
| implementation freeze commit (ancestor) | `24287844cb85477392f8f45e8ee2064b98e26ff9` |

## Phase 0 — provenance and machine preflight

```bash
cd ~/noise-weighted-subspace-reconstruction
git fetch --all --prune || echo "ROUND10_HANDOFF_FETCH_FAILED"   # record, do not hide
test -z "$(git status --porcelain)"                # dirty tree => STOP, report paths
git switch codex/rps-round10-desk-integration || git switch -c codex/rps-round10-desk-integration --track origin/codex/rps-round10-desk-integration
git cat-file -e ROUND10_CONTENT_COMMIT_PLACEHOLDER^{commit}
git merge-base --is-ancestor c9686a0179bae0435d35bce1bbdaafb11839ef86 ROUND10_CONTENT_COMMIT_PLACEHOLDER
git switch --detach ROUND10_CONTENT_COMMIT_PLACEHOLDER
git switch -c round10/remote-corrections
```

If fetch fails for credentials, verify the pinned SHA against the local object
store exactly as Round 9 did, record `ROUND10_HANDOFF_FETCH_FAILED`, and
continue only if the SHA and ancestry checks pass locally.

Machine preflight (record all): host, kernel, CPU load (proceed only if the
1-minute load average leaves ≥8 free cores), free RAM ≥ 32 GB, free disk on the
repo filesystem ≥ 10 GB, no other user's processes touched, interpreter
`.venv/bin/python` with Python/NumPy/SciPy versions.

**Before/after digests.** Record SHA-256 of these files now and again at the
end; any change is a global stop:

```
results/rps2026/E1_nfpa_blockA/block_A_colored_raw.csv
results/rps2026/E8_semantics/archived_fit_reproduction.csv
results/rps2026/E9_blockA_replication/replication_raw.csv
results/rps2026/E9_blockA_replication/replication_summary.json
results/rps2026/E10_estimated_covariance/raw.csv
results/rps2026/E10_estimated_covariance/summary.json
```

## Phase 1 — tests, then smoke

```bash
export PYTHONPATH=src:experiments/synthetic
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=.venv/bin/python

# Full focused suites INCLUDING the full-config control test (no FAST gate).
nice -n 10 timeout 60m $PY -m pytest \
  experiments/synthetic/nfpa_confirmatory/tests/test_round10.py \
  experiments/synthetic/nfpa_confirmatory/tests/test_round9.py -q
```

Any test failure is a **global stop** (shared configuration/provenance
failure), not an experiment-specific one.

Smoke both runners (reserved seeds 910000/910001, reduced config, minutes):

```bash
nice -n 10 timeout 30m $PY -m nfpa_confirmatory.run_corrected_angles \
  --smoke --out results/rps2026/Round10_smoke/E11_server
nice -n 10 timeout 30m $PY -m nfpa_confirmatory.run_calibrated_estimated_covariance \
  --smoke --out results/rps2026/Round10_smoke/E10b_server
```

## Phase 2 — E10b (cap 3 h)

```bash
nice -n 10 timeout 3h $PY -m nfpa_confirmatory.run_calibrated_estimated_covariance \
  --out results/rps2026/E10b_calibrated_estimated_covariance
```

Sequential; do not parallelize seeds. If the cap kills it, resume with
`--resume` after re-verifying no config drift; the runner validates every
existing shard before reuse. The cap may be **raised only before launch**,
justified from measured smoke or archived per-seed timing, and never reduced
after an outcome.

Gates the runner enforces (a failure here stops E10b and is reported, but E11
still runs): Iso-MPCA archived reproduction ≤ 1e-9; seed set exactly 0–9;
manifest/schema/config-hash validation on resume.

## Phase 3 — E11 (cap 4 h)

```bash
nice -n 10 timeout 4h $PY -m nfpa_confirmatory.run_corrected_angles \
  --out results/rps2026/E11_corrected_angles
```

Same sequencing and resume rules. Gates (failure stops E11, is reported, and
does not invalidate a completed E10b): archived reproduction ≤ 1e-9 for
CW-PCA, NFPA and Iso-MPCA against E1 (seeds 0–9) and E9 (seeds 10–29); seed
set exactly 0–29.

**Continuation rule:** an experiment-specific *scientific* outcome — however
unfavorable — is never a reason to stop, rerun, or adjust anything. A shared
configuration, test, provenance, or archive-integrity failure stops everything.

## Phase 4 — close out

1. Re-record the Phase 0 digests; compare; any difference is a global stop
   reported loudly.
2. `git status` — only new files under
   `results/rps2026/{E10b_calibrated_estimated_covariance,E11_corrected_angles,Round10_smoke}/`
   may exist. `git diff --check` on the new changes.
3. Commit everything to `round10/remote-corrections` with a report commit
   message; `git push -u origin round10/remote-corrections`. If push fails for
   credentials, commit locally, report the exact SHA, and mark the push
   failure — never claim a synchronization that did not occur.
4. Write `docs/plans/rps2026/ROUND10_REMOTE_REPORT.md`: identifiers, machine
   state before/after, test results, smoke paths, per-experiment wall times,
   gates passed/failed, digests before/after, and the exact commands run.

## Required literal terminal fields (end of the report)

```text
scientific_thresholds_changed: false
archived_E1_to_E10_overwritten: false
manuscript_edited: false
angle_v2_used_for_new_outputs: true
e10b_scale_correction_frozen_before_full_results: true
```

Any field that would be untrue must be reported as its true value with an
explanation — never asserted falsely.
