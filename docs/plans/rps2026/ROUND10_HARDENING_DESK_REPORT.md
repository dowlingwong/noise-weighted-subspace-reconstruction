# Round 10 hardening — desktop report (rev 2)

**Date:** 2026-08-19 · **Branch:** `codex/rps-round10-handoff-hardening`
**Started from:** `873ad272721968c1634ac502decc29ed2262c86d`

## Status: complete locally, BLOCKED at publication

Phases 0–6 are complete and verified. **Phase 7 (push) failed for missing Git
credentials, which by the runbook's own rule is a stop for Phase 8: unpublished
handoff instructions must not be merged into `main`.** No merge was attempted.

| Item | Value |
|---|---|
| `NEW_ROUND10_IMPLEMENTATION_FREEZE` | **`048f1bd96acf943ad4d143ea50333b194099f170`** (rev 2) |
| `NEW_ROUND10_CONTENT_COMMIT` | **`3eee165ad1cab76cb61101ddce072b189cbdc453`** (rev 2) |
| metadata pin (branch tip) | **`6407ba79aa00a8f1cf63a248630055193de593b8`** (rev 2) |
| superseded rev-1 freeze / content / pin | `34c47ae` / `f9c844d` / `5d67c4b` — **do not dispatch** |
| evidence base | `c9686a0179bae0435d35bce1bbdaafb11839ef86` |
| `main` (untouched) | `220df0a375af226aa964811535c0316c49f2bea8` |

Ancestry verified: freeze → content → pin, and evidence base → content.

## What the user must run (authenticated machine)

```bash
cd ~/Documents/noise-weighted-subspace-reconstruction
git push -u origin codex/rps-round10-handoff-hardening
git fetch origin codex/rps-round10-handoff-hardening
test "$(git rev-parse codex/rps-round10-handoff-hardening)" \
   = "$(git rev-parse origin/codex/rps-round10-handoff-hardening)" && echo PUSH_VERIFIED

# only after PUSH_VERIFIED:
git switch main
git pull --ff-only origin main
git merge --no-ff codex/rps-round10-handoff-hardening
PYTHONPATH=src:experiments/synthetic python -m pytest \
  experiments/synthetic/nfpa_confirmatory/tests/test_round9.py \
  experiments/synthetic/nfpa_confirmatory/tests/test_round10.py \
  experiments/synthetic/nfpa_confirmatory/tests/test_round10_hardening.py -q
PYTHONPATH=src python -m pytest src/tidmad/tests -q
git diff --check
git push origin main
```

`main` diverges from the Round 10 line after `c1c4b75`; use `--no-ff` and do
**not** rebase, squash, cherry-pick or amend — the pinned SHAs above are
referenced by the dispatch prompt and must survive verbatim.

## Rev 2 — the seven review blockers

All seven were reproduced before being fixed.

| # | Blocker | Resolution |
|---|---|---|
| 1 | Remote flow wrote into the worktree, so E10b's clean-tree gate would reject it | `R10_RUNTIME_ROOT` permits fully out-of-tree execution; runbook runs tests, audits, both experiments and smoke in `$RUNTIME` and copies results in only at step 12 |
| 2 | `pytest \| tee` masked a failed or timed-out suite | redirect to a file, capture and test the exit status explicitly; `set -euo pipefail`; explicit "never pipe pytest into tee" |
| 3 | **Validators trusted stored error columns** — a corrupted metric with a small stored error was accepted (reproduced: both runners) | validators load the authoritative E1/E9 tables and **recompute** archived values, both reproduction errors and both deltas, requiring the stored columns to agree; pinned by tests that replay the original bypass |
| 4 | Resume did not bind shards to the run identity | in-progress manifest carries freeze/execution/config identity from the start; each shard registered with its SHA-256 at creation; resume rejects unregistered shards, digest mismatches, and foreign identities |
| 5 | Completion-time `git status` recorded the run's own outputs as source dirtiness | records `source_tree_dirty_at_start`, `source_commit`, and `expected_output_changes_at_completion` |
| 6 | Report committed before the push it was supposed to describe | report with `push_status: pending` → push → synchronization addendum with the verified SHA → push; evidence commit never amended |
| 7 | Not published | still blocked: no Git credentials in this environment (see below) |

Also fixed: the full suite rewrote the tracked `src/NFPA/nfpa_demo_results.png`,
dirtying the tree and blocking the runners. `nfpa_demo.py` honours
`NFPA_DEMO_OUT_DIR`; the test redirects it; default behaviour unchanged.

## Corrections implemented (rev 1, retained)

1. **Provenance** (`r10_provenance.py`) — non-smoke runs require
   `--implementation-freeze-commit`; the SHA must exist, be a commit, and be an
   ancestor of the execution HEAD, with a clean worktree. Never inferred.
   Freeze and execution SHAs recorded separately; smoke manifests carry
   `evidence_class=implementation_check_not_scientific_evidence`.
2. **Fail-closed validation** (`r10_validation.py`) — every gate re-derived
   from the final rows (fresh and resumed alike): seed sets, method sets,
   duplicates, schema, config hash, regime, label, finiteness, ≤1e-9
   reproduction, E10b delta self-consistency, E11 cohort/source and the CW-PCA
   empty-angle convention. `control_gate_passed` recomputed, never inherited.
3. **Authoritative archive** — all **15** Phase 0 paths pinned and audited
   before computation and before the manifest; a test asserts the list equals
   the preflight set. (Previously six files were checked while the prose said
   "E1–E10".)
4. **Output integrity** — `summary_digest`; summaries are pure functions of raw
   rows, re-derived and digest-compared after writing; the completed manifest
   is the final atomic write, so interrupted runs stay `status=in_progress`.
5. **Test-scope honesty** — the remote runbook now runs the complete suite from
   the repository root without `NFPA_CONFIRMATORY_FAST`; `export/` is excluded
   from collection (it was being collected as a duplicate suite, 374 → 364).
6. **Remote flow** — 17 steps; safe create-or-resume of
   `round10/remote-corrections` (never force-reset); **report written before
   the final commit**; eight literal terminal fields; success line forbidden
   when any gate fails; identifiers supplied by the dispatch prompt rather than
   a self-referential SHA.

## Test results (this desktop)

| Suite | Result |
|---|---|
| Focused R9 + R10 + hardening, **no FAST** | **94 passed** (112 s; includes the full-config Iso-MPCA control) |
| Hardening suite alone | **47 passed** (was 34) |
| Adversarial bypass replay | both reported bypasses now **rejected** |
| TIDMAD (`src/tidmad/tests`) | 10 passed, 5 skipped |
| Complete suite | **NOT run locally** — exceeds the desktop execution cap. Collection: **364 tests**, 2 collection errors from sandbox-only missing deps (`tomllib` needs Python ≥3.11, sandbox has 3.10; `numba`, a declared dev extra, absent). Both resolve on the server's `.venv` (Python 3.12.2). The complete-suite gate is the server's. |

## Archive and safety

- All **15** archive digests verified unchanged at Phase 0 and at completion.
- No seed-0–29 E10b/E11 output was run or inspected on this desktop.
- No archived evidence file and no manuscript file was modified.
- `git diff --check` clean.

## Deviations and observations

1. **`git fetch` and `git push` unavailable** (no credentials in this
   environment) — the same condition Round 9 hit. Verification used the local
   object store; the push and merge are the user's to perform.
2. **Pre-existing test hygiene issue (not fixed, out of scope):**
   `experiments/synthetic/nfpa_confirmatory/tests/test_correctness.py` executes
   `src/NFPA/nfpa_demo.py`, which rewrites the tracked binary
   `src/NFPA/nfpa_demo_results.png` as a side effect. It was restored, not
   committed. Worth fixing separately (have the demo write to a temp path under
   test), since it makes any full-suite run dirty the worktree — which now
   matters, because the runners refuse to start on a dirty tree.
3. The earlier dirty, pre-freeze smoke output was **deleted** rather than
   presented as frozen evidence; clean smoke was regenerated at the new freeze
   with the freeze SHA supplied.
