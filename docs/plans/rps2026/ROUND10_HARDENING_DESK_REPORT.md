# Round 10 hardening — desktop report

**Date:** 2026-08-19 · **Branch:** `codex/rps-round10-handoff-hardening`
**Started from:** `873ad272721968c1634ac502decc29ed2262c86d`

## Status: complete locally, BLOCKED at publication

Phases 0–6 are complete and verified. **Phase 7 (push) failed for missing Git
credentials, which by the runbook's own rule is a stop for Phase 8: unpublished
handoff instructions must not be merged into `main`.** No merge was attempted.

| Item | Value |
|---|---|
| `NEW_ROUND10_IMPLEMENTATION_FREEZE` | `34c47ae71a79b9569457dc54915e36ba5791ebb9` |
| `NEW_ROUND10_CONTENT_COMMIT` | `f9c844d967b4d268d1c5beaf5c52f59edacbb199` |
| metadata pin (branch tip) | `5d67c4b66a0deea862c0f71e8e6a10b6b1ea6ad0` |
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

## Corrections implemented

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
| Focused R9 + R10 + hardening, **no FAST** | **81 passed** (128 s; includes the full-config Iso-MPCA control) |
| Hardening suite alone | 34 passed |
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
