# Round 10 paper revision — QA record

**Date:** 2026-08-20 · **Branch:** `codex/rps-round10-paper-revision`
**Base:** `main` @ `62a330f83c1e49d386ca7868ccd114674f36e91c`

Manuscript integration only. No experiment was run, no experiment code was
modified, and no archived evidence file changed.

## Provenance

| Item | Value |
|---|---|
| evidence branch tip merged | `0e037b567f2cfd617e89c42e7839a4d9717d27a3` |
| evidence commit | `64df3c7b5da61f48aaaebbf31c0e8daf963d7c67` (ancestor of the tip, verified) |
| E10b `summary.json` digest | `1846242438133ce52c1dd4adf2d78f34cfc34570c4748ac3e71f50f26b0d496b` |
| E11 `summary.json` digest | `d4f9d004430f5bc9cf55a347c33bced1448190d1e1c9cc8678b1841c22bd6170` |
| authoritative archive | 15/15 unchanged, byte-identical before and after the revision |

`git diff --name-status main -- results/` is additions only: no archived file
was modified or deleted.

## Commits

1. `65bfbe1` — merge (`--no-ff`) of the synchronized Round 10 evidence
2. `442f9dd` — E10b into §4/§5, E11 into the appendix, table generator + table
3. `c70135e` — OpenReview metadata synced to the new abstract
4. `36bcb49` — claim audit reconciled
5. `77cd693` — final four-page QA build

## Acceptance gates

| Gate | Result |
|---|---|
| E10b/E11 digest verification | pass (values above) |
| 15-file authoritative archive audit | pass, identical to the pre-revision baseline |
| table regenerates deterministically from archived summaries | pass (byte-identical on re-run) |
| table generator fails closed on tampered evidence | pass (verified by mutating E11's summary; refused with a digest mismatch, then restored) |
| LaTeX clean build (`latexmk -C` then `make`) | pass |
| four main pages, references from page 5 | pass (10 pages total) |
| undefined references / citations | 0 |
| overfull boxes | 0 |
| `PENDING` / placeholder sweep | clean (only the unused `\PENDING` macro definition in `paper.tex`) |
| anonymity sweep of the built PDF | clean; empty author and title metadata |
| `git diff --check` | clean |
| working tree after commit | clean |
| `methods.basis_whitened` | unchanged; `ARCHIVAL` warning present |
| new angle references | resolve to `angles_v2` / `*_v2` columns only |
| visual inspection, pages 1–4 and the appendix table | pass |

## Test suite — scope and one known failure

The complete repository suite exceeds this desktop's execution cap, as in every
prior round; the authoritative complete-suite result is the server's, recorded
at `results/rps2026/Round10_remote/`: **417 passed, 9 skipped**,
`PYTEST_STATUS=0`, 1155 s. Nothing in this revision touches experiment code, so
that result still applies.

Focused Round 9 + Round 10 + hardening suite on this desktop: **93 passed, 1
failed** (107 s).

The one failure is a stale fixture assumption, not a regression:

```
test_round10_hardening.py::test_non_ancestor_freeze_sha_fails_closed
```

The test obtains "a commit that is not an ancestor of HEAD" by taking `main`,
with the comment *"main diverges from the Round 10 line"*. That was true on the
Round 10 branches. It is false here, because this revision branch was cut
**from** `main`, so `main` is an ancestor of HEAD and the gate correctly does
not raise.

The behaviour under test is fine. Verified directly with a genuinely divergent
commit (`6f5cd22`, the Round 9 overnight handoff tip):

```
IntegrityError: implementation freeze commit 6f5cd22... is not an ancestor of
the execution HEAD 77cd693...
```

Suggested fix, **not applied here** because the Round 10 test file is pinned and
this round authorizes no code change: have the test select a branch tip that is
verifiably not an ancestor of HEAD, and skip when the checkout contains none —
the same shape as the existing `HEAD is main` skip guard directly above it.

## Deviations from the plan

1. **Commit 3's content.** The plan grouped the appendix table with the
   submission metadata. The table and its generator went into commit 2 instead,
   so that every commit builds: §4 references `\ref{tab:round10}`, which would
   dangle if the table arrived a commit later.
2. **One compression donor outside the predeclared five.** The §5 limitations
   paragraph restated the rung-4 baseline comparison and the
   capacity/trajectory/regularization candidates almost verbatim from §4. That
   duplication was removed and the candidate list kept in §5 only. No protected
   statement was dropped; all eight remain present, including the two-seed
   rung-4 bound, the no-new-estimator boundary, the nonconvexity of the joint
   NFPA problem, and E8's absent recovery threshold.
3. **Per-seed E10b rows were read.** Computing the relative-degradation
   denominator from NFPA's archived oracle held-out values — required by the
   plan and by the caption — means reading `E10b/raw.csv`. The desk restriction
   on inspecting seed-level Round 10 output governed the period before the
   remote run completed; the results are now archived and synchronized. No
   experiment was executed here.

## Open author items, unchanged by this round

- `priorThesis` still needs a DOI or otherwise verifiable source.
- The anonymized code link remains no-go pending gates 3, 5 and 6.
- Per the plan, this branch is **not** merged into `main` until the rebuilt PDF
  and the claim audit have been independently reviewed.
