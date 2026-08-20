# Round 10 paper revision — QA record

**Date:** 2026-08-20 · **Branch:** `codex/rps-round10-paper-revision`
**Base:** `main` @ `62a330f83c1e49d386ca7868ccd114674f36e91c`

Manuscript integration and review response. No experiment was run and no
archived evidence file changed. No experiment code changed either: the second
revision touches one *test* file, whose fixture assumed a branch topology, and
the table generator, which is reporting rather than experiment code.

## Provenance

| Item | Value |
|---|---|
| evidence branch tip merged | `0e037b567f2cfd617e89c42e7839a4d9717d27a3` |
| evidence commit | `64df3c7b5da61f48aaaebbf31c0e8daf963d7c67` (ancestor of the tip, verified) |
| E10b `summary.json` digest | `1846242438133ce52c1dd4adf2d78f34cfc34570c4748ac3e71f50f26b0d496b` |
| E10b `raw.csv` digest | `f76c212f60e075a43196cd535f124262ccc6eb7f49bd5ce6734ff509851409ac` (equals the manifest's `raw_digest`) |
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
6. `0be876e` — this QA record (first revision)

### Second revision, 2026-08-21 — external review response

7. evidence authentication, topology-independent test, wording corrections,
   figure and appendix polish, and this corrected record

The external review returned an advanced-draft verdict with two P1 defects and
several presentation items. Both P1s were reproduced before being fixed.

| # | Finding | Reproduced | Resolution |
|---|---|---|---|
| P1 | The generator verified the E10b summary but parsed `raw.csv` unverified, and the printed percentages depend on it | Yes — editing one denominator cell moved the reported MSE degradation from `0.0345 %` to `0.0341 %` with both summary digests intact, and the build still succeeded | Every archived input is pinned and byte-verified before parsing; an unpinned path is refused; seven tests in `tests/test_round10_table_generator.py`, including value, whitespace-only and unpinned-path tampering |
| P1 | `test_non_ancestor_freeze_sha_fails_closed` assumed a branch graph | Yes | The non-ancestor is now constructed, not discovered |
| P2 | "measurement noise supplies it directly" overstated the contrast with augmentation-based self-supervision | — | §1 now separates the two roles: the measurement supplies the target, an independently calibrated noise model supplies the residual weighting |
| P2 | The covariance-eigenspace identity is exact only for the population covariance | — | §2 says "at the population level" and notes the finite-$N$ perturbation |
| P2 | §4 conflated delayed-predeclared and post-hoc contrasts | — | Two of seven named as delayed-predeclared, five as post-hoc; "across-seed bootstrap interval"; no multiplicity correction stated |
| P2 | Panel B mixes units on one axis | — | Caption states bars are comparable within a group only. Figure 1 itself is byte-identical to the reviewed version |
| P2 | The appendix table floated above its own heading | Yes | `[H]` instead of `[t]`; the B.3 heading now precedes Table 3 |
| P2 | `priorThesis` cites an unevaluated design | — | See the note below — resolved differently from the suggestion, deliberately |

**On `priorThesis`.** The review's suggested fix was to delete the
optimal-filter-injection sentence and the citation. Deleting the citation would
violate `PLAN_03_RUNG4_TIDMAD` §5 condition 2, which requires the prior work to
be *cited where the design is described*, not merely acknowledged — the
acknowledgement-rather-than-co-authorship decision rests on that condition. The
sentence has instead been rewritten to drop the unevaluated forward reference
and attach the citation to the architecture description, where the borrowed
design actually is, with "the architecture is a vehicle, not a contribution"
satisfying condition 1. Condition 3 concerns the `T-chi2+OF` arm, which this
paper does not run. **Whether an unpublished MSci report is a reviewer-verifiable
source remains an author decision and is still open.**

Compression to hold four pages after these additions: the CW-PCA-versus-NFPA
contrast moved from §4 to the appendix intact rather than being deleted, and
duplicated phrasing between §4 and §5 was removed. Figure 1's size and content
are unchanged.

## Acceptance gates

| Gate | Result |
|---|---|
| E10b/E11 digest verification | pass (values above) |
| 15-file authoritative archive audit | pass, identical to the pre-revision baseline |
| table regenerates deterministically from archived summaries | pass (byte-identical on re-run) |
| table generator authenticates **every** archived file it reads | pass — E10b `summary.json`, E10b `raw.csv`, E11 `summary.json`; an unpinned path is refused outright |
| table generator fails closed on tampered evidence | pass — verified against a mutated summary, a mutated `raw.csv` value, and a whitespace-only `raw.csv` edit; archive restored and re-verified each time |
| LaTeX clean build (`latexmk -C` then `make`) | pass |
| four main pages, references from page 5 | pass (10 pages total) |
| undefined references / citations | 0 |
| overfull boxes | 0 |
| `PENDING` / placeholder sweep | clean (only the unused `\PENDING` macro definition in `paper.tex`) |
| anonymity sweep of the built PDF | clean; empty author and title metadata |
| `git diff --check` (working tree) | clean |
| `git diff --check main...HEAD` restricted to manuscript, scripts and tests | clean |
| `git diff --check main...HEAD` unrestricted | **one reported occurrence**, see below |
| working tree after commit | clean |
| `methods.basis_whitened` | unchanged; `ARCHIVAL` warning present |
| new angle references | resolve to `angles_v2` / `*_v2` columns only |
| visual inspection, pages 1–4 and the appendix table | pass |

### The one whitespace occurrence

An earlier revision of this record claimed `git diff --check` was simply
"clean". That was true of the working tree and misleading about the branch
diff. Stated precisely:

```
$ git diff --check                                  # working tree
(clean)
$ git diff --check main...HEAD -- . ':(exclude)results'
(clean)
$ git diff --check main...HEAD
results/rps2026/Round10_remote/machine_preflight.txt:15: trailing whitespace.
```

The line is inside an archived machine log written by the remote agent and
committed at the evidence commit `64df3c7`. It is **inherited, not introduced**,
and it is deliberately not fixed: `machine_preflight.txt` is evidence, its
bytes are what the machine emitted, and editing it for cosmetics would break
the property that archived evidence is never rewritten. Nothing this revision
authored contains a whitespace error.

## Test suite — scope and one known failure

The complete repository suite exceeds this desktop's execution cap, as in every
prior round; the authoritative complete-suite result is the server's, recorded
at `results/rps2026/Round10_remote/`: **417 passed, 9 skipped**,
`PYTEST_STATUS=0`, 1155 s. Nothing in this revision touches experiment code, so
that result still applies.

Focused Round 9 + Round 10 + hardening suite, plus the new generator tests, on
this desktop: **101 passed, 0 failed** (115 s).

`test_round10_hardening.py::test_non_ancestor_freeze_sha_fails_closed` failed in
the first revision of this branch and has since been fixed. The failure was a
fixture assumption, never a provenance defect: the test obtained "a commit that
is not an ancestor of HEAD" by taking `main`, on the comment *"main diverges
from the Round 10 line"*. That held on the Round 10 branches and is false on a
branch cut **from** `main`, where `main` is an ancestor and the gate correctly
stays silent.

The test now **constructs** its non-ancestor with `git commit-tree` over the
empty tree: a real commit object, never an ancestor of anything, independent of
branch topology. It asserts both that the object is a commit and that
`merge-base --is-ancestor` rejects it before exercising the gate. The orphan
commit is unreferenced and collectable; nothing is written to the worktree.

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

- `priorThesis` still needs a DOI or otherwise verifiable source. The citation
  is now attached to the architecture description rather than to an unevaluated
  design, which satisfies the attribution obligation, but an unpublished report
  "available on request" is still not something a reviewer can check.
- The anonymized code link remains no-go pending gates 3, 5 and 6.
- Per the plan, this branch is **not** merged into `main` until the rebuilt PDF
  and the claim audit have been independently reviewed.
