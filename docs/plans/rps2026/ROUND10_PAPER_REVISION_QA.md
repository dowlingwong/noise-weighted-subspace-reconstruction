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
| P1 | `test_non_ancestor_freeze_sha_fails_closed` assumed a branch graph | Yes | A real commit is validated while only Git's non-ancestor response is simulated; the test no longer depends on branch topology or writes Git objects |
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

### Third revision, 2026-08-21 — second external review

A second audit returned **lean accept, 4.0/5**, with three positioning changes
and one attribution decision. All are applied.

**1. The rung-2 comparison returned to §4.** It had been moved to the appendix
in the second revision purely for space, and it answers the first objection
Proposition 2 provokes — which reviewers are not obliged to read an appendix to
find. The wording was tightened at the reviewer's insistence and they were right
to insist: "NFPA beats the unrestricted optimum" is scientifically wrong.
CW-PCA's optimum concerns its *fitted* objective; what NFPA has is a lower
*held-out* residual, which is finite-sample generalization. §4 now says exactly
that. The line was paid for from §5's opening sentence, which restated §4.

**2. Tokenization claim narrowed.** The R/PS call emphasizes discrete
tokenization of continuous physical systems. This paper selects a *physical*
token axis — the one in which `Σ⁻¹` is diagonal — and builds no vocabulary and
no quantizer. §3 now says "a token-axis design principle" and states plainly
that no learned discrete vocabulary is proposed. This lowers the claimed
theme-fit and removes a fair line of attack.

**3. Abstract closing softened**, and §1 now pre-empts the "isn't this
tautological?" reading. A skeptic can compress the result to *a model trained
under MSE does better under MSE*. What is not tautological is that the reversal
appears on **held-out** data, extends to **semantic recovery**, replicates on an
independent cohort, survives finite covariance estimation, and then **fails to
appear** at rung 4. §1's contribution sentence now names that chain. Use the
same chain in the author response and on the poster — not "we propose NFPA".

**4. `priorThesis` resolved as a camera-ready acknowledgement.** The decisive
input was an author clarification that could not be inferred from the
repository: **the rung-4 architecture here is ours, and the earlier MSci report
was an inspiration for posing the problem to an axis-factored attention model,
not a source model.** No code, model or result of it is used.

That changes the attribution analysis rather than overriding it.
`PLAN_03_RUNG4_TIDMAD` §5 condition 2 requires citing the prior work *where the
design is described*; the condition presupposes a borrowed design, so with the
premise false it does not apply on its own terms. Condition 1 (do not claim the
architecture as a contribution) still holds and §3 still says so. Condition 3
concerns the `T-chi2+OF` arm, which is not run.

Attribution is therefore an `ack` environment in `paper.tex`. `neurips_2026.sty`
does `\let\ack\hide` in the double-blind build and typesets it only for
camera-ready — **verified both ways**: the submission PDF contains no occurrence
of the name and no bibliography entry for the report, and a `final`-option build
in a scratch tree prints the acknowledgement under "Acknowledgments and
Disclosure of Funding". The `\citep` in §3 and the forward reference to an
unevaluated OF-injection design are both gone, which also removes an
unverifiable "available on request" source and a named institution from the
reviewed bibliography. The BibTeX entry is retained but uncited, with the
reasoning recorded above it; `plainnat` does not typeset uncited entries.

Space for the additions came from deleting §1's Section-4 roadmap sentence,
which the new contribution sentence had made redundant.

### Three defects found by our own read-through, not by the review

1. **A dangling cross-reference introduced by the compression above.** §3 said
   the rung-2/rung-3 gap is "a cost Section~4 measures", but that measurement
   had just moved to the appendix. LaTeX cannot catch this — the reference was
   prose, not a `\ref`. Now points at Appendix~B.2.
2. **An ambiguous exponent.** "$-1.15$ and $-1.17\times10^{-4}$" reads as
   though only the second value carries the exponent. Now "$-1.15$ and
   $-1.17$, both $\times10^{-4}$".
3. **A dangling participle** left by the §1 rewrite ("where X constructs its
   own nuisance structure, here Y supplies..."). Split into two clauses.

### Anonymity note for the author — resolved

An earlier revision flagged that `priorThesis` typeset as "[Sjostrom, 2026]" in
§3, with a reference-list entry naming Imperial College London beside an
unpublished source — not a direct deanonymization, since the cited author is not
an author here, but a weak affiliation signal in a double-blind submission.
Resolved by the acknowledgement route above: neither the name nor the entry
reaches the reviewed PDF.

## Acceptance gates

| Gate | Result |
|---|---|
| E10b/E11 digest verification | pass (values above) |
| 15-file authoritative archive audit | pass, identical to the pre-revision baseline |
| table regenerates deterministically from archived summaries | pass (byte-identical on re-run) |
| table generator authenticates **every** archived file it reads | pass — E10b `summary.json`, E10b `raw.csv`, E11 `summary.json`; an unpinned path is refused outright |
| table generator fails closed on tampered evidence | pass — verified against a mutated summary, a mutated `raw.csv` value, and a whitespace-only `raw.csv` edit; archive restored and re-verified each time |
| LaTeX clean build (`latexmk -C` then `make`) | pass; one known non-fatal `hyperref`/`lineno` engine warning at the page-2/3 boundary, with no visible or structural PDF defect |
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

Focused Round 9 + Round 10 + hardening suite, plus the new generator tests, in
a read-only checkout: **101 passed, 0 failed** (107 s). Pytest emitted one
non-test warning because it could not write its optional `.pytest_cache`.

`test_round10_hardening.py::test_non_ancestor_freeze_sha_fails_closed` failed in
the first revision of this branch and has since been fixed. The failure was a
fixture assumption, never a provenance defect: the test obtained "a commit that
is not an ancestor of HEAD" by taking `main`, on the comment *"main diverges
from the Round 10 line"*. That held on the Round 10 branches and is false on a
branch cut **from** `main`, where `main` is an ancestor and the gate correctly
stays silent.

An intermediate fix constructed an orphan commit with `git commit-tree`. That
removed the branch assumption but still failed in a read-only checkout because
it writes into `.git/objects`. The final test validates a real existing commit
and deterministically simulates only the non-ancestor response from
`merge-base --is-ancestor`. It exercises the fail-closed path without depending
on the repository graph or writing either the worktree or Git object store.

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

## Anonymous-submission audit, 2026-08-21

Audited against the R/PS 2026 call as published: Short Papers, **four pages
excluding references and appendices**, **single PDF**, **double-blind following
the NeurIPS main-track approach**, non-archival, no NeurIPS checklist required,
concurrent submissions allowed, optional link to *properly anonymized* code.

### Mechanics

| Requirement | State |
|---|---|
| Four pages of main content | ✅ references begin on page 5 |
| References excluded from the limit | ✅ pages 5–6 |
| Appendix excluded, unlimited, after references | ✅ pages 7–10 |
| Main paper self-contained | ✅ every main-text claim carries its own number; the appendix is supporting |
| Single PDF, no additional files | ✅ one 10-page, 293 kB file |
| NeurIPS-based template | ✅ `neurips_2026.sty`, `dblblindworkshop` — **verify against the workshop's own template zip**, see below |
| Paper checklist | not required by this workshop; none present |
| Anonymized code link | none present, consistent with the code link being no-go |

### Identity leaks — none found

Checked on the built PDF, including a `qpdf --qdf` decompression so that
compressed content streams and font programs were scanned as text, not just the
uncompressed shell.

| Check | Result |
|---|---|
| Author name, email, affiliation in body | none |
| `Dowling`, `Wong`, `dwong`, `wongdowling` | 0 occurrences in the decompressed PDF |
| `Karlsruhe`, `KIT`, `Imperial`, `DELight`, `Sjostrom` | 0 (an earlier scan reported `KIT: 1`; that was the byte sequence `xkIT` inside a compressed stream, not a word) |
| Absolute paths — `/Users`, `/home`, `/ceph` | 0. The only path-like strings that reach the page are arXiv URLs and "800/400/2000 training/validation/test events" |
| Source file names (`.tex`, `.bib`) | 0 in the decompressed PDF |
| Document Info dictionary | `/Author ()`, `/Title ()`, `/Subject ()`, `/Keywords ()` all empty; `/Creator (LaTeX with hyperref)`, `/Producer (pdfTeX-1.40.22)` |
| XMP metadata stream | none |
| Embedded files, attachments, annotations, JavaScript, forms | none |
| Embedded figure metadata (`figures/evidence.pdf`) | Matplotlib version string only — no path, no username |
| Every URI in the PDF | arXiv, GWOSC, MLR Press, matplotlib, font-licence sites. No repository, no personal page |
| Reference list | no entry by an author of this paper; no "unpublished", "on request", "in preparation" or "personal communication" source |
| Self-reference phrasing | no "our previous work"-style construction anywhere |
| Acknowledgement | present in source, hidden by the style file in this build — 0 occurrences of the name in the PDF; verified to print under a `final`-option build |
| `\thanks`, funding footnote | none |

The single word `grant` appears six times: URW font-licence boilerplate inside
embedded Type 1 programs. Not author content.

### Two residual risks the audit cannot close

**1. The companion preprint is a live deanonymization channel (author-controlled).**

The appendix paragraph *"Rung-2 numbers carried from the companion work"* prints
`48.419`, `47.869`, `0.550`, `86.25°` and `0.1209 → 0.0880`. Those exact figures
appear in `arXiv-v1-polished-extended-preprint/07_experiments.tex:74–81, 385`,
which renders with the author's name, department and institutional email.

Nothing here breaks a rule: the call permits concurrent submissions and NeurIPS
policy permits preprints. But the appendix hands a curious reviewer a search
string, and it says a companion exists.

Current state: that preprint is **not posted** — `ARXIV_CHECKLIST.md` still lists
author metadata as an open author action and the S5 provenance audit as not
started. So the channel is latent today and opens the moment it is posted. The
R/PS review window runs 29 Aug → 29 Sep 2026.

*Options, in order of cost:* hold the preprint until notification; or post it and
accept a weakened blind; or drop the five carried numbers from the RPS appendix.
The third is cheapest for anonymity and costs the least — those numbers are
superseded rung-2 evidence, not load-bearing for any main-text claim — but it
also removes an honest provenance note. **Author decision.**

**2. The workshop's own template has not been diffed against ours.**

The call links `assets/rps_2026_template.zip`, "based on the NeurIPS style". We
build with `neurips_2026.sty` and `\usepackage[dblblindworkshop]`, which is
almost certainly the same file, but the zip could not be unpacked and compared
here. Consequence if they differ: a visible formatting deviation, not an
anonymity failure.

Worth knowing either way: in submission mode this style prints the generic
footer *"Submitted to 40th Conference on Neural Information Processing Systems
(NeurIPS 2026). Do not distribute."* and does **not** print the workshop name —
`\workshoptitle` is consumed only in `final` mode. That is the style file's
designed behaviour, not a misconfiguration.

### Cosmetic, optional

`CreationDate` and `ModDate` carry a `+08'00'` timezone offset. A very weak
locality signal and nobody strips it, but `exiftool -all:all= ` or a re-build
with `SOURCE_DATE_EPOCH` would remove it if you want the metadata sterile.

## Open author items, unchanged by this round

- ~~`priorThesis` still needs a DOI or otherwise verifiable source.~~
  **Closed 2026-08-21.** See "Third revision" below.
- The anonymized code link remains no-go pending gates 3, 5 and 6.
- Per the plan, this branch is **not** merged into `main` until the rebuilt PDF
  and the claim audit have been independently reviewed.
