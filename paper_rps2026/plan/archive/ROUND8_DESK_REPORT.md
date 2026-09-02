# Round 8 desk revision — implementation report

**Date:** 2026-08-18. **Commit:** `b97c25d`. **Plan:** `ROUND8_DESK_PLAN.md`.
Ran in parallel with the remote Task B seed replication; nothing here depends on it.

## Final state, verified from a clean build

| Gate | Result |
|---|---|
| Main content | **4 pages**, references start p5 (9 pages total with appendix) |
| Template | `neurips_2026.sty` byte-identical to the official zip |
| Overfull boxes / undefined refs / `\PENDING` | 0 / 0 / 0 |
| Anonymity | PDF text sweep clean; empty Author field |
| Figure | in §4, regenerated at 9.6×2.1 in, numbers reproduce exactly, inspected at final size |

## What was done, item by item

### A — fit and framing

**A1 (tokenization) — done.** §3 rung 4 now states the principle the implementation
already embodied: *"This is a tokenization principle, not a convenience: choose the
token axis in which Σ⁻¹ is diagonal and the likelihood metric becomes a per-token
weight rather than a dense coupling across tokens,"* with the 328-token / 656-row
instantiation. §1 gains the clause that 𝓕 includes, for structured measurements, the
token axis. This is a direct answer to the workshop's tokenization theme ("naive
patching destroys geometric priors").

**A2 (self-supervision) — done.** The abstract now opens: *"Label-free scientific
reconstruction learns representations from measured observations alone ---
self-supervision in which the instrument-characterized noise covariance, not an
augmentation policy, decides which residuals are informative."* Shorter than the
sentence it replaced. §1's SSL paragraph compressed to match.

**A3 (transfer) — done.** The GWOSC paragraph is retitled **"Transfer of the noise
model"** and closes: implementation identity is not calibration, *"and the transfer
step is where a learned reconstruction geometry is falsified"* — near-verbatim the
theme's "falsifiable rather than empirically successful." The paper now speaks to
three of the four workshop themes by name (self-supervision, tokenization, transfer).

**A4 (title) — drafted, not changed.** Four options in `ROUND8_DESK_PLAN.md`;
recommendation: keep current or adopt option 4. **Your decision.**

### B — evidence presentation

**B1 (figure into §4) — done.** The evidence figure is now on page 4 where reviewers
are obliged to see it, with a compact caption (the prose carries the exact numbers).
Regenerated shorter (9.6×2.1 in); output values verified identical to the archived
CSVs. The appendix subsection that held it is removed.

**B2 (contrast table) — prototyped and dropped.** The figure carries the identical
three-rung comparison; the page could not hold both, and the figure is the stronger
form. Recorded as a decision, not an omission.

**B3 (stale comment) — done.** The FIGURE SLOT block referencing an internal plan
file is deleted.

### C — referee-proofing

**C1 (autocorrelation caveat) — done.** §4 now says, where the rung-4 intervals are
reported: *"These evaluations lie along one training trajectory and are
autocorrelated, so the intervals understate uncertainty; the last-ten contrast agrees
in sign on both metrics."* (Checked against the data: last-10 paired differences are
−8.44e-05 and −37.99, same signs as the full series.) The weakest methodological spot
is now disclosed rather than discoverable.

**C2 (contingency) — done.** `notes/round8_contingency.md` has drop-in §4/§5 text for
both Task-B outcomes (seeds agree / signs disagree), with page-budget donors listed.
The result can be integrated the day it syncs.

**C3 (code link) — prepared, decision yours.** `notes/anonymized_code_release.md`:
include list, verified leak vectors (git history, plan/notes dirs, `/ceph/dwong`
paths inside `train_tidmad_local.sh` and `paths.py`, wandb metadata), hosting
recommendation (anonymous.4open.science).

### D — new analysis

**D1 (cross-rung dispersion) — done, and it strengthens the paper.**
`scripts/rung3_whitening_range.py` → `results/rps2026/E6b_rung3_whitening_range.json`:

| Geometry | CV of inverse whitening weights | Reversal? |
|---|---|---|
| Rung 3 colored oracle Σ (eigenvalues of Σ⁻¹) | **1.10** | yes |
| Rung 4 TIDMAD readout (measured 1/J, E6) | **0.69** | no |
| White-noise control | 0 exactly | zero contrast |

The metric is comparably non-degenerate where the reversal appears and where it
doesn't — so the rung-4 absence is not a weaker metric, which sharpens the
capacity-absorption reading. §4 now states this with the numbers.

**D2 (Proposition 2 consequence) — done.** The proof sketch adds: the cost of the
Kronecker prior *"is measurable by fitting rungs 2 and 3 at matched rank and
comparing held-out likelihood residuals, which Section 4 does"* — tying the paper's
sharpest formal result to its own experiment.

### Found and fixed along the way

Two `\PENDING` cells had survived in the appendix baselines table (previous sweeps
covered sections only). Filled with the measured upstream scores (−2.7708, both
arms) and a caption explaining the bit-identity via the 0.01 quantisation, with the
unrounded in-band contrast and its zero-spanning interval. **The manuscript's last
placeholders are gone.**

## How it was paid for

The figure plus ~10 lines of additions cost roughly a third of a page. Recovered by:
inlining three short display equations (the tuple, normalized χ², the Grassmann
product — none needed display-level cross-referencing after rewiring), merging §1's
duplicated rung list and roadmap sentence, moving the mean-handling protocol
sentence to the appendix (text preserved there), and sentence-level compression in
§§1–5. **Nothing protected was cut**: every interval and number, both propositions,
the metric-reversal statement, the novelty boundary, the non-replication caveat, the
GWOSC dispersions, and all §5 caveats survive.

## Decisions taken (author, 2026-08-18, after external audit)

1. **Title: keep option 1** (current title). No file change needed.
2. **Code link: conditional GO.** The v1 export recipe was audited NO-GO (53 files
   with identity/host strings; leaks in four source files, not two;
   `pyproject.toml`/`LICENSE` name the author; `noise_geometry` dependency
   missing). `notes/anonymized_code_release.md` v2 now specifies a fresh
   whitelist-built repository with eight mandatory gates, and the fallback:
   **no link is better than a leaking link.**
3. **Task B: wait for validated E7 artifacts.** `notes/round8_contingency.md` v2
   adds the third gate (incomplete/mismatched runs → keep the one-seed account),
   softens Outcome 1's wording ("same qualitative signs", not "replicates"), and
   for Outcome 2 coordinates **four** edits — §4, abstract, Figure 1 panel C
   labeling, §5 — plus the claim audit, since a sign flip falsifies text beyond
   §4. The training-column sentence is weakened to what the evidence supports.

`notes/claim_audit.md` is brought current: Block A, the rung-4 contrast, and the
whitening refutation are recorded as archived experiments; `T-chi2+OF` is marked
not-run with a no-result-claimed control.

## Round 8b — the six executable items (2026-08-18, same day)

All six items that did not require the server or the author were implemented and
verified; `paper_rps2026/README.md` is brought current.

1. **Anonymized export built and gated.** `scripts/build_anonymized_export.sh`
   builds the whitelist release (78 files) with every neutralization as a visible
   sed and the anonymity grep inside the build (fails on any hit — its first run
   caught three survivors my sed rules missed, proving the gate earns its keep).
   From the export itself: Figure 1 reproduces with identical statistics, the
   rung-3 dispersion script reproduces 1.0964/0.0000, tidmad tests 10 pass /
   5 torch-skip, all 18 NFPA tests collect. The full NFPA suite (~12 min) exceeds
   the local sandbox's call cap; one server command finishes gate 4b. Gates 3,
   5, 6 are the author's.
2. **Both contingency swaps dry-run compiled — both fit.** Outcome 1 + donor 1,
   and Outcome 2 (all four coordinated edits) + donors 1–2, each: references
   start p5, 0 overfull, 0 undefined. Recorded in `round8_contingency.md`; sync
   day is substitution, not composition. Also noted: donor 1 removes the only
   `priorThesis` citation, so prefer the caption donor under Outcome 1.
3. **Figure script takes `--with-seed2 E7`.** Panel C then shows both seeds
   (seed 2 hatched), the title says "two seeds", the console prints per-metric
   sign AGREES/DISAGREES, and axis limits cover both seeds. Verified with
   stand-in data including a forced sign flip (clearly visible), then stand-ins
   deleted. **The default path reproduces the archived figure bit-identically**
   — an initial version changed the shared RNG's consumption order and shifted
   third-decimal interval bounds; panels now compute in fixed A→B→C order with
   seed 2 drawn after C.
4. **Cold referee pass.** Two real defects found and fixed: rung 2 used $P$
   without defining it (a compression casualty — now "for a learned-subspace
   basis $P$"), and §1 had an awkward clause splice from the roadmap merge (now
   two sentences). Everything else read cleanly.
5. **`chi2_of` sentence resolved.** Now: "Injecting the optimal-filter statistic
   into the forward pass \citep{priorThesis} is compatible with this
   architecture but not evaluated here." — attribution kept, no result promised.
   The cut-entirely alternative (which would drop the paper's only `priorThesis`
   citation and dissolve the anonymity question with it) remains available and
   is pre-tested as contingency donor 1.
6. **OpenReview metadata drafted** (`notes/openreview_metadata.md`): title,
   plain-text abstract (with the Outcome-2 substitution rule), TL;DR, keywords,
   theme mapping, and a submission-day checklist.

**Sjostrom PDF: not in any connected folder** (it was a chat upload in an
earlier session). To have the acknowledgements-stripped copy prepared, drop the
PDF into the workspace; otherwise the stripping and deposit stay with you.

After the fixes, re-verified: 4 pages main content, references from p5,
0 overfull, 0 undefined, 0 PENDING.

## Pre-submission repairs still open

1. **Sync `E6_whitening_range.json` from the server** — the paper's 0.69/441×
   claims currently trace only to the run log locally. It is in the Round 8
   sync-back list, so it arrives with Task B; verify it is committed.
2. **`priorThesis` needs a reviewer-verifiable source** — "available on request"
   is weak under double-blind. Options: deposit the (acknowledgements-stripped)
   report at a DOI service and cite the DOI, or include it as properly anonymized
   supplementary material if the venue's single-PDF rule allows an appendix link.
   Author action; preserves the attribution requirement either way.
3. When E7 syncs: apply the matching contingency per the gate, rebuild, re-verify
   (refs p5, 0 overfull, 0 undefined), and update the claim audit row.
