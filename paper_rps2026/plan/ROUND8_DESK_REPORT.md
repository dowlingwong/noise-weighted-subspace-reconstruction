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

## Open items (yours)

1. **Title** — options in the plan doc.
2. **Anonymized code link** — go/no-go and hosting.
3. When Task B syncs: apply the matching contingency paragraph from
   `notes/round8_contingency.md`, rebuild, re-verify the p5 gate.
