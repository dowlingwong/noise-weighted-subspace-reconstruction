# Paper 1 revision instruction — adding TIDMAD

Status date: 27 July 2026 (updated same day: Phase 3 taken to a genuine
confirmatory attempt, three files total, none confirmed; Phase 4 and 6 now
have real-data results). This supersedes nothing in
`REVISION_INSTRUCTION_2026-07-12.md`; it adds a new evidence stream (TIDMAD)
on top of that instruction's CRESST/GWOSC revision. Read both. Where they
conflict on manuscript length or structure, this document's TIDMAD-specific
guidance takes precedence for the TIDMAD material; the July 12 instruction's
guidance on CRESST, GWOSC, theory compression, and overall length still
applies.

Read `PENDING_RESULT_PLACEHOLDERS.md` and `MANUSCRIPT_EVIDENCE_MAP.md` (both
now have TIDMAD rows/sections) before drafting any TIDMAD prose. Numerical
claims must trace to the archived JSON in `data/tidmad/current/`, not to this
document's summary of it.

## Why TIDMAD is being added

Paper 1's real-data evidence has one structural gap CRESST cannot close: no
real-data test of the metric-reversal claim *against known ground truth* —
CRESST has no injected reference, so it can compare residuals but not show
that the MSE-preferred reconstruction actually recovers the true signal
worse. TIDMAD (ABRACADABRA, arXiv:2406.04378, NeurIPS 2025 spotlight) closes
this: its validation files carry an injected reference channel alongside the
detector readout, giving real ground truth. See
`docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md` in the parent repository for
the full design reasoning.

## What actually happened, in one paragraph

Gate G0 (reproducing the release's own published `none`/`savgol` baseline
scores, unmodified) passed on the calibration-anchor score (`none`) and
diagnosed a non-reproducibility in `savgol` down to the published baseline's
own numerically ill-conditioned filter design, not a flaw in this
reproduction. Phase 1 (estimator identity) passed all six gates on real
detector noise. **Phase 2 — the headline experiment — passed on the first
attempt**: on real TIDMAD data with the file's own real injected reference
channel as ground truth, the noise-weighted (EMPCA) reconstruction is
significantly closer to truth than the isotropic-MSE-preferred (PCA) one; the
qualitative direction replicated on two further files (not significant on
either). Phase 3 (the rank-2 prediction) was investigated on **three**
files — two diagnostic, one genuinely confirmatory (selected by a
predeclared, mechanically-applied rule before its outcome was known) — and
confirmed on **none of them**, each for a different, understood,
non-implementation reason. Phase 5 (the detector-facing denoising score) ran
and returned an honest negative result — the linear estimator scored below
"no denoising." Phase 4 (calibration boundary) ran on real data and returned
a genuine **positive** result, unlike GWOSC: a simple global PSD model
calibrates real TIDMAD SQUID noise adequately. Phase 6 (negative controls)
ran on real data: 3 of 5 controls degrade; 2 do not.

**The manuscript consequence is a positive TIDMAD section with real
limitations stated plainly, not a uniformly positive section.** Do not write
this as "TIDMAD confirms the theory on real data" in full generality — write
it as "TIDMAD Phase 2 confirms metric reversal against real ground truth for
the first time, and Phase 4 shows real SQUID noise (unlike GWOSC's real
strain) is adequately calibrated by a simple noise model; the rank
prediction, the denoising-score comparison, and two of five negative
controls did not confirm/degrade on real data, for identified reasons."

## Evidence to use

All in `data/tidmad/current/`; full context and numbers in
`MANUSCRIPT_EVIDENCE_MAP.md`'s TIDMAD rows. Do not restate any TIDMAD number
without checking it against the JSON — several numbers below use the same
symbol (RMSE, bias) at very different scales across files and it is easy to
transpose them.

**Phase 2 (the headline result), `TIDMAD_P2_real_seed20260727.json`, file
`abra_validation_0000.h5`, first attempt, no file selection or tuning:**

| Metric | PCA | EMPCA | winner |
| --- | --- | --- | --- |
| raw MSE | 182.40 | 212.44 | PCA |
| weighted residual | 9.499e13 | 9.432e13 | EMPCA |
| amplitude error vs. real injected truth | 0.1209 | 0.0880 | EMPCA |

Paired difference (truth-error, PCA − EMPCA): **+0.0190, 95% CI [0.0045,
0.0321]** window-level (n=150), **[0.0027, 0.0344]** block-level (block 16).
Both exclude zero. All three predicted orderings reversed.

**Phase 5, `TIDMAD_P5_file_seed20260727.json`:** Wiener-filter coarse
denoising score **-1.8025**, via the unmodified upstream `benchmark.py`.
Published scoreboard for comparison (arXiv:2406.04378 Table 1, coarse
column): none 1.10, moving average 0.64, SG filter -2.35 (see the
ill-conditioning caveat), FCNet 6.55, PUNet 3.84, Transformer 4.18.

**Phase 3, three files, none confirmed** (`TIDMAD_P3_real_seed20260727.json`,
`TIDMAD_P3_confirmatory_seed20260727.json`): see the placeholder entry in
`PENDING_RESULT_PLACEHOLDERS.md` for exact allowed wording if it is
mentioned at all. The genuinely confirmatory attempt used
`abra_validation_0005.h5`, selected by a predeclared file-selection rule
before its outcome was known; rank-2 amplitude RMSE (4.752) was
statistically indistinguishable from rank-1 (4.752), CI [−0.056, 0.098].

**Gate G0, `G0_baseline_reproduction.json` / `tidmad_data_contract.json`:**
`none` = 1.1026 vs. published 1.10.

**Phase 4, `TIDMAD_P4_real_seed20260727.json`, file `abra_validation_0000.h5`:**
global model chi2/dof 1.0031, score dispersion 1.0142, both within the
predeclared [0.9, 1.1] band — calibrates well, unlike GWOSC.

**Phase 6, `TIDMAD_P6_real_seed20260727.json`, same file:** `wrong_frequency`
(+0.357), `permuted_bins` (+0.016), `wrong_psd` (+0.017) degrade with CI
excluding zero; `noise_only_train` (−0.0025, CI spans zero) and
`random_weighted_basis` (−0.0062, CI entirely negative) do not.

## Required manuscript changes

1. Add a TIDMAD section (§7.4, following the roadmap's original numbering
   plan, or renumber to fit whatever CRESST/GWOSC restructuring the July 12
   instruction produced — the important part is TIDMAD sits alongside CRESST
   as a second real-data section, not folded into GWOSC's limitations
   material). Order within it: (a) gate G0 as a one-sentence implementation
   check, (b) Phase 2 as the headline positive result with the table and CIs
   above, (c) Phase 4 as a second positive result (real-noise calibration,
   unlike GWOSC), (d) Phase 5 as an honestly negative auxiliary result, (e)
   Phase 3 as a brief, honest non-confirmation across three files with its
   stated reasons, (f) Phase 6 as 3-of-5 controls degrading with both
   exceptions named — or compress (e)/(f) to a limitations paragraph if
   section length is tight, but do not omit either silently if rank or
   negative-control results are discussed anywhere else in the paper.
2. Add one sentence to the abstract along the lines of: "a second public
   real-detector dataset (TIDMAD/ABRACADABRA), which carries an injected
   reference signal, confirms the metric-reversal claim against known ground
   truth for the first time on real data." Do not claim TIDMAD validates the
   full theory — only the metric-reversal claim, from Phase 2.
3. GWOSC's demotion (per the July 12 instruction and the roadmap) can now
   proceed with TIDMAD as the reason: GWOSC's role becomes "the boundary
   case that motivated seeking a second real-data anchor with ground truth,"
   compressed further than the July 12 instruction alone required.
4. Add TIDMAD to the data/code availability statement: release
   `jessicafry/TIDMAD`, CC-BY-4.0, with the specific validation/science file
   indices used, the frozen operating point (4096-sample windows, rank 2,
   70/30 chronological split, 4-window guard gap, 300-resample bootstrap),
   and the git commit this ran at.
5. State the G0/`savgol` finding as a methodological aside, not a main
   result: reproducing a numerically ill-conditioned published baseline
   exactly is not possible in any implementation, and this repository's
   `none`-score match is the operative calibration check.
6. If reviewers are likely to ask why only one validation file was used for
   Phase 2: state plainly that this is a first real-data attempt at this
   operating point, not an exhaustive scan, and that extending it to more
   files is future work — do not imply a broader survey was done.

## Claim boundaries

Allowed:

- TIDMAD gate G0 as a positive implementation/data-integrity check, with the
  `savgol` non-reproducibility explained rather than hidden.
- TIDMAD Phase 1 as positive real-noise estimator-identity validation.
- **TIDMAD Phase 2 as a positive, real-data, ground-truth confirmation of
  metric reversal** — this is the strongest available real-data result in
  the paper and should be treated as such.
- TIDMAD Phase 5 as an honest negative result under its predeclared claim
  boundary (how much of the no-denoising-to-deep-denoiser gap a linear
  estimator explains — here, a negative amount).
- **TIDMAD Phase 4 as a second positive real-data finding**: a simple global
  Gaussian PSD model calibrates real SQUID noise adequately, unlike GWOSC's
  real-noise calibration failure. Do not generalize beyond the one file and
  ~0.2-second sample range checked.
- TIDMAD Phase 6 as 3-of-5 negative controls degrading, with both exceptions
  named plainly (matching CRESST's precedent of naming its one exception).

Not allowed:

- A general claim that TIDMAD confirms the paper's theory on real data.
- Any claim that the rank-2 quadrature-manifold prediction is confirmed on
  real data (Phase 3 — not confirmed on any of three files tried, including
  one genuinely confirmatory attempt).
- Any claim that the noise-aware linear estimator is competitive with deep
  denoising baselines on TIDMAD's detector-facing score (Phase 5 scored
  below "no denoising").
- Bit-exact reproduction claims for the `savgol` baseline.
- Extrapolating Phase 2's or Phase 4's results beyond the one file and
  operating point each was run at.
- A claim that all five Phase 6 negative controls degrade on real data (2 of
  5 do not).

## Readiness estimate

- TIDMAD experimental implementation and archived execution: **~90% ready**
  for what's in scope for this revision. All six phases now have real-data
  results (G0, Phase 1, Phase 2, Phase 4, Phase 5, Phase 6 done and
  documented; Phase 3 investigated across three files, including one
  genuinely confirmatory attempt, and honestly reported as not confirmed).
  Only a fresh, power-calibrated Phase 3 attempt (larger `n_windows` or a
  more carefully chosen SNR threshold) remains as genuinely open follow-up
  work, and it is optional.
- Evidence/claim alignment for what has been run: **high** — every number
  above traces to an archived JSON, and the placeholder/evidence-map
  documents make the confirmatory-vs-exploratory distinction explicit
  throughout (Phase 2's file-0000 result vs. its two replication attempts;
  Phase 3's three attempts).
- Manuscript integration: **not started** — this document is the first pass
  at instructing that work.

TIDMAD Phase 2 and Phase 4 do not require any new primary experiment to be
paper-ready today. Phase 3 has a completed, honest, non-confirmatory
finding across three files (acceptable as-is); a fresh attempt with a
power-calibrated selection threshold would only be needed if a confirmatory
rank-2 real-data result is specifically wanted. Phase 6's two non-degrading
controls are unexplained candidate mechanisms, not yet investigated further
— optional strengthening, not required for any claim to stand.
