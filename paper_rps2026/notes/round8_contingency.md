# Round 8 Task B contingency — drop-in text for the seed replication (v2)

Revised 2026-08-18 after external audit. **Do not insert anything until the E7
artifacts are synced and validated locally** (`E7_rung4_seed_replication.json`,
both `E7_*_s20260818_eval_series.csv`).

## Gate: which outcome applies

1. **Both seeds agree in sign on both validation metrics** → Outcome 1.
2. **Either metric changes sign between seeds** → Outcome 2.
3. **One arm incomplete, configurations differ beyond `loss`, or evaluation grids
   differ** → **neither.** Keep the current one-seed account unchanged; mention the
   incomplete run, if at all, only as a diagnostic. Do not average or mix seeds.

The C1 caveat sentence ("These evaluations lie along one training trajectory and
are autocorrelated...") stays under every outcome — it applies per seed.

---

## Outcome 1 — signs agree (§4 replacement for the final sentence)

> A degenerate metric would explain it, and we tested for one: $J$ spans a factor
> $441$ over the modelled band and the whitening weights have coefficient of
> variation $0.69$ --- the same order as the synthetic rung-3 geometry where the
> reversal \emph{is} observed ($1.10$; the white control is $0$ by construction).
> A second seed shows the same qualitative signs: paired differences
> $<\text{val\_chi2 diff, seed 2}>$ on the likelihood residual and
> $<\text{val\_mse diff, seed 2}>$ on raw MSE. Two seeds bound seed variability
> rather than establish replication, but the metric is comparably non-degenerate
> at both rungs, so we read the absence as a property of this instantiation, not
> of the ladder: at this capacity and budget the architecture absorbs the
> weighting.

§5 caveat: "Rung 4 is one seed per arm..." → "Rung 4 rests on two seeds per arm,
which bounds but does not characterize seed variability".

Figure 1 panel C: still shows seed 1 only. Either add the seed-2 contrast as a
second pair of bars (preferred; `make_rps_figure.py` gains an `E7` input) or state
"first seed shown; the second agrees in sign" in the caption.

## Outcome 2 — signs disagree (four coordinated edits, not one)

**Changing only §4/§5 would leave misleading claims elsewhere. All four:**

1. **§4** replacement for the final sentence:

> A degenerate metric would explain it, and we tested for one: $J$ spans a factor
> $441$ over the modelled band and the whitening weights have coefficient of
> variation $0.69$ --- the same order as the synthetic rung-3 geometry where the
> reversal \emph{is} observed ($1.10$; the white control is $0$ by construction).
> A second seed does not reproduce the signs: the paired difference on
> $<\text{metric(s)}>$ reverses ($<\text{seed-1}>$ vs.\ $<\text{seed-2}>$), so the
> rung-4 contrast is within seed variance at this scale. Two seeds do not resolve
> a consistent metric effect here: the metric is measurably non-degenerate, and
> the contrast it produces is smaller than the variation between training runs.

2. **Abstract**: "At the attention rung the reversal does not appear, and the
   obvious explanation --- a near-degenerate whitening there --- is ruled out by
   measurement." → "At the attention rung two seeds do not resolve a consistent
   metric effect, and the obvious explanation --- a near-degenerate whitening
   there --- is ruled out by measurement."

3. **Figure 1 panel C**: currently labeled as *the* rung-4 result ("no reversal").
   Show both seed-level contrasts, or relabel title/caption as "first seed"; the
   caption's "both point the same way --- no reversal" must not survive a sign
   flip.

4. **§5 caveat**: → "Rung 4's paired contrast is within seed variance (two seeds
   per arm), so the experiment bounds the effect rather than detecting it."

Also update `notes/claim_audit.md` rung-4 rows to the two-seed status under either
outcome.

## Training-column sentence (both outcomes, only if seed 2 reproduces the pattern)

> The $\chi^2$-trained arm shows no resolved training-MSE advantage in either
> seed, despite a resolved training-$\chi^2$ advantage.

(The earlier draft's "not explained by better optimisation" claimed more than the
evidence supports; do not use it.)

## Page budget — **both outcomes dry-run compiled 2026-08-18, both fit**

With realistic-width placeholder numbers:

- **Outcome 1** + donor 1 (the §3 OF-injection sentence): references start p5,
  0 overfull, 0 undefined.
- **Outcome 2** (all four edits: §4 text, abstract, caption relabel, §5 caveat)
  + donors 1 and 2 (§4's single-split pointer): references start p5, 0 overfull,
  0 undefined.

So on sync day: substitute the real numbers into the applicable version, apply
the listed donor(s), rebuild, re-verify the same three gates. The donor removals
are pre-tested; no new cutting will be needed. Note donor 1 removes the paper's
only `priorThesis` citation — if Outcome 1 (which needs just one donor line)
leaves room, prefer keeping donor 1 and trimming the caption's last sentence
instead, so the attribution stays.
