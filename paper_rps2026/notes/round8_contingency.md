# Round 8 Task B contingency — drop-in text for the seed replication

When `E7_rung4_seed_replication.json` lands, replace the rung-4 paragraph's final
sentence (currently ending "...the architecture absorbs the weighting, with one seed
per arm.") with ONE of the two versions below, filling the `<>` values from the JSON,
and update §5's first caveat as indicated. Nothing else in the paper changes.

The C1 caveat sentence ("These evaluations lie along one training trajectory and are
autocorrelated...") stays in both outcomes — it applies per seed.

---

## Outcome 1 — both seeds agree in sign on both metrics

§4 replacement:

> A degenerate metric would explain it, and we tested for one: $J$ spans a factor
> $441$ over the modelled band and the whitening weights have coefficient of
> variation $0.69$ --- the same order as the synthetic rung-3 geometry where the
> reversal \emph{is} observed ($1.10$; the white control is $0$ by construction).
> A second seed replicates the signature: paired differences
> $<\text{val\_chi2 diff, seed 2}>$ on the likelihood residual and
> $<\text{val\_mse diff, seed 2}>$ on raw MSE, agreeing in sign with the first
> seed on both metrics. The metric is comparably non-degenerate at both rungs and
> the finding replicates, so we read the absence as a property of this
> instantiation, not of the ladder: at this capacity and budget the architecture
> absorbs the weighting.

§5 caveat edit: change "Rung 4 is one seed per arm, so its held-out differences are
contrasts over evaluations rather than evidence about seed variability" to
"Rung 4 rests on two seeds per arm, which bounds but does not characterize seed
variability".

## Outcome 2 — signs disagree on either metric

§4 replacement:

> A degenerate metric would explain it, and we tested for one: $J$ spans a factor
> $441$ over the modelled band and the whitening weights have coefficient of
> variation $0.69$ --- the same order as the synthetic rung-3 geometry where the
> reversal \emph{is} observed ($1.10$; the white control is $0$ by construction).
> A second seed does not reproduce the signature: the paired difference on
> $<\text{metric(s) that flipped}>$ changes sign
> ($<\text{seed-1 value}>$ vs.\ $<\text{seed-2 value}>$), so the rung-4 contrast
> is within seed variance at this scale. The honest statement is that this
> experiment does not resolve whether the metric matters for this architecture at
> this budget --- the metric is measurably non-degenerate, and the contrast it
> produces is smaller than the variation between training runs.

§5 caveat edit: change the first caveat to "Rung 4's paired contrast is within
seed variance (two seeds per arm), so the experiment bounds the effect rather than
detecting it".

## Also report (both outcomes)

If the new seed reproduces the training-column pattern (training-MSE difference
unresolved, training-χ² resolved), add after the held-out numbers:

> The $\Sigma^{-1}$-trained arm's advantage on \emph{training} MSE is not resolved
> in either seed, so its held-out raw-MSE advantage is not explained by better
> optimisation of the raw objective.

## Page budget

Each replacement is 1–3 lines longer than the current sentence. Donors, in order:
the §3 sentence "Injecting the optimal-filter statistic..." (secondary contrast,
can move to the appendix); the §4 sentence "The single-split $C=k_c=2$ run is in
Appendix~B.2."; caption last sentence. Verify references still start on p5 after
the swap.
