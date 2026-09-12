# T19-S protocol amendment — 12 September 2026

The frozen `PROTOCOL_T19S.json` (EXP `bcf7f27`) is **not edited**. This dated
amendment records a correction to its part-1 derivation found by the local
review of 12 September 2026.

## What was wrong

`part_1_deterministic.derivation` specified the model `A + eps G` with symmetric
additive `G`. That is **not** the model the paper or the archived T19 use. The
fitted covariance-error subspace comes from `A (I + eps E)^{-1}`, whose
expansion carries higher-order operator terms `-eps A E`, `+eps^2 A E^2`, ...
that the additive recurrence omits entirely.

The additive calculation therefore could not establish the archived mechanism,
even though its stated conclusion happened to be correct.

`claim_c` also justified the cubic norm behaviour by "Frobenius-orthogonality"
and the accompanying docstring gave the fourth-order projector expansion with
the **wrong sign**, adding the squared diagonal-block contributions instead of
subtracting the normalization correction of the same order.

## What replaces it

The covariance-model recurrence is derived explicitly in `t19s_observables.py`
and reproduced in `FINDINGS_T19S.md`. The corrected projector identity is

    ||P(X) - P(0)||_F^2 = 2 tr[X^T X (I + X^T X)^{-1}]
                        = 2||X||_F^2 - 2 tr[(X^T X)^2] + O(||X||^6)

Both models are now implemented and **separately labelled**; the additive model
is retained only as a control.

## What is unchanged

The predeclared orders, the tolerance, the acceptance rule and the failure
branch of the frozen protocol all stand. The registered T19 coupling-family
slope prediction remains **not met**, and Proposition 5.2's `44 eta^2` bound
remains restricted to `eta <= 0.1`. Nothing here extends it.

## What was not executed

`part_2_grid` (120 estimator fits) was **not run**. The mission's eight-hour
elapsed ceiling expired at 06:39:12Z on 12 September 2026 before the grid
started, and no new scientific computation was authorised after that point.
The grid remains a frozen, executable design awaiting a new allocation.
