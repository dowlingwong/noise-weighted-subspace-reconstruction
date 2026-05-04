# Block 05: Linear Autoencoder Bridge

## Goal

Turn `05_linear_ae.tex` into the payoff section of the paper.

The reader should feel that the AE bridge is the inevitable conclusion of the previous sections, not a detached ML add-on.

## Main file

- `05_linear_ae.tex`

## Source material to reuse

- existing tied linear AE proof already in `05_linear_ae.tex`
- numerical equivalence table already present there
- EMPCA objective from `04_empca.tex`
- whitened-objective framing from `02_unified_objective.tex`

## Target structure

1. Encoder-decoder language
2. Noise-aware loss
3. Bridge theorem
4. EM as coordinate descent on the AE objective
5. Numerical verification: ablation
6. Connection to Paper 2

## Required content additions

### Formal bridge theorem

The theorem should say clearly:

- what the linear AE objective is;
- what constraints are assumed;
- why its minimizer spans the EMPCA subspace;
- in what sense the equivalence holds.

### Why noise-aware loss matters

Do not just replace MSE by a weighted norm and move on. Explain:

- standard MSE encodes isotropic noise;
- whitening changes geometry, not architecture;
- the paper's transferable principle is "noise in the loss, not in a more complicated model by default."

### Coordinate-descent interpretation

Explain the E-step and M-step as:

- latent-code solve;
- basis update;
- same objective, different parametrization.

## Implementation instructions

1. Keep the projector proof short and clean.
2. Add a formal theorem statement before or around the existing proof.
3. Put the numerical table immediately after the theorem logic, not pages later.
4. Add one ablation framing sentence: same linear architecture, different loss.
5. End with one paragraph only for Paper 2: the linear encoder is the part to relax next.

## Things to avoid

- Do not reintroduce duplicate AE content in another section.
- Do not let this section become a VAE or deep-AE survey.
- Do not oversell nonlinear implications that are not demonstrated yet.

## Done when

- the bridge theorem is explicit and defensible;
- the reader understands why AE enters the paper at all;
- the transition to Paper 2 is clean and restrained.
