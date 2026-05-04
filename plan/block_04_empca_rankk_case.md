# Block 04: EMPCA as the Rank-k Case

## Goal

Make `04_empca.tex` the paper's main theoretical engine for moving from rank-1 to rank-`k`.

This section should explain why EMPCA is not "another algorithm section" but the learned-subspace generalization of OF.

## Main file

- `04_empca.tex`

## Source material to reuse

- classic PCA and EMPCA material already in `04_empca.tex`
- weighted frequency-domain formulation already present there
- equivalence theorems and geometry from `OF_EMPCA.tex`
- numerical verification metrics already referenced elsewhere in the repo

## Target structure

1. Rank-`k` constraint
2. EMPCA algorithm
3. Main theorem: rank-1 EMPCA equivalent to OF
4. Proposition: `chi^2` improvement for `k > 1`
5. Geometric interpretation
6. Numerical verification
7. Limitation: algorithmic definition motivates AE framing

## Required content additions

### Formal theorem flow

This section needs a clean theorem sequence:

- main rank-1 equivalence theorem;
- exact algebraic degeneracy statement;
- `chi^2` monotonic improvement with increasing rank;
- remark on finite-sample caveats and subspace identification.

### Gauge and amplitude reconstruction

Do not leave the coefficient/gauge issue as a stub. Add:

- what is identifiable and what is not;
- how amplitudes are compared across bases;
- where regularized solves may be needed.

### Finite-sample realism

Add one remark on:

- small-`N` regime;
- spike separation / subspace recovery caveat;
- why theorem verification remains empirical in finite data.

## Implementation instructions

1. Reuse the current PCA and weighted EMPCA material as the algorithmic background.
2. Put the rank-1 theorem after the algorithm, not before it.
3. Keep the geometric interpretation visual and concise; it should set up the rank-`k` residual improvement claim.
4. Add a short transition from the EMPCA algorithm to the AE bridge: once the objective is linear-subspace reconstruction in whitened space, the encoder-decoder reframing becomes natural.
5. Move any excessive implementation detail to appendix if it interrupts the theorem flow.

## Things to avoid

- Do not let the algorithm subsection dominate the theorem section.
- Do not mix the numerical-verification evidence into the proof text.
- Do not promise exact finite-sample equivalence beyond the stated assumptions.

## Done when

- the rank-1 equivalence theorem reads as the main theorem of the linear paper;
- the case for `k > 1` is mathematically and physically motivated;
- the AE section now feels like the natural next step.
