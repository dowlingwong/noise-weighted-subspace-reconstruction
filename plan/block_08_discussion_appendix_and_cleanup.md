# Block 08: Discussion, Appendix, and Final Cleanup

## Goal

Finish the manuscript so it ends like a coherent MLST paper and not like a partially merged note set.

## Main files

- `08_discussion.tex`
- `appendix.tex`
- `main.tex`

## Source material to reuse

- current discussion stubs in `08_discussion.tex`
- detector/simulation material already present in `appendix.tex`
- old noise-model content in `noise.tex`
- file-organization evidence from the repo and `EMPCA_improvement.pdf`

## Target structure

### Discussion

1. The noise-aware loss as a transferable principle
2. Relation to PPCA and VAE
3. Limitations
4. Companion paper preview
5. Conclusion

### Appendix

1. proof of main theorem
2. proof of bridge theorem
3. convergence proof details
4. simulation and dataset details
5. structured-noise module details
6. implementation specifics that would overload the main text

## Required content

### Discussion

Add the missing conceptual positioning:

- PPCA as isotropic special case;
- factor-analysis link for structured noise;
- VAE as orthogonal latent-variable extension, not the main story here.

### Limitations

Be direct about:

- stationarity assumption;
- known or estimated covariance dependence;
- finite-sample subspace limits;
- pile-up / nonlinear pulse-family limits.

### Appendix cleanup

The appendix should absorb:

- detector physics setup;
- simulator pipeline details;
- K-alpha characterization;
- noise-generator / structured-noise module specification;
- proof overflow.

## Implementation instructions

1. Make the discussion widen the significance but keep it brief.
2. Move detector-heavy exposition out of the main paper if it is not needed for a theorem or experiment interpretation.
3. Check that duplicate or removed nonlinear sections do not resurface.
4. Verify every cross-reference, table label, and theorem label after section growth.
5. End the conclusion with the design principle plus one restrained sentence about Paper 2.

## Things to avoid

- Do not introduce new major results here.
- Do not turn the companion-paper preview into a second abstract.
- Do not leave appendix material unlabeled or disconnected from the main text.

## Done when

- the discussion clearly positions the paper in ML/ST language;
- the appendix carries the heavy technical load cleanly;
- the paper closes on one design principle and one realistic next step.
