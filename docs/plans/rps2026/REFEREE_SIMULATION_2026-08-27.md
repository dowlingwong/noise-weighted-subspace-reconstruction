# R/PS 2026 Referee Simulation — 2026-08-27

## Build fingerprint

**Result: PASS.** I reviewed `paper_rps2026/paper.pdf`, an 11-page build.

- `pdfinfo`: 11 pages (expected 11).
- Exact extracted-text count for `held-out ranking reverses`: 0 because the phrase is split at a PDF line/hyphen boundary; the permitted fallback `ranking reverses` occurs exactly once.
- `48.419`: 0 occurrences (expected 0).
- `T-mse) vs. 0.0231`: 1 occurrence (expected 1).

## Frozen independent review (written from the PDF alone)

### 1. Summary

This paper studies low-rank reconstruction under colored Gaussian measurement noise, organizing several classical estimators by the pair of a likelihood-derived residual geometry, \(\Sigma^{-1}\), and an admissible representation family, \(\mathcal F\). It gives a whitening-based geometric account connecting fixed-template optimal filtering, covariance-weighted PCA, a tied linear autoencoder, and a Kronecker-separable two-axis projection, then treats a factored-attention model as a nonlinear analogue. The main controlled result compares two identically structured separable linear models trained with covariance-weighted and isotropic objectives: their held-out ordering reverses between the likelihood residual and raw MSE, consistently across a frozen ten-seed cohort, a disjoint twenty-seed cohort, and one estimated-covariance condition. In a separate exactly separable synthetic condition, the covariance-weighted factors recover the planted channel subspace nearly exactly and improve several truth-referenced quantities, although temporal recovery remains weak. A two-seed public TIDMAD attention pilot does not exhibit the predicted reversal and is far below the published coarse-score baselines. The paper presents its contribution as a unifying measurement-geometry/architecture-capacity perspective and a controlled demonstration, not as a new estimator or competitive nonlinear reconstruction method.

### 2. Strengths

1. **The conceptual decomposition is clean and useful.** The pair \((\Sigma^{-1}, \mathcal F)\) distinguishes what comes from the measurement likelihood from what comes from architectural capacity. Proposition 1 and the whitening argument make the relationship among optimal filtering, covariance-weighted PCA, and a tied linear autoencoder easy to understand.

2. **The comparison underlying the primary reversal is unusually controlled.** In Section 4, NFPA and the isotropic baseline have the same factored architecture and differ only in the training metric. The frozen ten-seed result is reproduced in a separately reported twenty-seed cohort, and the white-noise identity control returns zero as the theory requires.

3. **The paper does not oversell an objective-dependent result.** The abstract explicitly says, “Each model leading under its own objective is the weaker claim,” then adds truth-referenced recovery measurements. That is the correct distinction: the channel projector error of 0.035 versus 1.021 is more scientifically informative than the reversal by itself.

4. **The scope and negative evidence are reported candidly.** The two-seed attention result is labeled an “open observation,” the coarse score's inability to separate the arms is explained in Appendix B.1, and the paper states that both attention arms are far below published baselines. Likewise, the limitations distinguish oracle covariance, one finite-covariance sensitivity condition, post-hoc recovery measurements, and instrument-specific covariance adequacy.

5. **The appendix supports reproducibility and interpretability.** It records sample sizes, ranks, seed counts, restarts, train/validation/test sizes, SNR definition, cohort separation, bootstrap unit, predeclared versus post-hoc analyses, and the nonconvex status of the separable optimization. Figure 2 also makes the Kronecker restriction much more concrete than the notation alone.

6. **The workshop fit is real on self-supervision.** The core observation—unlabeled noisy measurements provide a target while independently characterized instrument noise supplies the residual geometry—is relevant to scientific reconstruction, and the paper explicitly identifies covariance misspecification as a failure of measurement geometry rather than hiding it as an implementation detail.

### 3. Weaknesses

1. **The main empirical “reversal” is partly built into the setup and is not by itself strong evidence of better scientific reconstruction.** Two constrained models optimized under different metrics being ordered differently by those same metrics is plausible and, as the authors concede, the weaker claim. The truth-referenced result helps, but it is run under the distinct exactly separable condition \(\alpha=0\), whereas the primary reversal uses \(\alpha=0.1\). This separation makes the headline empirical story less unified than the abstract initially suggests.

2. **The strongest positive evidence is synthetic and narrow.** The primary study fixes one channel/time shape, rank, SNR, Brownian noise family, nonseparability level, and mostly oracle covariance. The estimated-covariance experiment uses one large calibration size (2,048 noise-only events) and essentially recovers the oracle covariance; it does not show behavior under realistic covariance estimation error or misspecification. The very small relative likelihood improvement (about 1.9%) is perfectly consistent across seeds but has no demonstrated downstream consequence for an actual instrument.

3. **The public-data nonlinear experiment does not validate the proposed architecture-level story.** Section 3 says that the factored attention model is “an analogous axis-factored architecture,” but the two-seed TIDMAD pilot shows no reversal, cannot be distinguished by the coarse score, and is far below published baselines. Thus the fourth “rung” currently functions more as motivation than evidence, and the four-class framing is stronger than the experiments support.

4. **Novelty is mainly organizational.** The manuscript states, “We introduce no new estimator,” and the linear results follow from whitening, Rayleigh–Ritz, and a Kronecker restriction of a classical low-rank problem. The unification is worthwhile for a workshop, but the technical contribution is modest, and Proposition 2 is close to a set-containment characterization: a restricted optimum equals the unrestricted optimum exactly when an unrestricted minimizer lies in the restricted set.

5. **The optimization comparison is not fully resolved by the theory.** The joint NFPA problem is nonconvex, even if each alternating subproblem is solved exactly. Appendix B.2 reports ten restarts, but the paper does not quantify solution variability or give evidence that the covariance-weighted and isotropic arms are comparably close to their respective global optima. With extremely narrow across-seed intervals, optimization stability is an important possible explanation to rule out explicitly.

6. **Some workshop-theme claims are stretched.** The tokenization claim is a sensible engineering rule—use physical-frequency tokens so diagonal inverse-noise weights act per token—but the paper itself says this “selects a physical token axis rather than proposing a learned discrete vocabulary.” Transfer is not demonstrated; the gravitational-wave example is a negative covariance-calibration observation rather than representation transfer. Sampling appears only through simulated training data and noise-only calibration, not through an adaptive simulator or experimental loop.

7. **The density of qualifications impairs the four-page narrative.** Phrases such as “the fixed-template OF case is recovered under Corollary 1’s assumptions” and the distinction among \(\alpha=0.1\), \(\alpha=0\), frozen, replication, estimated-covariance, and pilot conditions are all necessary, but the reader must work hard to track which claim belongs to which condition. Figure 1B's ratios visually emphasize the near-zero channel ratio, while the absolute temporal errors and the different experimental condition are only clear from the surrounding text.

### 4. Questions to the authors

1. Can you report the truth-referenced channel, temporal, and parameter-recovery results at the primary \(\alpha=0.1\) reversal condition, with the same uncertainty reporting, and explain why the main text instead foregrounds recovery at \(\alpha=0\)? I would particularly like to know whether the scientific advantage and the metric reversal coexist on the same data-generating condition.

2. How sensitive are the reversal and recovery conclusions to nonseparability, SNR, rank mismatch, and covariance calibration size or misspecification? If no sweep exists, which of these axes do the authors expect to be the first failure mode, and why?

3. What evidence rules out alternating-optimization quality as the source of the arm differences? Please provide restart-to-restart variability, convergence diagnostics, or objective gaps for both metrics, not only the fact that ten restarts were used.

4. What practical decision on TIDMAD or another instrument would change because the likelihood residual improves by roughly 1.9%? Is there any held-out signal-detection, amplitude-estimation, or calibration result connecting the representation geometry to a downstream scientific quantity under a non-oracle covariance?

### 5. Limitations

The paper acknowledges most important limitations adequately and more explicitly than is typical for a short workshop paper: the unresolved two-seed nonlinear pilot, lack of a competitive public-data result, one unreplicated linear TIDMAD file, per-instrument stationarity assumptions, oracle covariance in the primary experiment, one finite-covariance condition rather than a sweep, absent multiplicity correction for five post-hoc recovery contrasts, and open distribution shift. The main underemphasis is that the strongest truth-referenced result and the primary reversal are obtained under different signal conditions. I would also prefer a sharper limitation statement about nonconvex optimization and about the absence of evidence that the observed residual differences matter for a downstream scientific task.

### 6. Theme fit

- **Self-supervision: central.** The noisy measurement is its own reconstruction target, and an independently characterized noise covariance supplies a physically meaningful residual geometry. This is the paper's clearest and best-earned fit.
- **Transfer: absent.** No learned embedding is shown to extrapolate beyond its training distribution. The gravitational-wave result demonstrates failed covariance calibration on a different real instrument, which is a useful warning but not transfer.
- **Sampling: partial.** The controlled simulator, per-seed resampling, white-noise control, and independent noise-only calibration records are important to the evidence. However, there is no adaptive sampling or experiment/simulator loop shaping training data.
- **Tokenization: partial.** Aligning tokens with the frequency basis in which \(\Sigma^{-1}\) is diagonal is a defensible physical design principle. It is one design choice in an unresolved pilot, not a developed tokenization contribution or a discrete vocabulary for continuous multiscale data.

The abstract's self-supervision positioning is earned. Section 3's token-axis claim is plausible but preliminary; sampling is supportive rather than central, and transfer is explicitly future work rather than an achieved theme contribution. Overall the submission fits the workshop through self-supervised scientific reconstruction even without strong coverage of the other themes.

### 7. Scores

- **Overall recommendation: 7/10 — Accept.** This is a sound, honest, and relevant workshop contribution with a useful unifying view and unusually disciplined controlled evidence. Its technical novelty and real-data impact are limited, but the workshop bar does not require a new SOTA estimator, and the paper clearly distinguishes demonstrated facts from open questions.
- **Confidence: 4/5.** I am confident in the assessment of the paper's mathematical and empirical claims as presented, though I am not a specialist in the TIDMAD instrument and cannot judge whether the chosen covariance model captures all domain-specific measurement issues.
- **Contributed-talk nomination: No.** I would accept it as a poster, but the main positive result is a narrow synthetic linear study built from classical tools, while the public-data nonlinear result is unresolved and noncompetitive; I would reserve a talk slot for a result with broader empirical or methodological consequence.

### 8. The decisive paragraph

This paper earns workshop acceptance because it turns a familiar statistical fact into a clear representation-geometry framework and tests the metric effect under a carefully controlled, replicated comparison. The authors are candid that the estimator is not new and that the nonlinear/public-data evidence is negative or inconclusive, which makes the narrow positive claims credible rather than inflated. The lack of same-condition truth recovery, robustness sweeps, and convincing real-instrument utility prevents a stronger score or a talk nomination.

### 9. What would flip me

- **One step upward:** A response showing uncertainty-quantified truth-referenced recovery at the primary \(\alpha=0.1\) condition, together with a concrete downstream scientific consequence, would unify the reversal and representation-quality stories and move me to 8/10.
- **One step downward:** Evidence that the arm difference is sensitive to alternating-optimization restarts, covariance regularization, or the single chosen synthetic configuration—and therefore is not stable under comparably optimized models—would move me to 6/10.

## Phase-2 spot-check log

No archived-result spot-checks were performed. The PDF gave enough detail to form the review, and a bounded real-reviewer read did not produce a single claim whose verification seemed more decision-relevant than the manuscript's acknowledged scope limitations. The review and score above were frozen before opening any prior review or author-side audit.

## Calibration appendix (read only after the review was frozen)

I calibrated against `docs/plans/rps2026/REFEREE_REPORT_2026-08-26.md` and `paper_rps2026/reviews/2026-08-21-paper-review.md`. The latter reviewed an older ten-page build; the former records both an eleven-page audit and the author-directed fixes that produced the build I read. I did not change the 7/10 score, confidence, theme ratings, or talk decision after reading them.

### Where the independent read agreed

The central agreement is on the submission's risk profile: correctness and evidence discipline are strengths, while ambition is the ceiling. Both prior reports and my blind review identify the contribution as an honest but technically modest organization of classical GLS/weighted-PCA/multilinear-PCA ideas; all regard the controlled cohort separation, white-noise identity check, negative attention result, and explicit limitations as unusually careful. We also independently converged on the same rejection risks: the headline is synthetic, the nonlinear public-data result is two-seed and noncompetitive, the likelihood-residual change is small in practical terms, and the four-page evidence section is dense. The 2026-08-21 report's overall 4/5 assessment is substantively consistent with my 7/10 workshop accept, although my refusal to nominate it for a talk makes the ceiling more explicit.

My reading of the paper's strongest evidence also agrees with the 2026-08-26 audit: the truth-referenced channel/temporal asymmetry is more persuasive than the objective-relative reversal. I likewise understood that rung 3 restricts rung 2, rung 1 is recovered only under Corollary 1's assumptions, rung 4 is analogous rather than nested, and the \(\alpha=0.1\) reversal and \(\alpha=0\) recovery conditions are not textually conflated. That comprehension is useful evidence that the post-audit wording fixes worked.

### Where the independent read diverged

The largest divergence is theme calibration. The 2026-08-21 review treated transfer as directly hit and sampling as absent. I rated transfer absent because a noise model calibrated on synthetic stationary noise and shown to fail on real strain is a negative calibration test, not an embedding that extrapolates beyond its training distribution. I rated sampling partial because the simulator and independent noise-only calibration records materially shape the training setup, although there is no adaptive experimental loop. I also rated tokenization only partial rather than direct: selecting a physically aligned frequency-token axis is sensible, but it is one design choice in an unresolved pilot and not a developed vocabulary or tokenization method.

The prior audits generally treat the clean separation between \(\alpha=0.1\) and \(\alpha=0\) as a resolved correctness issue. I agree that the wording is now correct, but as a referee I still regard the use of different conditions for the headline reversal and strongest semantic-recovery evidence as a narrative and evidential weakness. Correct non-conflation does not establish that both desirable phenomena coexist in the main condition.

### What this blind pass added

I did not find in either prior report an equivalent decision-level request for same-condition, uncertainty-quantified recovery at \(\alpha=0.1\); Appendix B.2 says the post-hoc analysis agrees in direction there, but the main paper does not let the referee assess its magnitude. That became my most important author question and my explicit one-step-up condition.

I also raised optimization comparability more strongly than the prior reports. They verify that the joint NFPA problem is nonconvex, each factor subproblem is exact, and ten ALS restarts were used, but they do not ask for restart-to-restart variability or comparable objective-gap diagnostics for the covariance-weighted and isotropic arms. A real reviewer can reasonably want this before interpreting exceptionally narrow seed intervals as solely a property of representation geometry.

Finally, I made practical consequence—not only statistical consistency—a scoring criterion. The prior reports note the roughly 1.9% likelihood change as a rejection risk, but my blind review asks what actual detection, parameter-estimation, or calibration decision changes under an estimated covariance. That distinction drove the poster-versus-talk judgment.

### What the blind pass missed, and why

I did not flag the abstract phrase “exact geometric characterization” as potentially too strong for a nonconvex joint fit. Because I read Section 1's qualifier and Appendix A alongside the abstract, I silently supplied the caveat; a reviewer who stops at the abstract may not. The 2026-08-26 audit is right that this remains a wording risk.

I praised Figure 2 after turning to the appendix but did not flag that it sits several pages away from the dense rung-3 paragraph. The current build's explicit “in the appendix” pointer was enough for my deliberate read, yet a tired four-page reviewer may not make that page turn. Similarly, I found the rung-2 comparison because I checked Appendix B.2; the 2026-08-21 report correctly warns that many referees will ask the question and miss the answer.

I did not notice the asymmetry in comparing rung 4's 441-fold weight range plus CV against rung 3's CV alone, nor the much larger archived rung-3 range. That requires opening result files or reading with audit-level numerical suspicion, and I intentionally performed no Phase-2 spot-check. I also missed the weak provenance of Appendix B.5's single-split numbers and all submission-metadata/QA-ledger issues; those are invisible from the PDF and therefore appropriately absent from a simulated blind review. Conversely, the current caption now labels the T-mse/T-chi2 in-band values, the contribution sentence now assigns replication and estimated-covariance evidence to the reversal rather than recovery, and the SSL-neighbor paragraph now says why those methods are context rather than comparators. Those fixes prevented the blind read from reproducing issues caught on the older builds, which is evidence that the prior passes improved the submitted artifact.
