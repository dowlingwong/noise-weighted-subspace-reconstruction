# Review prompt — Paper 1, Paper 3 and the experiment implementation (September 2026)

_Paste verbatim to an independent reviewing agent.  The reviewer edits nothing; it writes
one report to `docs/reviews/<date>_external_review.md` in the experiment repository._

---

You are an adversarial referee with three targets, reviewed in this order.  You have
read-only access to three repositories on this machine:

- `~/Documents/arXiv-v1-polished-extended-preprint` — **Paper 1**, "Noise Covariance Defines
  Reconstruction Geometry" (build `main.tex`; new material of 2026-09-02: §3.4, §6, §7 first
  proposition, Appendix E.3–E.4, `07b_separable_evidence.tex` which is deliberately *not*
  input yet; `CHANGELOG.md` top entry; `OPEN_ISSUES.md`).
- `~/Documents/ORACLE` — **Paper 3**, the proposal `latex/paper3_proposal.tex` (new §3
  "Mechanism"), `docs/EXPERIMENT_DESIGN.md`, `docs/IMPLEMENTATION_PLAN.md`.
- `~/Documents/noise-weighted-subspace-reconstruction` — the **experiments**:
  `docs/EXPERIMENT_PLAN_PAPER1.md`, `docs/EXPERIMENT_PLAN_PAPER3.md`,
  `docs/IMPLEMENTATION_CHECKLIST.md`, `docs/plans/oracle/PREREGISTRATION.md`,
  `docs/reviews/2026-09-02_experiments_review.md` (the author's own review — read it first so
  you do not re-raise what is already recorded there; do challenge its conclusions),
  code under `experiments/paper1/`, `experiments/oracle/`, results under `results/paper1/`,
  `results/oracle/`.

Rules.  Verify before you assert: run the tests
(`PYTHONPATH=src:. python -m pytest -q experiments/oracle/oracle_cov/tests`) and open the
result files you cite.  Every finding must name a file and line or a result key, state the
concrete failure scenario, and say whether it is CONFIRMED (you reproduced it) or PLAUSIBLE.
Rank findings by severity.  Do not pad: a section with no finding says "none".  Never
re-raise an item listed as resolved in `OPEN_ISSUES.md` or in the author's review §A.
Numbers under `results/oracle/*/dev/` and `results/paper1/P1-E2` (reduced restarts) are
development output and must not be treated as evidence for or against a claim; judge the
*design* and the *code*, and whether the dev output is consistent with the pre-registered
predictions.

## Part 1 — Paper 1 (the manuscript)

1. **Mathematics.**  Check Proposition 6.1 (Kronecker-restricted bridge), Corollary 6.2
   (rank-one NFPA ↔ OF), Proposition 3.x (stationary basis as precision eigenbasis) and
   Proposition 7.1 (architecture-neutral loss selection) line by line.  Do the stated
   hypotheses suffice?  Is the vectorisation convention (`vec(AZB) = (Bᵀ⊗A) vec Z`) used
   consistently between §6, Table 1 and Appendix E.3?  Is the claim "a full-rank factor does
   not deactivate the restriction" correctly stated?
2. **Story.**  State the paper's one-sentence claim in your own words after reading §1 and
   §7 only.  Does every section serve it?  Where does the text still argue with the reader
   (count negative-result clauses in the abstract and §7)?  Is the (Σ⁻¹, F) framing
   introduced early enough to organise §3–§6?
3. **Evidence wiring.**  `07b_separable_evidence.tex` is drafted for input after 29 Sep.
   Check each number against `paper_rps2026/sections/04_experiments_discussion.tex` and
   `paper_rps2026/notes/claim_audit.md`; list any that will need re-verification against the
   archived JSON digests.  Does the boundary paragraph on the rung-4 pilot over- or
   under-claim?
4. **Two decisions for the author** (answer with a recommendation and a reason): should
   Proposition 7.1 move before §6; should the three instrument subsections be replaced by
   the table in `results/paper1/P1-E4/instrument_table.tex`.

## Part 2 — Paper 3 (the ORACLE proposal)

1. **Mechanism section.**  Is the pullback/Fisher derivation correct and is its use as the
   consequence statistic justified?  The coverage statement: is the claim "support shift
   lives in the unexcited tangent complement, acquisition shift in the excited span"
   consistent with the author's own smoke finding B1–B3 (jitter is N by contract but S by
   geometry; amplitude/decay strata move along excited coordinates; the nonseparable
   deformation is invisible in the mean)?  Propose the minimal rewording of §3 that is true.
2. **Claims C1–C4.**  For each, name the single primary endpoint and check it is the one
   `PREREGISTRATION.md` §4 states.  Is the designed dissociation (`null`/`aligned`) an
   honest test given that it is constructible only at stages with a Jacobian null space?
3. **Scope.**  Five claims, three tiers, a repair arm, five pages.  Which one claim should
   the proposal lead with, and what would you cut?
4. **Relation to Paper 1.**  Is the division "Paper 1 owns (Σ⁻¹, F); Paper 3 owns Σ̂ ≠ Σ and
   P_train → P_target" stated in both papers, and is the pointer in Paper 1's Limitations
   sufficient?

## Part 3 — Experiment implementation

1. **Correctness.**  Read `experiments/oracle/oracle_cov/{generator,subjects,tangent,families,
   monitors,pipeline}.py`.  Specifically test: (a) `ForwardModel` reproduces `clean_train`
   bit-for-bit (it must); (b) `tangent_geometry` rank is 4 for the planted family and T_S ⟂
   T_U; (c) `PullbackWhitened.raw` equals the Mahalanobis norm of `J δ` at the linearisation
   point; (d) `designed_perturbation` produces a first-order null (Jacobian-projected score
   < 1e-8 at the window mean); (e) the clean-window standardisation in `tangent_statistics`
   gives z ≈ 0 on fresh clean windows; (f) the FAR threshold is set on windows disjoint from
   those that fitted the monitors.
2. **Leakage.**  Can any monitor see labels, truth amplitudes, or the deployment covariance?
   (`PullbackWhitened` must use only the assumed covariance.)  Does the attribution
   classifier's leave-one-seed-out split leak seeds through the clean reference?
3. **Statistics.**  Are the resampling units right (windows and seeds, never events)?  Is
   the C4 within-stratum AUROC computed only in strata containing both consequence classes?
   Is BH-FDR applied within claim?
4. **Paper 1 runs.**  `experiments/paper1/{fig1_factors,alpha_sweep,rank_one_check,s5_provenance}.py`
   and `scripts/make_instrument_table.py`: does each acceptance rule in
   `docs/EXPERIMENT_PLAN_PAPER1.md` map to a check in the code?  Does `P1-E1` really regress
   against the archived E8 values, and to what tolerance was it met?
5. **The author's open items C1–C7** in `docs/reviews/2026-09-02_experiments_review.md`:
   for each, say fix-now / fix-before-confirmatory / accept-with-caveat, with one sentence.
6. **What is missing** to reach the checklist's Tier-1 definition of done: list the
   concrete steps, ordered, with an effort estimate.

Report format: `# External review <date>`, then Parts 1–3 with numbered findings
`[SEVERITY] [CONFIRMED|PLAUSIBLE] file:line — finding — failure scenario — fix`, then a
final section "Three things I would do first".
