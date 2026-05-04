# Revision Plan — Remaining Blockers

Six issues to resolve, ordered from quickest to most involved.

---

## 1. Theorem naming fix (30 min) — `04_empca.tex` ~line 408

**The problem.** `thm:rank1-equivalence` is titled `[Rank-1 OF--EMPCA equivalence]` with no indication in the theorem block itself that the result is at the population level. The population caveat currently lives only in a downstream remark (line 476) and the finite-sample subsection (~line 643), so a skimming reviewer reads the theorem as stronger than it is.

**What to change.** Rename the theorem title and add a scoping clause to the conclusion:

```latex
% BEFORE
\begin{theorem}[Rank-1 OF--EMPCA equivalence]

% AFTER
\begin{theorem}[Rank-1 OF--EMPCA equivalence, population level]
```

Then in the conclusion block of the theorem (the three-item `enumerate`), prepend "At the population level," to item 1:

```latex
% BEFORE
\item the rank-1 EMPCA subspace equals $\mathrm{span}(\tilde{s})$ ...

% AFTER
\item At the population level, the rank-1 EMPCA subspace equals $\mathrm{span}(\tilde{s})$ ...
```

**What to proof-read after.** Grep every `\ref{thm:rank1-equivalence}` call site (01_intro.tex line 144, 03_optimal_filter.tex line 367, 05_linear_ae.tex lines 207 and 302, 08_discussion.tex line 198, OF_EMPCA.tex line 255, appendix.tex line 10) and check that the surrounding prose still reads correctly with the new title in a `\ref` cross-reference.

**How to improve further.** Consider adding a one-sentence preamble before the theorem:

> "The following theorem is a population-level result; finite-sample corrections are given in §\ref{subsec:empca-finite-sample}."

This makes the scoping explicit even when the theorem is read in isolation.

---

## 2. k > 1 containment caveat — `04_empca.tex` ~line 560–585

**The problem.** The proposition correctly conditions on $\mathcal{S}_1 \subseteq \mathcal{U}_k$ (line 562), but there is no sentence anywhere in the body that states learned EMPCA is *not* guaranteed to satisfy this containment unless the basis is constrained or initialized with the template. The reviewer specifically asked for this disclaimer, plus four validation metrics.

**What to change.**

After the discussion of eq. `\eqref{eq:equiv-k-chi2}` (the $\chi^2$ inequality), add a warning paragraph:

```latex
\paragraph{Caveat: containment is not automatic.}
The inequality $\chi^2_{\mathrm{EMPCA},k} \le \chi^2_{\mathrm{OF}}$ holds
only when $\mathcal{S}_1 \subseteq \mathcal{U}_k$, i.e.\ when the OF
template direction lies within the learned EMPCA subspace.
Standard unsupervised EMPCA minimizes the population reconstruction
error and will include $\mathcal{S}_1$ in $\mathcal{U}_k$ whenever the
signal component is the dominant variance direction---which is the
case when $\mathbb{E}[a^2]\,\|\tilde{s}\|^2$ exceeds the next-largest
noise eigenvalue.
If that SNR condition is not met, or if EMPCA is initialized without
reference to the template, the containment $\mathcal{S}_1 \subseteq \mathcal{U}_k$
may fail and the inequality can be violated in finite samples.
Constrained initialization (seeding $u_1 \leftarrow \tilde{s}/\|\tilde{s}\|$)
or a joint OF+EMPCA basis is then required to guarantee the bound.
```

**Four validation metrics (scaffold in §7 subsections).** These are currently only referenced in §7 TODOs. Add a forward-reference sentence immediately after the caveat paragraph:

```latex
Section~\ref{sec:experiments} verifies empirically that the containment
holds in our benchmark regime by reporting: (i)~weighted residual
$\chi^2_{\mathrm{EMPCA},k}$ versus $k$; (ii)~amplitude bias
$|\hat{a}_{\mathrm{EMPCA}} - \hat{a}_{\mathrm{OF}}|$ at $k=1$;
(iii)~energy resolution $\sigma_E(k)$; and (iv)~held-out log-likelihood
as a function of rank.
```

**What to proof-read after.** Re-read the entire §4 k>1 subsection end-to-end to verify the caveat paragraph doesn't contradict the SNR condition already stated in the finite-sample subsection (~line 643). Align the SNR notation ($\Delta_\lambda$ there vs. the energy-eigenvalue framing here).

---

## 3. Szymkowiak et al. 1993 citation — `references.bib` + body files

**The problem.** `grep` finds zero matches for "Szymkowiak" in `references.bib` and no `\cite` anywhere in the tex files. The reviewer called it out by name.

**What to add to `references.bib`:**

```bibtex
@article{Szymkowiak1993,
  author  = {Szymkowiak, A. E. and Kelley, R. L. and Moseley, S. H. and Stahle, C. K.},
  title   = {Signal processing for microcalorimeters},
  journal = {Journal of Low Temperature Physics},
  year    = {1993},
  volume  = {93},
  number  = {3--4},
  pages   = {281--285},
  doi     = {10.1007/BF00693433},
}
```

**Where to cite it in the body.** The natural home is `01_intro.tex` in the paragraph that introduces the optimal filter / matched filter in the microcalorimeter context — wherever the first mention of "optimal filter" or "matched filter for calorimetry" appears. A second cite is appropriate in `03_optimal_filter.tex` in the opening motivation paragraph. A third cite in the discussion (§8) when comparing to prior work is also standard.

**What to proof-read after.** Compile with `bibtex` / `biber` and confirm the entry resolves without error. Check the journal name renders correctly (some templates italicize, some abbreviate).

---

## 4. `07_experiments.tex` — six TODOs and the empty Study B table (days of writing)

This is the largest remaining blocker. The six TODO subsections and the empty table are listed below with specific guidance for each.

### 4a. Benchmark domain (~§7.1)

The stub says only "expand with concise experimental setup description." Write 2–3 paragraphs covering: detector type and operating temperature, digitizer sampling rate and trace length, number of events in the training set vs. held-out set (specify the train/held-out split the reviewer asked for), and the noise sources characterized in Study A. Cross-reference Appendix D for full simulation details.

### 4b. Evaluation metrics (~§7.2)

Write a compact enumeration (prose, not bullet-list) of the four metrics the reviewer requested:

1. **Weighted residual** $\chi^2_{\mathrm{EMPCA},k}$ and its ratio to $\chi^2_{\mathrm{OF}}$, as a function of $k$.
2. **Amplitude bias** $|\hat{a}_{\mathrm{EMPCA}} - \hat{a}_{\mathrm{OF}}|$ at $k=1$, verified to be $< 0.1\%$ (this is the parity table the reviewer asked for).
3. **Energy resolution** $\sigma_E(k)$ (FWHM in eV).
4. **Held-out log-likelihood** as a function of rank $k$, used for model-order selection.

State the train/test split explicitly here (e.g., 80/20 or a fixed held-out set of $N_{\rm test}$ events).

### 4c. Verification of equivalence theorems (~§7.3)

The comment already points to `tab:of-empca-verification` and `tab:empca_ae_primary`. Fill this subsection by:

1. Reporting the OF vs. rank-1 EMPCA amplitude parity table (the reviewer's first requested artifact). Show that estimated amplitudes agree to within numerical precision when whitening is matched.
2. Reporting the principal-angle vs. iteration plot (the reviewer's second artifact): plot the principal angle between the EMPCA subspace and the OF template direction as EM iterations proceed, showing convergence to zero.

If these tables/figures already exist as notebook outputs, export them and include them. If not, note exactly what script needs to run.

### 4d. Reconstruction quality vs. rank k (~§7.4)

Fill the stub with:

1. A table of $\chi^2_{\mathrm{EMPCA},k}$ and $\sigma_E(k)$ for $k = 1, 2, \ldots, k_{\max}$ (the reviewer's third artifact: weighted-residual/energy-resolution vs. $k$ table).
2. A brief narrative interpreting the elbow in the curve — where adding rank stops improving $\sigma_E$.

### 4e. Noise-aware loss vs. isotropic MSE (~§7.5)

This is the reviewer's fourth artifact: an ablation comparing PSD-weighted loss to plain $\ell_2$ loss. Write:

1. A description of the two training conditions (PSD-weighted EMPCA vs. unweighted PCA).
2. A table or figure comparing $\sigma_E$ and $\chi^2$ for both, on the held-out set.
3. A sentence explaining why the isotropic-MSE model is expected to underperform at low frequencies where the noise PSD is colored.

### 4f. Convergence behavior (~§7.6)

Expand the stub (which currently has only the cuBLAS enumeration) with:

1. A plot of the EMPCA objective vs. EM iteration number, for several values of $k$ (the reviewer's convergence-plot request).
2. Quantitative statement of the convergence criterion used and the number of iterations to reach it.

### Study B table (line 96)

The `% fill from notebook export` placeholder needs the actual scenario results. Export the leave-one-out and scale-down rows from the notebook and paste them in. The column headers (`Scenario`, `Relative sensitivity`, `Improvement vs baseline`) are already in place.

**What to proof-read after §7 is written.** Read §7 against the abstract and §1 introduction and verify every claim in the abstract is now backed by a number or figure in §7. In particular, check the abstract's wording about "experimental validation on cryogenic detector pulse data" and "energy resolution improvement" — both must be quantified somewhere in §7.

---

## 5. Appendix proof bodies — `appendix.tex` (major writing task)

The intro (§1, line 186) promises: "Technical proof details and detector-specific simulation material are collected in the appendix." Eight sections are currently TODO stubs. Priority order:

| Appendix section | Priority | Notes |
|---|---|---|
| Proof of Main Theorem (thm:rank1-equivalence) | **High** | Proof already exists inline in §4 (04_empca.tex ~line 440). The appendix version should be the extended proof — move or expand the inline proof here and add a pointer in §4: "Full proof in Appendix A." |
| Proof of Bridge Theorem (§5.3) | **High** | Write from scratch. Check 05_linear_ae.tex for the theorem statement. |
| Convergence proof details | **Medium** | Expand on §6 material; cross-ref app:convergence-proof. |
| LAMCAL and cryogenic detector response | **Medium** | Stub only. Write 1–2 pages of detector physics. |
| Quasi-particle signal formation | **Medium** | Stub only. Write 1 page; cross-ref the TraceSimulator description already present in §A.3. |
| NoiseGenerator implementation details | **Low** | The §A.3 summary paragraph exists; expand with pseudocode or equations. |
| OF implementation | **Low** | Write pseudocode block. |
| EMPCA implementation | **Low** | Write pseudocode block. |

**What to proof-read.** After filling the Main Theorem proof appendix section, compile and check that the `\label{app:convergence-proof}` label resolves from `\ref` calls in §6. Also verify the appendix section letters auto-assigned by LaTeX match any hard-coded cross-references (e.g., "Appendix A" written as prose in §1 or §4).

---

## 6. Cross-cutting: consistency check after all edits

Once issues 1–5 are addressed, do a final pass:

1. **Grep `TODO` and `fill from notebook export`** across all `.tex` files — confirm zero remaining hits in files that will be submitted.
2. **Grep `\ref{thm:rank1-equivalence}`** — verify all 7 call sites read naturally with the new "(population level)" title.
3. **Compile end-to-end** and check for undefined references (`?` in the PDF) and bibtex warnings.
4. **Read the abstract against §7** — every quantitative claim in the abstract must have a backing number in the paper.
5. **Read §1 introduction bullet list** (lines ~144–147) — the promise about Theorem 1 should mention the population-level scope after the rename.

---

## Summary priority table

| # | File | Location | Effort | Blocks submission? |
|---|---|---|---|---|
| 1 | `04_empca.tex` | line ~408 | 30 min | No, but creates review risk |
| 2 | `04_empca.tex` | line ~580 | 1–2 hr | Partially |
| 3 | `references.bib` + body | — | 30 min | Yes (reviewer called it out) |
| 4 | `07_experiments.tex` | all subsections | Days | **Yes** |
| 5 | `appendix.tex` | all TODO sections | Days | Yes (intro promises it) |
| 6 | All files | final grep/compile | 1 hr | — |
