# Desk revision log — R/PS 2026 short paper

**Date:** 2026-08-08
**Agent:** desk agent (manuscript-only scope)
**Source plan:** `plan/DESK_REVISION_PLAN_2026-08-08.md`
**Parallel agent:** experiment agent executing `plan/EXPERIMENT_PLAN_2026-08-08.md`

---

## 1. Task status

| ID | Task | Status | Files touched |
|---|---|---|---|
| D1 | GLS / matched-filter lineage | **done** | `sections/01_introduction.tex` |
| D2 | PPCA / FA novelty boundary | **done** | `sections/01_introduction.tex`, `sections/02_covariance_geometry.tex` |
| D3 | SSL anchoring | **done** | `sections/01_introduction.tex`, `references.bib` |
| D4 | Paper 3 GWOSC paragraph | **done** | `sections/04_experiments_discussion.tex`, `notes/claim_audit.md` |
| D5 | Provenance | **done** (fixes 2 + 3; fix 1 deferred) | `appendix/appendix_experiment_details.tex`, `sections/04_experiments_discussion.tex` |
| D6 | Figure slot | **done** | `sections/04_experiments_discussion.tex` |
| D7 | Raw-MSE re-export | **done** (values recovered) | `tables/results_summary.tex`, `appendix/appendix_experiment_details.tex`, `tmp/re_export_paper2_mse.py` |
| 8 | Page-budget pass and build | **done with a caveat** — see §3 | — |

### D3 — new citation verification (required by plan)

Every new bib entry was verified against a primary source before inclusion. Recorded here per plan §0.4.1:

| Key | Paper | Verification URL checked | Result |
|---|---|---|---|
| `chen2020simclr` | Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020. | `https://arxiv.org/abs/2002.05709` (arXiv page; comments field states ICML'2020) and `https://arxiv.org/bibtex/2002.05709` | Title, author list, year match. Venue (ICML 2020) confirmed by arXiv comments. |
| `he2022masked` | He, Chen, Xie, Li, Dollár, Girshick. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022. | `https://arxiv.org/abs/2111.06377` (arXiv page) and `https://arxiv.org/bibtex/2111.06377` | Title, author list match. Published at CVPR 2022 (pages 16000–16009). |

The plan allowed a third optional scientific-representation citation but only if verifiable against a primary source; I chose to keep two verified citations rather than risk an unverifiable third. This matches the plan's stated fallback ("If you cannot verify, omit it and keep two").

### D4 — source of truth for the GWOSC numbers

- `transfer_paper/tables/gwosc_filter_equivalence_summary.csv` (max_abs_score_difference: synthetic 4.98e-15, H1 1.60e-12, L1 1.45e-12; threshold 1e-10; accepted=True for all three).
- `transfer_paper/tables/gwosc_time_local_psd_summary.csv` (synthetic `local_radius_64s` score_std 1.1476, accepted=True; H1 score_std 9.4242, L1 8.8268, accepted=False; global leave-one-out H1 1.7713, L1 7.2880; coverage 1.0 for every row).

The paragraph reports the over-dispersion factor (seven to nine times) and the passing synthetic control, and makes exactly the one claim the plan permits ("implementation identity is not calibration"). A corresponding row was added to `notes/claim_audit.md` with status `experiment, real data, archived`.

### D5 — provenance

- **Fix 2 (done, unconditional):** added `\subsection{Predeclared confirmatory protocol}` with `\label{app:protocol}` to `appendix/appendix_experiment_details.tex`, condensing `notes/nfpa_submission_experiment_plan.md` (hypotheses H1–H4, primary configuration, acceptance gates). Changed the limitations line from "described separately" to "in Appendix~\ref{app:protocol}".
- **Fix 3 (done):** the forward reference is now a concrete `\ref{app:protocol}`.
- **Fix 1 (deferred):** anonymized repository link. `gh` is not installed on this machine and no anonymous.4open.science credentials are available, so I cannot create or verify an anonymized repo. Per plan, a dead anonymized link is worse than none, so this is left as an author action. **Action for authors:** create the anonymized repo, verify the URL resolves, and add the page-1 footnote before submission.

### D7 — re-exported raw-MSE values

The archived Paper 2 run is `src/NFPA/nfpa_demo.py` in `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction` (deterministic, `np.random.seed(42)`). It reproduces the archived table exactly (NFPA χ̄² 0.99161, IsoAE χ̄² 0.99949, angles `[0,59.3,64.6,73.8,84.8,87.8]`). `tmp/re_export_paper2_mse.py` re-imports that archived run and prints every held-out metric at full precision. Values substituted into `tables/results_summary.tex`:

| Method | χ̄² | Raw MSE (full precision) |
|---|---|---|
| CW-PCA (EMPCA) | 0.9929 | 4.949487390e-03 |
| NFPA | 0.9916 | 4.949487330e-03 |
| IsoAE | 0.9995 | **4.582994617e-09** |
| IsoPCA | 1.0007 | **4.333467677e-09** |

Traceability: numbers transcribed only from the archived run via `tmp/re_export_paper2_mse.py`. Not superseded (Block A has not landed; see handoff note).

---

## 2. Page count

| State | Total pages | Main content | References | Appendix |
|---|---|---|---|---|
| Before (baseline 2026-08-08) | 6 | 4 | 1 | 1 |
| After desk revisions | 9 | **5** (see §3) | 2 | 2 (proofs + experiment details) |

The reference and appendix growth is excluded from the four-page limit (plan §0.2.1). Bibliography grew by two entries and now spans two pages; appendix experiment-details now spans two pages because of the added protocol subsection.

---

## 3. Page-budget pass — status and the one open item

**Build status: clean.** `make` completes with **0 overfull boxes, 0 undefined citations, 0 LaTeX warnings** in the final `paper.log`. Verified with `pdfinfo paper.pdf` (9 pages).

**Main content is currently 5 pages, not ≤ 4.** Page 5 holds ~0.46 page of overflow (the tail of the first interpretation paragraph, the "small held-out difference / rank-one diagnostic" paragraph, Limitations, Takeaway) and the bibliography starts on page 6.

This is the **plan-sanctioned interim state**, not an overrun of my scope:

- Plan §0.3 budgets desk additions at ~0.45 pages and the experiment agent's §4 rewrite at freeing ~0.3 pages ("Do not perform emergency cuts on the assumption that you must absorb the full overflow alone").
- I applied the cut order from `notes/cut_list.md` for the items I own:
  1. **(cut item 3, done)** The tuple display equation `(\Sigma^{-1},\mathcal{F})` (`eq:tuple`) was folded into an inline phrase in `sections/01_introduction.tex` — it was unreferenced, saving ~4 lines.
  2. **(cut item 4, done)** The parameter-count sentence in `sections/03_nfpa.tex` was removed, saving ~2 lines.
- The remaining cut-order items 1 ("extra experimental setup detail") and 2 ("the rank-one diagnostic paragraph") live in `sections/04_experiments_discussion.tex` lines 6–13, which the desk–experiment interface contract (plan §0.3) assigns to the experiment agent. Touching them would create a merge conflict with the parallel agent, so I did not.

**Handover for the experiment agent:** your rewrite of lines 6–13 is expected to free ~0.3 pages. If after your rewrite the main text still exceeds four pages, the prescribed next cuts in `notes/cut_list.md` are (in order): cut item 1 (extra experimental setup detail) then cut item 2 (the rank-one diagnostic sentence at the end of line 13). Do not invent cuts beyond that order. My two cuts (tuple, parameter count) are already applied and need no further action.

---

## 4. Claim-audit adherence

- No claim was added beyond `notes/claim_audit.md`. The one new claim (D4, covariance adequacy on real data) was added to the audit with status, provenance, and controls, per plan §D4.
- No change contradicts the audit. The D2 boundary ("Do not claim full-covariance weighted PCA is novel") was respected — `bailey2012principal`/`allen2014gls` remain the full-covariance citations at the intro's "We ask" paragraph, and the new D2 text only states the PPCA/FA partial-relaxation boundary and the OF placement, exactly where the audit puts it.
- D7 numbers come only from the archived run (no fabrication).

---

## 5. Diff summary

```
git diff --stat  (working tree vs HEAD, paper subdirectory)
```

The paper directory is currently **untracked** in the parent repo (`rps_noise_covariance_representation/` and `sources/` are new), so `git diff` shows nothing until the directory is added. Files changed by the desk pass:

- `sections/01_introduction.tex` — D1+D2 paragraph, D3 sentence, tuple folded inline (cut item 3).
- `sections/02_covariance_geometry.tex` — D2 clause at the CW-PCA definition.
- `sections/03_nfpa.tex` — parameter-count sentence removed (cut item 4).
- `sections/04_experiments_discussion.tex` — D4 paragraph (after line 4, before line 6), D6 figure slot comment (after the table), D5 forward reference in Limitations.
- `tables/results_summary.tex` — D7 full-precision raw-MSE values.
- `references.bib` — D3 entries `chen2020simclr`, `he2022masked`.
- `appendix/appendix_experiment_details.tex` — D5 protocol subsection (`app:protocol`), D7 appendix note.
- `notes/claim_audit.md` — D4 claim row.
- `tmp/re_export_paper2_mse.py` — D7 provenance re-export script (new).

Rebuilt artifact: `paper.pdf` (9 pages) in the paper directory.

---

## 6. Points requiring escalation / author action

1. **Anonymized repository link (D5 fix 1)** — needs author action; `gh` unavailable. Add footnote once created and verified.
2. **Abstract D4 clause** — skipped deliberately. Plan §D4 says "only if the page budget allows"; it does not (see §3). The experiment agent also owns the abstract's evidence sentence, so this avoids a merge collision.
3. **`≤ 4` page gate** — not independently satisfiable during the desk phase; resolution is the experiment agent's §4 rewrite plus the recorded cut-order fallback (§3). The desk build itself is clean.
4. **Section 4 line 21 wording** ("IsoAE attains raw MSE below $10^{-5}$") remains true but less precise than the re-exported table values; it sits in the experiment agent's lines 6–13, so the experiment agent should align it with the new table (or the desk agent can on request).
