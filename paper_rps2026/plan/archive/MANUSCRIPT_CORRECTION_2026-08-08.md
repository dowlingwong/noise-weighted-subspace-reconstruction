# Manuscript correction plan — review of the restructure implementation

**Date:** 2026-08-08
**Reviews:** `plan/MANUSCRIPT_IMPL_LOG.md` and the current build (`paper.pdf`, 9 pages, main content 5)
**For:** the manuscript agent. Execute in order. Do not start new work outside this list.

---

## 0. Assessment

The implementation is good. All seven phases ran, the build is clean (0 overfull boxes, 0 undefined citations, 0 warnings), the ladder structure is in place, both key sentences were written, the tuple regression was correctly restored, **no number was fabricated**, and the page overflow was escalated rather than resolved by inventing cuts. The log's Phase 7 analysis is honest and correct.

Six defects remain. **Two of them are errors in the plans I wrote, not agent errors** — C1 and C6 below. The agent could not have resolved them.

Fix in the order given. C1–C3 are blocking.

---

## C1 — [BLOCKING, plan error] The representation is claimed to be shared and it is not

`sections/03_architecture_ladder.tex`, opening paragraph:

> "All four rungs share the representation used in Section~4: the short-time Fourier transform (STFT) of the readout channel as a band--frame plane $Z_i\in\C^{F\times M}$…"

**This is false as the evidence currently stands.** §4's linear metric-reversal result is 512 synthetic traces of dimension 256. §4's NFPA result is a synthetic `C=8` channel × `T=256` time configuration. Only rung 4 is on the STFT band–frame plane. A referee who reads §3 and then §4 will find the mismatch immediately.

The cause is mine: `PLAN_00` §1 renamed rung 3 to "band × frame" when the dataset changed, but the NFPA experiment (Block A) is still synthetic channel × time. The plan created an inconsistency the agent faithfully implemented.

**Fix — make the factorization generic and name its two instantiations.**

1. In §3, define NFPA over two generic axes rather than band/frame specifically. Use neutral names — `U_1\in\R^{n_1\times k_1}` and `U_2\in\R^{n_2\times k_2}`, or keep `U_c, U_t` and call them "the two structured axes". State that the Kronecker restriction is a statement about *any* two-axis measurement, and that this paper instantiates it twice: **sensor channel × time** in the controlled synthetic study, and **frequency band × time frame** on the public benchmark.
2. Replace the false "all four rungs share" sentence with the true one: rungs 1–3 are analysed in general two-axis form and evidenced synthetically; rung 4 is evidenced on the STFT band–frame plane, where `Σ⁻¹` is diagonal by measurement.
3. Keep the χ²/dof = 1.0002 justification, but attach it to **rung 4's instantiation**, not to all rungs.

This is a genuine improvement, not a concession: the proposition is stronger stated over generic axes, and it makes the two instantiations a feature.

---

## C2 — [BLOCKING] The abstract asserts results that do not exist

`sections/00_abstract.tex` contains no `\PENDING` while §4 and §5 contain five. It currently states:

> "…and **show** the metric question recurs at every rung…"
> "The first three rungs admit closed-form analysis; the fourth **is compared against** published baselines on a public benchmark."

Both assert a completed rung-4 comparison. The rung-4 arms have not been trained. An abstract that claims unproduced results is the exact failure mode the placeholder discipline exists to prevent — it simply escaped the grep because the grep covered `sections/`, `tables/`, `appendix/` and the abstract's claim is prose, not a number.

**Fix.** Rewrite the two clauses so the rung-4 status is accurate at submission time and carries a `\PENDING` for the verdict. The first three rungs may be stated as established. Do not write "we show" for rung 4 until the numbers exist.

---

## C3 — [BLOCKING] `references.bib` contains a stub that renders in the bibliography

```bibtex
@mastersthesis{priorThesis,
  author = {PENDING, Author},
  title  = {PENDING: unpublished MSc thesis on ...},
  school = {PENDING: institution},
  year   = {n.d.},
```

This renders in the built PDF as a bibliography entry reading "PENDING, Author." The Phase 6 grep did not catch it because my prompt scoped the sweep to `sections/ appendix/ tables/` and omitted `references.bib`. My omission.

**Fix.**

1. Extend the placeholder sweep to `references.bib` and record the result in the log.
2. Supply real thesis metadata: **Sjostrom, S., *Adapting Transformers for Light Dark Matter Event Reconstruction*, MSci project report, Imperial College London, 2026.** Verify against the source PDF before committing. Note it is an **MSci** report, not MSc — the current `type` field is wrong.
3. **Flag for the authors, do not resolve yourself:** citing this thesis is required by the attribution conditions (`PLAN_03_RUNG4_TIDMAD` §5), but the thesis acknowledges one of this paper's authors by name. Under double-blind review that is a deanonymization vector a diligent reviewer could follow. This is an author decision between competing goods — proper attribution versus blind review — and it must be made explicitly rather than by default. Record it as an open item in the log.

---

## C4 — The paper describes a model that does not exist

`sections/03_architecture_ladder.tex`, rung 4:

> "…with rotary position embeddings on both axes \citep{su2024roformer,nie2023patchtst}."

The implementation applies RoPE on the **temporal axis only**. Cross-channel layers are constructed with `use_rope=False` and use a learned `AbsolutePositionalEmbedding` (`reconstruction_model/model.py:161–166`). Two-axis RoPE was a *suggestion* in my plan, never a fact.

Also `nie2023patchtst` is cited for rotary embeddings; PatchTST is the patched-embedding reference, not the rotary one.

**Fix.** Either (a) describe what the model does — rotary embeddings on the temporal axis, learned positional embeddings across bands — or (b) leave the sentence pending until the code patch (`PLAN_04`) settles it. **(a) is correct today; choose (a) and revisit only if the port changes it.** Move `nie2023patchtst` to the patched-embedding clause where it belongs.

---

## C5 — Redundancy and a misplaced forward reference in §3

1. The parallel sentence ("NFPA restricts to `range(U_f)⊗range(U_m)`: separable frequency-band and time-frame factors. The factored attention of rung 4 performs…") **repeats verbatim** the definition given four paragraphs earlier in the same section. It is the paper's most important sentence and currently reads as an accidental duplication.
   **Fix:** cut the restatement and open the sentence at the comparison — the two constructions, one claim. Consider moving it to open rung 4 rather than closing rung 3, so it introduces the thing it is about.
2. §3's opening quotes χ²/dof = 1.0002 before §4 introduces TIDMAD. Add a three-word forward pointer, or move the justification to where rung 4 is defined (which C1 makes natural anyway).

---

## C6 — [plan error] The limitations requirement pointed at the wrong section

My prompt's Phase 4 said "all four items from `PLAN_03_RUNG4_TIDMAD` §6." That plan's §6 is *Scope and schedule*; the four-item limitations list existed in the superseded DELight document and was lost in the rewrite. The agent flagged this correctly in log §7.1 and reconstructed a sensible four-item list.

**The reconstruction is accepted.** The four caveats now in §5 — paired contrast not leaderboard rank, null-contrast-is-a-finding, TIDMAD rung-2 non-replication, per-instrument PSD adequacy — are the right ones.

**Fix:** none in the manuscript. I am adding a Limitations section to `PLAN_03_RUNG4_TIDMAD` so the reference resolves for anyone reading later. **Add one item the reconstruction missed:** rung 4's evidence will be **simulation-free but single-run** — no training-seed variance — which must be stated once the arms exist.

---

## C7 — Page budget: the decision the agent correctly escalated

Main content is 5 pages against a 4-page limit. The agent applied every sanctioned cut and stopped rather than inventing more. That was right.

**Decision, made here so the agent is unblocked:**

1. **Move `tables/rung4_baselines.tex` to the appendix.** It is the single largest float in the main text, and its content is five published numbers plus two pending cells. In the main text, replace it with one sentence: the published baselines span roughly −2 to 6.6 on the release's coarse score, the archived linear estimator sits below the no-denoising anchor, and the two rung-4 arms are `\PENDING`. Cross-reference the appendix table.
2. Re-run the budget pass after that. If still over, the next cut is §1's third paragraph, which currently spends four sentences on Noise2Self / SSDU / Noisier2Inverse; two will do.
3. **Do not cut** the tuple, either proposition, metric reversal, the parallel sentence, or the rung-4 metric contrast.
4. Record the estimated recovery when placeholders resolve (log §7.1 estimates 0.15–0.2 page) and the 0.3 page the rung-4 figure will consume. **Net: the figure roughly cancels the placeholder recovery**, so the table move is doing the real work and should not be skipped on the assumption that numbers will save you later.

---

## C8 — Minor, fix while you are in the files

- `appendix/appendix_experiment_details.tex` mixes notations: the first paragraphs use `(k_c,k_t)` and `C`, the last uses `k_f, F, k_m, M`. After C1 makes §3 generic, make the appendix consistent — synthetic study in channel/time, rung 4 in band/frame, with the generic form referenced once.
- The `\C` macro is defined in `paper.tex` but becomes unused if C1 removes complex-valued notation from §3. Remove it if so; keep the preamble honest.

---

## Execution order and gates

| # | Item | Gate |
|---|---|---|
| 1 | C3 — bib stub, extended grep, flag the anonymity conflict | No `PENDING` string in `references.bib`; anonymity conflict logged as an open author decision |
| 2 | C2 — abstract accuracy | Abstract asserts nothing about rung 4 that is not archived |
| 3 | C1 — generic two-axis NFPA; fix the false shared-representation claim | §3 and §4 describe the same method in reconcilable terms |
| 4 | C4 — describe the model that exists | Rung 4's architecture sentence matches `reconstruction_model/model.py` |
| 5 | C5, C8 — redundancy, forward reference, notation | `make` clean |
| 6 | C7 — table to appendix, re-run budget pass | Main content ≤ 4 pages, bibliography starts on page 5 |

**Do not fabricate any number.** Everything unproduced stays `\PENDING`. If C7 cannot reach four pages without cutting protected content, stop and report rather than cutting.

Append a "Correction round" section to `plan/MANUSCRIPT_IMPL_LOG.md` covering each item, the final page count, the full placeholder list including `references.bib`, and the anonymity decision as an explicit open item.
