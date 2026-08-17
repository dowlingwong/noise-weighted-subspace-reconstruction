# Master restructure plan — the architecture ladder

**Date:** 2026-08-08
**Status:** governing document. Supersedes the *scope* of the two plans dated the same day; their task content remains valid.
**Deadline:** 29 Aug 2026 AoE (= 30 Aug 2026 11:59 UTC)

---

## 1. The decision

The paper is being restructured around a single organizing claim, already present as Eq. (1) but currently underused:

> A reconstruction method is a pair `(Σ⁻¹, F)`. The measurement likelihood fixes the residual geometry `Σ⁻¹`. The architecture fixes the admissible representation class `F`.

The current draft holds `F` almost fixed and varies little. The restructured paper **holds `Σ⁻¹` fixed as the organizing principle and walks `F` up a ladder of four architecture classes**, showing that the metric question recurs at every rung and that the answer does not become irrelevant as the architecture becomes expressive.

| Rung | Architecture class `F` | Status |
|---|---|---|
| 1 | rank-one fixed template — optimal filter | classical, exact |
| 2 | unrestricted rank-`k` linear — CW-PCA / tied linear AE | Proposition 1 |
| 3 | Kronecker-separable band × frame — NFPA | Proposition 2, Corollary 1 |
| 4 | factored attention — intra-band then cross-band transformer | empirical, **TIDMAD** |

> **Amendment, 2026-08-08.** Rung 4 was originally planned on DELight simulation data.
> That is void — see `PLAN_03_DELIGHT_INTEGRATION_2026-08-08.md`. Rung 4 now runs on
> **TIDMAD** (public, CC-BY-4.0), where Paper 1 already holds archived results, a
> reproduced calibration anchor, and where published deep baselines exist for
> comparison. **No DELight data, simulation, physics, or checkpoint appears anywhere
> in this paper.** The scientific claim is unchanged and the evidence is stronger.
>
> Consequence for the ladder: all four rungs now live on **one public dataset**, in
> **one representation** — the STFT band × frame plane. That is a materially more
> coherent paper than four rungs across four datasets.

### Why this is the right move

- **It answers the paper's single biggest weakness.** A referee reading the current draft can say "everything here is linear, so this is classical statistics relabelled." The ladder makes the linearity a *choice about tractability* rather than a limitation, because rung 4 shows the same phenomenon nonlinearly at scale.
- **Rungs 3 and 4 share an inductive bias.** NFPA restricts to `range(U_f) ⊗ range(U_m)` — separable frequency-band and time-frame factors. The transformer performs intra-band temporal attention followed by cross-band attention. These are the same factorization at different levels of nonlinearity, in the same coordinates. **State this parallel explicitly — it is the paper's most valuable single sentence.**
- **The metric and the factorization now live in the same basis.** `Σ⁻¹` is diagonal in frequency — justified on TIDMAD by measurement, not assumption (Paper 1 Phase 4: `χ²/dof = 1.0002`, all gates passed) — and the first factor indexes exactly that basis. This is a tighter instantiation than a sensor axis would have been.
- **It rescues NFPA rather than demoting it.** NFPA becomes the rung where the restricted-bridge proposition is provable. It no longer has to carry the empirical weight of the paper alone, which its current single-seed evidence cannot do.
- **Paper 1 already posed the question rung 4 answers.** Its archived Phase 5 record states the boundary directly: *"This measures how much of the gap between no denoising and a trained deep denoiser is explained by using the correct noise geometry with a linear model. It is not a claim to win the benchmark."* The linear estimator scored −1.8025 against a published Transformer at 4.18. Rung 4 asks whether the metric still matters once the architecture can compete. **Motivating an experiment from your own archived diagnostics is the strongest available framing.**

---

## 2. New paper skeleton and page budget

Four pages of main content. References and appendices excluded.

| § | Content | Budget |
|---|---|---:|
| 1 | Introduction. Motivation; the `(Σ⁻¹, F)` tuple as the organizing idea; compressed related work (tasks D1–D3); contributions stated as the ladder. | 0.85 |
| 2 | Covariance-weighted representation geometry. Gaussian objective; whitening normal form; **Proposition 1** (OF / CW-PCA / tied linear AE bridge); metric reversal. | 0.70 |
| 3 | The architecture ladder. `F` as the varying element across four rungs. **Proposition 2** (Kronecker-restricted bridge). The NFPA / factored-attention parallel. | 0.90 |
| 4 | Evidence. (a) linear metric reversal; (b) NFPA consolidated; (c) DELight three-way ablation, energy crossover, channel masking; (d) covariance adequacy on real data (GWOSC). | 1.20 |
| 5 | Discussion, limitations, takeaway. | 0.35 |
| | **Total** | **4.00** |

### Required cuts to fit

Cut in this order — this **replaces** the order in `notes/cut_list.md`, which predates the restructure:

1. **Corollary 1 (rank-one correspondence) → appendix.** Keep one clause in §3 noting the degenerate case recovers OF, with a `\ref{}`.
2. **The parameter-count sentence** in §3 (`O(k_c C + k_t T)` vs `O(k_c k_t CT)`) → appendix.
3. **The `C = k_c = 2` proof-of-concept** → appendix as an implementation check (this was already planned).
4. **Compress the TIDMAD single-file result** to one sentence. It is now the third-strongest real-data item, behind DELight and GWOSC.
5. If still over: **compress §2's centering and covariance-estimation contract** to one sentence plus an appendix pointer.

**Do not cut:** the `(Σ⁻¹, F)` tuple (now load-bearing), Proposition 1, metric reversal, Proposition 2, the NFPA/attention parallel, or the DELight fixed-architecture ablation.

### Title

The current title, *"Likelihood-Aware Self-Supervision: Noise Covariance as Representation Geometry"*, becomes partly inaccurate once the headline evidence is supervised downstream tasks (see `PLAN_03` §4). Replace. Candidates, in preference order:

1. **"Noise Covariance as Representation Geometry: One Metric, Four Architecture Classes"**
2. "The Measurement Likelihood Fixes the Geometry; the Architecture Fixes the Representation"
3. "Likelihood Geometry for Scientific Representations, from Optimal Filters to Attention"

Do not keep "Self-Supervision" in the title unless §4's headline evidence is label-free. It is not.

---

## 3. Track routing

Three parallel tracks. **Ownership is exclusive — do not edit outside your set.**

| Track | Plan document | Owns |
|---|---|---|
| **A — Desk** | `DESK_REVISION_PLAN_2026-08-08.md` | §1 introduction, §2, §5, `references.bib`, the GWOSC paragraph in §4 |
| **B — Experiments** | `EXPERIMENT_PLAN_2026-08-08.md` (rescoped, see §4 below) | NFPA rung of §3, the NFPA paragraph of §4, `tables/`, the metric-reversal figure |
| **C — Rung 4** | `PLAN_03_RUNG4_TIDMAD_2026-08-08.md` | rung 4 of §3, the TIDMAD paragraph and table of §4, the STFT data path |

**Track C is now the critical path.** It carries the paper's headline evidence and owns the STFT representation that rungs 2–4 share. Prioritise it over Track B when they compete for time.

**Shared files requiring coordination:** `sections/03_nfpa.tex` becomes `sections/03_architecture_ladder.tex` and is co-owned by B and C. **Track C creates the new file and the rung-4 content; Track B ports the NFPA content into it.** Track C goes first, because the section's structure depends on the ladder framing.

`sections/00_abstract.tex` is written **last**, by whoever finishes, and must be reviewed by all three. It currently claims "A controlled, single-split proof-of-concept illustrates metric reversal" — that sentence dies in every scenario.

---

## 4. Scope change to the experiment track

**Decision: reduce the NFPA sprint.** With rung 4 supplying the paper's headline empirical evidence, NFPA no longer has to carry it. The consolidation still matters — the `C = k_c = 2` degeneracy is a real defect and a referee will find it — but it no longer justifies a three-week sprint.

**Revised scope — cut harder than the first amendment.**

- **Block A at 10 seeds**, primary configuration `C = 8 > k_c = 2` unchanged. The referee-facing defect is the `C = k_c = 2` degeneracy, not the seed count. Ten paired seeds with an interval is enough to retire it.
- **Block B (white-noise control) — unchanged and non-negotiable.** Small, cheap, evidentially disproportionate.
- **Block H dropped entirely.** Truth-referenced recovery is now covered by rung 4's amplitude error against TIDMAD's injected reference channel, which is real data rather than synthetic latents. Block H would be the weaker version of a claim rung 4 makes better.
- **Phase 0 audit unchanged.** Do not skip the correctness tests. The degenerate-case reproduction check is what tells you the codebase is trustworthy.

**Freed time: roughly two weeks. Reallocate to Track C**, which is now the critical path.

### Stretch goal — NFPA on the TIDMAD spectrogram

Once Track C's STFT data path exists, running NFPA and CW-PCA on the same whitened spectrograms is cheap: alternating least squares plus an eigendecomposition, no neural training. If it lands, **rungs 2, 3 and 4 all sit on one public dataset in one representation**, which is the difference between a coherent paper and a very coherent one.

Conditions: attempt only after Track C's data path is verified and Block A/B are complete; it must not delay rung 4. Treat the synthetic Block A as the primary NFPA evidence regardless — it is the controlled setting where the restricted-bridge proposition can be checked against known planted factors.

---

## 5. Supersession table

| Document | Status |
|---|---|
| `PLAN_00_MASTER_RESTRUCTURE_2026-08-08.md` | **Governing.** |
| `PLAN_03_RUNG4_TIDMAD_2026-08-08.md` | **Active. Track C, critical path.** |
| `PLAN_03_DELIGHT_INTEGRATION_2026-08-08.md` | **VOID.** No DELight material in this paper. |
| `DESK_REVISION_PLAN_2026-08-08.md` | Active. Tasks D1–D7 all remain valid. §0.3 interface note is superseded by §3 above. |
| `EXPERIMENT_PLAN_2026-08-08.md` | Active with reduced scope per §4 above. Its own in-file amendment banner says 20 seeds and Block H reduced; **§4 above supersedes it again — 10 seeds, Block H dropped.** Protocol definitions unchanged. |
| `notes/nfpa_submission_experiment_plan.md` | Authoritative for *definitions*. Its scope is superseded. |
| `notes/cut_list.md` | **Superseded** by §2 above. Its Paper 3 exclusion is already reversed by task D4. |
| `notes/claim_audit.md` | **Still authoritative for claims.** Must be updated by every track, not bypassed. |
| `RPS_MINIPAPER_HANDOFF.md` | Stale. Rewrite at the end. |

---

## 6. Schedule

| Dates | Track A (desk) | Track B (NFPA) | Track C (rung 4 — critical path) |
|---|---|---|---|
| 8–9 Aug | D1, D2 | Phase 0 audit | Reconstruction head; smoke test |
| 10–12 Aug | D5, D3 | `C > k_c` harness | TIDMAD STFT dataset; reproduce `none` anchor at 1.1026 |
| 13–14 Aug | D4, D6 | Freeze config; Blocks A (10 seeds) + B | Weighted loss; config-equality assertion; **freeze config 14 Aug** |
| 15–19 Aug | Reserve — assist C | Results audit; `sections/03` rung 3 | **Train `T-mse` and `T-chi2`; score through unmodified script** |
| 18 Aug | — | — | **Checkpoint: if not training by now, escalate** (see `PLAN_03_RUNG4_TIDMAD` §6) |
| 20–21 Aug | Page-budget pass | Stretch: NFPA on TIDMAD spectrogram | `T-chi2+OF` if on schedule; paired statistics; figure |
| 22–23 Aug | — | Table | Draft rung 4 and §4 evidence paragraph |
| 24–25 Aug | Abstract, title, final build | — | Verify every number against archived output |
| 25 Aug | **Freeze** | **Freeze** | **Freeze** |
| 26–29 Aug | Review, corrections, submit | | |

**No permissions gate remains.** TIDMAD is CC-BY-4.0 and Paper 1 has already run its benchmark script. The only external dependency is attribution wording (`PLAN_03_RUNG4_TIDMAD` §5), which does not block work.

---

## 7. Definition of done

- `make` builds cleanly; main content ≤ 4 pages; bibliography starts on page 5.
- Every claim in the PDF appears in an updated `notes/claim_audit.md` with its status and controls.
- **No DELight data, simulation, physics, figure, number, or checkpoint appears anywhere** — including in any published or anonymized repository. Verify with `git log --all --name-only` before publishing anything.
- Every rung-4 number was produced by TIDMAD's unmodified `benchmark.py` or by an archived run file, and is traceable to the command that produced it.
- No number anywhere was transcribed by hand from a source that is not archived.
- The three attribution conditions in `PLAN_03_RUNG4_TIDMAD` §5 are all satisfied in the text.
- Acknowledgements are correct for the camera-ready — and **absent** from the anonymous submission.
