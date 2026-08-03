# Publication portfolio — five papers

**Compiled 2026-08-03.** Scores are my inferred judgment of *quality and value if
completed*, deliberately ignoring current completeness — a strong idea with no
runs scores high, a finished paper with a thin claim scores low. They are not
venue acceptance probabilities. Where your own
`PUBLICATION_PLAN_EVALUATION.md` gives a probability, I quote it rather than
substitute my own.

Deadlines verified 2026-08-03. Journal venues (MLST, JINST, TMLR, DMLR) are
rolling; only the conference and workshop rows carry dates.

---

## Master table

| # | Paper | Main topic | Main contribution | Score | Type | Nearest deadline | Three publication pipes | Canonical reference |
|---|---|---|---|---|---|---|---|---|
| **1** | **Noise Covariance Defines Reconstruction Geometry** | Likelihood geometry of linear detector reconstruction under colored/correlated noise | Σ⁻¹ *is* the reconstruction metric. OF, EMPCA and the tied noise-aware linear AE are the rank-1, rank-k and reparameterized cases of one projection. Bridge Theorem verified numerically. Four public datasets with predeclared gates, and negatives reported as negatives | **8.5 / 10** | Journal (+ preprint) | Rolling. Workshop option: **NeurIPS 2026 workshops ≈ 29 Aug 2026** | ① arXiv now → MLST full paper ② JINST (detector-algorithm framing) ③ NeurIPS 2026 *Sim2Science* / AI4Science workshop for the metric-reversal result | `arXiv-v1-polished-extended-preprint/main.pdf` (58 pp, clean). Plan: `SUBMISSION.md`, `ARXIV_CHECKLIST.md` |
| **2** | **Architecture-Constrained Likelihood Geometry** (NFPA) | Which architecture priors are justified once the metric is fixed | The (Σ⁻¹, F, P_train→P_target) tuple; NFPA as a Kronecker-restricted projector class; rank-one NFPA ≡ OF under a separable whitened template | **5.5 / 10** as drafted · **7.5** if the approximation bound is proven and the N-sweep is run | Journal | Rolling | ① MLST (companion to Paper 1) ② TMLR (rolling, method framing) ③ ICLR 2027 if the bias–variance crossover lands — **abstract 18 Sep 2026, paper 25 Sep 2026 AOE** | `paper2-nonlinear-reconstruction/main.pdf` (63 pp). Plan: `paper2/PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md` |
| **3** | **Consequence-Aware Failure Diagnostics** | Can frozen representations separate acquisition corruption from unsupported physics, and do alarms track scientific loss | Declared N/S/E intervention families; abstention with selective risk; the detectability-versus-consequence axis that makes the likelihood geometry load-bearing | **7.0 / 10** as a design · results TBD | Concept note → journal or workshop | Rolling; workshop **≈ 29 Aug 2026** | ① MLST if the cross-domain monitoring method is central ② JINST if the point-cloud detector case dominates ③ NeurIPS/ICML AI4Science workshop as a de-risking pilot | `paper3/latex/paper3_proposal.pdf` (3 pp). Plan: `paper3/PAPER3_PLAN.md` |
| **4** | **Failure Dynamics in Scientific Agents** (R1) | Causal benchmark for onset, propagation and recoverability of long-horizon agent failure | Governed benchmark: seven fault-mechanism families, four workload levels, deterministic injectors with clean counterfactuals, independent adjudication separating execution correctness from scientific validity | **6.5 / 10** — value hinges on whether one non-obvious failure law survives model/task/scaffold changes | Journal (artifact) | Rolling. Alternative: **KDD Datasets & Benchmarks Cycle 2** | ① DMLR (primary) ② KDD D&B Cycle 2 (deadline-based alternative) ③ TMLR if the transferable failure law dominates the artifact | `agent_sys_paper1_2/latex/r1_dmlr/main.tex`. Plan: `WRITING_AND_SUBMISSION_PLAN.md` §R1, `PUBLICATION_PLAN_EVALUATION.md` |
| **5** | **Learning When to Verify and How to Recover** (R2) | Cost-sensitive process supervision: when to buy evidence, which recovery is safe | Sequential decision framework over partially observed trajectory state; action space spanning verify/continue/recover/escalate/abstain/terminate with adjudicated postconditions; randomized matched-budget policy comparison | **5.5 / 10** — algorithmically narrow, and depends on R1's frozen artifact | Journal | Rolling | ① TMLR (primary, rolling) ② COLM or main ML track if evidence is strong ③ Reliable-agents workshop to validate the action space and cost model | `agent_sys_paper1_2/latex/r2_tmlr/main.tex`. Plan: `WRITING_AND_SUBMISSION_PLAN.md` §R2 |

---

## Your own acceptance estimates, for comparison

From `PUBLICATION_PLAN_EVALUATION.md` (dated 2026-08-01) — these are archival
acceptance ranges conditional on completing the priority revisions:

| Project | Realization | Archival acceptance | Workshop acceptance | Verdict recorded |
|---|---|---|---|---|
| AI R1 (#4) | 70% | 30–45% | 55–75% | Advance after substrate freeze |
| AI R2 (#5) | 45% | 25–40% | 45–65% | Stage behind R1 |

Note the asymmetry these numbers imply: for the agent papers, workshop
acceptance is roughly twice archival. That argues for a workshop pilot first on
#4/#5, and against one for #1, which is already strong enough to go archival
directly.

---

## Verified deadlines

| Venue | Type | Date | Note |
|---|---|---|---|
| MLST | Journal | rolling | Scope explicitly covers physics-motivated ML, robustness, architectures, benchmark studies, particle physics |
| JINST | Journal | rolling | Explicitly covers detector algorithms, software, data-reduction methods |
| TMLR | Journal | rolling | Flexible length; guidance warns against unnecessarily long submissions |
| DMLR | Journal | rolling | Requires reviewer access, licensing, maintenance plan, reproducibility materials, broader impact |
| NeurIPS 2026 workshops | Workshop | **≈ 29 Aug 2026** | Confirmed for AI for Meta-Science; Climate Change AI uses 29 Aug (80%) + 15 Oct (100%). Conference 6–12 Dec 2026, Sydney. **Check each workshop's own page — dates vary** |
| ICLR 2027 | Conference | **abstract 18 Sep 2026 · paper 25 Sep 2026 AOE** | |
| KDD D&B Cycle 2 | Conference | see current call | Named in your plan as the R1 alternative; verify the 2027 cycle date |

**26 days to the NeurIPS workshop window.** That is the only near-term
deadline, and *Sim2Science: ML with Imperfect Scientific Models* is an unusually
precise fit for Paper 1's core result — a loss function encoding a noise model,
and the benchmark ranking inverting when that model is wrong.

---

## Reading of the portfolio

**Paper 1 is the asset.** It is the only one with verified theory, four public
datasets, and a real-data confirmation whose interval excludes zero. It also
carries the evidence discipline the other four can inherit. Post it.

**Papers 2 and 3 are one strong paper each, currently under-powered in
different ways.** Paper 2 has the machinery but not the theorem or the sweep;
Paper 3 has the design but no results. Both are cheap to fix relative to their
value.

**Papers 4 and 5 are a different research program.** They share no technical
object with 1–3 — different data, different claims, different reviewers. Worth
being deliberate about that: five papers across two disconnected programs is a
lot of simultaneous front, and your own evaluation already stages R2 behind R1.

**The sequencing I would choose:** Paper 1 to arXiv this month and to a NeurIPS
workshop before 29 Aug; Paper 3's concept note to Junjie now, since it has the
longest external feedback latency; Paper 2's approximation bound and N-sweep
next, targeting ICLR 2027 in late September if the crossover appears, MLST
otherwise; Papers 4–5 after the substrate freeze, R1 first.

---

## Caveats

- Scores are mine and are judgments, not measurements. Treat them as a ranking
  device, not a metric.
- Workshop deadlines move and vary per workshop. The 29 Aug figure is confirmed
  for one NeurIPS 2026 workshop and used by another; verify on each workshop's
  own call before relying on it.
- KDD Datasets & Benchmarks Cycle 2 is carried from your own plan; I did not
  verify its 2027 date.
- Papers 4 and 5 are judged from abstracts, section structure, and your
  publication-plan evaluation. I have not read their methods sections in the
  detail I read Papers 1–3.
