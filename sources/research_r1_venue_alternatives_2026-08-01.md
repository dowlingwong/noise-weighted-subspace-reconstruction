# R1 venue alternatives after NeurIPS E&D 2026

_Checked 2026-08-01 against official venue and workshop pages._

## Rolling archival venues

### Data-centric Machine Learning Research

- DMLR is part of the JMLR family and accepts submissions through OpenReview.
- Its stated scope explicitly includes ML benchmarks, benchmarking tools and
  methods, data generators and RL environments, experimental-design and
  evaluation methodology, and systematic analyses that yield new insight.
- It has no page limit, requires benchmark access, documentation, licensing and
  maintenance information, and expects reproducible code and evaluation
  procedures.
- The stated expected decision time is usually 4–6 months, with exceptional
  cases taking up to 10 months.
- DMLR is open access and states that it charges no author fees.

Source: <https://data.mlr.press/submissions.html>

### Transactions on Machine Learning Research

- TMLR accepts submissions year-round and is continuously published.
- It accepts empirical studies that yield insight into intelligent systems,
  formalizations of learning tasks and evaluation methods, and analyses of the
  behavior of learning systems.
- Its acceptance criteria emphasize whether claims are supported by convincing
  evidence and whether the findings are of interest, rather than requiring a
  new state of the art.
- The FAQ states a target of approximately nine weeks to a final decision,
  without a guarantee; manuscripts over 12 main-body pages may take longer.
- Accepted papers can be considered for a Journal-to-Conference certification
  connected to NeurIPS, ICML, and ICLR.
- TMLR does not accept an archival expanded version of prior published work, so
  a non-archival workshop version or preprint is safer than an archival
  preliminary publication.

Sources:

- <https://www.jmlr.org/tmlr/faq.html>
- <https://www.jmlr.org/tmlr/editorial-policies.html>
- <https://www.jmlr.org/tmlr/acceptance-criteria.html>

## Conference-track alternatives

### KDD 2027 Datasets and Benchmarks

- KDD 2027 has a dedicated Datasets and Benchmarks track.
- Its scope includes new benchmarks, evaluation methodologies and frameworks,
  benchmarking tools, data generators, environments, and responsible dataset
  development.
- The 2027 first-cycle abstract and paper deadlines were July 19 and July 26,
  2026, so they have passed.
- KDD states that it has two submission cycles per year. The Research-track CFP
  explicitly refers to a February 2027 cycle; a separate Datasets and Benchmarks
  Cycle 2 CFP was not yet posted on the checked page. Treat February as the
  planning assumption, not a confirmed D&B deadline.
- The current D&B format is eight content pages plus references and appendix.
  Code and data must become public upon acceptance.

Sources:

- <https://kdd2027.kdd.org/datasets-and-benchmarks-track-call-for-papers/>
- <https://kdd2027.kdd.org/research-track-call-for-papers/>

### ICLR 2027

- ICLR 2027 lists datasets and benchmarks, infrastructure, safety, uncertainty,
  planning, and scientific applications among its subject areas.
- Abstract deadline: September 18, 2026 AoE.
- Paper deadline: September 25, 2026 AoE.
- This is only suitable if the full confirmatory benchmark and principal result
  can be completed without compressing the evidence plan.

Source: <https://iclr.cc/Conferences/2027/CallForPapers>

## Near-term non-archival feedback venues

### Who Verifies the Agents? — NeurIPS 2026 workshop

- Direct fit for environment-grounded verification, runtime observability,
  benchmark/environment design, heterogeneous verification signals, and
  long-horizon agent evaluation.
- Submission deadline: August 29, 2026 AoE.
- Accepts 4–9 page papers and up-to-4-page demos.
- Explicitly non-archival and permits dual submissions, so it can be used for a
  pilot without consuming the later DMLR, TMLR, or KDD paper.

Source: <https://verify-agents-workshop.github.io/>

### AI for Meta-Science — NeurIPS 2026 workshop

- Fits the scientific-agent evaluation, reproducibility, and governance angle.
- Submission deadline: August 29, 2026 AoE.
- Accepts four-page concise contributions or eight-page substantial/position
  papers. It is a weaker technical match than Who Verifies the Agents? for the
  failure-dynamics benchmark itself.

Source: <https://ai4metascience.org/>

## Planning conclusion

1. Use **DMLR** as the default archival target if the released benchmark,
   trace contract, documentation, and failure-dynamics analysis are the center.
2. Use **TMLR** instead if the main claim is the transferable empirical law and
   calibrated early-warning method, with the benchmark serving as evidence.
3. Submit a compact, non-archival pilot to **Who Verifies the Agents?** only if
   meaningful pilot results can be produced by August 29 without weakening the
   confirmatory design.
4. Keep **KDD 2027 Datasets and Benchmarks Cycle 2** as the conference-track
   option, pending its official second-cycle CFP.
5. Do not force **ICLR 2027** unless the evidence is already nearly complete;
   the September deadline is too close for the current proposal-stage R1.
