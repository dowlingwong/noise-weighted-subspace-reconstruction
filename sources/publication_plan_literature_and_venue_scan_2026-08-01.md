# Publication-plan literature and venue scan

## 📍 Scope and date

This lookup was performed on 1 August 2026 to support the comparative evaluation in `agent_sys_paper1_2/PUBLICATION_PLAN_EVALUATION.md`. It is a targeted collision and venue-fit scan, not a systematic literature review. Search emphasis was placed on primary papers, proceedings pages, and official venue pages.

## 🔬 Detector reconstruction and representation diagnostics

- Weighted and noisy subspace estimation is established terrain. Relevant anchors include heteroskedastic/noisy PCA, weighted PCA, multilinear PCA, Tucker decomposition, and tensor factor models. The closest interpretation of NFPA is therefore a likelihood-aware or whitened Tucker-2/product-subspace model; novelty must come from the detector-likelihood formulation, a nontrivial approximation theorem, or empirical consequences rather than the existence of a Kronecker factorization itself.[^behne2022] [^mpca2008]
- Internal-layer anomaly and OOD detection is also established. A prior ICML framework explicitly aggregates internal representations for anomalous-input detection, and newer work studies layerwise attribution for OOD detection.[^raghuram2021] [^splitting2025]
- Recent work argues that classifier uncertainty and feature distance can answer the wrong question for OOD detection. This supports Paper 3's decision to distinguish acquisition/noise shift from under-supported but physically valid signal variation and to score downstream consequence separately.[^wrongquestion2025]
- Recent detector-reconstruction work highlights robustness to process-dependent event-topology changes, confirming that topology/composition shift is an active collision axis for Paper 3.[^detectoraware2026]

## 🤖 Scientific-agent reliability and post-training

- Paper A1 faces direct collisions from MAST-Data, AgentRx, Aegis, AgentBoard, and newer long-horizon/security benchmarks. MAST-Data already contains more than 1,600 annotated multi-agent traces and a failure taxonomy; AgentRx provides critical-step localization and root-cause categories for 115 failed trajectories; Aegis pairs a failure taxonomy with environment interventions.[^mast2025] [^agentrx2026] [^aegis2025]
- Scientific-agent benchmarking is also moving quickly. MLR-Bench evaluates open-ended ML research agents, and SciAgentArena reports roughly 200 interactive scientific tasks with stepwise verification and failure characterization.[^mlrbench2025] [^sciagentarena2026]
- These collisions mean Paper A1 cannot claim novelty from a taxonomy, trace dataset, or scientific workload alone. A defensible gap is the predeclared measurement of failure *dynamics*: survival/hazard, propagation, recoverability, calibration, and fault-composition generalization under governed execution.
- Paper A2 also overlaps AgentRx's trajectory diagnosis and Aegis's intervention logic. Its strongest differentiator is a calibrated, cost-sensitive active-verification policy evaluated on unseen faults, not a larger module inventory.[^agentrx2026] [^aegis2025]
- Paper A3 enters an active post-training field that already includes execution-feedback RL, trajectory-level learning, self-verification, agent RL objectives, and verifier-guided repair. CAPF is especially close because it turns failed, zero-reward attempts into verifier-guided repair trajectories; Afterburner compares SFT, DPO, and GRPO under execution feedback.[^capf2026] [^afterburner2025]
- A3 therefore needs a paper-pointed differentiator: learning *calibrated detection, selective verification, and recovery policies* from independently adjudicated scientific-agent traces, with explicit regression and reward-channel audits.

## 🎯 Venue evidence

- JINST explicitly covers detector methods, algorithms, software, and data-reduction methods, making it a strong detector-community venue for Papers 1 and 3 and a plausible home for a narrowed Paper 2.[^jinst]
- Machine Learning: Science and Technology accepts physics-motivated methodological advances, robustness, architectures, benchmark studies, and particle-physics applications. It is the strongest cross-disciplinary journal fit for Papers 2 and 3 and a plausible alternative for Paper 1.[^mlst]
- IEEE Transactions on Signal Processing expects original and significant theory, algorithms, performance analysis, and signal-processing applications. Paper 1 would need sharper novelty positioning and a tighter signal-processing presentation to compete there.[^tsp]
- NeurIPS 2026 renamed its Datasets & Benchmarks track to Evaluations & Datasets and explicitly welcomes evaluation methodology, audits, negative results, data generators, and executable benchmark artifacts. This is the best archival-track fit for a mature Paper A1.[^neuripsed]
- COLM explicitly covers post-training and agent-relevant language-model research. It is a natural archival target for a mature A3 and a possible target for a narrowed A2.[^colm]
- The 2026 Lifelong Agents workshop at COLM welcomed agent post-training, agentic RL, self-evolving agents, and agents for science. The exact workshop may not recur in 2027, so the report treats it as a venue family rather than a guaranteed future call.[^lifelong]

## ⚠️ Interpretation limits

The acceptance ranges in the evaluation report are judgmental planning ranges conditional on completing the stated revisions. They are not inferred from historical venue acceptance rates, and they should be updated when the actual submission cycle, call for papers, author team, and completed evidence are known.

[^behne2022]: J. K. Behne and G. Reeves, “Fundamental limits for rank-one matrix estimation with groupwise heteroskedasticity,” AISTATS 2022. https://proceedings.mlr.press/v151/behne22a.html
[^mpca2008]: H. Lu, K. N. Plataniotis, and A. N. Venetsanopoulos, “MPCA: Multilinear principal component analysis of tensor objects,” IEEE Transactions on Neural Networks 19(1), 2008. Bibliographic anchor as cited in Paper 2; overview available through https://ieeexplore.ieee.org/document/4358916
[^raghuram2021]: J. Raghuram et al., “A General Framework for Detecting Anomalous Inputs to DNN Classifiers,” ICML 2021. https://proceedings.mlr.press/v139/raghuram21a.html
[^splitting2025]: J. Zhang et al., “Splitting & Integrating: Out-of-Distribution Detection via Adversarial Gradient Attribution,” ICML 2025. https://proceedings.mlr.press/v267/zhang25by.html
[^wrongquestion2025]: Y. L. Li et al., “Position: Supervised Classifiers Answer the Wrong Questions for OOD Detection,” ICML 2025. https://proceedings.mlr.press/v267/li25ec.html
[^detectoraware2026]: K. Schäuble et al., “Detector-aware target definitions for full-event particle reconstruction,” arXiv:2606.30485. https://arxiv.org/abs/2606.30485
[^mast2025]: M. Cemri et al., “Why Do Multi-Agent LLM Systems Fail?” NeurIPS 2025 Evaluations and Datasets. https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html
[^agentrx2026]: S. Barke et al., “AgentRx: Diagnosing AI Agent Failures from Execution Trajectories,” arXiv:2602.02475. https://arxiv.org/abs/2602.02475
[^aegis2025]: K. Song et al., “Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures in LLM Agents,” arXiv:2508.19504. https://arxiv.org/abs/2508.19504
[^mlrbench2025]: “MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research,” NeurIPS 2025 Evaluations and Datasets. https://proceedings.neurips.cc/paper_files/paper/2025/hash/ab8dd000d6f87f40061a73f8bca7fae4-Abstract-Datasets_and_Benchmarks_Track.html
[^sciagentarena2026]: T. Liu et al., “Benchmarking AI Agents for Addressing Scientific Challenges Across Scales,” arXiv:2606.12736. https://arxiv.org/abs/2606.12736
[^capf2026]: B. Chen et al., “CAPF: Guiding Search-Agent Rollouts with Credit-Attenuated Privileged Feedback,” arXiv:2606.01830. https://arxiv.org/abs/2606.01830
[^afterburner2025]: M. Du et al., “Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization,” NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/0d1d415fb1a14ba3bb56b831d1eecbe7-Abstract-Conference.html
[^jinst]: Journal of Instrumentation, “About JINST: Scope and editorial policy.” https://jinst.sissa.it/jinst/help/JINST/JINST_about.jsp
[^mlst]: IOP Publishing, “About Machine Learning: Science and Technology.” https://publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/about-machine-learning-science-technology/
[^tsp]: IEEE Signal Processing Society, “IEEE Transactions on Signal Processing: Scope.” https://signalprocessingsociety.org/publications-resources/ieee-transactions-signal-processing
[^neuripsed]: NeurIPS 2026, “Call for Evaluations & Datasets.” https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets
[^colm]: COLM 2026, “Call for Papers.” https://colmweb.org/cfp.html
[^lifelong]: COLM 2026 Workshop on Lifelong Agents, “Call for Papers.” https://lifelongagent.github.io/
