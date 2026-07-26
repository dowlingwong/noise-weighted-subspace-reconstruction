# Academic Evaluation of the Experiments and Noise Modules

Date of audit: 2026-07-26  
Repository revision: `34112bf` (`main`)  
Primary sources: [Paper 1](../paper1.pdf), [Paper 2](../paper2_nonlinear_reconstruction.pdf), source code, checked-in configurations, machine-readable results, notebooks, and tests.

## 📌 Executive verdict

Paper 1 has a strong theoretical core and a substantially better experimental foundation than a typical methods manuscript. The OF–EMPCA–tied-linear-AE equivalence is implemented coherently, the synthetic gates test the right mathematical invariants, CRESST supplies a positive public-data result, and the failed GWOSC calibration is reported honestly. It is nevertheless **not ready for submission without major revision** because one supplementary result is contradicted by its source data, the manuscript describes the obsolete non-Gaussian Fourier sampler, one metric-reversal sentence overstates the computed quantity, and the primary CRESST interval does not model detector/channel-level sampling.

Paper 2 is currently a good research program and a useful theoretical framework, but not a validated nonlinear-reconstruction paper. Its completed NFPA case study is explicitly a single-seed proof of concept. The newer neural experiments have uncontrolled random initialization, repeated test-set evaluation during training, a broken timing metric, a weak coverage intervention, and a Transformer that has neither positional encoding nor cross-channel attention. Existing neural result artifacts therefore should not be used as paper evidence.

The upgraded noise package is scientifically sound at the level of **controlled stochastic simulation and second-order validation**. Its stationary Gaussian Fourier synthesis, PSD-density normalization, CSD sampling, explicit circulant-versus-Toeplitz distinction, and distributional tests are strong. It is not, by itself, a physical noise-decomposition method. A total PSD or covariance can be diagonalized and decorrelated, but its statistical eigenmodes are not uniquely identifiable physical sources.

| Artifact | Internal assessment | Publication status |
| --- | --- | --- |
| Paper 1 theory | Strong | Defensible after wording and consistency fixes |
| Paper 1 synthetic experiments | Strong implementation validation | Increase repetitions and clarify held-out configuration provenance |
| Paper 1 CRESST | Positive and useful | Add hierarchical sensitivity; retain detection-sensitivity limitation |
| Paper 1 GWOSC | Valid negative stress test | Keep as a boundary result, not detection validation |
| Paper 1 Study A noise budget | **Contradicted by checked-in data** | Replace Table 8 and associated prose/figures before submission |
| Paper 2 NFPA | Coherent proof of concept | Run the manuscript-prescribed sweeps and at least about 20 seeds/splits |
| Paper 2 neural/coverage runs | Exploratory and currently invalid for inference | Repair implementation and rerun from a frozen protocol |
| Upgraded trace generator | Strong simulator | Suitable for controlled validation after manuscript synchronization |
| Upgraded PSD/CSD/calibration stack | Strong second-order toolkit | Requires acquisition-response assumptions and empirical held-out validation for detector claims |

An internal five-point ScholarEval-style triage gives Paper 1 about **3.7/5 (major revision)** and Paper 2 about **2.5/5 (promising framework, evidence incomplete)**. These are readiness judgments, not bibliometric scores.

## 🧭 What each paper requires

### Paper 1

Paper 1 claims that, under a linear Gaussian model, inverse noise covariance defines reconstruction geometry; OF is the fixed rank-one solution, EMPCA is the learned rank-k solution, and a tied linear autoencoder with the correct noise-weighted adjoint learns the same subspace. The minimum sufficient evidence is therefore:

1. analytic and numerical OF normalization/CRB agreement;
2. recovery of the OF direction by rank-one EMPCA;
3. independent optimization of the tied AE reaching the EMPCA optimum;
4. PCA/EMPCA agreement under white noise;
5. divergence or metric reversal under colored noise;
6. a rank requirement when timing or pulse shape varies;
7. convergence with covariance-estimation sample size;
8. calibrated whitened residuals under a correct model;
9. benefit from full rather than diagonal multichannel covariance;
10. public-data evidence with independent noise calibration and evaluation samples, strong baselines, negative controls, and detector-facing metrics.

The repository covers items 1–9 with S1–S9. CRESST covers most of item 10 for reconstruction, but its trigger-efficiency result is inconclusive. GWOSC demonstrates model inadequacy rather than positive calibration, which is scientifically useful provided it remains a negative result.

### Paper 2

Paper 2 broadens the object of study to the tuple `(noise metric, model class, training coverage)`. Only the NFPA component is currently claimed as experimentally completed. The paper itself correctly asks for:

- at least about 20 seeds and multiple train/test splits;
- low/medium/high SNR;
- white, pink, Brownian, and measured noise;
- channel counts 2, 4, 8, and 16;
- matched total ranks and alternative `(k_c, k_t)` factorizations;
- a controlled nonseparability sweep;
- explicit sample counts, rank, whitening, seeds, and confidence intervals.

The nonlinear architecture and coverage claims additionally require a factorial design in which metric, architecture, coverage, and noise mismatch are changed independently. Current real-data neural runs do not yet satisfy this design.

## 🧪 Experiment registry: description, method, and target

### Paper 1 controlled experiments

| ID | Description and method | Scientific target | Current result and status |
| --- | --- | --- | --- |
| S0 | Small end-to-end synthetic smoke run through configuration, simulation, reconstruction, metrics, and serialization. | Computational integration, not a scientific claim. | Passing smoke coverage. |
| S1 | Generate a fixed pulse in known Gaussian noise; estimate amplitude with OF across replicates; compare empirical standard deviation with `(sᴴΣ⁻¹s)⁻¹/²`. | OF unbiasedness, normalization, and CRB efficiency. | Positive; gate covered by tests. |
| S2 | Fit rank-one weighted PCA/EMPCA to noisy copies of a fixed template and compare its whitened direction with OF. | Learned rank-one subspace recovers the OF direction. | Positive; ten-seed mean angle about 3.86°. Finite-sample recovery is good but not exact. |
| S3 | Optimize an independently parameterized tied linear AE with the weighted loss and compare objective/subspace with EMPCA. | Numerical verification of the linear AE–EMPCA bridge. | Positive; ten-seed relative objective gap about `2×10⁻¹⁶`; reported maximum angle about `1.9×10⁻⁶` degrees. |
| S4 | Run PCA and EMPCA on white-noise data with the same rank and held-out evaluation. | Negative control: the two geometries should coincide when covariance is isotropic. | Positive; sweep maximum angle about `1.5×10⁻⁶` degrees. |
| S5 | Train matched-rank PCA and EMPCA under colored noise; compare held-out raw MSE and weighted residual. | Demonstrate that Euclidean and likelihood orderings can disagree. | Positive, but modest in the likelihood metric: PCA `48.490`, EMPCA `47.945`, paired difference `0.546` with 95% CI `[0.519, 0.571]`; raw-MSE ordering reverses. The manuscript must not call `48.49` a “times reference” ratio because the code does not compute that ratio. |
| S6 | Sweep reconstruction rank for pulse families with timing variation; evaluate held-out clean MSE/residual. | Timing variation creates a higher-dimensional signal manifold. | Positive; best clean-MSE rank averages about 4.6 across ten seeds. |
| S7 | Estimate covariance using increasing calibration samples and compare uncertainty with an oracle covariance. | Quantify covariance-estimation error and convergence. | Positive; at 1,000 calibration samples the mean estimated/oracle uncertainty ratio is about `1.032`. |
| S8 | Whiten residuals under the correct Gaussian model and compare quadratic energy with its χ² expectation. | Distributional calibration, not merely mean PSD agreement. | Positive; mean χ² per degree of freedom about `0.997`. This experiment requires the upgraded Gaussian-bin generator. |
| S9 | Simulate correlated channels; compare full-CSD/full-covariance and diagonal weighting. | Full cross-channel covariance improves calibrated uncertainty when correlations matter. | Positive; diagonal/full uncertainty ratio about `1.025` in the current gate. |

The canonical descriptions and commands are in [Experiment Protocols](EXPERIMENT_PROTOCOLS.md), with outputs under [`results/`](../results/) and the sweep implementation in [`scripts/sweep_all.py`](../scripts/sweep_all.py).

### Paper 1 public-data and supplementary experiments

| ID | Description and method | Scientific target | Current result and status |
| --- | --- | --- | --- |
| CRESST-A | Seven public detector channels. Pulse and noise streams are each split chronologically 70/30 within channel. Calibration uses 3,195–12,692 pulses and 512–537 noise records per channel; held-out evaluation uses 1,369–5,440 pulses. Compare OF, multi-template OF, PCA, rank-6 EMPCA, tied AE, and five negative controls. | External validation of the linear hierarchy on a real pulse-shape family. | Positive for reconstruction. Pooled EMPCA improvement over OF `1012.27`, event-bootstrap 95% CI `[927.90, 1097.61]`; margin over multi-template OF `352.17`, CI `[272.40, 424.01]`. PCA–EMPCA same-rank benefit is the cleaner covariance-specific evidence. Trigger sensitivity remains inconclusive. See [CRESST result](CRESST_RESULT.md). |
| CRESST post-freeze sensitivity | Reweight channels equally and perform channel-then-event hierarchical bootstrap. | Test whether pooled event counts let large channels dominate and whether effects survive detector clustering. | Positive but wide: hierarchical CIs remain above zero for OF–EMPCA, multi-OF margin, and PCA–EMPCA. This is post-freeze sensitivity and should be labeled as such. |
| GWOSC-A | Global median Welch PSD on H1/L1 off-source data, random and chronological calibration/evaluation splits; compare null score dispersion with model prediction. | Test whether a stationary global PSD calibrates real interferometer noise. | Negative. Null standard-deviation/prediction ratios: H1 `1.986` random and `3.183` chronological; L1 `1.263` and `2.964`. |
| GWOSC-F1 | Apply identical FIR coefficients through two implementations and compare scores. | Isolate implementation mismatch from statistical model mismatch. | Positive implementation identity; it does not establish physical calibration. |
| GWOSC-L1 | Re-estimate a local 64-second PSD and repeat null calibration, with a synthetic local control. | Test whether simple time-local stationarity fixes GWOSC failure. | Synthetic control passes (`1.148`), real H1/L1 fail strongly (`9.424`, `8.827`). Retain as a negative boundary result. See [GWOSC result](GWOSC_RESULT.md). |
| Study A | Square component ASDs, sum component PSDs, compare with total ASD, then report component fractions, dominant bands, and crossovers. | Noise-budget decomposition and detector/readout interpretation. | Arithmetic consistency passes, but the manuscript interpretation is wrong; details are in the critical findings below. |
| Study B | Construct `W(f)=|S(f)|²/P_total(f)` and remove or rescale components to assess the change in OF information/sensitivity. | Determine which modeled component most limits amplitude resolution. | Implemented in a notebook and reported in Paper 1, but not archived as a clean machine-readable, versioned result. It must be regenerated after correcting Study A labels and the acquisition/aliasing assumptions. |
| Supplementary pulse/rank diagnostics | Mean/derivative/width alignment, template mismatch, time shifts, rank saturation, jitter, convergence, and residual calibration. | Explain which physical variations produce additional subspace directions and verify optimizer behavior. | Useful supporting diagnostics, but they should not be presented as independent external validation. |
| TIDMAD-A | Optional denoising extension. | Transfer beyond the validated detector domains. | Planned/optional, not evidence. |

### Paper 2 experiments

| ID | Description and method | Scientific target | Current result and status |
| --- | --- | --- | --- |
| NFPA main demo | Two channels, 256 samples, Brownian noise, 800 train/300 test, `(k_c,k_t)=(2,3)`, total rank 6, one seed. Compare EMPCA, NFPA, isotropic AE/PCA; also run rank-one NFPA/OF. | Restricted bridge, cost of Kronecker structure, likelihood-vs-MSE reversal, and rank-one OF recovery. | Valid proof of concept only. EMPCA/NFPA whitened residuals `0.9929/0.9916`; rank-one cosine and amplitude correlations are `1.000`. One seed cannot support comparative superiority. |
| NPML-A metric ablation | PCA versus EMPCA under white, pink, and Brownian simulated noise. | Metric choice independent of architecture. | Historical notebook output exists, but the notebook imports a missing `implementation` package and is not runnable now. Its white-noise table reports an implausible `93,084,020` versus `2,687` weighted residual despite the white-noise equivalence target, so it must be treated as stale or convention-broken evidence. |
| NPML-B synthetic coverage ablation | Add explicit amplitude, timing, shape, and position-like basis directions; set one latent to zero in restricted training; test on full latent support; compare EMPCA/PCA and linear probes. | Demonstrate that correct metric cannot recover an unobserved tangent direction. | Conceptually useful synthetic prototype. “Position” is a broadened-waveform proxy, not detector geometry; single seed and missing current dependency prevent paper use. |
| NPML-C NFPA nonseparability sweep | Rank-one NFPA and rank-one EMPCA on separable, mildly nonseparable, and strongly nonseparable four-channel simulations. | Measure degradation of a Kronecker prior as separability fails. | Single-seed prototype. Reported principal angles remain about 3° and NFPA slightly beats EMPCA even as distortion increases, which does not visibly expose the intended cost; redesign the perturbation and repeat. |
| NPML-D architecture readiness | Inventory linear AE, CNN, and Transformer support. | Establish whether a controlled architecture ladder can run. | The exported status is stale. A newer trainer exists, but the comparison is not scientifically ready. |
| NPML-E Transformer readiness | Planned raw/prewhitened × MSE/Mahalanobis matrix. | Test attention under detector likelihood geometry. | New result directories exist, but the design and implementation defects below invalidate comparison. |
| Real AE 2×2 | Train convolutional AE with raw/prewhitened input and MSE/Mahalanobis losses on K-alpha traces. | Separate input coordinate system from loss metric. | Exploratory. Nominally identical run 01/run 02 configurations differ by many orders of magnitude because model/training RNGs are not seeded. |
| Real metric×coverage 2×2 | Full or central-quantile-restricted K-alpha training crossed with MSE/Mahalanobis. | Test the paper’s coverage condition on real data. | Invalid as a coverage test: the timing 20th and 80th quantiles are both `4035`, and restriction also halves training size (`1434` to `711`), confounding support with sample count. |
| Real Transformer run | Patch Transformer with prewhitened input and Mahalanobis loss. | Test a nonlinear global architecture. | Not interpretable: no positional encoding, channels processed independently, no controlled seed ensemble, and unstable training metrics. |

## 🔬 Scientific-rigor assessment

### What is done well

- The theoretical claims and synthetic gates are unusually well aligned. Each core equivalence has a direct numerical invariant rather than only a downstream task score.
- White-noise controls, wrong-PSD controls, random/noise-only bases, permuted bins, and mismatched templates are appropriate falsification tests.
- The project preserves failed GWOSC gates and does not convert them into a positive narrative.
- CRESST uses held-out pulses and independent noise calibration records, a strong multi-template baseline, paired comparisons, and predeclared negative controls.
- The upgraded noise tests go beyond mean PSD matching: they check Fourier Gaussianity, exponential periodograms, χ²-relevant behavior, covariance/CSD, endpoints, and reproducible random streams.

### Critical findings that must be fixed

1. **Paper 1 Study A is inconsistent with the data.** The checked-in export [`studyA_band_summary_AL.csv`](../data/noise_sample/Study_A/studyA_band_summary_AL.csv) says SQUID is dominant in every band from 1 Hz to 30 MHz, with median fractions about 0.867–0.967. [`studyA_crossover_summary_AL.csv`](../data/noise_sample/Study_A/studyA_crossover_summary_AL.csv) reports no SQUID–Johnson, SQUID–TD, or SQUID–Er crossing. Paper 1 Table 8 instead reports “1/f phonon,” “vibrational,” “TES Johnson,” and SQUID dominance in different bands with several crossovers. Those labels do not correspond to the actual files `SQUID`, `Johnson`, `TD`, and `Er`, and the numerical regimes are absent. Replace the table, figures, crossovers, interpretation, and any downstream Study B wording.

2. **Paper 1 describes the obsolete random-phase sampler.** Equation 81 fixes each Fourier-bin magnitude and randomizes only phase. That produces the desired second-order spectrum but not a Gaussian Fourier ensemble; per-bin periodogram power is degenerate, so χ² and goodness-of-fit tests are invalid. The upgraded implementation correctly uses complex Gaussian interior bins in [`NoiseGenerator.py`](../noise_module/noise_module/NoiseGenerator.py), lines 325–368. Update the equation and regenerate every result that used the old sampler. The manuscript also calls the FFT-grid covariance Toeplitz; it is circulant/periodic. Exact finite Toeplitz sampling is a separate path in the new code.

3. **Paper 2 experiments are not reproducible.** [`train_reconstruction.py`](../paper2/trainers/train_reconstruction.py) reads a data split seed but never seeds Torch, NumPy, or Python before model initialization and shuffled training. This explains orders-of-magnitude differences between nominally identical runs. Seed all RNGs, pass a seeded `torch.Generator` to loaders, record deterministic settings, and run a seed ensemble.

4. **Paper 2 evaluates the test set repeatedly.** Lines 448–460 of [`train_reconstruction.py`](../paper2/trainers/train_reconstruction.py) evaluate test data each time validation improves. Although selection is based on validation loss, repeated test access invites adaptive inspection and the saved metrics overwrite the validation fields. Save the best validation checkpoint, restore it, and evaluate the test set exactly once.

5. **The real-data timing metric is invalid.** [`reporting.py`](../paper2/analysis/reporting.py), lines 284–340, compares an OF-fitted relative shift directly with metadata `t0` near absolute sample 4035. The resulting RMSE near 4035 is a coordinate mismatch, not timing performance. Define the template origin, sign, units, and conversion explicitly and unit-test known injected shifts.

6. **The current Transformer cannot test the proposed representation claims.** [`transformer_encoder.py`](../paper2/models/transformer_encoder.py) embeds patches and applies self-attention without positional encoding. It reshapes `(B,C,P,D)` to `(B*C,P,D)`, so channels never attend to one another. Add temporal position, channel identity/geometry, and a deliberate cross-channel interaction scheme; then ablate them as Paper 2 proposes.

### Major design limitations

- Ten synthetic seeds are acceptable for regression testing but thin for publication inference. Use at least 20–30 independent seeds, report raw seed rows, paired effect distributions, and a predeclared smallest relevant effect.
- S4–S6 sweep code uses held-out fractions, but the frozen standalone YAML defaults do not always encode the same held-out design. Archive the exact generated configs or command manifest used for every table and figure.
- CRESST’s pooled event bootstrap is conditional on seven observed channels and lets high-count channels dominate. Add equal-channel and channel-then-event hierarchical intervals to the main sensitivity analysis. Do not claim population-level detector generalization from seven clusters.
- CRESST pulse and noise streams are split chronologically but separately. There is no direct row/hash leakage, yet some evaluation pulses precede the last calibration-noise time. A future replication should use a shared acquisition cutoff or grouped acquisition blocks.
- Paper 1’s S5 effect is statistically clear but smaller than its prose suggests: the likelihood metric changes by about 1.1% (`0.546/48.49`), while the dramatic reversal is in raw MSE/subspace angle. Report both scales without calling a metric value a ratio.
- The real K-alpha “restricted coverage” experiment restricts a central observed box, not a designed physical latent. It changes support and sample size simultaneously. Use equal-n controls, explicit held-out tails, and a simulator or labeled dataset with independent energy, time, position, sharing, and shape coordinates.
- Random event splits should be replaced or supplemented by acquisition-time, run, detector, or source-group splits wherever neighboring/duplicate records could leak.
- Paper 2 result folders need immutable checkpoint, predictions, git revision, environment lock, data checksums, split indices, seed state, and separate validation/test metrics. Some promised artifacts are absent from current result folders.

## 🛠️ Implementation assessment

### Paper 1 implementation

The linear algebra is internally coherent. OF, weighted subspace fitting, whitening, principal angles, and the tied-AE projection are implemented in conventions that agree with the positive gates. The full test suite passed during this audit (`224 passed`; warnings only), and the upgraded noise-module suite passed (`99 passed`; warnings only). Passing tests verify the coded contracts; they do not resolve the manuscript/data inconsistency or experimental-design limitations.

The main implementation correction is provenance. A paper table should be traceable to one immutable configuration, source rows, code revision, and rendering script. At present, some paper evidence comes from sweeps, some from standalone configs, and some from stateful notebooks. Consolidate these into one results manifest.

### Paper 2 implementation

The current four-way neural experiment matrix should be considered contaminated by uncontrolled initialization and metric bugs. The following repairs are minimum requirements before rerunning:

1. initialize and record Python, NumPy, CPU Torch, and CUDA seeds before model construction;
2. use deterministic split and loader generators, and record split indices;
3. restore the best validation checkpoint and touch the test set once;
4. retain separate `best_validation` and `final_test` dictionaries;
5. correct timing coordinates and add injection-based unit tests;
6. define a physical Mahalanobis normalization if values are to be called χ² or NLL;
7. add position/channel encodings and cross-channel attention to the Transformer;
8. match parameter count, optimization budget, early stopping, rank/bottleneck, and training samples across architecture comparisons;
9. run divergence checks and report failed seeds rather than silently selecting stable runs.

The whitening code preserves relative precision geometry but normalizes positive weights to mean one and clips their dynamic range in [`whitening.py`](../paper2/data/whitening.py), lines 41–60. This is acceptable for optimization geometry. It is not an absolute Gaussian likelihood, calibrated χ², or cross-dataset NLL unless the missing constants, endpoint conventions, effective degrees of freedom, and clipping are explicitly restored.

## 🌊 Upgraded noise-module review

### Part I: noise-trace generation

The stationary Gaussian core is scientifically correct under its declared discrete convention:

- inputs are one-sided PSD densities with explicit units;
- interior rFFT bins use independent circular complex Gaussians with expected power `S(f) f_s N / 2`;
- DC and Nyquist are real Gaussians with endpoint scaling;
- `irfft` normalization and expected time variance are consistent;
- the default FFT-grid process is correctly labeled periodic/circulant;
- an exact finite Toeplitz sampler is supplied separately;
- random streams can be seeded and split reproducibly.

This is a material improvement over the sampler printed in Paper 1 and makes S8-style residual calibration meaningful.

The multichannel CSD path is also sound at the second-order level. It checks Hermitian positive-semidefinite matrices, preserves the PSD cone under linear interpolation, treats real endpoints correctly, supports explicit repair policies, and can sample directly from low-rank factors without materializing a dense CSD. The `shared_private` and low-rank generators are useful controlled models, but their latent factors are simulated causes, not fitted physical source identities.

The temporal, artifact, and non-Gaussian extensions are appropriate **stress-test generators**. Piecewise scaling, evolutionary overlap-add, slow envelopes, drift, lines, glitches, bursts, and heavy tails let the reconstruction pipeline be challenged under declared violations. They should not be called a calibrated detector-noise model unless their parameters and joint distributions pass held-out real-data checks.

Important cautions:

- per-realization channel variance normalization, if enabled, constrains the sample norm and therefore changes the Gaussian distribution; keep it off for likelihood-calibration experiments;
- crossfades and multiplicative envelopes introduce spectral convolution and boundary correlations, so local target PSDs are approximate;
- additive artifacts plus Gaussian background do not reproduce detector causality unless morphology, rate, amplitude, state dependence, and dead time are empirically fitted;
- one small documentation bug says custom mode `multiply`, while the configuration uses `scale`.

### Part II: PSD simulation, fitting, resampling, and calibration

The PSD code has strong engineering contracts: density versus rFFT power is separated, component ASDs are squared before addition, custom out-of-band behavior is explicit, composite contributions are recorded, and deterministic interpolation, alias folding, and time-domain polyphase resampling are separate operations.

The Al₂O₃/Al athermal component fit is mathematically consistent with the supplied files. The nearly exact quadrature validation, however, is not independent empirical validation: the provided total is effectively the quadrature construction of the same component curves. It proves correct parsing and arithmetic, not that the four components uniquely or physically explain a detector measurement.

The default alias-folded artifact is a hypothesis about acquisition. Aliasing depends on the analogue transfer function and anti-alias filter. With no measured electronics response, “fold everything above target Nyquist” and “remove everything out of band” are two bounding assumptions, neither a validated detector truth. Study B sensitivity should be reported under both, plus the actual transfer function when available.

The empirical calibration layer is useful but its current gates are engineering heuristics. It randomly splits records, bootstraps records as independent, and applies thresholds such as 50% interval coverage and 0.75 CSD relative error. For publication calibration:

- split chronologically or by acquisition/run groups;
- use block/hierarchical bootstrap when records or channels are dependent;
- report frequency-resolved uncertainty and simultaneous bands, not only aggregate coverage;
- predeclare gates from the downstream scientific tolerance;
- validate generated and real distributions on independent records using PSD/CSD/coherence/phase, whitened autocorrelation, tails, line occupancy, drift, and downstream OF residuals.

`CalibrationPreset.sample_parameters()` samples each marginal log-PSD bin independently and uniformly between bootstrap bounds. That produces jagged spectra and is not a posterior draw. Replace it with bootstrap spectrum draws, a smooth log-spectral Gaussian process, or a low-dimensional component-parameter posterior that preserves frequency correlation and, for multichannel data, PSD-constrained CSD structure.

### Scientific boundary of the module

```mermaid
flowchart LR
    accTitle: Evidence ladder for the upgraded noise module
    accDescr: A sequence from physical assumptions through PSD and CSD fitting, stochastic generation, held-out validation, reconstruction tests, and representation diagnostics; physical-source claims require independent sensors or interventions.

    A["Physical sources and electronics"] --> B["Transfer functions and auxiliary sensors"]
    B --> C["PSD/CSD or component model"]
    C --> D["Gaussian, nonstationary, or artifact generator"]
    D --> E["Held-out real-noise validation"]
    E --> F["OF/EMPCA/neural reconstruction"]
    F --> G["Stability, calibration, and coverage diagnostics"]
    C -. "not identifiable from total PSD alone" .-> A
```

The module covers nodes C–G well as a controlled framework. It cannot infer node A uniquely without node B or equivalent interventions.

## 📐 Noise decomposition and decorrelation: what linear methods can do

Linear methods can do the following rigorously:

- whiten noise using a fitted covariance or CSD;
- eigendecompose covariance or perform frequency-dependent CSD factorization;
- identify low-rank correlated statistical modes;
- construct null channels or generalized eigenmodes relative to a signal covariance;
- measure coherence, phase, shared/private variance, and the sensitivity of OF/EMPCA to removing a fitted component;
- regress against auxiliary/environmental channels and quantify the held-out residual reduction.

They generally **cannot** identify physical sources from the observed total covariance alone. If `Σ = Σ₁ + Σ₂`, many alternative positive-semidefinite pairs produce the same total. Rotations of latent factors leave a low-rank covariance unchanged. Decorrelation gives statistically orthogonal modes, not labels such as SQUID, vibration, Johnson, microphonics, or unknown physics.

Physical categorization becomes defensible only with additional structure: known spectral/transfer-function families, auxiliary sensors, spatial/channel topology, multiple operating conditions, injected calibration lines, hardware interventions, or non-Gaussian/temporal signatures. Even then, fit competing models on calibration data and compare predictive performance on held-out runs. Report identifiability intervals and correlations among fitted components rather than one deterministic decomposition.

For detector design, the easier and more useful target is often **actionable sensitivity attribution**, not unique source recovery. Compute how each well-defined nuisance model changes Fisher information, OF variance, trigger threshold, and reconstruction stability under a declared intervention. This can guide shielding, bandwidth, sensor layout, or readout changes without pretending that covariance eigenvectors are physical causes.

## 🤝 Fit to the proposed collaboration topics

The upgraded noise module is directly relevant to the proposed joint direction, but its best role is a **controlled perturbation and calibration engine**.

| Topic | Relation to the module | Recommended scientific test |
| --- | --- | --- |
| Noise decomposition and decorrelation | Direct for second-order statistical modes; limited for physical attribution. | Compare whitening, frequency-dependent CSD eigenmodes, constrained component fits, and auxiliary-channel regression on held-out runs. |
| Embedding separability | Direct. Noise state can be treated as a nuisance label while energy/time/position/shape are signal labels. | Measure linear-probe performance, conditional mutual information or classifier accuracy, and subspace angles after matched and mismatched noise injection. Separability should preserve signal coordinates while suppressing noise-state predictability. |
| Clustering robustness | Direct. | Cluster clean/reference embeddings, then perturb with PSD scale, slope, lines, drift, glitches, CSD modes, and calibration error. Report adjusted Rand/variation-of-information stability with bootstrap intervals and null label permutations. |
| Perturbation stability | Very direct and probably the strongest first application. | Track reconstruction change, latent displacement, local Jacobian norm, principal-angle rotation, nearest-neighbor retention, and downstream amplitude/timing shift as a function of perturbation magnitude. |
| Uncertainty or score calibration | Direct if the generator is independently calibrated. | Reliability diagrams, coverage of amplitude/timing intervals, probability-integral-transform or standardized residual tests, and false-positive calibration under matched versus shifted noise. Use real held-out noise as the final judge. |
| Performance near interaction vertices | Potentially important but not supported by the present single-channel K-alpha dataset. | Use position-aware multichannel simulation and real vertex labels; stratify performance by vertex distance and channel-sharing topology. Do not substitute waveform broadening for a physical vertex claim. |
| Fitted correlated noise categorization | Direct as a model class, not automatic source discovery. | Fit low-rank frequency-dependent CSD factors plus private PSDs, constrain them with topology/auxiliary sensors, and validate coherence, phase, and downstream residuals on held-out runs. |
| Artificial noise injection probe | Excellent fit. | Predeclare a nuisance library and severity grid; inject only into held-out clean/simulated events or real noise windows; test monotonic response, recovery, false positives, and method ranking. |
| Calibration verification | Excellent fit. | Blindly mix matched, misscaled, wrong-slope, wrong-CSD, nonstationary, and artifact-contaminated samples. Test whether diagnostics detect the known mismatch at controlled type-I error. |
| Unknown-physics anomaly | Possible but high risk. | Keep anomaly injections absent from training, include hard nuisance-only shifts, calibrate background false-positive rates on independent runs, and require signal-efficiency gains beyond a noise-mismatch detector. Call these blind synthetic anomalies, not evidence for new physics. |

### Recommended joint-paper thesis

The proposed title direction, **“Noise and Coverage Diagnostics for Learned Representations in Hierarchical Particle Reconstruction,”** is plausible if the paper makes one narrow claim:

> A representation should preserve identifiable physical coordinates under adequate training coverage while remaining stable and calibrated under a predeclared family of measured and synthetic noise shifts.

This is stronger and more testable than attempting universal physical noise decomposition. The paper should not be a catalogue of unrelated embedding plots. It needs a factorial experiment and one central failure mechanism: metric mismatch, coverage failure, or noise-domain shift.

### Four-application experimental program

1. **Representation separability:** generate or select events with labeled energy, time, position/vertex, sharing, and shape; cross these with independent noise-state labels. Compare OF, EMPCA, NFPA, CNN, and Transformer embeddings at matched bottleneck/compute. Use held-out linear probes and nuisance predictability.
2. **Robustness:** apply the noise module’s PSD, CSD, temporal, line, glitch, and heavy-tail perturbations on a log-spaced severity grid. Measure latent and output stability, clustering retention, and detector-facing amplitude/timing/trigger metrics.
3. **Coverage:** train on full versus one-latent-restricted support with equal sample counts, then test on the same full factorial target distribution. This directly tests Paper 2’s coverage proposition.
4. **Calibration/anomaly probe:** blind the analyst to a mixture of nominal backgrounds, nuisance shifts, known signal injections, and held-out synthetic anomalies. Evaluate false-positive calibration, power, and whether anomaly scores are merely detecting PSD/CSD mismatch.

Use at least 20 independent seeds/splits, group-aware real-data splits, paired comparisons, hierarchical intervals across detectors, and a locked test set. The noise model should be fitted only on calibration data; perturbation severity and success thresholds should be frozen before test evaluation.

## 🚀 Prioritized action list

### Stop-ship before Paper 1 submission

1. Replace Study A Table 8 and all associated labels/crossovers with values regenerated from the checked-in component files; audit Study B downstream conclusions.
2. Replace Paper 1 Equation 81 with the complex-Gaussian Fourier-bin construction and change the covariance statement from Toeplitz to periodic/circulant, mentioning the exact Toeplitz alternative.
3. Establish which archived experiments used fixed-modulus versus Gaussian bins; rerun S8 and every distributional result that used the old generator.
4. Correct the S5 “48.49 times” wording to a metric value and report the paired difference/effect size.
5. Add equal-channel and hierarchical CRESST sensitivity intervals to the main evidence, clearly marked post-freeze; preserve the trigger-efficiency limitation.

### Before using Paper 2 neural results

1. Fix seeding, test-set handling, timing coordinates, metric naming, and artifact provenance.
2. Replace the present coverage restriction with a designed latent intervention and equal-n control.
3. Add Transformer positional/channel geometry and cross-channel interaction; predeclare ablations.
4. Rerun the NFPA required sweeps and the architecture×metric×coverage×noise experiment across at least about 20 seeds/splits.
5. Keep current result folders labeled exploratory and exclude their numbers from claims.

### Before claiming detector noise decomposition

1. Obtain acquisition transfer functions and anti-alias response.
2. Add independent auxiliary channels or controlled operating-condition/hardware interventions.
3. Fit constrained component/CSD models on calibration runs and compare predictive likelihood on held-out runs.
4. Report statistical modes and actionable sensitivity attribution separately from physical source labels.

## 📂 Evidence and reproducibility map

- Paper 1 protocols: [EXPERIMENT_PROTOCOLS.md](EXPERIMENT_PROTOCOLS.md)
- Current claim status: [CURRENT_STATUS.md](CURRENT_STATUS.md)
- CRESST predeclaration: [CRESST_PREDECLARED_GATES.md](CRESST_PREDECLARED_GATES.md)
- CRESST outcome: [CRESST_RESULT.md](CRESST_RESULT.md)
- GWOSC outcome: [GWOSC_RESULT.md](GWOSC_RESULT.md)
- Synthetic implementation: [`src/noise_geometry/experiments/synthetic.py`](../src/noise_geometry/experiments/synthetic.py)
- Noise core: [`noise_module/noise_module/NoiseGenerator.py`](../noise_module/noise_module/NoiseGenerator.py)
- PSD resampling: [`noise_module/noise_module/psd_resampling.py`](../noise_module/noise_module/psd_resampling.py)
- Multichannel CSD: [`noise_module/noise_module/multichannel_noise.py`](../noise_module/noise_module/multichannel_noise.py)
- Empirical calibration: [`noise_module/noise_module/calibration.py`](../noise_module/noise_module/calibration.py)
- Noise validation: [`noise_module/noise_module/validation.py`](../noise_module/noise_module/validation.py)
- Al₂O₃ component fit: [`noise_module/noise_module/fit_al2o3_athermal.py`](../noise_module/noise_module/fit_al2o3_athermal.py)
- Paper 2 trainer: [`paper2/trainers/train_reconstruction.py`](../paper2/trainers/train_reconstruction.py)
- Paper 2 whitening: [`paper2/data/whitening.py`](../paper2/data/whitening.py)
- Paper 2 Transformer: [`paper2/models/transformer_encoder.py`](../paper2/models/transformer_encoder.py)
- Paper 2 experiment artifacts: [`paper2/results/`](../paper2/results/)

The audit distinguishes mathematical implementation tests, controlled synthetic evidence, public-data validation, negative stress tests, and proposed experiments. Only the first four categories have current evidence; the collaboration program is a recommended design, not a completed result.
