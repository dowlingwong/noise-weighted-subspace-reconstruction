# Independent Evaluation: Paper 1 Experiments, Noise Module, and Collaboration Fit

Date: 2026-07-26 (second, independent audit pass; verified against the revised
`paper1.pdf` built 2026-07-26 and the current repo state). Where the earlier
`ACADEMIC_EXPERIMENT_AND_NOISE_MODULE_EVALUATION.md` made claims, I re-verified
the load-bearing ones directly against code and archived artifacts instead of
taking them on faith. This document reports what I confirmed, what is new, and
my own judgment on the collaboration questions.

---

## Part 1 — Do the experiments match what Paper 1 needs, and are they rigorous?

### 1.1 Requirements coverage

Paper 1's claims decompose into: (i) estimator identities (OF = rank-1 GLS at
the CRB; rank-1 EMPCA ≡ OF; tied weighted linear AE ≡ EMPCA), (ii) the
loss-geometry consequence (MSE is the likelihood only for white noise; metric
reversal under colored noise), (iii) model-consequence claims (rank under
timing jitter, covariance-sample convergence, residual calibration, full vs
diagonal multichannel Σ), (iv) real-data reconstruction evidence (CRESST), and
(v) an honest model-adequacy boundary (GWOSC). S1–S9 map one-to-one onto
(i)–(iii); CRESST-A with negative controls and the post-freeze hierarchical
analysis covers (iv); the GWOSC global/local PSD failures cover (v). Coverage
is complete for the paper's (deliberately narrow) claim set. The revised
manuscript's scoping language — within-channel only, no transfer, no detector
sensitivity, GWOSC diagnostic-only — matches what the repo can actually
support.

### 1.2 Design rigor — verdict

Above field standard. Specific strengths I verified in code, not just docs:

- Each theorem has a direct numerical invariant (optimality gap to a closed
  form, principal angle, CRB ratio), not a downstream proxy.
- S4 (white-noise) is a true internal control for S5, so the colored-noise
  divergence is attributable to the metric rather than rank/optimizer/data.
- CRESST: predeclared gates (`docs/CRESST_PREDECLARED_GATES.md`), paired
  event-level comparisons, a strong multi-template OF baseline (which honest
  papers usually omit), five falsification controls, chronological
  time-disjoint splits with zero row/hash overlap (checked in
  `cresst/analysis.py`: `time_disjoint_split`, trace-SHA audit), and a
  post-freeze channel-then-event hierarchical bootstrap.
- The failed GWOSC calibration is preserved as a negative result with the
  predeclared 64 s local-PSD hypothesis explicitly falsified. This is how it
  should be done and is rare in practice.

Remaining design limitations (all acknowledged in the paper's Limitations, so
these are residual risks rather than misrepresentations): only 7 CRESST
channel clusters from 2 runs bounds external validity; pulse and noise streams
split chronologically but *separately* (no common cross-stream cutoff — 9
evaluation pulses precede the last calibration-noise timestamp across two
channels); the OF-vs-EMPCA headline mostly measures signal-model capacity, with
the smaller PCA-vs-EMPCA contrast carrying the covariance-weighting claim
(correctly stated in §8); trigger efficiency inconclusive.

### 1.3 Implementation accuracy — what I verified myself

- `src/canonical/OptimumFilter.py`: covariance-form OF matches Eq. (12)–(13)
  of the paper, with correct two-sided PSD unfolding, DC exclusion via an
  infinite PSD bin, and the standard χ² decomposition. Consistent with the
  microcalorimetry convention (Szymkowiak-style).
- `src/noise_geometry/experiments/synthetic.py`: each S-runner implements
  exactly what the paper text describes; S5 uses the likelihood-preserving
  Re/Im-stacked rFFT representation (the doc trail shows the earlier
  real-part-only defect was found and fixed); S1 routes OF through the same
  whitened-real primitive as EMPCA, so the S2 equivalence is not circular at
  the code-path level while still sharing one convention module.
- Empirically re-derived (in an isolated script, not repo code) the noise
  generator convention used by S8: time-domain variance matches the PSD
  integral to 0.2%, interior rFFT bin power matches `f_s·N·P_k/2` to 0.03%,
  and the periodogram coefficient of variation is 0.993 vs 1.0 expected for
  complex-Gaussian bins. The S8 χ² gate is therefore meaningful with the
  current generator.

### 1.4 New findings from this pass (not in the previous audit)

1. **S5 confirmatory numbers in the revised PDF do not match the archived
   30-seed artifact.** The paper (§7.2) reports: PCA 48.418 / EMPCA 47.878,
   paired difference 0.540, 95% CI [0.523, 0.560], raw MSE 2367.99 / 5388.75,
   angle 86.07°. The only archived 30-seed confirmatory run
   (`evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/`) gives:
   PCA 48.419 / EMPCA 47.869, difference 0.5503, CI [0.530, 0.571], raw
   2330.27 / **7412.67**, angle 86.25°. The frozen 10- and 20-seed sweeps do
   not match either (0.5455 / 0.5446). The prose appears to mix numbers from
   different runs, or cites a run not archived in this repo. None of this
   changes any conclusion (all variants agree in sign and magnitude), but for
   a paper whose selling point is provenance discipline, the S5 paragraph must
   be regenerated verbatim from one archived JSON — especially the raw-MSE
   EMPCA value (5388.75 vs 7412.67 is a 38% discrepancy).
2. **Latent crash bug in `OptimumFilter.sliding_fit(chisq_mode='none')`.** The
   call to `_slide_block_hop1_amponly` passes 12 arguments to a 13-parameter
   numba kernel (`F_imag` is missing, so `E_real`→`F_im`, `E_imag`→`E_re`,
   `X`→`E_im`…). First invocation will raise a numba `TypingError`. No test
   exercises this path and no paper result uses it, but it should be fixed and
   covered before anyone uses the streaming amplitude-only mode.
3. **S5 weight/noise scale mismatch (cosmetic).** The synthetic noise is
   scaled by 0.25 while weights derive from the unscaled PSD. Subspace fits
   and residual orderings are invariant to a global Σ scale, so no claim is
   affected; just do not read S5 weighted residuals as χ²-calibrated per dof.
4. Confirmed (from the earlier audit, independently re-checked): the v1
   standalone S4/S5/S6 configs are in-sample; the v2 configs and the
   confirmatory sweep correctly bake in `test_frac=0.5` and disjoint seed
   ranges with a predeclared smallest-relevant-effect. The remediation is
   sound; keep citing only v2/sweep artifacts for held-out claims.

### 1.5 Overall verdict on Part 1

Theory–experiment alignment is excellent; implementation is accurate where it
matters; the evidence chain is unusually disciplined. The single stop-ship item
is the S5 paragraph provenance (finding 1). With that fixed and the seven-
cluster caveat retained, the experimental foundation supports every claim the
revised paper actually makes.

---

## Part 2 — Experiment registry

Moved to its own deliverable: `docs/EXPERIMENTS_OVERVIEW.md` (description,
method, target, and status for every implemented experiment).

---

## Part 3 — Noise module review (trace generator + PSD simulation)

Scope: `noise_module/noise_module/`. Two parts as you framed them: (I) the
stochastic trace generator (`NoiseGenerator.py`, `multichannel_noise.py`,
`temporal_noise.py`, `artifact_injector.py`, `non_gaussian.py`, `streaming.py`)
and (II) the PSD simulation/fitting stack (`spectral_models.py`,
`psd_resampling.py`, `fit_al2o3_athermal.py`, `calibration.py`,
`validation.py`).

### 3.1 Part I — trace generation: scientifically sound

Verified directly:

- **Fourier-domain convention is correct and now Gaussian.** Interior rFFT
  bins are circular complex Gaussian with `E|X_k|² = f_s·N·P_k/2`; DC and
  Nyquist are real Gaussian with `f_s·N·P_k`; odd-N traces correctly treat the
  last bin as complex (both in `sample_stationary_gaussian_from_rfft_power`
  and the vectorized `generate_ensemble`; the `upper = N//2+1-(N+1)%2` index
  logic is right in both). I confirmed variance, per-bin power, and the
  exponential periodogram distribution numerically. This replaces the
  fixed-modulus random-phase sampler of the old Paper 1 Eq. 81, which had
  correct mean PSD but degenerate per-bin power — the module's own
  `PAPER_CONVENTION.md` documents this precisely and should be lifted into the
  manuscript appendix as planned.
- **The circulant-vs-Toeplitz distinction is handled honestly.** FFT synthesis
  is labeled a periodic/circulant finite-grid process; `sample_from_covariance`
  provides exact finite-Toeplitz sampling via circulant embedding with a dense
  eigendecomposition fallback, and I verified the embedding algebra (real
  symmetric circulant square root ⇒ exact target covariance when the embedding
  is PSD). Most published noise generators silently conflate these; this one
  does not.
- **Density vs bin-power units are separated** (`build_psd_density` vs
  `build_rfft_power`), with the legacy name deprecated loudly. This is the
  kind of convention bug that quietly ruins χ² studies; the module engineers
  it away.
- **Multichannel:** shared/private and low-rank latent constructions have
  correct implied covariances recorded in metadata; the target-CSD path
  enforces Hermitian PSD matrices per bin, requires real endpoint bins,
  supports explicit repair policies, and can sample from low-rank factors.
  Sound at second order.

Cautions (agreeing with, and adding to, the earlier audit):

- `normalize_channel_variance` conditions on the realized sample norm and
  breaks Gaussianity; keep it off for any likelihood-calibration experiment
  (default is off — good).
- Evolutionary/temporal and artifact layers are stress-test generators, not a
  calibrated detector model; crossfades and envelopes make local PSDs
  approximate. Fine for robustness grids, not for χ² gates.
- `CalibrationPreset.sample_parameters()` draws log-PSD bins independently —
  jagged spectra, not a posterior. Replace with bootstrap spectrum draws or a
  smooth log-spectral GP before using it to propagate PSD uncertainty.

### 3.2 Part II — PSD simulation: sound, with clearly declared hypotheses

- `alias_fold_psd_density`: I checked the folding algebra — summing the
  one-sided source density over `|f + n·f_s|` reproduces the one-sided PSD of
  the ideally-sampled process, and the DC/Nyquist double-count guard is
  correct. The module rightly treats "fold everything" vs "in-band only" as
  two *bounding acquisition hypotheses* rather than truth; Study B reports
  both branches. This is the correct epistemic posture given no measured
  anti-alias response.
- `synthetic_resample_psd_density` is a genuinely operational Monte Carlo path
  (generate at source rate → `resample_poly` → Welch at target rate), with
  edge trimming and segment boundaries aligned to trace boundaries so Welch
  never straddles independent traces. Good.
- The Al₂O₃/Al athermal fit is internally consistent, but note the supplied
  "total" is itself the quadrature sum of the same component curves — the
  near-exact agreement validates parsing and arithmetic, not physics. The repo
  and revised paper both now say this correctly (Study A as an *audit* of
  supplied curves).

### 3.3 Contribution to Paper 1 and Paper 2

Paper 1 — direct, in three places: (a) the Gaussian-bin generator is what
makes S8 and any χ²/KS distributional gate valid — this is a correctness
dependency, already integrated; (b) `PAPER_CONVENTION.md` /
`paper_convention.json` is the ready-made replacement for old Eq. 81 and the
Toeplitz→circulant wording fix; (c) the component spectra + resampling +
Study A/B machinery is the entire noise-budget appendix. In short, the module
is already load-bearing for Paper 1; what remains is manuscript
synchronization, not new capability.

Paper 2 — this is where the upgrade has the most *unrealized* value. Paper 2's
tuple (Σ⁻¹, F, P_train→P_target) needs exactly what the module provides:
matched/mismatched Σ pairs (train under one PSD/CSD, evaluate under a
perturbed one), separable vs non-separable multichannel covariances for the
NFPA Kronecker prior (the target-CSD and low-rank factor paths can generate
controlled violations of separability — precisely what the weak NPML-C sweep
needs), coverage interventions with independent latent controls, and the
temporal/artifact/non-Gaussian layers as a predeclared robustness grid. None of
the current Paper 2 experiments exploit this yet; wiring the noise module into
a rerun of the NFPA and coverage suites is the highest-leverage next step for
Paper 2.

---

## Part 4 — Collaboration topics (Junjie) vs the upgraded noise module

### 4.1 The core question first: noise decomposition/decorrelation — 线性方法能做到哪一步

There is a hard identifiability wall, and it is worth being precise about
where it sits. What linear/second-order methods CAN do rigorously:

1. Whitening / decorrelation: given a fitted Σ or CSD, produce statistically
   orthogonal modes; exact and useful (this is Paper 1's whole geometry).
2. Frequency-resolved CSD eigenanalysis: per-bin eigenmodes, coherence, phase,
   shared-vs-private variance splits across channels; rank estimation of
   common-mode noise.
3. Constrained component fits: if component *shapes* are known up to
   parameters (SQUID floor, Johnson, TES thermal, 1/f, lines — i.e., a
   transfer-function family), fitting the total PSD/CSD to a sum of
   parameterized templates is well-posed and testable on held-out runs.
   This is Study A/B's regime, conditional on the declared model.
4. Auxiliary-channel regression: project out environmental/witness channels
   (the LIGO-style approach); attribution is causal to the extent the witness
   is upstream.
5. Sensitivity attribution: for any declared decomposition, compute how
   removing/rescaling a component changes OF variance, Fisher information,
   trigger threshold (Study B). Actionable for detector design without
   claiming uniqueness.

What they CANNOT do: recover uniquely labeled physical sources from one total
Σ alone. If Σ = Σ₁ + Σ₂, infinitely many PSD-cone pairs give the same sum;
low-rank factors are rotation-degenerate. Eigenmodes ≠ sources. So your
instinct is right: "decompose the noise to help detector design" via
unconstrained decomposition is not just hard — it is ill-posed. It becomes
well-posed only with added structure: known spectral families (3), auxiliary
sensors (4), multiple operating conditions / hardware interventions, injected
calibration lines, or non-Gaussian/temporal signatures that break the
rotation degeneracy. The productive reframing for the collaboration is
**sensitivity attribution under declared models** (what the module already
does) rather than **source discovery**.

Paper 1 already contains the right language for this (§9: "decorrelated
statistical modes, not uniquely labelled physical sources") — that paragraph
is the honest boundary of the linear story and can be quoted to Junjie as-is.

### 4.2 Topic-by-topic fit with the upgraded module

- **Embedding separability** — good fit. The module supplies controlled
  nuisance labels (noise state: PSD tilt, line content, CSD rank) crossed
  with signal labels; test whether learned embeddings keep signal coordinates
  linearly decodable while noise-state predictability stays at chance.
- **Clustering robustness** — good fit; the perturbation grids (PSD scale/
  slope, lines, drift, glitches, CSD modes) are exactly the severity axes;
  report ARI/VI stability with seed ensembles.
- **Perturbation stability** — the strongest first application. Inject
  calibrated perturbations at known severity; measure latent displacement,
  reconstruction drift, amplitude/timing bias. The module's severity is exact
  by construction, which real data never gives you.
- **Uncertainty / score calibration** — direct fit and doubly interesting
  because Paper 1's GWOSC result is a real example of calibration failure:
  you can show the diagnostic detects mismatch that implementation checks
  miss. Use real held-out noise as the final judge.
- **Performance near interaction vertices** — weakest fit today: nothing in
  the current stack encodes geometry/vertex structure; waveform broadening is
  not a vertex. Needs a position-aware multichannel simulation or labeled
  data first.
- **Fitted correlated noise categorization (ask IMS)** — feasible as
  *constrained model class* (low-rank CSD factors + private PSDs + known
  families), not as automatic discovery; validate by held-out predictive
  likelihood, and report factor-rotation uncertainty honestly.
- **Artificial noise injection probe / calibration verification ("assume
  non-physics, unknown-physics anomaly")** — excellent fit and the cleanest
  joint-paper backbone: a blind mixture of nominal, nuisance-shifted, and
  anomaly-injected samples, with predeclared type-I error. The one scientific
  trap: an "anomaly detector" that is really a PSD-mismatch detector. The
  design must include hard nuisance-only shifts as foils, and any claimed
  anomaly sensitivity must exceed the best noise-mismatch baseline.

### 4.3 On the joint-paper title

"Noise and Coverage Diagnostics for Learned Representations in Hierarchical
Particle Reconstruction" is viable if it commits to one thesis: *a
representation should preserve identifiable physical coordinates under
adequate coverage while remaining stable and calibrated under a predeclared
family of noise shifts*. That makes the noise module the paper's instrument
(perturbation + calibration engine), Paper 2's tuple the organizing theory,
and the five bullet topics become sections of one factorial experiment instead
of five disconnected studies. Avoid putting "noise decomposition" in the
claims; put "injection-calibrated diagnostics" there instead. Minimum
standards to agree with Junjie up front: ≥ 20 seeds/splits, group-aware
real-data splits, matched compute/bottleneck across architectures, frozen
severity grids and thresholds before test evaluation, and a locked test set.

### 4.4 Bottom line

Your design intuition is correct on both counts: the noise module's
architecture is right (and now verifiably correct at the distributional
level), and pure decomposition-for-detector-design is the one topic on the
list I would *not* anchor the collaboration on — it is identifiability-limited,
not effort-limited. Anchor on perturbation stability + calibration
verification (where the module is uniquely strong), use sensitivity
attribution as the detector-design deliverable, and keep decomposition claims
conditional on declared component families exactly as the revised Paper 1
already phrases them.
