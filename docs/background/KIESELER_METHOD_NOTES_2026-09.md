# What to take from Kieseler's two papers (notes, 2026-09-03)

Sources: Qasim, Kieseler, Iiyama, Pierini, *Learning representations of irregular
particle-detector geometry with distance-weighted graph networks* (GravNet/GarNet,
arXiv:1902.07987); Kieseler, *Object condensation: one-stage grid-free multi-object
reconstruction in physics detectors, graph and image data* (arXiv:2002.03605).
Read in full; this file records what transfers to Papers 1 and 3 and what does not.

## 1. Where the content touches ours

**GravNet's learned space S is the object our Σ⁻¹ replaces.** GravNet maps every
sensor to a low-dimensional coordinate vector S = F_S(x) and weights messages by
V(d_jk) = exp(−d²_jk) in that space; the geometry of "which sensors talk to which" is
*learned*. Paper 1's position is that for a calibrated detector this geometry is
not free: the likelihood fixes the inner product to Σ⁻¹, and for stationary noise
its eigenbasis is Fourier with weights 1/J_k (`prop:stationary-pe`). GravNet is the
"assumed graph" of Bronstein §5.4 with the graph itself learned; Paper 1 is the case
where the graph is measured. That is a one-sentence related-work contrast, and it
is the honest reason our PE result exists: in a noise-dominated readout a learned S
must rediscover 1/J_k or lose to the OF.

**Object condensation is a loss-geometry paper, and the argument style is ours.**
Its content is: pick a potential (attractive/repulsive, charge q = arctanh²β + q_min),
show that the minimiser has the structure you want (one condensation point per
object), and let the network find it. Paper 1 §7 does the same move with a group
instead of a potential: the residual loss invariant under G_noise is unique. The two
are complementary — his is a construction, ours a uniqueness — and the comparison
sharpens our framing: "we do not design the loss; we derive it."

**Object condensation's β/charge is a resolvability weight; ORACLE's Fisher rank is
the frozen-model version.** β is learned confidence that a vertex can represent its
object; ORACLE asks, for a frozen representation, whether the excited tangent span
resolves the perturbation (Fisher-rank abstention). Different objects, same question
("can this element carry the information?"), and his inference thresholds
(t_β ≈ 0.1, t_d) are the operational analogue of our abstention thresholds — with the
same weakness: fixed by hand, reported once, never swept. We can do better by
reporting the sweep (risk–coverage curve), which Tier 1 already does.

**Not related:** grid-free multi-object assignment, calorimeter clustering,
particle-flow candidates. Do not force a connection in the papers.

## 2. Methodology worth copying

1. **Matched-parameter baselines.** Every model in GravNet has ≈100k parameters and
   the table says so. Our Tier-1 `mlp_ae` and the P1-E8c ladder must report the
   parameter count per arm and match it within a few percent, otherwise a referee
   reads "NFPA beats AE" as "small model beats untrained model".
2. **A resource section.** Inference time, memory and their dependence on input
   size (his Fig. 6/Table 3) is a standard section, not an afterthought. Add a
   `resources.csv` per learned experiment and one table per paper.
3. **Inclusive and subset metrics side by side.** He reports μ_L, σ_L, μ_R, σ_R
   inclusive *and* for overlapping showers, because the overlap subset is where the
   methods differ. Ours: report on all events and on the low-SNR / boundary /
   nonstationary subset, with the same metric names.
4. **Extrapolation beyond the training regime.** Object condensation is trained at one
   multiplicity and tested at higher ones, and says so. Our ladders train at SNR 10;
   test at 3 and 30 and say so in the caption.
5. **Hyperparameter table with the values used.** q_min, s_B, s_C, t_β, t_d are in one
   place. Ours: α, k_c, k_t, whitening rank, calibration size, abstention thresholds,
   in one table per paper.
6. **Proof-of-concept on a toy, then the physics application, same metrics.** Object
   condensation: images first, then particle flow against a baseline PF with
   efficiency / fake rate / response / jet resolution. ORACLE's Tier 1 → Tier 2 is
   the same shape; the bridge table must reuse the metric names, and the Tier-2
   baseline must be the community one (NuBench DynEdge), not our own.
7. **Physics quantities as the headline metric.** His headline is jet momentum
   resolution, not a loss value. Paper 1's headline is already truth-referenced
   recovery (0.035 vs 1.021); keep it in physical units (amplitude error / CRB,
   principal angle) and never quote a loss.
8. **Verification is a picture of the mechanism, then a table.** Fig. 3/4 of the
   GravNet paper show what the learned space does before any number. P1-E1 (Figure 1:
   planted vs learned factors) is that picture; P1-E8a (kernel invariance defects) is
   the same for the PE claim.

## 3. Framing and writing

- Both papers state the problem in one paragraph with a physical picture (irregular
  detector; overlapping showers) before any equation. Paper 1's §1 should open with
  the detector, not with the Grassmannian.
- The method section is *constructive*: what the network computes, in order, with
  one figure of the architecture. Paper 1 §6 (NFPA) and ORACLE §3 should each have
  one such figure: (Σ⁻¹, F) as a pipeline, and the frozen-subject / monitor stack.
- Claims are narrow and each has a table. He never claims a general principle from
  one benchmark; the general principle is stated as the reason for the construction,
  and the experiments show the construction works *here*. Our boundary paragraphs
  already do this; do not let the abstract outrun them.
- Limitations are concrete (memory scaling with sensor count; multiplicity beyond
  which the potential saturates). Ours must be equally concrete: calibration size
  below which the metric hurts (P1-E3), SNR below which the amplitude readout is
  noise-dominated (Tier-1 K_recon finding).
- Two to three hyperparameters, all tabulated, all swept once. Not more.

## 4. What we should *not* copy

- Learned geometry as the default. For a calibrated readout, the metric is known;
  Paper 1's point is that learning it is a cost, not a feature.
- Single-threshold operating points reported without the curve.
- Comparisons against the authors' own reimplementation of the baseline only. The
  Tier-2 baseline must be the published DynEdge configuration.
