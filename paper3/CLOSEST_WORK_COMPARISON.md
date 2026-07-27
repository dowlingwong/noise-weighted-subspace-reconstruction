# Closest-work and novelty comparison

This comparison supports proposal review; it is not yet a systematic review.
Citation keys refer to `references.bib`.

## Where the proposal sits

The proposal is at the intersection of (i) public neutrino-telescope event
reconstruction benchmarks, (ii) detector-systematics studies, (iii) common
corruption/distribution-shift benchmarks, and (iv) representation diagnostics.
None of the works below alone establishes novelty. The defensible gap is the
combination of **paired, physically declared input interventions; frozen public
reconstruction checkpoints; conditional task/nuisance diagnostics; internal
representation response; and a separately standardized replication track on a
real reconstruction pipeline**.

| Closest work | What it already establishes | What it does not supply for this proposal | Consequence for our scope |
| --- | --- | --- | --- |
| NuBench [orsoe2026nubench] | Seven open simulated datasets, six geometries, four model families, released checkpoints/predictions, and up to five reconstruction tasks | A benchmark of controlled measurement shifts or paired representation stability | Use its fixed checkpoints and test events; do not rebrand its cross-geometry benchmark as causal geometry attribution |
| GraphNeT / DynEdge [sogaard2022graphnet; orsoe2025graphnet2] | Detector-agnostic graph reconstruction software and public model components | A predeclared perturbation-and-representation protocol | Treat GraphNeT as infrastructure and cite exact release compatibility; the paper's contribution cannot be “we ran DynEdge” |
| IceCube CNN systematics [icecube2021cascade] | Reconstruction can be evaluated against simulation systematic variations and experimental data | A public cross-model representation audit or matched severity-response benchmark | Compare our perturbations with collaboration-standard systematics; avoid claiming systematics evaluation itself is new |
| Robust real-time IceCube reconstruction [sommani2023robust] | Reconstruction and angular uncertainty can be optimized for reduced sensitivity to systematic variations | Frozen-representation diagnostics or a modality-spanning replication contract | Frame our work as diagnosis/measurement, not a superior reconstruction method unless retraining is later added |
| ImageNet-C/P [hendrycks2019corruptions] | Standard severity sweeps and common, non-adversarial corruptions are useful robustness benchmarks | Detector-valid interventions, conditional physics controls, or event-level latent analysis | Borrow the severity-curve logic, not image corruption semantics |
| WILDS [koh2021wilds] | Realistic distribution shifts require standardized datasets, metrics, and evaluation splits | Paired interventions on the same physical event or internal representation analysis | Distinguish observational domain shift from paired controlled perturbation |
| Ovadia et al. [ovadia2019uncertainty] | Predictive uncertainty and calibration may deteriorate under dataset shift | Verification of NuBench checkpoint restoration or detector-specific calibration targets | Evaluate calibration drift separately from software correctness; clean miscalibration can be genuine |
| Guo et al. [guo2017calibration] | Reliability diagrams, ECE limitations, and post-hoc calibration are established tools | Regression/directional calibration and detector-shift semantics | Use task-appropriate coverage/PIT or directional concentration diagnostics, not classification ECE by default |
| CKA [kornblith2019cka] | Representation comparisons require stated invariances; CKA handles rotation/isotropic scale | Paired physical perturbations or task/nuisance meaning | Do not compare raw latent Euclidean distances across models; use standardized paired distances and alignment-aware cross-model measures |
| STIR [nanda2022stir] | Representation robustness can be formalized through shared invariances | A neutrino reconstruction benchmark with declared detector interventions | Cite as conceptual precedent; our novelty must come from the physics contract and evidence, not the generic idea of latent robustness |
| Controlled probing [hewitt2019probes] | Probe accuracy alone can reflect probe capacity; control tasks contextualize decodability | Physics covariate controls or measurement-quality interpretation | Use fixed-capacity probes, held-out splits, label/control nulls, and conditional analyses; do not demand that noise be undecodable |

## Novelty statement suitable for the proposal

> We will test whether frozen neutrino-event reconstruction representations
> preserve task information and calibrated downstream behavior under paired,
> detector-interpretable measurement shifts. Unlike existing reconstruction
> benchmarks, the primary unit is the same event before and after a declared
> intervention; unlike generic corruption benchmarks, nuisance association is
> conditioned on energy, topology, multiplicity, and containment; unlike
> representation-similarity studies, each latent response is tied to a
> downstream reconstruction consequence. NuBench provides a public,
> reproducible development track, while the collaborator pipeline provides a
> separately standardized replication test rather than pooled evidence.

## Claims that would overreach the literature

- “First robustness study in neutrino reconstruction.” Existing IceCube work
  evaluates systematic robustness.
- “First study of representation robustness.” CKA/STIR and a broad literature
  precede this proposal.
- “NuBench proves geometry causes robustness differences.” Its geometries are
  separately simulated and models are separately trained; geometry, medium,
  event distribution, optimization, and checkpoint are entangled.
- “A representation should remove all noise information.” Measurement quality
  may be useful for uncertainty and reconstruction. The test is retained task
  information and conditional stability, not forced nuisance blindness.
- “A null curve is publishable.” A useful null requires a validated positive
  control, adequate power, and a predeclared equivalence margin.

## Literature work remaining before submission

Before manuscript submission, add detector-specific sources for every chosen
perturbation, confirm whether NuBench authors have released follow-up robustness
analyses, and search the target venue for representation/systematics studies
published after this audit date. A NuBench author or detector-domain
collaborator should review the physical interpretation of every transform.
