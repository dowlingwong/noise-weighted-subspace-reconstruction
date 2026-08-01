# Research-move library

Use these as composable operators. Select by structural fit to the diagnosed bottleneck. A candidate must produce the listed artifact and defend against the listed failure; merely naming a pattern does nothing.

## Contents

1. Diagnostic and reframing moves
2. Construction and representation moves
3. Decomposition, adaptation, and unification moves
4. Composition guidance

## 1. Diagnostic and reframing moves

### Audit and pivot a load-bearing assumption

- **Fit:** Existing methods share a necessary-looking assumption that may be relaxed, relocated, or violated.
- **Artifact:** A weaker condition, counterexample, exploit, or re-derived guarantee showing the assumption was binding.
- **Differentiate by:** Demonstrating that the new condition holds in the target setting and that the outcome changes because of the pivot.
- **Failure:** Replacing one strong assumption with another, relaxing a non-binding condition, or presenting an isolated exploit without an adaptive test or mechanism.

### Reframe the unsolved as a solvable object

- **Fit:** The native formulation is intractable, ill-posed, or hides a tractable surrogate with preserved semantics.
- **Artifact:** A new object or formulation plus a mapping that preserves the property that matters.
- **Differentiate by:** Proving what the reformulation retains, loses, or makes computable.
- **Failure:** Renaming the problem, optimizing a loose proxy, or regressing to a formulation an older method already solved.

### Design a confound-isolating diagnostic

- **Fit:** Competing explanations produce the same aggregate metric or benchmark result.
- **Artifact:** A controlled intervention, item set, or measurement instrument varying one causal factor while holding alternatives stable.
- **Differentiate by:** Showing that the diagnostic identifies a failure hidden by existing evaluations.
- **Failure:** A new benchmark with uncontrolled difficulty, label leakage, weak construct validity, or no link from the diagnostic to the claimed mechanism.

### Characterize a limit, then surpass it

- **Fit:** A formal or empirical boundary is suspected but not sharp.
- **Artifact:** A lower bound, impossibility result, scaling law, or boundary characterization paired with a method that strictly improves the relevant regime.
- **Differentiate by:** Matching upper and lower evidence or isolating the exact condition under which the limit changes.
- **Failure:** Beating a loose bound, changing the setting so the comparison is invalid, or claiming generality from a thin regime.

## 2. Construction and representation moves

### Substitute the operator or representation

- **Fit:** A standard operator or representation creates the bottleneck through sensitivity, scaling, locality, or inductive bias.
- **Artifact:** A replacement operator or encoding with a clear property-level advantage.
- **Differentiate by:** Isolating the property gained and controlling parameter count, compute, and surrounding architecture.
- **Failure:** Architecture swapping without causal isolation, novelty-by-component, or trading one unresolved cost for another.

### Encode structure by construction

- **Fit:** The target has known symmetry, topology, invariance, conservation, or compositional structure that generic learning must rediscover.
- **Artifact:** An architecture, parameterization, or constraint that guarantees or strongly biases the desired property.
- **Differentiate by:** Showing the encoded structure does identifiable work over strong generic alternatives.
- **Failure:** Adding an ornate prior with no necessity test, mismatching the assumed structure to data, or attributing gains to topology without isolation.

### Liberate a fixed generative component

- **Fit:** A generative pipeline treats a schedule, prior, corruption process, decoder, or transition as fixed even though it determines performance.
- **Artifact:** A learned, adaptive, or redesigned component with a controlled interface to the rest of the process.
- **Differentiate by:** Identifying why the fixed choice is limiting and showing the new freedom does not destroy stability or identifiability.
- **Failure:** Extra flexibility without a mechanism, uncontrolled training cost, or gains explainable by capacity alone.

### Manufacture the supervisory signal

- **Fit:** Direct labels, rewards, or annotations are missing, costly, delayed, or misaligned.
- **Artifact:** A derived signal from invariance, agreement, ordering, structure, perturbation, or process traces.
- **Differentiate by:** Establishing why the signal tracks the target and where it fails.
- **Failure:** Proxy optimization, self-confirming pseudo-labels, leakage, or a signal whose cost rivals direct supervision.

### Design a property-targeting pretext objective

- **Fit:** A representation should acquire a specific downstream property not encouraged by generic self-supervision.
- **Artifact:** A pretext task whose optimal behavior entails or predicts that property.
- **Differentiate by:** Connecting the objective to the property through analysis or targeted diagnostics.
- **Failure:** Another auxiliary loss with no causal bridge, benchmark tuning, or gains confined to one evaluation.

## 3. Decomposition, adaptation, and unification moves

### Unify heterogeneous inputs in one space

- **Fit:** Modalities, domains, schemas, or populations cannot interact because their representations are incompatible.
- **Artifact:** A shared space with explicit alignment, preservation, and missing-modality behavior.
- **Differentiate by:** Demonstrating what becomes possible after unification beyond generic embedding similarity.
- **Failure:** Alignment collapse, erasure of modality-specific information, or another contrastive objective without a new capability.

### Prove equivalence to unify

- **Fit:** Seemingly different objectives, algorithms, or probabilistic views may be transformations of one underlying object.
- **Artifact:** An exact equivalence, correspondence, or common objective that transfers guarantees or algorithms.
- **Differentiate by:** Using the equivalence to obtain a result unavailable from either view alone.
- **Failure:** A known algebraic restatement, equivalence under unrealistic assumptions, or no downstream consequence.

### Decompose for differentiated treatment

- **Fit:** A population, error, signal, or objective is heterogeneous and uniform treatment creates interference.
- **Artifact:** A principled partition with component-specific operations and a recombination rule.
- **Differentiate by:** Showing why the partition corresponds to distinct mechanisms rather than convenient buckets.
- **Failure:** Hand-designed subgrouping, unstable routing, fairness regressions, or gains from extra capacity rather than differentiated treatment.

### Decompose and delegate to solvers

- **Fit:** A complex task contains subproblems requiring different tools, models, or reasoning modes.
- **Artifact:** A decomposition criterion, routing policy, specialist interfaces, and aggregation or verification mechanism.
- **Differentiate by:** Making delegation itself reliable and showing it beats a strong monolithic or generic-agent baseline.
- **Failure:** Agent theater, unbounded orchestration cost, error propagation, or a pipeline that is merely the sum of off-the-shelf parts.

### Relax discrete search to a continuous problem

- **Fit:** Combinatorial choices dominate cost but admit a useful differentiable or convex surrogate.
- **Artifact:** A relaxation plus a rounding, projection, or recovery procedure.
- **Differentiate by:** Bounding the relaxation gap or identifying regimes with exact recovery.
- **Failure:** Optimizing an easy surrogate that rounds poorly, hiding search cost in tuning, or comparing against weak discrete methods.

### Adapt by conditioning, not retraining

- **Fit:** A mature model must adapt repeatedly across contexts while full updates are costly, risky, or unavailable.
- **Artifact:** A conditioning interface, context code, prompt, adapter state, or test-time signal that changes behavior without retraining the core.
- **Differentiate by:** Preserving base capabilities while establishing controllability and low adaptation cost.
- **Failure:** Moving training into the conditioner, brittle context dependence, contamination, or weaker performance than a simple lightweight update.

## 4. Composition guidance

Prefer a coherent two-move chain when the bottleneck supports it:

- **Diagnose then construct:** audit an assumption + substitute an operator; isolate a confound + manufacture a signal.
- **Reframe then solve:** reframe as a solvable object + relax discrete search; characterize a limit + encode structure.
- **Partition then specialize:** decompose for differentiated treatment + delegate to solvers.
- **Unify then exploit:** prove equivalence + liberate a fixed component; unify inputs + design a targeted pretext objective.

Use three moves only when each has a distinct role. Reject compositions that read as a component bundle, lack one shared thesis, or require several independent papers to validate.
