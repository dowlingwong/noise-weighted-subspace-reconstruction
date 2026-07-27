# NuBench feasibility audit

Audit date: 2026-07-27. Release: NuBench 1.1 / GraphNeT 1.8.0 artifacts.
Status: the smallest direction-reconstruction cell is verified through
checkpoint restoration and a paired development pilot. The strict clean
identity gate remains open; see `results/nubench_hexagon_ice_le_dynedge/`.

## Decision

Use **Hexagon Ice LE × DynEdge × direction** for V0 and the first pilot. It is
the smallest raw archive (6.38 GiB compressed), has a published direction
result, supplies event-level released predictions, and uses the same task as
all four benchmarked architectures. It is a feasibility cell, not a claim that
DynEdge or this geometry is representative of the whole benchmark.

## Dataset and artifact inventory

Sizes below are exact HTTP `Content-Length` values converted to GiB
(2^30 bytes). The SQLite and Parquet columns are alternative raw-data
deliveries; predictions and models are separate archives.

| Dataset | Medium / energy | Events | SQLite GiB | Parquet GiB | Predictions GiB | Models GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cluster | water / 10–100,000 GeV | 22.9 M | 174.34 | 102.70 | 1.59 | 1.16 |
| Flower L | water / 10–100,000 GeV | 24.0 M | 92.71 | 64.18 | 1.52 | 1.16 |
| Flower S | water / 10–1000 GeV | 20.5 M | 277.43 | 178.57 | 1.12 | 0.85 |
| Flower XL | water / 10–100,000 GeV | 10.1 M | 78.50 | 58.27 | 1.62 | 1.10 |
| Hexagon | water / 10–100,000 GeV | 20.5 M | 127.32 | 89.05 | 1.57 | 1.11 |
| **Hexagon Ice LE** | **ice / 10–1000 GeV** | **8.6 M** | **6.38** | **4.75** | **1.35** | **1.09** |
| Triangle | water / 10–100,000 GeV | 23.1 M | 42.54 | 27.04 | 1.61 | 1.16 |

Total download for the selected SQLite route is 8.82 GiB compressed (SQLite,
predictions, and all model artifacts). Selective extraction occupies about
17.84 GiB for `merged.db`, 0.38 GiB for the DynEdge prediction Parquet, and
5.7 MiB for the chosen pickled checkpoint. The model archive is not split by
checkpoint, so the full 1.09 GiB archive must be downloaded once.

## Checkpoint, task, and geometry availability

Task codes: D = direction, E = energy, T = track/cascade topology, V = vertex,
I = visible inelasticity. `D only` for DeepIce reflects its benchmark scope.
The matrix reports paper/release availability, not a complete Cartesian
product guaranteed by the repository.

| Geometry | ParticleNeT | DynEdge | GRIT | DeepIce | Important qualification |
| --- | --- | --- | --- | --- | --- |
| Cluster | D,E,T,V,I | D,E,T,V,I | D,E,T,V,I | D only | five-task cell |
| Flower L | D,E,T,V,I | D,E,T,V,I | D,E,T,V,I | D only | five-task cell |
| Flower S | D,E,T,V,I | D,E,T,V,I | — | D only | GRIT was not trained |
| Flower XL | D,E,T,V,I | D,E,T,V,I | E,T,V,I; D failed/not compared | D only | GRIT direction convergence issue |
| Hexagon | D,E,T,V,I | D,E,T,V,I | D,E,T,V,I | D only | five-task cell |
| **Hexagon Ice LE** | **D,E,T,V** | **D,E,T,V** | **D,E,T,V** | **D only** | I is not defined/released for this sample |
| Triangle | D,E,T,V,I | D,E,T,V,I | D,E,T,V,I | D only | five-task cell |

Archive inspection of the selected cell confirms twelve ParticleNeT, DynEdge,
and GRIT task artifacts (D/E/T/V) plus the DeepIce direction artifact. The
selected DynEdge files are:

- `training_hexagonice_full_direction_model.pth` (trusted official pickle),
  SHA-256 `8f04373407d78b5cea43fcc1a3be3a878c3b43f28a31209622a8d835ae4c4163`;
- configuration SHA-256
  `6e73fc73af1caa791ccbb08e735b990a30732ecc1d696e9c13e4f78c79eb8d36`;
- state dictionary SHA-256
  `f27672adb0072241ee65ea717e2149d124d074e2a3673b3f0dfd09d60580e160`;
- released prediction file SHA-256
  `13a50cc7a8800e4ac05aa1c65af83ba310c31cbbe9a3bb1886c71873e835d4ad`;
- SQLite archive SHA-256
  `1973d899cc1a0c99638ac90c6f1513338d0297d1514067f5e056ff604427b406`.

## Selected data contract

| Field | Verified value |
| --- | --- |
| Raw pulse table | `pulses_no_noise`, 89,619,466 rows |
| Truth table | `mc_truth`, 8,614,339 events |
| Released test predictions | 2,989,339 events, three row groups |
| Input features | sensor x/y/z, charge, time |
| Test input | noise-free pulses; no training-time stochastic augmentation |
| Training augmentation in config | time jitter σ=1 ns; charge jitter σ=0.25 p.e. |
| Graph construction | percentile-cluster nodes; 8-nearest-neighbor edges |
| DynEdge size | 1,358,099 trainable and frozen parameters restored |
| Task output | Cartesian direction plus concentration κ |

The paper and official current GraphNeT source specify the coordinate/feature
standardization. The checkpoint embeds the same constants, but its custom
detector's Python 3.10 bytecode does not execute correctly under the available
Python 3.11 release container. The smoke script therefore reconstructs the
official transforms explicitly and tests against released predictions.

## V0 clean-reproduction gate

| Field | Decision/evidence |
| --- | --- |
| Cell | Hexagon Ice LE / direction / DynEdge |
| Published metric | Table 6 median opening angle: 23.60° for CC/track and 56.12° for NC/cascade |
| Secondary values | ≤1°: 2.26% / 0.04%; ≤5°: 22.24% / 1.03% |
| Reproduction tolerance | absolute 0.01° for reported medians and 0.01 percentage point for threshold rates |
| Full-file released-prediction reproduction | 23.6548° / 56.1553°; maximum table discrepancy 0.055° or 0.051 percentage point |
| Checkpoint restoration | pass; official pickle loads as `StandardModel`, 1,358,099 parameters |
| Checkpoint-versus-release subset identity | 256 events: median 0.944°, p95 7.708°, max 21.834° direction disagreement |
| Gate | **open / strict tolerance failed**; author/version/backend clarification required |

The checkpoint predates a GraphNeT package refactor. Three historical module
paths must be aliased to their GraphNeT 1.8.0 locations before loading, and the
embedded older-Python detector methods must be replaced by the matching
official source implementation. This is an implementation-compatibility fact,
not evidence of model miscalibration.
Because `torch.load`-style pickles can execute code, only the checksummed
official release artifact is accepted.

## Embedding hook

| Field | Contract |
| --- | --- |
| Extraction point | output of `StandardModel.backbone` before `DirectionReconstructionWithKappa` |
| Meaning | pooled event-level DynEdge representation |
| Shape | event × 128 (verified on restored model) |
| Hook | PyTorch forward hook on the named backbone module |
| Scale handling | per-coordinate center/SD fitted on clean development events; report RMS-standardized paired displacement |
| Rotation/scale-safe comparison | paired standardized displacement and k-NN retention; use CKA/Procrustes only for cross-model representation comparisons |
| Zero-severity contract | same event order, released clean metric, checkpoint output, and embedding within numerical tolerance |

This hook is architecture-specific. A future cross-architecture study needs an
explicit event-level hook and pooling contract for every model.

## Compute feasibility

| Resource | Measured or bounded value |
| --- | --- |
| Environment | official GraphNeT 1.8.0 CPU Apptainer image; PyTorch 2.6.0+cpu; PyG 2.7.0 |
| Local accelerator | none visible during audit |
| Development size | 256 events, balanced 128 CC/track and 128 NC/cascade |
| Pilot severities | module dropout 0, 0.10, 0.25, 0.50 |
| Pilot output/runtime | 256 paired events; 3.0–5.8 CPU s per severity (44–86 events/s) |
| Full V0 storage | about 28 GiB after selective extraction plus compressed archives |
| Full benchmark storage | not approved; all SQLite archives alone total 799.2 GiB compressed |
| Confirmatory compute | must be projected from measured CPU/GPU throughput before registration |

The development pilot is deliberately CPU-capable. Confirmatory inference over
millions of events should use a GPU allocation and cache selected events in a
compact, immutable shard; it should not repeatedly query the 19 GB SQLite file.

## First perturbation validity: observed-module dropout

For each event, identify unique observed sensor positions, draw a deterministic
fraction without replacement, remove all pulses on those modules, and rebuild
the percentile-cluster graph. At least nine observed modules are retained so
the 8-NN graph remains defined. The random seed is a function of the global
seed, event ID, and severity, making reruns exact.

This is a controlled sensor-availability stress test. It does **not** emulate a
specific hardware failure rate or claim a causal detector-design effect. It is
outside the checkpoint's listed charge/time-jitter augmentation and therefore
serves as a simple positive-control shift. Severity zero is the identity. The
pilot reports paired downstream angular error, prediction drift, standardized
embedding displacement, and 10-nearest-neighbor retention.

## Go/no-go

Proceed to a registered E1 only after (1) the full released metric is within
tolerance, (2) clean checkpoint outputs agree with released predictions on the
selected events, (3) severity zero is exactly identity-preserving, (4) the
positive-control severity is detectable, and (5) projected confirmatory
storage and accelerator time are approved. A failed matched-noise calibration
test is a scientific result only after these implementation checks pass; it is
not automatically a software bug.
