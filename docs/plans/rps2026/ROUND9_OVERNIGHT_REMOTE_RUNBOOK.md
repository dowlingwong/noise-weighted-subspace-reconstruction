# Round 9 overnight remote runbook

*Fail-closed implementation and execution of the remaining submission evidence on the remote server. Scientific interpretation and manuscript edits remain desk work.*

---

## 🎯 Mission and terminal state

Start from the exact committed Round 9 revision, preserve every archived result,
finish or recover E7, implement and run E8–E10, and return a traceable evidence
bundle. Do not edit the manuscript and do not make the `priorThesis` publication
decision.

The unattended run is successful when either:

1. every eligible experiment completes and its validation gates pass; or
2. a gate fails, downstream work that depends on it stops, independent work
   continues, and the failure is recorded without tuning or threshold changes.

The terminal report must state the status of E6–E10, every failed gate, the exact
commit used, output paths, wall times, and final machine state. A disappointing
scientific result is not a failed run. A data, configuration, reproduction, or
schema mismatch is.

## 🔒 Authority and prohibitions

Authority order:

1. this file for Round 9 execution mechanics;
2. `ROUND9_RUNBOOK.md` for scientific motivation and outcome-to-paper mapping;
3. `ROUND8_RUNBOOK.md` for E7;
4. `AGENT_PROMPT_REMOTE_EXPERIMENTS.md` for shared-server safety;
5. the frozen configuration in `experiments/synthetic/nfpa_confirmatory/config.py`.

Never:

- overwrite or append to `results/rps2026/E1_nfpa_blockA/block_A_colored_raw.csv`;
- edit data under `/ceph/dwong/datasets` or the resolved TIDMAD raw-data tree;
- loosen a tolerance, change a seed, choose a subset, tune shrinkage, or rerun
  because a result is unfavorable;
- pool the original and replication seeds before reporting each subset;
- launch a duplicate E7 if an E7 process, PID, checkpoint, or screen session exists;
- edit LaTeX, figures, tables, the claim audit, or bibliography;
- push directly to `main` or force-push;
- claim that E8 formally confirms H3. The original protocol supplied no
  quantitative recovery threshold, so E8 reports continuous errors and paired
  contrasts; the desk decides the wording.

## 🧭 Ownership and dependency map

```mermaid
flowchart TD
    A[Remote reconnaissance] --> B[E6 artifact verification]
    A --> C{E7 already active or complete?}
    C -->|active| D[Monitor; do not duplicate]
    C -->|complete| E[Validate and summarize E7]
    C -->|absent| F[Launch per Round 8]
    A --> G[Implement E8-E10 runners and tests]
    G --> H[Commit implementation freeze]
    H --> I[E8 archived-fit gate and semantic runs]
    H --> J[E9 isolated replication]
    H --> K[E10 estimated covariance]
    I --> L[Evidence audit]
    J --> L
    K --> L
    B --> L
    D --> L
    E --> L
    F --> L
    L --> M[Remote handoff; no manuscript edits]

    classDef start fill:#dbeafe,stroke:#1d4ed8,color:#172554
    classDef gate fill:#fef3c7,stroke:#b45309,color:#451a03
    classDef work fill:#e0f2fe,stroke:#0369a1,color:#082f49
    classDef stop fill:#fee2e2,stroke:#b91c1c,color:#450a0a
    classDef done fill:#dcfce7,stroke:#15803d,color:#052e16
    class A start
    class C,H gate
    class B,D,E,F,G,I,J,K,L work
    class M done
```

The CPU experiments are independent of E7 and may run at `nice -n 10` with
four BLAS threads while E7 occupies GPUs, provided the initial load and memory
checks show headroom.

## 📋 Phase 0 — repository and machine preflight

Use the actual checkout path; do not assume `~` if the repository lives
elsewhere.

```bash
export CODE_ROOT=/ceph/dwong/noise-weighted-subspace-reconstruction
test -d "$CODE_ROOT/.git" || export CODE_ROOT="$HOME/noise-weighted-subspace-reconstruction"
cd "$CODE_ROOT"

git fetch --all --prune
git status --short --branch
git log -1 --format='%H %cI %s'
nvidia-smi
who
uptime
free -g
df -h /ceph "$CODE_ROOT"
screen -ls || true
ps -u "$USER" -o pid,etime,cmd
```

Required starting commit ancestry: `c1c4b75`. If it is not present, stop and
report `ROUND9_COMMIT_NOT_AVAILABLE`. If the tree is dirty, preserve and report
the changes; do not stash or discard unknown work. Create the remote work branch
on a fresh run, or switch to it when resuming an interrupted run:

```bash
if git show-ref --verify --quiet refs/heads/round9/remote-evidence; then
  git switch round9/remote-evidence
else
  git switch -c round9/remote-evidence
fi
export PYTHON="$CODE_ROOT/.venv/bin/python"
test -x "$PYTHON"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT"
export RPS_RESULTS="$CODE_ROOT/results/rps2026"
```

Run the NFPA correctness suite before implementation. A failure blocks E8–E10
but does not block monitoring an already-running E7.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  timeout 30m nice -n 10 "$PYTHON" -m pytest \
  experiments/synthetic/nfpa_confirmatory/tests -q
```

Record the complete preflight in
`results/rps2026/Round9_remote/preflight.txt`.

## 📦 Phase 1 — reconcile E6 and E7 without duplication

### E6: whitening-range evidence

Locate the existing remote `E6_whitening_range.json`. Copy it into
`results/rps2026/` only if the tracked destination is absent. Do not recompute
unless the artifact is genuinely missing and the exact Round 7 data root and
estimator are available.

Validate all of the following in code:

- `excluding_dc.cv_inverse_J`, not `all_bands.cv_inverse_J`, rounds to the
  manuscript's `0.69`;
- the stored verdict is `collinearity_supported: false`;
- the file records 328 bands before DC exclusion and the frozen PSD estimator;
- the source and destination SHA-256 digests are recorded.

Any mismatch is `E6_VARIANT_MISMATCH`; preserve the file, but do not silently
rename a statistic or update prose.

### E7: second rung-4 seed

Inspect screens, PIDs, checkpoints, logs, and output files before launching
anything. If E7 is active, monitor it. If complete, run the four Round 8 gates
and generate the registered summary. If absent, launch exactly the two
same-seed arms from `ROUND8_RUNBOOK.md` with its timeouts and GPU pinning.

Do not launch `chi2_of`. Do not compare incomplete or unequal step grids. A
failed or timed-out arm is reported as partial evidence, never restarted from
zero when a valid checkpoint exists.

## 🧱 Phase 2 — implement execution-safe E8–E10 drivers

Implementation must be committed before any new full result is inspected.
Use tests and reduced synthetic fixtures only during development.

### Shared output contract

Each runner must:

- accept an explicit output directory;
- refuse to overwrite a non-empty final output directory;
- write one atomic shard per seed before aggregation;
- validate config hash, seed, regime, method set, and schema when resuming;
- use a temporary file plus atomic rename for each shard and summary;
- record the code commit, frozen config hash, Python/NumPy versions, hostname,
  thread limits, start/end timestamps, wall time, and command;
- produce deterministic CSV/JSON summaries from seed shards;
- support `--smoke` without using confirmatory seeds or changing the frozen
  primary configuration;
- exit nonzero on a failed integrity gate.

Add focused tests for overwrite refusal, resume validation, deterministic
aggregation, exact seed sets, and malformed/missing shards.

### E8: semantic completion

Implement `experiments/synthetic/nfpa_confirmatory/run_semantics.py` with two
separate outputs:

1. `primary_alpha_0p1`, seeds 0–9, for semantics at the archived primary
   configuration; and
2. `separable_alpha_0`, seeds 0–9, for the separable condition named by H3.

At `alpha=0.1`, refit the same five registered methods: `CW-PCA`, `NFPA`,
`Iso-MPCA`, `IsoPCA`, and `OF`. The name `PCA` is invalid. Before writing any
semantic result, reproduce every archived `test_normalized_chi2` at the same
seed and method with relative error at most `1e-9`. If any row misses, stop E8
and write an audit table of expected, observed, absolute error, and relative
error. Do not loosen the tolerance overnight.

The semantic utilities must be method-aware:

- `NFPA` and `Iso-MPCA`: latent decoding plus channel, temporal, and joint
  planted-projector errors;
- `CW-PCA` and `IsoPCA`: latent decoding from `fit.P` plus joint planted-product
  projector error; channel/temporal factor errors are null because an
  unrestricted subspace has no identified axis factors;
- `OF`: one-dimensional amplitude decoding only; pulse-shape and planted-factor
  fields are null.

Do not make null mean zero. Tests must cover every method and assert that no
metric is silently fabricated.

For `alpha=0`, use `dataclasses.replace(PRIMARY,
nonseparability_alpha=0.0)` and compute a fresh config hash embedded in every
row. This condition has no archived-fit reproduction gate. Report the
continuous recovery metrics and seed-level paired contrasts; do not invent a
post-hoc threshold or a binary H3 verdict.

Write:

```text
results/rps2026/E8_semantics/
  manifest.json
  primary_alpha_0p1/seed_00.csv ... seed_09.csv
  primary_alpha_0p1/raw.csv
  primary_alpha_0p1/summary.json
  separable_alpha_0/seed_00.csv ... seed_09.csv
  separable_alpha_0/raw.csv
  separable_alpha_0/summary.json
  archived_fit_reproduction.csv
```

Bootstrap only across seeds, 10,000 resamples, seed `20260819`. Report NFPA
minus Iso-MPCA contrasts for amplitude RMSE, rise-time RMSE, decay-time RMSE,
and the available projector errors. Keep all individual seed values.

### E9: isolated 20-seed replication

Implement a dedicated runner or an explicit output-path option. It must execute
colored Block A for seeds exactly `10..29`, all five methods, with the unchanged
frozen `PRIMARY` configuration. It must never resolve to the E1 directory.

Write seed shards and final output under:

```text
results/rps2026/E9_blockA_replication/
  seed_10.csv ... seed_29.csv
  replication_raw.csv
  replication_summary.json
  pooled_summary.json  # only when the direction-agreement rule permits it
  manifest.json
```

The replication summary is generated first and contains the three registered
paired contrasts. Compare their signs with seeds 0–9. If either primary
NFPA−Iso-MPCA contrast reverses sign, mark `pooling_permitted: false`, omit
pooled estimates, and report both subsets. If directions agree, the pooled
summary may be generated but must display original, replication, and pooled
results side by side and label the latter two post-hoc.

### E10: estimated covariance

Implement `run_estimated_covariance.py` using the following frozen choices;
the remote agent has no discretion to substitute another estimator:

- seeds 0–9 and `N_cal=2048` independent noise-only events per seed;
- calibration RNG initialized from
  `numpy.random.SeedSequence([20260819, seed, 10])`, separate from dataset RNG;
- generate calibration noise from the oracle colored model, with no signal
  events and no train/validation/test observations;
- estimate the channel covariance from rows shaped `(N_cal*T, C)` and the
  temporal covariance from rows shaped `(N_cal*C, T)` using
  `noise_geometry.noise.estimate_covariance(center=True)`;
- apply `regularize_covariance(shrinkage=0.0, floor=None)` to each axis: the
  existing data-scaled eigenvalue floor is the only regularization;
- generate train/validation/test data with the oracle covariance, fit NFPA
  using the estimated axis covariances, and reconstruct through those same
  estimated covariances;
- evaluate the resulting reconstruction only with the oracle covariance and
  the unchanged held-out test set;
- refit `Iso-MPCA` on the identical data as the covariance-independent control;
  it must reproduce its archived primary metrics within the E8 reproduction
  tolerance.

Report per-axis covariance error, minimum eigenvalue, condition number,
oracle-whitening residual, NFPA and Iso-MPCA held-out raw MSE and normalized
chi-square, and seed-paired NFPA−Iso-MPCA contrasts. Write only to
`results/rps2026/E10_estimated_covariance/` using the shared shard/manifest
contract.

This is one fixed estimated-covariance condition, not the full originally
proposed covariance-robustness sweep. The paper may call it a measured
sensitivity check, not a general robustness boundary.

### Implementation freeze

Run the new focused tests plus the existing NFPA suite. Then inspect the diff
for scientific choices and commit before launching the ten- or twenty-seed
runs:

```bash
git diff --check
git status --short
git add experiments/synthetic/nfpa_confirmatory docs/plans/rps2026
git commit -m "round9: freeze output-safe semantic and covariance runners"
git log -1 --format='%H %cI %s'
```

Record this as `implementation_freeze_commit` in every new manifest. If tests
do not pass, do not execute E8–E10.

## 🚀 Phase 3 — unattended execution order

Use `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` and
`nice -n 10` for every CPU job. Give every command a timeout and capture its
return code. Use a named `screen` session with logging for the overnight CPU
sequence; do not use untracked `nohup` processes.

Run in this order:

1. E8 one-seed primary reproduction gate. If it fails, stop E8 only.
2. E8 full `alpha=0.1`, then `alpha=0`.
3. E9 seeds 10–29.
4. E10 seeds 0–9.
5. deterministic summaries and evidence audit.

E9 and E10 may continue if E8 fails because they do not depend on semantic
code. A shared infrastructure or frozen-config failure blocks all three.

Suggested hard caps, to be replaced only by larger caps justified from a
recorded one-seed timing (never smaller scientific budgets):

| Work | Expected | Hard cap |
|---|---:|---:|
| E8 primary, 10 seeds | 40–60 min | 2 h |
| E8 separable, 10 seeds | 40–60 min | 2 h |
| E9 replication, 20 seeds | 75–120 min | 4 h |
| E10 estimated covariance, 10 seeds | 40–75 min | 3 h |
| E7, if not already complete | 4.5–6 h on two GPUs | 8 h per arm |

The timing basis for E8/E9 is the archived Block A run: approximately 226 s
per seed across all five methods on the prior server execution. Record new
measurements rather than assuming that rate.

## 🧪 Phase 4 — evidence audit

Before reporting completion, independently verify:

- archived E1 SHA-256 is unchanged from the preflight digest;
- exact seed sets: E8 `{0..9}`, E9 `{10..29}`, E10 `{0..9}`;
- no duplicate `(seed, method, condition)` keys;
- expected row counts and method sets;
- config hashes agree within each condition and differ when `alpha` differs;
- summaries regenerate byte-for-byte from raw shards;
- E8 bootstrap unit is seed, not event;
- E9 replication is reported before any permitted pool;
- E10 evaluation uses oracle covariance and calibration data are independent;
- E7 arms, if compared, have identical seed, data root, model parameters,
  budget, and step grid, differing only in loss;
- every long-running process has exited and no stale PID or screen remains.

Run `git diff --check`, the NFPA tests, and any new Round 9 tests once more.
Do not run the paper build remotely as a substitute for desk reconciliation.

## 📤 Phase 5 — handoff

Append a factual Round 9 section to
`docs/plans/rps2026/EXPERIMENT_RUN_LOG.md`. Create
`docs/plans/rps2026/ROUND9_REMOTE_REPORT.md` with:

- start, implementation-freeze, and final commit hashes;
- machine and environment summary;
- E6 variant verification;
- E7 state and gate results;
- E8 archived-fit gate plus both semantic conditions;
- E9 original-versus-replication direction comparison and pooling decision;
- E10 estimator diagnostics and paired contrasts;
- all commands, return codes, wall times, output paths, and SHA-256 digests;
- failures and skipped dependents;
- final `nvidia-smi`, `screen -ls`, relevant `ps`, and repository status;
- a short `DOES_SUPPORT` / `DOES_NOT_SUPPORT` section without manuscript prose.

Commit code, small evidence, logs, and reports to `round9/remote-evidence`.
Do not include checkpoints, multi-gigabyte denoised files, or raw datasets.
Push only that branch. The final line must be one of:

```text
ROUND9_REMOTE_COMPLETE_PENDING_DESK_RECONCILIATION
ROUND9_REMOTE_PARTIAL_WITH_RECORDED_GATE_FAILURES
ROUND9_REMOTE_BLOCKED_BEFORE_EXECUTION
```

## 🖥️ Desk-only work after sync

The desktop author/revision agent must:

1. audit the remote branch and independently regenerate summaries;
2. apply the prewritten E7 contingency matching the measured sign result;
3. decide E8 wording, respecting the absent predeclared recovery threshold;
4. update the manuscript, appendix, figure/table, claim audit, source map, and
   review only from validated artifacts;
5. decide `priorThesis`: the recommended submission-safe choice is to remove
   the unevaluated OF-injection compatibility sentence and citation unless a
   stable reviewer-accessible anonymized source can be supplied;
6. rebuild the four-page paper and rerun anonymity, citation, traceability,
   undefined-reference, and page-limit checks;
7. merge the remote branch only after this audit.
