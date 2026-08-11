# Agent prompt — remote experiment execution (plain Linux server)

*Lives in the code repository at `docs/plans/rps2026/`. Everything you need is in
this one repository — you do not need the paper repository.*

---

## Your task

Execute the experiments for a NeurIPS 2026 workshop paper on a **shared Linux GPU server**. Your task list is `docs/plans/rps2026/EXPERIMENT_CHECKLIST_2026-08-08.md`. Read it fully before doing anything.

Supporting documents, all in `docs/plans/rps2026/`, in authority order:

1. `EXPERIMENT_CHECKLIST_2026-08-08.md` — **your task list**: environment, prerequisites, experiments, gates, order.
2. `EXPERIMENT_PLAN_2026-08-08.md` — NFPA protocol definitions (E1, E2).
3. `PLAN_03_RUNG4_TIDMAD_2026-08-08.md` — rung-4 design and acceptance gates (E3–E7).
4. `ROUND2_CORRECTION_2026-08-08.md` — outstanding corrections; §5 architecture decisions.
5. `PLAN_04_TIDMAD_CODE_PATCH_2026-08-08.md` — how the package is put together.
6. `TIDMAD_PATCH_LOG.md` — what the code patch already established.

**One repository. Everything is here.**

| Path | Contains |
|---|---|
| `src/tidmad/` | the rung-4 experiment (data, psd, model, loss, train, infer, score) |
| `src/noise_geometry/` | Paper 1 machinery the experiment imports |
| `experiments/synthetic/nfpa_confirmatory/` | the NFPA harness (E1, E2) |
| `scripts/train_tidmad_local.sh` | the training entry point |
| `docs/plans/rps2026/` | this prompt and every plan document |
| `results/rps2026/` | **where your outputs go** |

`python -m tidmad.*` requires `PYTHONPATH=src` from the repository root — the
driver script sets it for you.

> **Results go to `results/rps2026/` and nowhere else.** The repository's
> `.gitignore` excludes `results/*` generally, with an explicit exception for
> `results/rps2026/**`. Anything written elsewhere under `results/` is silently
> excluded and **will never sync back** — you would finish the work and lose it.

There is no batch system and no HTCondor material; anything you may have read
elsewhere about `submit_ref/` or `run.sh` refers to files that no longer exist.

---

## THE SAFETY RULES — read before anything else

There is **no batch system**. Nothing will enforce a walltime, reserve a GPU for you, queue you behind other users, or clean up after a crash. Every limit is one you impose yourself. This makes discipline more important here than on a scheduled cluster, not less.

**1. Find out who else is on the machine before you take anything.**

```bash
nvidia-smi                  # which GPUs are busy, and whose processes
who                         # who else is logged in
uptime                      # load average
free -g                     # RAM headroom
df -h /ceph /home .         # disk headroom
```

If another user's process is on a GPU, **that GPU is not yours.** Do not assume idle-looking memory means idle.

**2. Pin yourself to exactly one GPU.** Never let a job see all of them:

```bash
export CUDA_VISIBLE_DEVICES=<one free index>
```

Set it in the same command as the run so it cannot be forgotten. Taking every GPU on a shared box is the fastest way to make enemies.

**3. Every long-running command gets a hard wall-clock cap.** There is no scheduler to kill a runaway:

```bash
timeout 6h nice -n 10 python -m tidmad.train ...
```

`timeout` is not optional. A training loop with a bad learning rate will otherwise sit on a GPU until someone notices. Pick the cap from measured throughput (rule 6), not optimism.

**4. Long runs go in a named `screen` session — use one deliberately, and clean it up.**

Unlike a cluster, you *do* need the process to survive disconnect. Use exactly one `screen` session per experiment, named for the experiment, with logging on:

```bash
screen -S rps_t_mse -L -Logfile screenlog_t_mse.txt ./scripts/train_tidmad_local.sh mse
# detach: Ctrl-A then D
screen -ls                        # list — must show only sessions you created
screen -r rps_t_mse               # reattach
screen -S rps_t_mse -X quit       # clean up when done
```

`train_tidmad_local.sh` writes its PID to `<out_dir>/run.pid` and removes it on exit, so anyone — you tomorrow, or the author — can find and stop the run:

```bash
kill $(cat artifacts/tidmad/t_mse/<stamp>/run.pid)
```

**Never** use bare `nohup ... &` with no session, no PID file and no log. That is precisely the zombie this rule prevents. One session per experiment, killed when finished. **`screen -ls` must be empty before you report done.**

**5. Bound your resource use.** Dataloader workers, batch size and memory are yours to limit:

- Keep `num_workers` modest (≤ 4). Unbounded workers can OOM the machine for everybody.
- Check RSS a few minutes in: `ps -o pid,rss,etime,cmd -p $(cat run.pid)`.
- Run under `nice -n 10` so interactive users are not starved.
- If `free -g` shows the box near its limit, stop. Do not let the OOM killer decide which process dies.

**6. Checkpoint often — nothing requeues you.** If a run dies at hour five with no checkpoint, five hours are gone. Verify the checkpoint period is set and that a checkpoint actually appears on disk early in the run, before you walk away.

**7. Measure before you commit.** `num_steps = 150_000` may be hours or days. Smoke-test, record **measured steps/second**, and compute the wall-clock estimate before launching the full run. If the estimate exceeds your `timeout` cap, reduce steps or plan resumable segments. Never launch a run whose duration you have not estimated.

**8. One experiment at a time until one has succeeded end to end.** Do not launch E1–E7 in parallel on first attempt. Launch one, watch it, inspect the output, then the next.

**9. Never modify anything under `/ceph/dwong/datasets`.** Read-only input. Write only to your own output paths.

**10. Leave the machine as you found it.** Before finishing: no processes of yours on any GPU, no screen sessions left running, no stale PID files. Verify with `nvidia-smi`, `screen -ls`, `ps -u $USER`.

**11. If you are unsure whether an action affects other users, stop and ask.** Waiting costs hours. A wedged or OOM-killed shared box costs your colleagues days.

---

## Order of work

Follow the checklist's §4 table. Nothing here overrides it.

### Phase 0 — Reconnaissance. Change nothing.

Record in your log, before touching anything:

- hostname, OS, kernel, CUDA and driver version;
- `nvidia-smi` in full — GPU count, model, memory, **and every process currently on them with its owner**;
- `who`, `uptime`, `free -g`;
- disk: free space and quota on `/ceph`, `/home`, and the repo's filesystem;
- Python version, whether `uv` is present, whether `.venv` resolves;
- any pre-existing process of yours (`ps -u $USER`) — if there is one you did not knowingly start, stop and report.

**Gate:** the log records the starting state of a machine you have not yet modified.

### Phase 1 — Environment and data

Per checklist §0. Set `TIDMAD_DATA_ROOT=/ceph/dwong/datasets/tidmad/raw` and companions; never hard-code a path.

Run the §0.2 integrity check. Confirm `channel0001` / `channel0002` exist. **If the channel names differ, stop** — the data contract no longer holds and every downstream number is suspect.

Resolve the §0.1 path-provenance decision (symlink from `/ceph/dwong/paper1_dataset`, or a `path_migration` note in every new run record) and state which you chose. Paper 1's archived numbers are cited in the manuscript; their provenance must stay checkable.

Then set up the environment (`uv sync`, activate `.venv`) and **smoke-test before anything real**: a handful of steps, confirming the loop runs, a checkpoint lands on disk, and — critically — **recording measured steps/second**. Rules 3 and 7 depend on that number.

### Phase 2 — Blocking prerequisites

**P1 — characterise injection frequencies for all 20 validation files.** Cheap FFT scan; wrap in `timeout 30m`. This decides whether the 0–800 kHz band coverage is valid, and therefore whether the published-baseline comparison in Table 1 survives. Record the decision with evidence.

**P2 — reproduce the `none` anchor** through the unmodified upstream `benchmark.py`. Gate: 1.1026 or a documented equivalent. If it does not reproduce here, **stop** — environment or channel semantics differ from Paper 1's, and every downstream number is suspect.

**P3 — architecture decisions** in `ROUND2_CORRECTION` §5 / checklist §5: stacking real/imaginary as features rather than bands (D1), and the near-vestigial 12-position temporal axis (D2). Both change the frozen config, so settle them now. These are code changes; smoke-test after each.

**P4 — verify the training loop is resumable.** The five gaps documented in checklist §2 P4 have been fixed: checkpointing with a SIGTERM handler, `--resume auto`, gradient accumulation over `device_batch_size`, held-out evaluation on the chronological split, and steps/second logging. **You still have to prove it on this machine.** Gate: a smoke run writes a checkpoint, is killed mid-run, resumes from it, and reports steps/second.

```bash
./scripts/train_tidmad_local.sh smoke               # confirm a checkpoint lands
# interrupt, then:
TIDMAD_RESUME=auto ./scripts/train_tidmad_local.sh smoke
```

Safety rules 3, 6 and 7 all depend on this passing. Do not launch a full arm until it does.

Use `./scripts/train_tidmad_local.sh` as the entry point; it handles GPU pinning, the GPU and data preflight, `PYTHONPATH`, `nice`, the `timeout` cap, the PID file and log capture. Profiles: `check`, `smoke`, `mse`, `chi2`, `chi2_of`. Env: `GPU_INDEX`, `TIDMAD_TIMEOUT`, `TIDMAD_RESUME=auto`, `TIDMAD_WANDB=1`.

### Phase 3 — Freeze

Timestamped commit of the frozen configuration, **before any aggregate result is inspected**. This is what makes the experiment predeclared rather than retrospective.

### Phase 4 — Runs

E1/E2 (NFPA, CPU-bound, cheap) and E3/E4 (rung-4 arms, GPU) are independent. Even so, obey rule 8: one at a time until one has succeeded end to end.

Each GPU run: one named `screen` session, `CUDA_VISIBLE_DEVICES` pinned, `timeout` cap set from measured throughput, `nice -n 10`, PID file written, checkpointing verified.

**The load-bearing gate for E3/E4:** the two `run_config.json` files must differ **only** in `train.loss`. The assertion is implemented; confirm its output appears in the log. If anything else differs, the contrast is worthless.

**If a gate fails, report it — do not tune to pass.** A null rung-4 contrast is a publishable finding. Only one outcome triggers escalation: metric reversal failing at the *linear* rungs, which would contradict a theorem and means a bug.

### Phase 5 — Results, for sync-back

**This determines whether the work survives the round trip. Get it right.**

Small, tidy results go to **`$RPS_RESULTS`** — the one tracked results path:

```
$RPS_RESULTS/<experiment_id>/            # = <repo>/results/rps2026/<experiment_id>/
    run_config.json
    metrics.json          # or tidy CSV, one row per (seed|arm, method, condition, split)
    benchmark_stdout.txt  # rung-4: raw output of the unmodified scoring script
    run.log               # trimmed training log — not gigabytes
    figures/              # generated by script, never by hand
```

Large artifacts stay on the server and are **referenced, not copied**: checkpoints under `$CODE_ROOT/artifacts/`, denoised `.h5` under `$TIDMAD_DENOISED_DIR`. Record their absolute paths in `run_config.json`. Multi-gigabyte files in either repo will make the sync fail or crawl.

Every record carries: git commit, config hash, seed, realised SNR where applicable, array shapes, data checksums, the resolved dataset root, and the `path_migration` note.

Tables and figures are generated from these files by a deterministic script. **Never transcribe a number by hand.**

**Before you finish:**

```bash
du -sh $RPS_RESULTS                                  # MB, not GB
cd $CODE_ROOT && git status --short                  # what will sync back
nvidia-smi                                           # no processes of yours
screen -ls  # no sessions left
ps -u $USER                                          # nothing unexpected
```

---

## Deliverable

`docs/plans/rps2026/EXPERIMENT_RUN_LOG.md`, inside the repo:

1. Phase 0 reconnaissance — the machine's starting state, including who else was using it.
2. Measured steps/second and the wall-clock arithmetic that justified each `timeout` cap.
3. One section per experiment: command, GPU index, screen session name, PID, wall time, gate result, output paths.
4. P1's injection survey and the band-coverage decision, with evidence.
5. P2's anchor reproduction: exact command, returncode, raw stdout.
6. The frozen-config commit hash and timestamp, demonstrating it preceded aggregate inspection.
7. Every gate that failed and what you did — never what you tuned.
8. Final state: GPUs clear, no sessions, results directory size, what is in the repo versus on `/ceph`.
9. A plain statement of what the runs do and do not support, written so it can be pasted into the paper's limitations.

---

## Done when

- All checklist gates passed or explicitly reported as failed.
- No process of yours on any GPU; no screen session left running; no stale PID files.
- Results are inside the repo, small, and traceable to the commands that produced them.
- Nothing under `/ceph/dwong/datasets` was modified.
- No number was transcribed by hand.
