# Next steps → first-round final draft — R/PS 2026

**Written:** 2026-08-15
**Deadline:** 29 Aug 2026 AoE · **Results frozen:** 25 Aug 2026
**Current stage:** manuscript structurally complete; experiments in **round 3** (the whitening fix has landed and the NFPA suite passes 18/18 locally, but Block A/B and the rung-4 arms have not produced valid results yet).

---

## Status snapshot

| Strand | State |
|---|---|
| Manuscript (`rps_noise_covariance_representation/`) | Complete: D1–D7, ladder restructure, C1–C8 correction. Clean build. ~8 `\PENDING` placeholders remain. |
| Code (`noise-weighted-subspace-reconstruction`) | Complete: `src/tidmad/` + NFPA harness migrated; D1/D2 adopted; HEAD `89a4a06`, config hash `98001a62855cb9a5`; NFPA suite 18/18. |
| Experiments (`deepthought`) | Round 3 active. Round 1/2 produced no valid result (whitening bug). |

**Locked-in (do not revisit):** `none` anchor = 1.1026; P1 → files 0016–0019 out-of-band → **Route C** (drop published-baseline comparison); D1 (re/im as features → 328 bands) + D2 (patch_len 4 → 24 temporal positions); budget re-frozen 25,000 steps; Route A probe = exit 2. The whitening bug does **not** affect the manuscript's appendix values (`nfpa_demo.py` is a separate implementation).

---

## A — Server: finish round 3 (critical path)

Follow `docs/plans/rps2026/ROUND3_CHECKLIST.md` in order. Each step gated.

- [ ] **1. Full suites** (NFPA + `tidmad`) with the two new invariant tests (`test_whiten_unwhiten_round_trip_is_identity`, `test_cw_pca_attains_the_rank_k_whitened_optimum`). **Stop if either fails.**
- [ ] **2. E1 smoke** — check the predicted outcome (CW-PCA train χ² ≈ 0.9959 ≤ IsoPCA ≈ 1.0202; NFPA lower than Iso-MPCA on χ², higher on MSE). **Stop if CW-PCA train χ² > IsoPCA's.**
- [ ] **3. Block A** (10 seeds) → NFPA metric-reversal paired intervals.
- [ ] **4. Block B** (20 seeds) → white-noise control (non-negotiable).
- [ ] **5. Rung-4 arms** — train `T-mse` + `T-chi2` (25k steps, ~4.4 h each, two GPUs); `run_config.json` differ **only** in `train.loss`.
- [ ] **6. Inference + scoring** — in-band (16 files, reportable) and all-20, unmodified `benchmark.py`.
- [ ] **7.** *(only if 1–6 done)* E7 `T-chi2+OF`.
- [ ] **8. Sync back + clean state** — tidy rows to `results/rps2026/`, large artifacts referenced by path, `nvidia-smi` / `screen -ls` / `ps` clean.

---

## B — Manuscript: do now (unblocked, independent of training)

- [ ] **1. Reframe Table 1 per Route C** — published scores become *context*, not a comparison column. Drop the §4 sentence that reads like a like-for-like comparison.
- [ ] **2. Reserve the rung-4 figure budget** (~0.3 page) in the page plan.

---

## C — Manuscript: after round-3 numbers sync back

- [ ] **1. Replace every `\PENDING`** with archived numbers (Block A/B intervals, rung-4 in-band + all-20 scores, second-seed status).
- [ ] **2. Update the abstract** rung-4 verdict.
- [ ] **3. Generate the two-panel figure** from archived rows (scripted, never hand-transcribed).
- [ ] **4. Page-budget pass** → main content ≤ 4 pages, bibliography on page 5.
- [ ] **5. Write limitations** to the actual outcome — a **null** rung-4 contrast is stated as "capacity can absorb a misspecified metric," not hidden.
- [ ] **6. Final review** — no claim absent from `notes/claim_audit.md`; anonymity verified against the built PDF; clean `make`; rebuild `paper.pdf`.

---

## Blocking decision point

If rung-4 comes back **null** and, separately, if the fixed-architecture NFPA reversal intervals do not exclude zero in the predicted directions, both are **publishable findings** — report, do not tune. The only outcome that means "bug" is metric reversal failing at the *linear* rungs after the E1 smoke gate has passed.
