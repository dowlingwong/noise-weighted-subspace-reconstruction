# Remote upgrade handoff — Paper 1, 11 September 2026

_Written on the KIT server after the upgrade round. Everything below is the
result of work done here; **none of it has been independently reviewed**, and
nothing in this document should be read as approval. The self-checks described
are self-checks._

---

## 1. Revisions, branches and working-tree state

| | repository | start | end | branch |
|---|---|---|---|---|
| MS | `arXiv-v1-polished-extended-preprint` | `e0b130e1771949623dcc5d27cee7958d1b9cd5cd` | `ab18213d74c867af5327f805c7dbabf50baec132` | `upgrade/remote-2026-09-11` (new; `main` untouched) |
| EXP | `noise-weighted-subspace-reconstruction-mlst` (worktree of `noise-weighted-subspace-reconstruction`) | `a7125ea27890bf51d190087b0b03fac17d9653e2` | `5d731d0f4f685861d013626b2cc4c85b384a6d56` | `mlst/t22-2026-09-11` (new; `mlst/exp-phase-20260910` untouched) |

**Starting state as found**, which differs from the local audit's reference
points and is recorded because the difference matters:

* MS was at `e0b130e` (one commit past the audited `bf7d5f6`), working tree
  **clean**. The audited breakage was present: 49 claim-validation errors, and a
  fresh build failing on `transfer_paper/figures/likelihood_geometry_hierarchy.pdf`.
* EXP's audit reference `ea076ff` **does not exist on this machine**. The main
  checkout is at `8ee9eb1` (the R/PS lineage, clean) and the MLST worktree at
  `a7125ea`, whose branch `mlst/exp-phase-20260910` ends at **T14/T15** — the
  Part II runs (T18–T21, T23) and their tree `results/mlst_2026-09-11` are not
  here. That is recorded as gap **G1** (§7).
* No user changes, deletions or untracked files were discarded. `transfer_paper/`
  is present and untouched in both EXP checkouts; the only EXP files changed are
  the three new T22 files, `.gitignore`, `tests/test_mlst_t15_generator.py` and
  `results/t22_2026-09-11/`.

**Commits, MS (5):**

```
ab18213 evidence: track the archived console log that .gitignore's *.log rule hid
8cd8adc docs, packaging, maths: reconcile status, repair Prop. 5.2's remainder, fix the package include list
c9328a4 MLST: separate main paper and technical supplement; T22 integrated everywhere
1d56c4d T22: stage-0 protocol, selected-band predeclaration (B=9), amendments A1-A4
42104af evidence: complete the transfer_paper migration to a canonical, versioned evidence root
```

**Commits, EXP (3):**

```
5d731d0 tests(T15): raise two tolerances to the principal-angle estimator's own floor
55175b1 T22: correct a mislabelled point summary and add a zero-fit recombine stage
928715a T22: narrowband TIDMAD runner, tests and the frozen run archive
```

### Change summary

1. **Evidence migration completed.** `evidence/` is the canonical manuscript
   evidence root (259 files, 21 MB, complete SHA-256 manifest), recovered from
   MS `c826ef6` and checked against the recorded `BUNDLE_MANIFEST.json`
   (327/327 listed entries match). One resolution rule
   (`scripts/evidence_root.py`), one rebuild script
   (`scripts/materialize_evidence.py`), one verifier
   (`scripts/verify_evidence.py`, the migrated `verify_bundle.py`). Claim
   validation went 49 errors → 0.
2. **Fig. 10's transcribed inputs replaced** with full-precision rows exported
   from the T10–T12 archive, which was located on this host.
3. **T22 implemented, frozen twice, executed once**, with a stage-0 protocol, a
   selected-band predeclaration, four recorded amendments and a full report.
4. **MLST main paper and technical supplement created** as separate deliverables;
   the extended preprint is preserved unchanged in role.
5. **Prop. 5.2's remainder order repaired** (a demonstrable error); Prop. 6.3
   checked and correct; both now machine-checked.
6. **Status documents reconciled**, including an explicit exit-tier amendment
   with the cause of each change.

---

## 2. Requirement-by-requirement checklist

Legend: **done** = completed and self-checked here · **deferred** = deliberately
not done, with the reason · **needs review** = done, but a human should judge it ·
**blocked** = could not be done here.

### §2 Evidence migration

| requirement | status | evidence |
|---|---|---|
| Inventory active consumers of the removed directory | **done** | 55 distinct references across TeX, claim table, three generators, validator, revision checker, packaging list, `SUBMISSION.md`, `ARXIV_CHECKLIST.md`, `OPEN_ISSUES.md`, `archive/README.md` |
| One canonical versioned source + reproducible export step | **done** | `evidence/` + `scripts/materialize_evidence.py`; `evidence/PROVENANCE.md` states the policy |
| Configurable roots, no hardcoded paths in tracked source | **done** | `scripts/evidence_root.py`; `grep -rn '/Users/\|/ceph/' --include='*.py' --include='*.tex'` over tracked source returns nothing outside documentation of the data root |
| Recover bytes from a verified archive, checking hashes | **done** | MS `c826ef6`; 327/327 manifest entries match on sha256 and byte count |
| EXP stays the source for runs; MS carries compact inputs | **done** | `evidence/` is a declared export; EXP keeps the full run trees |
| Clean local checkout rebuilds figures without raw data | **done** | fresh `git clone` → `make_paper1_evidence.py` and `make_part2_figures.py` both succeed; only PDF timestamps differ |
| Manifest verification restored/migrated | **done** | `scripts/verify_evidence.py --require-complete [--compare] [--fail-on-scientific-diff] [--require-shared]` |
| All affected consumers updated together | **done** | single commit `42104af` |
| No silenced checks, removed rows or placeholder figures | **done** | claim table gained rows, lost none; no `\includegraphics` points at a placeholder |
| `t19_sminus_v2`, summary v2, T20 bounds, T21 post-hoc in the manifest | **done** | all 24 Part II files listed; the old bundle manifest listed none of them |
| Locate the T10–T12 archive; replace the three transcribed CSVs | **done** | `scripts/export_tidmad_heldout.py`; numerical consequences in §6 |
| **Exit:** integrity checks pass, figures resolve and regenerate, fresh build succeeds | **done** | §5 |

### §3 T22

| requirement | status | evidence |
|---|---|---|
| Dated stage-0 protocol/configuration before development | **done** | `plan/T22_STAGE0_PROTOCOL_2026-09-11.md` + `plan/mlst_protocol_2026-09-11-t22-stage0.json`, written before any T22 code |
| Identify the actual server T10–T12 implementation, roots, hashes, exposure | **done** | protocol §10; every instrument file hashed in the characterisation output |
| One configuration source, or automatic runner/protocol equality | **done** | the runner reads the JSON and *derives* the allocation offsets, asserting them against it |
| Meaningful tests (splits, calibration-only, Fourier, matched preprocessing, baselines, bootstrap) | **done** | 32 tests in `tests/test_mlst_t22_narrowband.py`; all named cases covered |
| Smoke tests on synthetic fixtures; log every exposure | **done** | all tests are synthetic; **no smoke run touched instrument data**; exposure record in the predeclaration §4 |
| Keep the design unless amended and frozen | **done** | four amendments A1–A4, each dated, caused and recorded before the affected outcome |
| B=1 identity control within 1e-12 | **done** | asserted on all six development stretches; measured −1.2e−18 … +2.9e−17 |
| Baselines preserved | **done** | all eight estimators; only PCA/EMPCA see the band |
| Allocation, cap, honest fit counting | **done** | 240 fits of a 250 cap; development executed twice (352 executions) recorded as a resource figure |
| Verify hold-out offsets and non-overlap from manifests | **done** | start 37 027 840 = 36 995 072 + 2×16 384; prior exposure `[0, 36 995 072)`; stated as temporal hold-out, not replication |
| Resolve items 1–7 before the affected stage | **done** | protocol §§5–9 |
| Development-only run, then a dated selected-band predeclaration | **done** | `reviews/2026-09-11-PREDECLARED_T22.md`, actual date, not backdated |
| Validation and final test run once, no silent retuning | **done** | each stage refuses to start without the freeze file; the final test ran once |
| Archive everything needed for local recomputation | **done** | `EXP:results/t22_2026-09-11/` (23 files, 6.0 MB) incl. the two draw `.npz` |
| Three verdicts reported separately | **done** | §3 below |

### §4 Scope preservation and optional items

| item | status |
|---|---|
| T19 conditional sample-basis framing, corrected decoder, raw/effective distinction | **preserved** — wording unchanged in substance in both versions |
| T20 fresh-seed bracket, best-found ≠ global, c/N refuted, N*≈960 is interpolation | **preserved**, and the exit tier restated to match |
| T21 registered predictions failed; explanation exploratory; no ORACLE threshold | **preserved**; canonical §11's handoff-curve box amended with its cause |
| T23 rejected the registered gap ordering; other spectral mechanisms untested | **preserved** |
| Whole-window TIDMAD: no useful recovery | **preserved** |
| T21 successor | **deferred by decision** — T21 stays a negative boundary plus an explicitly post-hoc explanation; no successor preregistered or run |
| T13-b/c | **deferred** — the MLST version makes no CRESST time-cut/rank/shrinkage robustness claim, and the limitations section says so |
| T16, further T20 refinement, finite-perturbation theorem, rung-4 GPU | **deferred** — named extensions, none started |
| Props. 5.2 and 6.3 assumption/proof check | **done, with one repair** — §7 |

### §5 Manuscript upgrade

| requirement | status |
|---|---|
| Extended preprint preserved as a deliverable | **done** — `main.tex`, 116 pp |
| Separate MLST main-paper entry point and technical supplement | **done** — `main_mlst.tex` (26 pp), `supplement_mlst.tex` (50 pp) |
| ~20–25 pp of main narrative, 7–9k words, measured separately | **done, at the edge** — 23 narrative pages (2–24; title 1, references 25–26); **9.4k words**, ~4 % above the 9k top. §6 explains the layout and the word count |
| Not achieved by smaller fonts or cramped layout | **done** — 11pt/1in/1.05 is a normal journal layout, not a squeeze; the preprint keeps 12pt/1.1in/1.15. Pages rendered and inspected |
| Required section organization | **done** — question/contributions; model and target risks; recovery theory; estimator instances; controlled evidence; instrument evidence incl. T22; discussion/limitations |
| Derivations, implementation, grids, diagnostics to the supplement | **done** — incl. the two three-panel diagnostic figures |
| Lead with geometry/class/support and the two propositions | **done** — §1 contributions (i)–(iii)/(iv), §3 carries both propositions |
| Abstracts, intro, results, captions, discussion, claim table consistent | **done** — both abstracts, contributions (vi), discussion, instrument table, claim rows 25/26 |
| Unified three-instrument table, untested marked, detection sensitivity unclaimed | **done** — MLST Table 2 |
| Scale invariance, encoder-update, token-axis with correct scope | **done** — §3.3 and §4.1 |
| Figures/tables regenerated from canonical evidence; source maps | **done** — `generated/SOURCE_MAP.md`, `generated/SOURCE_MAP_part2.md` |
| Post-hoc analyses labelled | **done** — T21 panel (b), the TIDMAD sensitivities, the flat-in-band explanation |
| References, labels, cross-document references correct after moving | **done** — 0 undefined references in all four builds and in all three gated builds |
| Programme, canonical status, TODO, OPEN_ISSUES, CHANGELOG, SUBMISSION reconciled | **done** |
| Obsolete dispatch/status paragraphs marked historical | **done** |
| Exit-tier amendments explained rather than redefined silently | **done** — canonical §8, with the cause of each of the three changes |

### §6 Verification — see §5 of this document.

### What was *not* done

* **No release action.** The gate is untouched (`decision_received: false`),
  `\gatedevidencefalse` in all entry points, nothing pushed, published,
  submitted or packaged. `scripts/package_arxiv.py --dry-run` reports
  `gate: HELD` and writes nothing.
* **The R/PS decision has not arrived** (today is 11 September; the decision is
  expected 29 September). Nothing to record.
* **No new architecture, no ORACLE monitoring, no unbounded search, no rerun of a
  completed study.**

---

## 3. T22 — the three verdicts

**All three are `inconclusive`.** The frozen dependence assessment did not
stabilise, and amendment A3 — declared before development ran — withholds every
benefit verdict in that case, not only the limited-recovery one. The estimates
below are reported beside the verdicts and are **not** verdicts.

| claim | contrast | equal-file estimate | 95 % interval | raw p | Holm p | verdict | would have been |
|---|---|---|---|---|---|---|---|
| limited recovery | Δ(zero, band-EMPCA) | +0.3927 | [+0.00453, +0.798] | 0.0102 | 0.0204 | **inconclusive** | supported |
| | Δ(cal-mean, band-EMPCA) | +0.3935 | [+0.00455, +0.798] | 0.008399 | | | |
| denoising benefit | Δ(bin read, band-EMPCA) | +0.001583 | [+0.000352, +0.00324] | 0.0002 | 0.0005999 | **inconclusive** | supported |
| metric contribution | Δ(band-PCA, band-EMPCA) | -3.788e-05 | [-0.000231, +0.00013] | 0.6987 | 0.6987 | **inconclusive** | not supported |

Primary estimand: the equal-file mean of the per-file paired squared-error
contrasts over the eligible final-test files, with a two-stage percentile
bootstrap (10 000 replicates, seed 20260911) that resamples files and, within
each drawn file, draws from that file's own joint-bootstrap replicate array.
Secondary sensitivities (inverse-variance with Q and I², DerSimonian–Laird,
equal-file t-interval) are in `evidence/data/paper1/t22_2026-09-11/combined.csv`.

**Eligibility and dependence.**

* Frozen band **B = 9** (18 real coordinates against 350 fit windows,
  p/n = 0.051 against the whole window's 187), selected on four eligible-coupled
  development stretches by the declared smallest-within-one-bootstrap-sd rule
  (best B = 17, threshold 0.005432, B = 5 short at 0.005387, B = 9 at 0.005636).
* Block lengths b_eval = 32, b_cal = 8. **Dependence verdict: inconclusive** —
  the aggregate normalised variance curve is still rising at the largest usable
  block (adjacent ratios 2.089, 2.072, 1.731, 2.128, 1.745; V(32)/V(16) = 1.745),
  leaving **nine complete blocks**, the same limiting count as T10's 150/16.
  Per curve: {'inconclusive': 107, 'stabilised': 112, 'stabilised-at-endpoint': 1}.
* Validation `0012`/`0013`/`0014` (status complete); final test
  `0015`–`0019`. **`0016` is chirped in the new time region** (drift 5.54 bins
  against the declared 0.5) although it was eligible-coupled in the T10 region,
  so by the declared rule it is evaluated, reported and excluded — leaving
  **four** eligible final-test files, exactly the declared threshold below which
  a combined interval would be labelled indicative only.
* Δ(bin read, band-EMPCA) is positive on **all seven** eligible stretches, with
  each file's own interval excluding zero on six of them.

**Fit counts.** characterisation 0 · development 176 · validation
24 · final test 40 · **total 240** of a hard cap
250. Development was *executed* twice (352 executions) for amendment A4;
that is a resource figure, not a second evaluation — the contrasts reproduce
exactly (0 differences). Resampling is not fitting: 2000 per-file joint
replicates, 2000 band-selection paired replicates, 10 000 file-level replicates,
400 block-variance-curve replicates.

**Amendments and exposures.**

| id | what | when, and why |
|---|---|---|
| A1 | development set 3 → 6 stretches; full grid on the first four eligible in a declared priority order; nominal fits 184 → 240 (cap 250 unchanged) | stage 0, before any code. Only one of the three original development files is eligible-coupled at w = 65536, so the selection rule would have had a single input; the three added files were fully exposed in T10 and are **not** hold-out |
| A2 | evaluation block grid gains 32 | stage 0. The 300-window allocation exists precisely so b = 32 leaves nine complete blocks |
| A3 | the dependence withholding applies to **all three** verdicts; the combined interval becomes a two-stage bootstrap carrying within-file uncertainty | stage 0. Audit item (b) and prompt item 5 |
| A4 | the second-line admissibility test restated as a local-maximum test beyond a one-bin guard | **on development material, before any hold-out outcome was read.** The first form compared observed band power against the *idealised boxcar Dirichlet* skirt and rejected every band of width ≥ 3 on three of four development stretches; the measured profiles decay monotonically outward on both sides — leakage, not a second line — but sit ~4× above the idealisation. Development was re-scored once with identical seeds; all contrasts reproduce exactly |

One further **post-processing correction, zero fits**: the first summarisation
reported the equal-file mean of each file's *bootstrap* mean under the field name
`equal_file_mean_of_point_estimates`. The label was wrong, not the inference —
intervals, p-values, Holm adjustment and all three verdicts are unchanged
(+0.39202 against +0.392717 on Δ(zero, band-EMPCA)). A `recombine` stage that
opens no data file and fits nothing (`fits_performed: 0`) was run once per
hold-out stage into `T22_*_v2`, leaving `*_v1` untouched.

**Exposure record.** (1) T10–T12 read `[7 097 152, 44 092 224)` of
`0000`/`0005`/`0010` and `[0, 36 995 072)` of every other validation file.
(2) T22 characterisation read the 64 transfer-calibration windows of all eleven
T22 stretches and fitted no method. (3) T22 development scored 22 cells on six
development stretches, twice. (4) Validation and the final test each ran once.
No hold-out stretch was scored before the freeze file existed.

**The one result worth the reviewer's attention.** Over the seven eligible
stretches the learned rank-2 reconstruction's mean amplitude RMSE falls from
0.655 at the whole window to **0.249** in the band, while the zero predictor
stays at 0.654–0.689 and the bin read at 0.246–0.261: it moves from
indistinguishable-from-zero and 2.7× worse than a bin read to *better* than the
bin read. That is the dimensional diagnosis of the whole-window stage confirmed
in direction. In the same run the **metric's contribution vanishes**, because
inside a nine-bin band the noise PSD is nearly flat (in-band dynamic range
1.16–4.66 against 4.77 × 10⁴ over the interior bins of the whole window). The
restriction that repairs the estimation problem removes most of the geometry's
leverage. That explanation is **post hoc**.


---

## 4. Where everything is

### Preregistrations and protocols
| what | path |
|---|---|
| T22 stage-0 protocol (narrative) | `MS:plan/T22_STAGE0_PROTOCOL_2026-09-11.md` |
| T22 stage-0 protocol (machine-readable, the runner's only configuration) | `MS:plan/mlst_protocol_2026-09-11-t22-stage0.json` |
| T22 selected-band predeclaration | `MS:reviews/2026-09-11-PREDECLARED_T22.md` |
| T22 report | `MS:reviews/2026-09-11-T22-report.md` |
| Programme (status table, II.T22 outcome) | `MS:plan/MLST_EXPERIMENT_PROGRAMME.md` |

### Evidence, manifests and compact artifacts
| what | path |
|---|---|
| manuscript evidence root (259 files, 21 MB) | `MS:evidence/` |
| manifest with sha256, sizes, exclusions, producers | `MS:evidence/MANIFEST.json` |
| provenance, policy, recovery, recorded gaps | `MS:evidence/PROVENANCE.md` |
| T22 compact export (7 files + README) | `MS:evidence/data/paper1/t22_2026-09-11/` |
| Part II archives (T19/T20/T21) | `MS:evidence/data/paper1/mlst_2026-09-11/` |
| full T22 run archive incl. bootstrap draw arrays | `EXP:results/t22_2026-09-11/` |
| T10–T12 run archive | `EXP:results/mlst_2026-09-10/` |
| historical bundle documents | `MS:archive/transfer_paper_docs/` |

### Figures, source maps, PDFs
| what | path |
|---|---|
| generated figures and tables | `MS:generated/` |
| source maps | `MS:generated/SOURCE_MAP.md`, `MS:generated/SOURCE_MAP_part2.md` |
| new T22 figure | `MS:generated/figures/fig_t22_narrowband.pdf` |
| built PDFs | `MS:main.pdf` (116 pp), `MS:main_mlst.pdf` (26 pp), `MS:supplement_mlst.pdf` (50 pp), `MS:supplement_configuration.pdf` (15 pp) |

### Logs
`EXP:results/t22_2026-09-11/*.log` (five stage logs, tracked). The historical
check logs `MS:reviews/2026-09-11-upgrade-{build-console,claim-validation}.txt`
are the pre-round state and are left as the record of what was broken.

---

## 5. Commands that work after synchronization

```bash
# ---- environment ----------------------------------------------------------
# Python 3.12, numpy 1.26, scipy 1.13, h5py 3.11, matplotlib 3.9; TeX Live 2026
# with multirow and placeins (tlmgr install multirow placeins).
MS=/path/to/arXiv-v1-polished-extended-preprint
EXP=/path/to/noise-weighted-subspace-reconstruction      # the MLST branch/worktree

# ---- evidence -------------------------------------------------------------
cd $MS
python3 scripts/verify_evidence.py --require-complete
python3 scripts/materialize_evidence.py --from-git c826ef6 --check     # or --source $EXP/transfer_paper
python3 scripts/export_tidmad_heldout.py --results $EXP/results/mlst_2026-09-10 --check
python3 scripts/export_t22.py            --results $EXP/results/t22_2026-09-11 --check

# ---- checks ---------------------------------------------------------------
python3 scripts/protocol_dry_run.py
python3 scripts/protocol_dry_run.py --part2
python3 scripts/validate_claim_table.py --exp $EXP
python3 scripts/check_revision_claims.py
python3 scripts/package_arxiv.py --dry-run          # must print 'gate: HELD'

# ---- regenerate figures and tables ---------------------------------------
python3 figures_src/make_paper1_evidence.py
python3 figures_src/make_part2_figures.py
OUT=/tmp/figcheck; mkdir -p $OUT; (cd figures_src && OUT=$OUT python3 make_figures.py)

# ---- builds (order matters: the supplement imports main_mlst's labels) ----
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_mlst.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement_mlst.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement_configuration.tex

# ---- experiment tests -----------------------------------------------------
cd $EXP && python3 -m pytest tests/ -q --ignore=tests/test_public_data_smoke.py

# ---- recompute T22 independently of the report ---------------------------
cd $MS && python3 scripts/recompute_t22.py --results $EXP/results/t22_2026-09-11
```

### What those produced here

| check | result |
|---|---|
| `verify_evidence.py --require-complete` | **OK**, 259 entries, 66 recorded excluded |
| `materialize_evidence.py --check` | **OK** |
| `export_tidmad_heldout.py --check`, `export_t22.py --check` | **unchanged** on every file |
| `protocol_dry_run.py`, `--part2` | **PASS** |
| `validate_claim_table.py` | **0 errors**, 75 path references, 0 EXP paths unchecked, 9/9 concrete values re-read |
| `check_revision_claims.py` | **ALL CHECKS PASSED**, including the six new Prop. 5.2/6.3 checks |
| `package_arxiv.py --dry-run` | `gate: HELD`, 36 files, `missing: none`, nothing written |
| EXP test suite | **305 passed, 11 skipped** |
| T22 implementation tests | **32 passed** |
| `recompute_t22.py` | reproduces every interval, p-value and all three verdicts **exactly** |
| four builds, fresh clone, no auxiliary files | 116 / 26 / 50 / 15 pp, **0 undefined references** each |
| three gated (private-preview) builds in a temporary copy | 126 / 33 / 50 pp, **0 undefined references** each |
| overfull boxes | `main` 2 (≤ 34 pt, pre-existing) · `main_mlst` **0** · `supplement_mlst` 24 (all ≤ 3.4 pt, table-of-contents) · `supplement_configuration` 0 |

### Page and word counts, and what each counts

| document | total pages | narrative pages | narrative words |
|---|---|---|---|
| `main.pdf` (extended preprint, 12pt/1.1in/1.15) | 116 | — (not a target) | ~29.5k main text |
| `main_mlst.pdf` (11pt/1in/1.05) | 26 | **23** (pp. 2–24; title 1, references 25–26) | **9 384** + 185-word abstract |
| `supplement_mlst.pdf` | 50 | — | — |
| `supplement_configuration.pdf` | 15 | — | — |

Word counts are from `strip commands, math, and non-prose environments`; they
exclude equations, tables, figures and captions, and they exclude references and
the supplement. The narrative is **23 pages against a 20–25 target** and
**9.4k words against a 7–9k target — about 4 % over the top of the range**. I
did not cut further because the remaining compressible text is the
qualifications that constrain the claims, which the same instructions require to
stay beside them. Where to spend the last 400 words is a judgement call worth
making locally.

---

## 6. Substantive claims that changed, and why

| claim | source | field / statistic | LaTeX location | what changed |
|---|---|---|---|---|
| T22 limited recovery | `evidence/data/paper1/t22_2026-09-11/combined.csv` | `equal_file_mean`, `ci_low/high`, `p_two_sided` for `zero\|empca` and `calibration-mean\|empca` | `mlst/06b_t22.tex` Table 3; `07_experiments.tex` T22 paragraph | **new** |
| T22 denoising benefit | as above, `raw-bin\|empca` | as above | as above | **new** |
| T22 metric contribution | as above, `pca\|empca` | as above | as above | **new**; registered direction not met |
| T22 RMSE by method | `…/rmse_by_method.csv` | `mean_rmse`, 7 eligible stretches | `mlst/06b_t22.tex`; Fig. `fig_t22_narrowband` panel (a) | **new** |
| T22 band selection | `…/band_selection.csv` | `equal_stretch_mean`, `bootstrap_sd` | Fig. panel (c); predeclaration | **new** |
| in-band PSD dynamic range 1.16–4.66 vs 4.77e4 | science-file noise bin powers, computed from the same `noise_bin_power` the runner uses | max/min over the B=9 band, and over interior bins | `mlst/06b_t22.tex`; `07_experiments.tex` | **new, post hoc** |
| TIDMAD post-hoc DerSimonian–Laird interval | `…/mlst_2026-09-11/tidmad_heldout_perfile_2026-09-10.csv` | `bootstrap_sd_zero_empca` (new column) | `07_experiments.tex` §9.5; `mlst/06_instruments.tex` | **[+2.0e−03, +1.1e−02] → [+1.9e−03, +1.1e−02]** |
| TIDMAD inverse-variance weight of file `0019` | as above | `weight_share_pct` | Fig. 10 panel (b) | 98.58 % → **98.65 %** (the archived value) |
| recomputed θ_IV and Q in the sensitivity table | as above | `generated/tables/tidmad_heldout_sensitivity.md` | — | +2.406e−06, Q = 359.4 → **+1.585e−05, Q = 328.3**, which now reproduce the archive exactly |
| Prop. 5.2 remainder order | analysis + `scripts/check_revision_claims.py` | — | `05_linear_ae.tex`, `mlst/03_theory.tex`, `appE_symmetry_geometry.tex` | **O(ε²λ₁/δ) → O(ε²λ₁²/δ²)** |
| exit tier *strong* contents | canonical plan | — | `plan/PAPER1_CANONICAL_PLAN_2026-09-10.md` §8 | c_F/c_U → x_F/x_U; "located or placed" → "located or bracketed"; "excitation curve" → "predictions scored as they fell" |

**Why the transcription replacement moved numbers.** The three Fig. 10 CSVs were
transcribed from the printed phase report at two significant figures, with some
interval endpoints blank. The post-hoc sensitivity had to infer each file's
bootstrap standard deviation from those rounded endpoints
(half-width / 1.96), which reproduced the archived precision-weighted estimate as
+2.406e−06 (Q = 359.4) instead of the archived +1.5845e−05 (Q = 328.33). With the
archived standard deviations the recomputation now reproduces both exactly.
Every value the transcription *carried* agrees with the archive at the
transcribed precision; nothing was re-run and no estimand changed. The
downstream consequence is the one interval endpoint above.

---

## 7. Open questions for the independent review

These are mine to raise, not to settle.

**Statistical.**

1. **T22's dependence problem is unsolved, and 300 evaluation windows were not
   enough.** Nine complete blocks at b = 32 is the same limiting count as the
   whole-window stage. The variance is *still rising* at the top of the grid, so
   this data does not bound the correlation length — it does not tell you how
   much longer a stretch would need to be. Every one of the three verdicts turns
   on this.
2. **Four eligible final-test files.** The equal-file *t*-interval on the point
   estimates alone includes zero on every contrast; the two-stage bootstrap
   excludes it on three. They answer different questions — the *t*-interval
   discards each file's within-file precision, the bootstrap keeps it — and
   which a referee should weigh more with n = 4 is a genuine question.
3. **Holm across three claims.** I chose Holm–Bonferroni step-down at familywise
   0.05, with the limited-recovery claim taking the larger of its two
   constituent p-values. That is a defensible choice, not the only one, and it
   was declared before the hold-out opened.
4. **The T10–T12 post-hoc DL interval excludes zero while the equal-file
   t-interval does not**, on eight files with I² ≈ 0.98. Both are reported; the
   predeclared primary stands as executed. Whether either deserves weight is
   open.

**Mathematical.**

5. **Prop. 5.2's remainder was wrong and is now repaired**
   (O(ε²λ₁/δ) → O(ε²λ₁²/δ²)); the appendix proof's own expression was already
   correct, so the statement and the proof disagreed. The repair is verified
   numerically, but the *rest* of the proposition — the Sylvester expansion, the
   four items, the exactness claim in item 3 — deserves an independent read.
   **Prop. 6.3** was checked in full (identity, the three signs, the Ky Fan
   partial-trace cap, the Δ_k lower bound) and I found nothing wrong; that is
   also a self-check.
6. **A finite-perturbation statement remains open**, as before. Nothing here
   changes that.

**Negative and inconclusive findings that must survive review.**

7. All three T22 verdicts are inconclusive. **The metric contribution is
   absent** and its registered direction was not met. The flat-in-band
   explanation is post hoc and untested.
8. T21's five registered predictions all failed; no abstention threshold exists.
9. T20's registered c/N law is refuted; N* is bracketed, not located.
10. Whole-window TIDMAD still shows no useful recovery.
11. T22 is a **temporal hold-out of files already examined**, not
    independent-file replication, and it is written that way everywhere.

**Provenance.**

12. **Gap G1 (open).** The Part II run tree `EXP:results/mlst_2026-09-11` at EXP
    `ea076ff`, branch `mlst/exp-phase-20260911`, is **not on this host**; the
    manuscript's registered copies are the only accessible instance. They carry
    the source tree's own `RESULTS_SHA256` and those verify, so the files are
    intact — what cannot be re-derived here is the run tree. Claim rows 22–24
    now cite the `MS:evidence/…` paths, which resolve. **Smallest action to
    close:** bundle that branch from the local phase host and fetch it into the
    experiment repository.
13. **Gap G2 (pre-existing, not closable).** The September 2026 Paper-1 runs
    executed at EXP `3026d25` with a dirty tree.
14. The T22 runner and tests were untracked at the moment of the freeze, so the
    run records EXP `a7125ea` with `tree_dirty: true`; they were committed
    immediately after, and the binding record is the code SHA-256 list inside
    the predeclaration.

**Scope.**

15. The MLST narrative is 23 pages and 9.4k words — inside the page target,
    ~4 % above the word target. See §5.
16. T13-b/c, T16, the T21 successor, further T20 refinement, a
    finite-perturbation theorem and rung-4 GPU training were all **deliberately
    not started**; the manuscript makes no claim that needs them, and the
    limitations say so.

---

## 8. Synchronization inventory — both repositories

The private transfer package is committed into the experiment repository at
**`EXP:transfer/mlst_upgrade_return_2026-09-11/`** (`SHA256SUMS` inside,
`README.md` with apply instructions). It is **not** an arXiv release package.

**Both branches are on their remotes**, so the bundles and patches below are a
redundant offline fallback rather than the transfer mechanism:

```bash
git -C <MS>  fetch origin upgrade/remote-2026-09-11 && git -C <MS>  checkout upgrade/remote-2026-09-11
git -C <EXP> fetch origin mlst/t22-2026-09-11       && git -C <EXP> checkout mlst/t22-2026-09-11
```

| file | bytes | what |
|---|---|---|
| `ms-upgrade-remote.bundle` | 9 409 266 | MS `e0b130e..6fbde0c`, branch `upgrade/remote-2026-09-11` (7 commits) |
| `ms-patches/0001..0007` | 7 files, 25 MB | the same commits as patches; redundant once the branch is fetched |
| `exp-t22.bundle` | 1 787 423 | EXP `a7125ea..5d731d0`, branch `mlst/t22-2026-09-11` (3 commits) |
| `exp-patches/0001..0003` | 3 files, 6.6 MB | the same commits as patches; redundant once the branch is fetched |
| `pdfs/main.pdf` | 116 pp | extended preprint |
| `pdfs/main_mlst.pdf` | 26 pp | MLST main paper |
| `pdfs/supplement_mlst.pdf` | 50 pp | MLST technical supplement |
| `pdfs/supplement_configuration.pdf` | 15 pp | configuration inventory |
| `REMOTE_UPGRADE_HANDOFF_2026-09-11.md` | — | a copy of this document |

The two bundle files and the two patch directories are a **snapshot taken before
the commit that added the package to the experiment repository**, so each is one
commit behind its branch tip; the branches themselves on the remotes are
authoritative. The patch directories (31.6 MB of the package) duplicate the
bundles, which duplicate the repositories; they can be deleted without losing
anything now that both branches are pushed.

### Everything scientific is inside Git — deliberately

* **MS:** `evidence/` is tracked (259 files, 21 MB), including the previously
  gitignored `data/cresst/…/console.log` that a fresh clone exposed as missing.
  `.gitattributes` keeps the tree byte-exact.
* **EXP:** `results/t22_2026-09-11/` is tracked (23 files, 6.0 MB), including
  both bootstrap draw arrays — `t22_validation_draws.npz` (488 296 B) and
  `t22_final-test_draws.npz` (811 576 B) — which are what
  `scripts/recompute_t22.py` needs. `.gitignore` was amended to allow them.

So **no result file needs copying outside Git**. To reproduce rather than copy:
every T22 stage command is in `plan/T22_STAGE0_PROTOCOL_2026-09-11.md` §12 and in
the stage logs; they need `/ceph/dwong/paper1_dataset/tidmad/raw` and about 25
minutes of CPU.

### Deliberately excluded from the package

* **Raw instrument data** — TIDMAD 171 GB, CRESST 2.3 GB, GWOSC 21 MB. They stay
  on the server and in their public releases; the manuscript archives
  checksums and source locations, not bytes.
* **Caches and credentials** — `__pycache__`, `.pytest_cache`, `.venv`, none
  tracked.
* `data/gwosc/runs/` (66 files, 5.16 MB) — three superseded GWOSC run
  directories, retained in `EXP:transfer_paper/` and hashed into
  `evidence/MANIFEST.json → excluded_retained_in_exp`.
* The **private-preview PDFs** (gated builds) — built in a temporary copy and
  discarded, as the procedure requires. Rebuild them locally from
  `SUBMISSION.md` if they are wanted.

### Pre-existing local work preserved

* MS `main` is untouched at `e0b130e`; all work is on `upgrade/remote-2026-09-11`.
* EXP `main` (`8ee9eb1`, the R/PS lineage) and the worktree branch
  `mlst/exp-phase-20260910` (`a7125ea`) are untouched; work is on
  `mlst/t22-2026-09-11`.
* `transfer_paper/` is present and unmodified in both EXP checkouts.
* Nothing was reset, cleaned or stashed; no untracked file was removed.

---

## 9. One-paragraph summary for whoever reads only this

The evidence migration is finished and verified: `evidence/` is the single
manuscript-side source, 259 files with a complete manifest, every consumer reads
it through one resolution rule, claim validation went from 49 errors to zero, and
all four documents build from a fresh clone with no undefined references. T22 was
designed, frozen twice, implemented with 32 tests, and executed once. It confirms
the whole-window stage's dimensional diagnosis in direction — the learned rank-2
reconstruction moves from the trivial-predictor level to below a no-denoising bin
read — and then reports **all three registered verdicts as inconclusive**,
because the dependence assessment it declared in advance did not stabilise, and
reports **no metric contribution at all**, because narrowing the band to fix the
estimation problem flattens the very PSD structure the metric acts on. The
manuscript now exists in three forms with the T22 result and its withheld
verdicts integrated everywhere. One real mathematical error was found and
repaired. The release gate is untouched. What needs a second pair of eyes is in
§7, and the largest single item there is that nine complete blocks was not
enough, twice.

