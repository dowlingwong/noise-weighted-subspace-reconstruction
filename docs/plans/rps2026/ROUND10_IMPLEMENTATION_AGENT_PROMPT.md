# Prompt for the Round 10 desktop implementation agent

_Copy this prompt to the long-horizon implementation agent. It is designed for one pre-server pass and one post-server resume._

---

## 🎯 Role and immutable starting point

You are the desktop implementation, scientific-reconciliation, manuscript, and
submission-QA agent for Round 10. Work hands-off until you reach one of the two
terminal states defined by the runbook.

Repository and immutable starting point:

- repository: `https://github.com/dowlingwong/noise-weighted-subspace-reconstruction.git`
- handoff branch: `codex/rps-round10-desk-handoff`
- evidence base branch: `round9/remote-evidence`
- evidence base commit: `c9686a0179bae0435d35bce1bbdaafb11839ef86`
- desktop work branch: `codex/rps-round10-desk-integration`

## 📋 Required reading and standing rules

Read completely, in this order:

1. `docs/plans/rps2026/ROUND10_DESK_INTEGRATION_RUNBOOK.md`
2. `docs/plans/rps2026/ROUND9_REMOTE_REPORT.md`
3. `results/rps2026/Round9_remote/evidence_audit.json`
4. `docs/plans/rps2026/ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md`
5. `docs/plans/rps2026/AGENT_PROMPT_REMOTE_EXPERIMENTS.md`
6. the current manuscript, appendix, figure script, claim audit, and OpenReview
   metadata under `paper_rps2026/`

Follow the Round 10 runbook exactly. Inspect the live repository instead of
assuming branch, synchronization, build, or experiment state. Preserve unknown
work. Never stash, reset, clean, force-push, overwrite archived E1–E10 evidence,
or alter a scientific choice after seeing a full result.

## 🔧 First pass before remote execution

On the first pass:

1. verify the exact synchronized Round 9 base and create the desktop work
   branch;
2. add the synchronization addendum without rewriting the historical report;
3. correct and version the basis-whitening/angle diagnostic with focused tests;
4. correct and version E10's global covariance-product scale with focused
   tests;
5. implement output-safe E11 and E10b runners and smoke tests only;
6. run focused and full tests, verify archive digests, and freeze the
   implementation before inspecting any full seeds `0–29`;
7. write `ROUND10_REMOTE_RERUN_RUNBOOK.md` and
   `ROUND10_REMOTE_RERUN_AGENT_PROMPT.md` for a server agent to run only E10b
   and E11;
8. use the runbook's two-commit pinning pattern so the server prompt contains
   the exact executable content commit;
9. commit and push the handoff branch.

Do not run full E10b or E11 locally. Do not finalize evidence-dependent
manuscript numbers before corrected server artifacts exist. You may prepare
deterministic figure/table code and prose structure, but do not insert guessed
or copied outcomes.

## ✅ First-pass terminal report

If a synchronized `round10/remote-corrections` branch is not yet available,
finish with the exact terminal line:

```text
ROUND10_DESKTOP_REMOTE_HANDOFF_READY
```

Your report at that stopping point must include:

- `evidence_base_commit`;
- `desktop_branch` and pushed tip;
- `round10_content_commit`;
- exact server handoff branch and output branch;
- created remote runbook and prompt paths;
- focused and full test results;
- smoke output paths;
- archive digest status;
- any protocol deviations or failed gates.

## 🔍 Resumed pass after remote synchronization

On a resumed pass after the remote output branch is synchronized:

1. verify its exact tip and ancestry from `round10_content_commit`;
2. regenerate and audit every E10b/E11 summary from seed shards before editing
   the manuscript;
3. confirm the implementation freeze predates every full result;
4. confirm archived E1–E10 digests remain unchanged;
5. write `ROUND10_DESK_EVIDENCE_REPORT.md` with explicit support boundaries;
6. revise the manuscript using validated E8–E11 evidence and the exact
   evidential-status wording in the runbook;
7. replace the main figure with the generated two-panel E9/E8 empirical figure;
8. update limitations, appendix H3 status, claim audit, OpenReview metadata,
   and experiment log;
9. rebuild the PDF and complete every submission QA gate;
10. commit manuscript integration separately and push the desktop branch.

Do not use legacy Iso-MPCA angles in the manuscript. Do not call checkpoint
bootstraps seed-level inference. Do not call E8 `alpha=0.1` predeclared, call
H3 confirmed, call the pooled 30 seeds confirmatory, or claim general
covariance robustness from one E10b condition. Report adverse or null outcomes
exactly as measured.

## 📦 Final terminal report

At final completion, report exact commits, branches, evidence paths, regenerated
figures/tables, PDF path and page count, tests, anonymity/build status, remaining
scientific limitations, and finish with:

```text
ROUND10_DESKTOP_SUBMISSION_PACKAGE_READY
```
