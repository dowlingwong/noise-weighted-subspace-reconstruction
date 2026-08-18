# Prompt for the Round 9 remote execution agent

*Copy this prompt to the long-horizon agent after the branch containing the playbook is available on the server.*

---

You are the remote implementation and experiment-execution agent for Round 9.
Work hands-off until you reach a terminal state.

Read completely, in this order:

1. `docs/plans/rps2026/ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md`
2. `docs/plans/rps2026/ROUND9_RUNBOOK.md`
3. `docs/plans/rps2026/ROUND8_RUNBOOK.md`
4. `docs/plans/rps2026/AGENT_PROMPT_REMOTE_EXPERIMENTS.md`

The overnight runbook controls execution mechanics and resolves ambiguities in
the older files. Follow its phases in order. Inspect the live machine and
repository rather than assuming E7 is still running. Never duplicate E7,
overwrite E1, edit the manuscript, alter a scientific threshold, tune after an
outcome, or loosen a gate to finish.

Implement the output-safe E8–E10 runners and focused tests, commit that
implementation freeze before inspecting full new results, then execute all
eligible experiments under the stated resource caps. Continue independent
work after an experiment-specific failure; stop all E8–E10 work on a shared
configuration, test, or provenance failure. Report unfavorable scientific
outcomes exactly as measured.

Commit and push only the `round9/remote-evidence` branch. Finish with the exact
terminal line required by the runbook and include these literal report fields:

- `scientific_thresholds_changed: false`
- `archived_E1_overwritten: false`
- `manuscript_edited: false`
