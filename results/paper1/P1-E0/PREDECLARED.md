# PREDECLARED — E-0 (environment and regression baseline)

Frozen before the first evaluation run of this card.

## Task card (verbatim)

> **E-0 — Environment and regression baseline (do first; 30 min)**
> Steps. `uv sync --extra dev`; `uv run pytest -q`;
> `uv run python experiments/paper1/fig1_factors.py --seeds 0 --smoke`;
> confirm `results/paper1/P1-E1/manifest.json` `max_regression_abs_diff` <= 1e-6
> against `results/rps2026/E8_semantics/separable_alpha_0`.
> Pass. Test suite green; smoke run reproduces the archived E1 projector errors to 1e-6.
> Fail. Stop and report; nothing downstream is trustworthy.

## Configuration

- Config: `PrimaryConfig` with `nonseparability_alpha = 0.0`
- config SHA-256: `09f7dbd811814c55636a1f36a6419a401d5ff57033ce13c36b97b61104fbe1fa`
- short `config_hash`: `09f7dbd811814c55` (identical to the archived
  `results/rps2026/E8_semantics/separable_alpha_0/seed_XX.csv` rows)
- Seeds: 0 (the seed the card names)
- Methods: NFPA, Iso-MPCA
- Regression reference: `results/rps2026/E8_semantics/separable_alpha_0/seed_00.csv`
- Compared quantities: `channel_projector_error`, `temporal_projector_error`
- Gate: `max_regression_abs_diff <= 1e-6`

## Deviation from the card, declared before the run

The card's literal step cannot decide its own gate: in `--smoke` mode
`fig1_factors.py` sets `archived = None`, so `regression_abs_diff` is never
computed and `max_regression_abs_diff` is vacuously `0.0`. The smoke run is
therefore executed as a wiring check only, and the gate itself is evaluated by
`experiments/paper1/e0_regression.py`, which refits the *frozen* (non-smoke)
configuration at seed 0 and compares against the same archive. This is
non-destructive: `results/paper1/P1-E1` (a 4-seed archived run) is neither
overwritten nor deleted (ground rule 5).
