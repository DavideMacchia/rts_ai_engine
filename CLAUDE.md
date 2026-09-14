# Working rules for this repo

The plan and current state live in `ml_training/docs/roadmap.md`; the *why* behind
choices in `ml_training/docs/design_decisions.md` (a chronological log). Read those
first; this file is only *how to work*, not *what*.

## Measurement (the discipline that matters most)
- Judge changes on **conquests**, never a proxy. Ablate on conquests too.
- **n=20 is noise; a mirror match is a coin flip.** Report ≥100 seeded games per
  opponent, opponent held **fixed**, as mean ± CI. An "improvement" needs
  non-overlapping CIs. Never state a result without this.

## Config (`simulator/config.py`)
- **Anchor → ratio → derive.** Never hand-tune a derived number; edit the ratio.
- The invariants at the bottom run at import — don't weaken them to make a change fit.

## Code & tests
- Comments carry rationale cross-referenced to Dxx; match the surrounding density.
- A test **asserts** or it's a script — no assert-free print-"tests".
- Run tests via `python run_tests.py`. The suite is **stochastic**: some behavioral
  tests flake and the failing set rotates on unchanged code — rerun/seed before
  calling anything a regression.

## Docs
- `design_decisions.md` is append-only history — don't rewrite past entries.
- Keep forward-facing docs (README, docs/README, roadmap, architecture status)
  honest and in sync with reality; `roadmap.md` is the source of truth for plan + state.

## Data & artifacts
- `game_data/` is **owned by the game repo**. Edit it there and sync one-way with
  `scripts/sync_game_data.sh --check`; never hand-edit it here to diverge.
- Model checkpoints are **not committed** (regenerable) — reproduce via `pretrain_bc`.
- **Retrain on every feature**: a new action/observation invalidates prior weights.
