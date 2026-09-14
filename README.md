# RTS AI — RL agent for the macro layer of a strategy game

A reinforcement-learning agent that runs the **economic and military macro layer** of a
real-time strategy game: exploit a plot of land, climb a production chain from raw resources to
finished goods and iron military kit, feed and grow a population, and turn the surplus into an
army. It trains against a **purpose-built Python simulator** of the game's macro dynamics
(sharing balance data with the Rust game it models), with **MaskablePPO warm-started by
behavioral cloning** of a scripted expert.

## How it works

```
        game_data/                 balance data (JSON), shared with the Rust game
            │
            ▼
   ml_training/simulator/          a macro simulation of the game
     ├─ game_state, professions    people, skills/careers, an age pyramid
     ├─ config                     the anchored economy (+ invariants that run at import)
     ├─ map                        the district's territory / plot
     ├─ managers, policies         production, population, combat, the policy levers
     └─ opponents                  scripted bots (behavior trees) — training foils & BC expert
            │
            ▼
   ml_training/agents/district/    the RL agent
     ├─ env                        Gym environment: observation, action mask, graded reward
     ├─ pretrain_bc                behavioral cloning + critic warm-up
     ├─ train / train_ppo          MaskablePPO training / fine-tuning
     └─ evaluate                   measure conquests vs each scripted opponent
```

The agent is scored on a **graded valuation** of what its territory is worth — people (by skill
tier), army (by kit), and goods (by how deep in the production chain they sit) — so it is paid
to exploit its land, climb to high-grade production, and arm its surplus population. There is no
population cap: the real limit is the plot's carrying capacity.

**Status.** The infrastructure — simulator, training pipeline (behavioral cloning → MaskablePPO),
and evaluation — runs end to end. The agent is mid-redesign and currently weak: under the
district-unified objective it does not reliably build military and loses most games. Trained
checkpoints are not committed; reproduce one with `pretrain_bc` (a few minutes).

## Quickstart

```bash
cd ml_training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python run_tests.py                                                    # test suite
python -m agents.district.pretrain_bc --out checkpoints/district/agent.zip   # train a policy
python -m agents.district.evaluate  --model checkpoints/district/agent.zip   # measure it
```

## Repository layout

| Path | What it is |
|------|------------|
| `ml_training/simulator/` | The macro simulation: state, economy, population, combat, map, scripted opponents |
| `ml_training/agents/district/` | The RL agent: environment, behavioral cloning, training, evaluation |
| `ml_training/docs/` | Architecture, the training pipeline, and the design-decision log |
| `ml_training/tests/` | Test suite (the simulator's invariants and behavior) |
| `game_data/` | Balance data (JSON), owned by the game; `scripts/sync_game_data.sh` keeps this copy in step |

## Documentation

- [`docs/architecture.md`](ml_training/docs/architecture.md) — the tiered design and roadmap.
- [`docs/rl_training_system.md`](ml_training/docs/rl_training_system.md) — how the pipeline works and how to run it.
- [`docs/design_decisions.md`](ml_training/docs/design_decisions.md) — the diagnostic log: why each choice was made (the depth of the project).

## License

[MIT](LICENSE)
