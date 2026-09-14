# RTS AI — a reinforcement-learning agent for the macro layer of a strategy game

A reinforcement-learning agent that learns to run the **economic and military macro layer**
of a real-time strategy game: exploit a plot of land, climb a production chain from raw
resources to finished goods and iron military kit, feed and grow a population, and turn the
surplus into an army — well enough to outlast scripted opponents.

The agent trains against a **purpose-built simulator** of the game's macro dynamics (written
here, in Python), sharing its balance data with the Rust game it models. It is trained with
**MaskablePPO warm-started by behavioral cloning** of a scripted expert.

> **What this repository is really about.** The interesting part isn't that a bot plays a
> game — it's the *engineering discipline* behind getting a sparse-reward, hard-exploration
> RL problem to work without fooling myself about whether it did. If you read one thing, read
> **[`ml_training/docs/design_decisions.md`](ml_training/docs/design_decisions.md)** — a
> diagnostic log (D1–D38+) that walks, in order, the chain of independent bugs behind a single
> symptom (*"the agent refuses to build any military"*), and how each was measured, not guessed.

---

## Why it was hard, and what the log shows

The headline symptom was simple: the agent learned an economy-only policy and **never built
military**. The fix was a chain of independent problems, each hiding the next. A few, as a
taste of the log:

- **The win rate was an illusion.** Plain PPO doesn't call the environment's `action_masks()`,
  so invalid actions (attack with no army) were selectable and the deterministic policy
  collapsed onto `ATTACK` ~3000×/game. Switching to `MaskablePPO` made the training signal
  mean what it said. *(D1)*
- **Behavioral cloning trained the actor but not the critic.** The real cause of the refusal:
  the cloned policy acted well but its value head was random, so PPO fine-tuning immediately
  destroyed it. *(D7)*
- **The measurements were noise.** `n=20` games with a stochastic sim and a mirror match is a
  coin flip; the RNG seed wasn't even reaching the simulator. Every earlier "result" had to be
  re-measured once the seed was wired through and the metric was **conquests**, not a proxy.
  *(D11, D12)*
- **Rewards punished the wrong thing.** A diversity penalty punished idling; a long horizon
  vanished the win signal. Reward shaping is where most of the bugs lived. *(D5, D6)*

The same discipline shows up in the code. [`ml_training/simulator/config.py`](ml_training/simulator/config.py)
derives every game constant from a single anchor (one adult working life) via stated ratios —
never a hand-tuned number — and ends with a block of **invariants that run at import** and
refuse to load a self-contradictory economy (a building that produces but employs nobody, a
settlement that asks for more workers than it can ever have).

---

## How it works

```
        game_data/                 balance data (JSON), shared with the Rust game
            │
            ▼
   ml_training/simulator/          a macro simulation of the game
     ├─ game_state, professions    people, skills/careers, an age pyramid
     ├─ config                     the anchored economy (+ invariants)
     ├─ map                        the district's territory / plot
     ├─ managers, policies         production, population, combat, the policy levers
     └─ opponents                  scripted bots (behavior trees) — the training foils & BC expert
            │
            ▼
   ml_training/agents/district/    the RL agent
     ├─ env                        Gym environment: observation, action mask, graded reward
     ├─ pretrain_bc                behavioral cloning + critic warm-up  (the deliverable)
     ├─ train / train_ppo          MaskablePPO training / fine-tuning
     └─ evaluate                   measure conquests vs each scripted opponent
```

The agent is scored on a **graded valuation** of what its territory is worth — people (by
skill tier), army (by kit), and goods (by how deep in the production chain they sit) — so it is
paid to exploit its land, climb to high-grade production, and arm its surplus population, with
no population cap: the real limit is the plot's carrying capacity.

**Status.** Stage 0: the **district-tier** agent plays economy *and* military. The behavioral-cloning
deliverable (`ml_training/checkpoints/district/`) matches or beats the scripted expert; RL
fine-tuning currently finds no headroom over it (see design log §D8). The macro tier that
commands multiple districts is the next stage.

---

## Quickstart

```bash
cd ml_training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# Run the test suite (the simulator's invariants and behavior)
python run_tests.py

# Behavioral cloning + critic warm-up — the deliverable
python -m agents.district.pretrain_bc

# MaskablePPO training / fine-tuning
python -m agents.district.train

# Measure the trained policy: conquests vs each scripted opponent
python -m agents.district.evaluate --model checkpoints/district/district_agent_bc_v14.zip
```

---

## Repository layout

| Path | What it is |
|------|------------|
| `ml_training/simulator/` | The macro simulation: state, economy, population, combat, map, scripted opponents |
| `ml_training/agents/district/` | The RL agent: environment, behavioral cloning, training, evaluation |
| `ml_training/docs/` | **Start here.** Architecture, the training pipeline, and the design-decision log |
| `ml_training/checkpoints/` | Trained policies |
| `ml_training/tests/` | Test suite (the simulator's invariants and behavior) |
| `game_data/` | Balance data (JSON), the source of truth owned by the game |

`game_data/` is owned by the game repository; `scripts/sync_game_data.sh` keeps this repo's
copy in step (`--check` reports drift). See the script header for details.

## Documentation

Read the docs in this order — they carry the depth this README only gestures at:

1. [`ml_training/docs/architecture.md`](ml_training/docs/architecture.md) — the tiered design and the roadmap.
2. [`ml_training/docs/rl_training_system.md`](ml_training/docs/rl_training_system.md) — how the pipeline works and how to run it.
3. [`ml_training/docs/design_decisions.md`](ml_training/docs/design_decisions.md) — **why** every choice was made: the diagnostic log.

---

*A solo research project. The simulator models the macro layer of a companion RTS game; the
numbers are uncalibrated by design (there is no finished game to fit them to) but the ratios
are stated and the invariants are enforced.*
