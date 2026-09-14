# RL Training System — How It Works

> How the DISTRICT-tier agent training pipeline is built and how to run it.
> (What exists today is the district tier, not the macro brain — see architecture.md §2.)
> For *why* the choices were made, see [design_decisions.md](design_decisions.md).
> For the *big-picture* architecture, see [architecture.md](architecture.md).

---

## Pipeline overview

```
game_data/*.json ──► simulator (RealTimeRTSSimulator)
                          │
                          ▼
        RealTimeRTSEnv (Gymnasium)  ◄── ScriptedOpponent (faction 1)
          │  obs (22 features)          (NormalBot / AggressiveBot)
          │  action_masks()
          ▼
        MaskablePPO policy
          MLPExtractor(22→256→256) → actor(pi) + critic(vf)
          │
          ▼
        RewardCalculator (delta-based)  + SessionLogger (session_log.json)
```

Training has three stages (see [design_decisions.md](design_decisions.md) for why):
1. **Behavioral cloning** (`pretrain_bc.py`) — imitate a scripted expert. Trains the **actor**.
2. **Critic warm-up** (same script) — fit the **value function** to the BC policy's
   returns. Skipping this makes RL fine-tuning destroy the actor (see D7).
3. **RL fine-tune** (`train_bot.py --resume`) — optional; only useful when the
   expert is *not* already optimal (see D8).

> **Current policy: a BC + critic warm-up on the current 37-action env** (train one with
> `pretrain_bc` — checkpoints are not committed, they regenerate in minutes). It runs, but is
> **weak**: it clones a scripted expert that idles ~86% and conquers ~38%, so it builds almost no
> army and loses most games. The `bc_v1` era ("10/10 vs every bot") is retired — those checkpoints
> predate the district-unified action space and no longer load. Restoring military play under the
> new objective is open work.

---

## Repository layout

Organized by **agent tier**, with tier-agnostic pieces factored into `common/`:

```
rl/
  simulator/            the game simulation — shared by every tier
    managers/           resource, building, training, population, combat, reward
    opponents/          scripted policies (= district-tier baselines / BC experts)
    behavior_tree/
  agents/
    common/             tier-agnostic building blocks
      extractors.py     MLPExtractor (flat obs) + TransformerExtractor (future graph obs)
      bc.py             behavioral_clone() + warmup_critic()  ← reusable by any tier
      callbacks/        session_logger, episode_logger
    district/           the settlement / district-tier agent  ← what exists today
      env.py            RealTimeRTSEnv
      pretrain_bc.py    demo collection + BC + critic warm-up
      train.py          MaskablePPO trainer / fine-tuner
      train_ppo.py, train_curriculum.py
    macro/              empire brain — NOT BUILT (see architecture.md §3-4)
    camp/               military camp tier — NOT BUILT (§5)
  checkpoints/
    district/district_agent_bc_v1.zip     ← the deliverable
  config/               env_config, reward_config, ml_training_config, transformer_config
  docs/  tests/  utils/
```

Run entry points as modules from `rl/`: `python -m agents.district.train …`

> **Note:** SB3 stores the feature-extractor's *module path* inside the saved `.zip`.
> Moving `agents/common/extractors.py` breaks loading old checkpoints — regenerate
> them by re-running `pretrain_bc` (~3 min).

---

## Components

### Simulator — `simulator/`
- `realtime_simulator.py` — `RealTimeRTSSimulator`, continuous time stepping. A
  `step(actions, delta_time)` advances the game, runs the managers, computes rewards.
- `managers/` — one system each: `resource_manager`, `building_manager`,
  `training_manager`, `population_manager`, `combat_manager`, `reward_calculator`.
- `game_state.py` — `Faction` (resources, buildings, units, population) and `GameState`.
- `config.py` — loads `game_data/*.json`, applies the 30× production speedup.
- Combat is abstract: `combat_manager.py` compares strengths (defender ×1.3),
  applies casualty rates, 30% random building damage. Treated as a swappable oracle.

### Environment — `agents/district/env.py`
- **Observation**: 22 normalized features (resources, population, building counts,
  own & opponent military). Normalization constants come from `config/env_config.json`
  (`normalization` block) — kept in ONE place to avoid the earlier mismatch bug.
- **Action space**: `Discrete(20)` — one per `ActionType` (build X, train soldier/
  archer, attack, do nothing).
- **`action_masks()`**: returns a boolean mask of valid actions (affordable,
  prerequisites met). **Consumed by `MaskablePPO`** — invalid actions can't be chosen.
- **`decision_interval`**: seconds of game time per agent step. Drives episode
  length = `max_game_time / decision_interval`. (This has a big effect on
  credit assignment — see design_decisions "horizon".)

### Feature extractor — `agents/common/extractors.py`
- **`MLPExtractor`** (default): `22 → 256 → 256 → features_dim`, LayerNorm + ReLU.
  Correct choice for a flat state vector.
- `TransformerExtractor` retained for future entity/sequence inputs (the spatial brain).

### Reward — `simulator/managers/reward_calculator.py` + `config/reward_config.json`
- **Delta-based**: rewards positive *changes* (population growth, building completed,
  unit trained, strength gained) — NOT absolute state. Idling → ~0 reward.
- **Military milestones**: one-time bonuses (first barracks, first unit, army-size,
  superiority) to guide the long build sequence.
- **Victory/defeat** terminal rewards; **timeout** scored by development
  (military weighted 100×).
- Diversity/monotony penalties are **disabled** (config v5) — they punished the
  necessary idle action.

### Opponents — `simulator/opponents/`
- `NormalBot` — balanced behavior-tree bot. Note: economy-heavy, **rarely attacks**.
- `AggressiveBot` — military rusher; minimal economy → barracks → constant soldiers →
  attacks. Parameterized by difficulty via the registry:
  `aggressive_easy|medium|hard` (attack time 1200/800/500 s).
- Registered in `simulator/opponents/registry.py`; selected by `opponent_type`.

### Behavioral cloning + critic warm-up — `agents/district/pretrain_bc.py`
- `collect_demonstrations()` runs a scripted expert (`AggressiveBot`) in the agent's
  slot, recording `(obs, action, mask)`.
- `behavioral_clone()` supervised-trains the **actor** (masked cross-entropy via
  `policy.evaluate_actions`).
- `warmup_critic()` then rolls out the BC policy, computes discounted returns-to-go,
  and regresses the **value head** to them. Only critic-branch params are optimized,
  and the policy uses `share_features_extractor=False`, so the actor is untouched.
  Watch `explained_var` — it should reach ~0.9. (Skip it and RL will wreck the actor.)

### Training — `agents/district/train.py`
- `MaskablePPO` + `MLPExtractor`, `SubprocVecEnv` parallel envs,
  `share_features_extractor=False`.
- Callbacks: `WinRateCallback`, `ActionDistributionCallback`, `SessionLoggerCallback`,
  `CheckpointCallback`.
- On `--resume`, `gamma`/`lr`/`ent_coef` are re-applied to the loaded model.

### Logging — `agents/common/callbacks/session_logger.py`
- Writes `<log_dir>/session_log.json`: per-snapshot win rate, reward, game metrics
  (population/military/buildings), action entropy/top actions, plus trends and an
  auto-diagnosis. **This is the file to read to reason about a run.**

---

## Configuration files — `config/`

| File | Controls |
|---|---|
| `env_config.json` | decision_interval, obs features, **normalization constants** |
| `ml_training_config.json` | timesteps, n_envs, ent_coef, PPO hyperparams, extractor |
| `reward_config.json` | all reward/penalty values (delta rewards, milestones, victory) |
| `transformer_config.json` | Transformer extractor architecture (only if used) |

---

## How to run

```bash
cd rl

# 1) Behavioral cloning + critic warm-up  (produces the current deliverable)
python -m agents.district.pretrain_bc \
    --opponent aggressive_hard --episodes 60 --epochs 15 \
    --decision-interval 2.5 --gamma 0.999 \
    --out ./checkpoints/district/district_agent_bc_v1.zip

# 2) (Optional) RL fine-tune from the warm-start.
#    Only worthwhile against an opponent the expert does NOT already beat -- see D8.
python -m agents.district.train \
    --resume ./checkpoints/district/district_agent_bc_v1.zip \
    --opponent aggressive_hard \
    --timesteps 500000 --n-envs 8 --batch-size 256 --n-steps 2048 \
    --lr 1e-4 --ent-coef 0.03 --decision-interval 2.5 --gamma 0.999 \
    --log-dir ./logs_bc_finetune

# Train from scratch vs an opponent (no warm-start; will NOT learn military -- see D4)
python -m agents.district.train \
    --opponent aggressive --timesteps 500000 --log-dir ./logs_run

# Inspect a run
python -c "import json; print(json.load(open('logs_bc_finetune/session_log.json'))['final_summary'])"
```

Key flags: `--opponent {normal,aggressive,aggressive_easy,aggressive_medium,aggressive_hard}`,
`--resume <model.zip>`, `--lr`, `--ent-coef`, `--n-envs`, `--decision-interval`, `--gamma`.

**`decision_interval` and `gamma` must match between BC and fine-tune** (episode
length changes with the interval, and the discount is tuned to it).

---

## Evaluating a policy (deterministic — the real test)

Stochastic training win rates can be misleading; always check the deterministic policy:

```python
from sb3_contrib import MaskablePPO
from agents.district.env import RealTimeRTSEnv
model = MaskablePPO.load('checkpoints/district/district_agent_bc_v1.zip')
env = RealTimeRTSEnv(opponent_type='aggressive_hard', decision_interval=0.5)
obs,_ = env.reset(seed=0); done=False
while not done:
    mask = env.action_masks()
    a,_ = model.predict(obs, deterministic=True, action_masks=mask)
    obs,r,done,_,info = env.step(int(a))
# then inspect env.sim.state.factions[0]: buildings, units, military_strength
```

---

## Gotchas

- **Use `MaskablePPO`, not `PPO`.** Plain PPO ignores `action_masks()` and the
  deterministic policy collapses to ATTACK.
- **Never re-implement the observation.** Drive `RealTimeRTSEnv` — it owns
  normalization (from `env_config.json`) and masking. Duplicating that logic with
  stale constants has silently broken evaluation before.
- **BC alone is not enough — always run the critic warm-up** before any RL
  fine-tune, or the random critic will destroy the cloned policy (D7).
- **Evaluate the DETERMINISTIC policy.** Stochastic training win rates flattered a
  policy whose argmax just idled or attacked. `model.predict(..., deterministic=True,
  action_masks=mask)` is the only honest test.
