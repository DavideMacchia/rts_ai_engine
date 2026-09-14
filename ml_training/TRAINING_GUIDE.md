# RTS AI Training Guide

> ⚠️ **OUTDATED — superseded by [`docs/`](docs/README.md).**
> This guide describes an older setup (plain PPO + Transformer extractor,
> `tutorial/easy/medium/hard/very_hard` opponents, `do_nothing: 25.0`) that no
> longer matches the code. The current system uses **MaskablePPO + MLP extractor**,
> **delta-based rewards**, `normal`/`aggressive` opponents, and a **behavioral
> cloning** warm-start. See:
> - [`docs/architecture.md`](docs/architecture.md) — overall architecture
> - [`docs/rl_training_system.md`](docs/rl_training_system.md) — how it works / how to run
> - [`docs/design_decisions.md`](docs/design_decisions.md) — why the choices
> Kept for historical reference only.

Complete guide for training your RTS AI with the refactored codebase.

## 🚀 Quick Start

### Basic Training (Default Settings)

```bash
cd ml_training
python train.py
```

This will:
- Load default configs from `src/assets/constants/`
- Train with balanced opponent
- Use PPO algorithm with transformer feature extractor
- Save model to `training/ppo_model/`

### Training Against Specific Opponent

The refactored environment now supports configurable opponents!

Edit your training config or modify the environment creation:

```python
# In train_ppo.py, modify create_env function:
return RealTimeRTSEnv(
    decision_interval=decision_interval,
    reward_config_path=reward_config,
    opponent_type='hard'  # Choose: tutorial, easy, medium, hard, very_hard
)
```

## 📋 Training Options

### Command Line Arguments

```bash
# Custom training config
python train.py --training-config path/to/config.json

# Custom reward config
python train.py --reward-config path/to/reward.json

# Override specific parameters
python train.py --timesteps 1000000 --ent-coef 0.02 --n-envs 8
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--training-config` | Auto-loaded | Path to training config JSON |
| `--reward-config` | Auto-loaded | Path to reward config JSON |
| `--ent-coef` | From config | Entropy coefficient (exploration) |
| `--timesteps` | From config | Total training timesteps |
| `--n-envs` | From config | Number of parallel environments |

## 🎓 Curriculum Training

Train progressively against harder opponents:

```bash
python train_curriculum.py
```

This will automatically:
1. Start with `tutorial` opponent (DoNothingBot)
2. Progress to `easy` (EasyBot)
3. Progress to `medium` (RushBot)
4. Progress to `hard` (BalancedBot)
5. Finally train against `very_hard` (AdaptiveBot)

Each stage requires achieving a minimum win rate before advancing.

## 🎯 Opponent Types

| Difficulty | Bot | Strategy | Win Rate Target |
|------------|-----|----------|-----------------|
| `tutorial` | DoNothingBot | Does nothing | 90% |
| `easy` | EasyBot | Basic economy, no military | 80% |
| `medium` | RushBot | Rush to military | 70% |
| `hard` | BalancedBot | Balanced build order | 60% |
| `very_hard` | AdaptiveBot | Adapts to opponent | 55% |

## 📊 Monitoring Training

### TensorBoard

Training metrics are logged to TensorBoard:

```bash
# In a separate terminal
cd ml_training
tensorboard --logdir=training/ppo_model/logs
```

Then open http://localhost:6006

### Metrics Tracked

- **Reward**: Episode rewards over time
- **Action Diversity**: Entropy and unique actions used
- **Episode Length**: How long games last
- **Win Rate**: Percentage of games won
- **Policy Stats**: Value loss, policy gradient loss

## 🔧 Configuration Files

### Training Config (`ml_training_config.json`)

Located in `src/assets/constants/` or `ml_training/config/`:

```json
{
  "training": {
    "total_timesteps": 1000000,
    "decision_interval": 1.0,
    "n_envs": 4,
    "learning_rate": 0.0003,
    "ent_coef": 0.01,
    "gamma": 0.99
  }
}
```

### Reward Config (`reward_config.json`)

Located in `ml_training/config/`:

```json
{
  "action_validity": {
    "valid_action": 25.0,
    "do_nothing": 0.0
  },
  "economic_progress": {
    "population": 0.5
  }
}
```

### Gameplay Config (`gameplay_config.json`)

Located in `ml_training/config/`:

```json
{
  "game_phases": {
    "early_game_end": 7200,
    "mid_game_end": 14400
  },
  "opponent_strategy": {
    "detection_delay_seconds": 1800
  }
}
```

## 📁 Output Files

Training produces these files:

```
ml_training/training/ppo_model/
├── rts_realtime_final.zip       # Final trained model
├── rts_realtime_checkpoint_*.zip # Periodic checkpoints
├── logs/                         # TensorBoard logs
│   └── PPO_*/
└── vecnormalize.pkl             # Normalization stats
```

## 🧪 Testing Your Model

After training, test your model:

```bash
cd ml_training
python tests/model/test_trained_model.py training/ppo_model/rts_realtime_final.zip
```

Options:
```bash
# Quick test (10 games vs medium opponent)
python tests/model/test_trained_model.py model.zip --test quick --opponent medium

# Full benchmark (100 games vs all opponents)
python tests/model/test_trained_model.py model.zip --test benchmark --games 100

# Watch a single game
python tests/model/test_trained_model.py model.zip --test watch --opponent hard
```

## 💡 Training Tips

### 1. Start Small
```bash
# Quick test run (1000 steps)
python train.py --timesteps 1000 --n-envs 1
```

### 2. Use Curriculum Learning
- Trains faster than jumping straight to hard opponents
- More stable learning
- Better final performance

### 3. Tune Exploration
```bash
# More exploration (better for early training)
python train.py --ent-coef 0.05

# Less exploration (better for fine-tuning)
python train.py --ent-coef 0.001
```

### 4. Monitor Diversity
- Check TensorBoard for action entropy
- If entropy is too low (<1.0), increase `ent_coef`
- If model isn't learning, decrease `ent_coef`

### 5. Save Checkpoints
Models are automatically checkpointed every N steps (configured in training config).

## 🐛 Troubleshooting

### Import Errors

If you see import errors, make sure you're running from the correct directory:

```bash
# Always run from project root
cd /path/to/rts_rust_game
python ml_training/train.py
```

### GPU Memory Issues

If training crashes with GPU memory errors:

```bash
# Reduce parallel environments
python train.py --n-envs 2

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
python train.py
```

### Low Win Rate

If your model isn't winning:

1. Check opponent difficulty (start with `tutorial`)
2. Increase training timesteps
3. Adjust reward config (increase action rewards)
4. Check action diversity (should use 15+ actions)

## 📚 Advanced Usage

### Custom Opponents

Create your own opponent in `simulator/opponents/`:

```python
# simulator/opponents/my_opponent.py
from .base import ScriptedOpponent
from ..actions import Action, ActionType

class MyCustomBot(ScriptedOpponent):
    def __init__(self, faction_id):
        super().__init__(faction_id)
        self.name = "MyCustomBot"
        self.difficulty = "custom"

    def act(self, game_state, game_time):
        # Your custom strategy here
        return Action(faction_id=self.faction_id, action_type=ActionType.BUILD_FARM)
```

Register in `simulator/opponents/registry.py`:

```python
from .my_opponent import MyCustomBot

OPPONENT_POOL['custom'] = MyCustomBot
```

### Resume Training

```bash
# Load existing model and continue training
python -c "
from stable_baselines3 import PPO
model = PPO.load('training/ppo_model/checkpoint_500000.zip')
model.learn(total_timesteps=500000)
model.save('training/ppo_model/continued.zip')
"
```

## 🎉 Success Criteria

Your model is well-trained when:

- ✅ Wins 90%+ vs tutorial opponent
- ✅ Wins 70%+ vs medium opponent
- ✅ Wins 50%+ vs hard opponent
- ✅ Uses 15+ different actions
- ✅ Action entropy > 1.5
- ✅ Builds economy AND military

Happy training! 🚀
