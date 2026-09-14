# ML Training Utilities

Comprehensive utilities for debugging, monitoring, and improving machine learning training for RTS games.

## 📚 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [Reward Debugging](#1-reward-debugging)
  - [Action Monitoring](#2-action-monitoring)
  - [Episode Statistics](#3-episode-statistics)
  - [Curriculum Learning](#4-curriculum-learning)
  - [Modular Rewards](#5-modular-rewards)
  - [Testing Utilities](#6-testing-utilities)
  - [Checkpoint Comparison](#7-checkpoint-comparison)
- [Examples](#examples)
- [Integration Guide](#integration-guide)

## Overview

This refactored utilities package provides professional-grade tools for ML training:

### ✅ What's Included

1. **Reward Debugging** - Track reward components, detect anomalies, understand what drives learning
2. **Action Monitoring** - Analyze action distributions, detect stuck patterns, ensure exploration
3. **Episode Statistics** - Comprehensive tracking of episode metrics, trends, and performance
4. **Curriculum Learning** - Progressive difficulty scheduling and performance-based advancement
5. **Modular Rewards** - Separate, trackable reward components for better understanding
6. **Testing Utilities** - Unit tests for environments and reward functions
7. **Checkpoint Comparison** - Compare model checkpoints to track improvement

### 🎯 Use Cases

- **Debugging**: Why isn't my AI learning?
- **Monitoring**: Is training progressing well?
- **Analysis**: What actions is my AI taking?
- **Optimization**: Which checkpoint performs best?
- **Testing**: Is my environment working correctly?

## Installation

The utilities are already part of your project:

```python
from utils import (
    RewardDebugger,
    ActionMonitor,
    EpisodeStatsTracker,
    # ... and more
)
```

### Dependencies

Already in your environment:
- `numpy`
- `matplotlib`
- `stable-baselines3`

## Quick Start

### 1. Run the Demo

```bash
cd src/ml_training/examples
python utils_demo.py
```

This interactive demo shows all utilities in action.

### 2. Basic Usage

```python
from utils import ActionMonitor, RewardComponentTracker

# Monitor actions
monitor = ActionMonitor()
monitor.record('BUILD_FARM')
monitor.record('TRAIN_SOLDIER')
monitor.print_distribution()

# Track rewards
tracker = RewardComponentTracker()
tracker.log('economic', 150.0, {'resources': 1000})
tracker.log('military', 75.0, {'soldiers': 5})
tracker.end_episode(episode_num=0)
tracker.print_summary()
```

## Modules

### 1. Reward Debugging

**Purpose**: Understand what's contributing to your reward signal.

#### RewardComponentTracker

Tracks individual reward components over episodes.

```python
from utils import RewardComponentTracker

tracker = RewardComponentTracker()

# During training
tracker.log('survival', 100.0, {'time': 30})
tracker.log('economic', 250.5, {'resources': 1500})
tracker.log('military', 75.0, {'soldiers': 5})

# End of episode
tracker.end_episode(episode_num)

# Analyze
tracker.print_summary(last_n_episodes=100)
tracker.save_to_file('rewards.json')
```

**Output Example:**
```
📊 REWARD ANALYSIS (100 episodes)
🎯 Total Reward per Episode:
   Mean:  4679.42

📈 Component Breakdown:
Component                 Mean        % of Total  Range
----------------------------------------------------------------------
survival                  1200.00     25.6%       1100.0 → 1300.0
economic                   950.50     20.3%        800.0 → 1100.0
military                   600.00     12.8%        400.0 → 800.0
```

#### RewardDebugger

Real-time debugging of reward behavior.

```python
from utils import RewardDebugger

debugger = RewardDebugger(buffer_size=1000)

# During training
debugger.log_step(
    action='BUILD_FARM',
    reward=10.5,
    state={'population': 10},
    reward_components={'economic': 8.0, 'building': 2.5}
)

# Analyze
debugger.print_recent(n=10)
debugger.print_action_analysis()
anomalies = debugger.detect_anomalies()
```

### 2. Action Monitoring

**Purpose**: Understand what actions your AI is taking.

#### ActionMonitor

Track action frequency and distribution.

```python
from utils import ActionMonitor, ActionDistributionAnalyzer

monitor = ActionMonitor(window_size=1000)

# Record actions
monitor.record('BUILD_FARM')
monitor.record('TRAIN_SOLDIER')
monitor.record('DO_NOTHING')

# Analyze
monitor.print_distribution()
diversity = monitor.get_diversity_score()  # Shannon entropy

# End episode
monitor.end_episode(episode_num)

# Diagnose issues
analyzer = ActionDistributionAnalyzer(monitor)
analyzer.print_diagnosis()
analyzer.suggest_fixes()
```

**Output Example:**
```
🎮 Action Distribution (recent window - 1000 actions)
Diversity Score: 2.450 (higher = more exploration)

Action                         Frequency    Bar
----------------------------------------------------------------------
BUILD_FARM                        25.5%    ████████████
TRAIN_SOLDIER                     18.3%    █████████
BUILD_HOUSE                       15.2%    ███████
DO_NOTHING                        12.0%    ██████
```

**Diagnostic Output:**
```
🔍 ACTION DISTRIBUTION DIAGNOSIS
✅ POSITIVE INDICATORS:
   ✅ Good diversity (entropy=2.45)
   ✅ Low DO_NOTHING (8.2%) - Good action diversity
```

### 3. Episode Statistics

**Purpose**: Track comprehensive episode-level metrics.

#### EpisodeStatsTracker

```python
from utils import EpisodeStatsTracker, EpisodeAnalyzer

tracker = EpisodeStatsTracker(window_size=100)

# Start episode
tracker.start_episode(episode_num=0)

# During episode
tracker.record_step(action='BUILD_FARM', reward=10.5, state={...})

# End episode
tracker.end_episode(
    total_reward=1500,
    win=True,
    final_state={...}
)

# Analyze
tracker.print_summary()
tracker.plot_trends(save_path='training_trends.png')

# Advanced analysis
analyzer = EpisodeAnalyzer(tracker)
analyzer.print_analysis()
analyzer.compare_periods((0, 100), (100, 200))
```

**Output Example:**
```
📊 EPISODE STATISTICS (last 100 episodes)

🎯 Total Reward per Episode:
   Mean:   4679.42
   Median: 4650.00

⏱️  Episode Length (steps):
   Mean:   1800.0

🏆 Win/Loss Record:
   Win Rate:    35.0%
   Wins:        35
   Losses:      65
```

### 4. Curriculum Learning

**Purpose**: Gradually increase difficulty as agent improves.

#### DifficultyScheduler (Step-based)

```python
from utils import DifficultyScheduler

# Create scheduler
scheduler = DifficultyScheduler()

# Or custom schedule
scheduler = DifficultyScheduler(schedule={
    0: 'idle',
    100_000: 'easy',
    500_000: 'medium',
    1_000_000: 'hard'
})

# During training
difficulty = scheduler.update(current_step=150_000)

# Print schedule
scheduler.print_schedule()
```

#### CurriculumManager (Performance-based)

```python
from utils import CurriculumManager

manager = CurriculumManager(
    initial_difficulty='idle',
    advancement_threshold=0.7,  # 70% win rate to advance
    min_episodes_before_advance=50
)

# After each episode
manager.record_episode(reward=1500, win=True)

# Check if should advance
if manager.should_increase_difficulty():
    manager.advance_difficulty()

# Get current difficulty
difficulty = manager.get_difficulty()
manager.print_status()
```

### 5. Modular Rewards

**Purpose**: Separate reward calculation into trackable components.

#### ModularRewardCalculator

```python
from utils import (
    ModularRewardCalculator,
    SurvivalReward,
    EconomicReward,
    MilitaryReward,
    create_default_reward_calculator
)

# Option 1: Use default configuration
calculator = create_default_reward_calculator()

# Option 2: Build custom calculator
calculator = ModularRewardCalculator()
calculator.add_component(SurvivalReward(reward_per_second=1.0, weight=1.0))
calculator.add_component(EconomicReward(resource_value=0.01, weight=0.5))
calculator.add_component(MilitaryReward(per_strength=5.0, weight=1.2))

# Calculate reward
total_reward, breakdown = calculator.calculate_with_breakdown(
    state=current_state,
    previous_state=prev_state,
    action='BUILD_FARM',
    delta_time=30.0
)

# Print configuration
calculator.print_configuration()
```

**Available Components:**
- `SurvivalReward` - Reward for staying alive
- `EconomicReward` - Reward for resources
- `PopulationReward` - Reward for population growth
- `MilitaryReward` - Reward for military strength
- `BuildingReward` - Reward for buildings
- `ActionValidityReward` - Penalties for invalid actions
- `EfficiencyReward` - Reward for resource efficiency
- `CombatReward` - Reward for combat

### 6. Testing Utilities

**Purpose**: Unit test your environment and reward functions.

#### EnvironmentTester

```python
from utils import EnvironmentTester

env = YourEnvironment()
tester = EnvironmentTester(env)

# Run all tests
tester.run_all_tests()

# Individual tests
tester.test_reset()
tester.test_step()
tester.test_episode_completion()
tester.test_action_masking()
```

#### RewardTester

```python
from utils import RewardTester

def your_reward_function(state, prev_state, action):
    return calculate_reward(...)

tester = RewardTester(your_reward_function)

# Test determinism
tester.test_determinism(test_state, prev_state, 'BUILD_FARM')

# Test bounds
test_cases = [(state1, prev1, 'ACTION1'), ...]
tester.test_bounds(test_cases, expected_min=-100, expected_max=1000)

# Test no NaN/Inf
tester.test_no_nan_inf(test_cases)

# Print results
tester.print_results()
```

### 7. Checkpoint Comparison

**Purpose**: Compare different model checkpoints to track progress.

#### CheckpointComparator

```python
from utils import CheckpointComparator, quick_compare

env = YourEnvironment()

# Option 1: Quick compare
quick_compare(env, {
    '100k': 'models/ppo_100k.zip',
    '500k': 'models/ppo_500k.zip',
    '1M': 'models/ppo_1m.zip'
}, n_episodes=20)

# Option 2: Detailed comparison
comparator = CheckpointComparator(env)
comparator.add_checkpoint('100k', 'models/ppo_100k.zip')
comparator.add_checkpoint('500k', 'models/ppo_500k.zip')

results = comparator.compare(n_episodes=20)
comparator.print_comparison()
comparator.plot_comparison(save_path='comparison.png')
comparator.export_results('comparison.json')
```

**Output Example:**
```
📊 CHECKPOINT COMPARISON

Checkpoint           Mean Reward     Win Rate    Avg Length
----------------------------------------------------------------------
1M                    5234.5 ± 423.1     45.0%     1850.0
500k                  4123.2 ± 512.3     28.0%     1780.0
100k                  2456.1 ± 678.9     12.0%     1650.0

🏆 Best Checkpoint: 1M
   Mean Reward: 5234.5
   Win Rate: 45.0%

📈 Improvement (100k → 1M):
   Reward: +2778.4 (+113.1%)
   Win Rate: +33.0%
```

## Examples

### Example 1: Basic Monitoring

```python
from utils import ActionMonitor, RewardComponentTracker

# Setup
action_monitor = ActionMonitor()
reward_tracker = RewardComponentTracker()

# Training loop
for episode in range(100):
    # ... your training code ...

    # Track actions
    for action in episode_actions:
        action_monitor.record(action)

    # Track rewards
    reward_tracker.log('economic', economic_reward)
    reward_tracker.log('military', military_reward)

    # End episode
    action_monitor.end_episode(episode)
    reward_tracker.end_episode(episode)

# Analyze
action_monitor.print_distribution()
reward_tracker.print_summary()
```

### Example 2: Complete Integration

See `examples/training_with_utils.py` for a complete integration example with custom callbacks.

## Integration Guide

### Step 1: Add to Training Loop

```python
from stable_baselines3.common.callbacks import BaseCallback
from utils import ActionMonitor, RewardComponentTracker, EpisodeStatsTracker

class MonitoringCallback(BaseCallback):
    def __init__(self, action_monitor, reward_tracker):
        super().__init__()
        self.action_monitor = action_monitor
        self.reward_tracker = reward_tracker

    def _on_step(self):
        # Track actions and rewards
        # ...
        return True
```

### Step 2: Periodic Analysis

```python
# Every N steps
if step % 10000 == 0:
    action_monitor.print_distribution()
    reward_tracker.print_summary()
```

### Step 3: Final Analysis

```python
# After training
action_monitor.save_to_file('actions.json')
reward_tracker.save_to_file('rewards.json')
episode_tracker.plot_trends(save_path='training.png')
```

## Troubleshooting

### Issue: Rewards are flat / not improving

**Diagnosis:**
```python
analyzer = ActionDistributionAnalyzer(action_monitor)
analyzer.print_diagnosis()
analyzer.suggest_fixes()
```

**Common causes:**
- DO_NOTHING dominance (>50%)
- Low action diversity (entropy <2.0)
- Reward components not balanced

### Issue: AI stuck on one action

**Diagnosis:**
```python
stuck_action = action_monitor.is_stuck_on_action(threshold=0.7)
if stuck_action:
    print(f"Stuck on: {stuck_action}")
```

**Solutions:**
- Increase entropy coefficient
- Add penalty for DO_NOTHING
- Check action masking

### Issue: Don't know which checkpoint is best

**Solution:**
```python
from utils import find_best_checkpoint

best_path, best_reward = find_best_checkpoint(
    env,
    checkpoint_dir='models/',
    n_episodes=20
)
```

## API Reference

See individual module docstrings for complete API documentation.

## Contributing

To add new utilities:

1. Create module in `utils/`
2. Add to `__init__.py`
3. Add documentation here
4. Add example to `examples/utils_demo.py`

## License

Part of the RTS training project.
