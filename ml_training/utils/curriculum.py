"""
Curriculum Learning Utilities
Gradually increase difficulty as the agent improves
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from enum import Enum


class DifficultyLevel(Enum):
    """Difficulty levels for opponent AI"""
    NORMAL = "normal"      # Balanced gameplay with behavior tree


class DifficultyScheduler:
    """
    Schedule difficulty progression during training

    Usage:
        scheduler = DifficultyScheduler()
        difficulty = scheduler.get_difficulty(current_step=50000)
    """

    def __init__(
        self,
        schedule: Optional[Dict[int, str]] = None,
        default_schedule: bool = True
    ):
        """
        Initialize scheduler

        Args:
            schedule: Custom schedule {step: difficulty_level}
            default_schedule: Use default progressive schedule if True
        """
        if schedule:
            self.schedule = schedule
        elif default_schedule:
            # Default schedule - only Normal difficulty for now
            self.schedule = {
                0: DifficultyLevel.NORMAL.value,      # Always use normal opponent
            }
        else:
            self.schedule = {0: DifficultyLevel.NORMAL.value}

        self.current_step = 0
        self.current_difficulty = self._get_difficulty_for_step(0)

    def _get_difficulty_for_step(self, step: int) -> str:
        """Get difficulty level for a given step"""
        # Find the highest threshold that's <= current step
        applicable_steps = [s for s in self.schedule.keys() if s <= step]
        if not applicable_steps:
            return DifficultyLevel.NORMAL.value

        threshold = max(applicable_steps)
        return self.schedule[threshold]

    def update(self, step: int) -> str:
        """
        Update current step and return appropriate difficulty

        Args:
            step: Current training step

        Returns:
            Difficulty level string
        """
        old_difficulty = self.current_difficulty
        self.current_step = step
        self.current_difficulty = self._get_difficulty_for_step(step)

        # Notify if difficulty changed
        if old_difficulty != self.current_difficulty:
            print(f"🎚️  Difficulty increased: {old_difficulty} → {self.current_difficulty} at step {step}")

        return self.current_difficulty

    def get_difficulty(self, step: Optional[int] = None) -> str:
        """Get difficulty for a step (or current if not specified)"""
        if step is not None:
            return self._get_difficulty_for_step(step)
        return self.current_difficulty

    def print_schedule(self):
        """Print the difficulty schedule"""
        print("=" * 80)
        print("📅 CURRICULUM SCHEDULE")
        print("=" * 80)

        sorted_schedule = sorted(self.schedule.items())
        for i, (step, difficulty) in enumerate(sorted_schedule):
            if i < len(sorted_schedule) - 1:
                next_step = sorted_schedule[i + 1][0]
                step_range = f"{step:,} - {next_step-1:,}"
            else:
                step_range = f"{step:,}+"

            marker = "📍" if self.current_step >= step else "⏳"
            print(f"{marker} Steps {step_range:>25}: {difficulty.upper()}")

        print(f"\nCurrent Step: {self.current_step:,}")
        print(f"Current Difficulty: {self.current_difficulty.upper()}")
        print("=" * 80)


class CurriculumManager:
    """
    Advanced curriculum manager with performance-based progression

    Usage:
        manager = CurriculumManager()
        manager.record_episode(reward=1500, win=True)
        should_advance = manager.should_increase_difficulty()
    """

    def __init__(
        self,
        initial_difficulty: str = DifficultyLevel.NORMAL.value,
        performance_window: int = 100,
        advancement_threshold: float = 0.7,
        min_episodes_before_advance: int = 50
    ):
        """
        Initialize curriculum manager

        Args:
            initial_difficulty: Starting difficulty
            performance_window: Number of episodes to evaluate
            advancement_threshold: Win rate to advance (0-1)
            min_episodes_before_advance: Minimum episodes before considering advancement
        """
        self.current_difficulty = initial_difficulty
        self.performance_window = performance_window
        self.advancement_threshold = advancement_threshold
        self.min_episodes = min_episodes_before_advance

        self.episode_results = []  # List of (reward, win) tuples
        self.difficulty_history = []

        # Difficulty progression order - only Normal for now
        self.progression = [
            DifficultyLevel.NORMAL.value,
        ]

    def record_episode(self, reward: float, win: bool):
        """Record episode result"""
        self.episode_results.append((reward, win))

    def get_recent_performance(self) -> Dict:
        """Get performance metrics for recent episodes"""
        if not self.episode_results:
            return {
                'num_episodes': 0,
                'win_rate': 0.0,
                'avg_reward': 0.0
            }

        recent = self.episode_results[-self.performance_window:]
        wins = sum(1 for _, win in recent if win)
        rewards = [reward for reward, _ in recent]

        return {
            'num_episodes': len(recent),
            'win_rate': wins / len(recent) if recent else 0.0,
            'avg_reward': np.mean(rewards) if rewards else 0.0,
            'reward_std': np.std(rewards) if rewards else 0.0
        }

    def should_increase_difficulty(self) -> bool:
        """
        Check if agent is ready for increased difficulty

        Returns:
            True if should advance
        """
        # Need minimum episodes
        if len(self.episode_results) < self.min_episodes:
            return False

        # Can't advance past max difficulty
        current_idx = self.progression.index(self.current_difficulty)
        if current_idx >= len(self.progression) - 1:
            return False

        # Check performance
        perf = self.get_recent_performance()
        return perf['win_rate'] >= self.advancement_threshold

    def advance_difficulty(self) -> bool:
        """
        Advance to next difficulty level

        Returns:
            True if advanced, False if already at max
        """
        current_idx = self.progression.index(self.current_difficulty)

        if current_idx >= len(self.progression) - 1:
            return False

        old_difficulty = self.current_difficulty
        self.current_difficulty = self.progression[current_idx + 1]

        self.difficulty_history.append({
            'episode': len(self.episode_results),
            'from': old_difficulty,
            'to': self.current_difficulty
        })

        print(f"🎚️  DIFFICULTY ADVANCED: {old_difficulty} → {self.current_difficulty}")
        perf = self.get_recent_performance()
        print(f"    (Win rate: {perf['win_rate']:.1%}, Avg reward: {perf['avg_reward']:.1f})")

        return True

    def get_difficulty(self) -> str:
        """Get current difficulty level"""
        return self.current_difficulty

    def print_status(self):
        """Print current curriculum status"""
        perf = self.get_recent_performance()

        print("=" * 80)
        print("📚 CURRICULUM STATUS")
        print("=" * 80)

        print(f"\nCurrent Difficulty: {self.current_difficulty.upper()}")
        print(f"Total Episodes:     {len(self.episode_results)}")

        print(f"\n📊 Recent Performance (last {perf['num_episodes']} episodes):")
        print(f"   Win Rate:     {perf['win_rate']:>7.1%}")
        print(f"   Avg Reward:   {perf['avg_reward']:>10.1f}")
        print(f"   Reward Std:   {perf['reward_std']:>10.1f}")

        # Progress to next level
        if self.should_increase_difficulty():
            print(f"\n✅ Ready to advance to next difficulty!")
        else:
            current_idx = self.progression.index(self.current_difficulty)
            if current_idx < len(self.progression) - 1:
                needed = self.advancement_threshold - perf['win_rate']
                print(f"\n⏳ Need {needed:.1%} more wins to advance")
                print(f"   (Current: {perf['win_rate']:.1%}, Target: {self.advancement_threshold:.1%})")
            else:
                print(f"\n🏆 At maximum difficulty!")

        # Show progression
        print(f"\n📈 Difficulty Progression:")
        for i, diff in enumerate(self.progression):
            if diff == self.current_difficulty:
                print(f"   📍 {diff.upper()} ← Current")
            else:
                status = "✓" if self.progression.index(self.current_difficulty) > i else " "
                print(f"   {status} {diff.upper()}")

        print("=" * 80)


class AdaptiveOpponentAI:
    """
    Opponent AI that adapts to player skill level

    Usage:
        opponent = AdaptiveOpponentAI()
        action = opponent.get_action(game_state, difficulty='normal')
    """

    def __init__(self):
        # NOTE: This class is deprecated - use scripted opponents from simulator.opponents instead
        # Keeping for backward compatibility only
        self.action_probabilities = {
            DifficultyLevel.NORMAL.value: {
                'DO_NOTHING': 0.2,
                'BUILD_FARM': 0.15,
                'BUILD_HOUSE': 0.15,
                'BUILD_BARRACKS': 0.1,
                'TRAIN_SOLDIER': 0.15,
                'BUILD_LUMBERYARD': 0.1,
                'BUILD_QUARRY': 0.1,
                'ATTACK': 0.05
            }
        }

    def get_action(self, game_state, difficulty: str, valid_actions: Optional[List] = None) -> str:
        """
        Get action for opponent based on difficulty

        Args:
            game_state: Current game state
            difficulty: Difficulty level
            valid_actions: List of valid action names (for masking)

        Returns:
            Action name
        """
        if difficulty not in self.action_probabilities:
            difficulty = DifficultyLevel.NORMAL.value

        probs = self.action_probabilities[difficulty].copy()

        # Filter to only valid actions if provided
        if valid_actions:
            probs = {action: prob for action, prob in probs.items() if action in valid_actions}

        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {action: prob / total for action, prob in probs.items()}
        else:
            return 'DO_NOTHING'

        # Sample action
        actions = list(probs.keys())
        probabilities = list(probs.values())

        return np.random.choice(actions, p=probabilities)
