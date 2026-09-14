"""
Action Monitoring Utilities
Track what actions the AI is taking to understand its behavior
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from datetime import datetime


class ActionMonitor:
    """
    Monitor action distribution during training

    Usage:
        monitor = ActionMonitor()
        monitor.record('BUILD_FARM')
        monitor.record('DO_NOTHING')
        monitor.print_distribution()
    """

    def __init__(self, window_size: int = 1000):
        self.action_counts = defaultdict(int)
        self.action_history = deque(maxlen=window_size)
        self.episode_actions = []
        self.current_episode_actions = []
        self.total_actions = 0

    def record(self, action: str):
        """Record a single action"""
        self.action_counts[action] += 1
        self.action_history.append(action)
        self.current_episode_actions.append(action)
        self.total_actions += 1

    def end_episode(self, episode_num: int):
        """Mark end of episode"""
        if self.current_episode_actions:
            self.episode_actions.append({
                'episode': episode_num,
                'actions': self.current_episode_actions.copy(),
                'distribution': self._get_distribution(self.current_episode_actions)
            })
            self.current_episode_actions = []

    def _get_distribution(self, actions: List[str]) -> Dict[str, float]:
        """Get action distribution for a list of actions"""
        if not actions:
            return {}

        counts = defaultdict(int)
        for action in actions:
            counts[action] += 1

        total = len(actions)
        return {action: count / total for action, count in counts.items()}

    def get_distribution(self, recent: bool = True) -> Dict[str, float]:
        """
        Get action distribution

        Args:
            recent: If True, use only recent actions in window. If False, use all actions.

        Returns:
            Dictionary of action -> probability
        """
        if recent:
            actions_to_analyze = list(self.action_history)
        else:
            actions_to_analyze = [action for action in self.action_counts.keys()]
            # Reconstruct from counts
            total = sum(self.action_counts.values())
            if total == 0:
                return {}
            return {action: count / total for action, count in self.action_counts.items()}

        return self._get_distribution(actions_to_analyze)

    def get_diversity_score(self, recent: bool = True) -> float:
        """
        Calculate action diversity using Shannon entropy
        Higher = more diverse actions, Lower = repetitive

        Returns:
            Entropy score (0 = all same action, higher = more diverse)
        """
        dist = self.get_distribution(recent=recent)
        if not dist:
            return 0.0

        entropy = 0.0
        for prob in dist.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)

        return entropy

    def is_stuck_on_action(self, threshold: float = 0.7) -> Optional[str]:
        """
        Check if AI is stuck repeating one action

        Args:
            threshold: If one action is >threshold% of all actions, consider stuck

        Returns:
            Action name if stuck, None otherwise
        """
        dist = self.get_distribution(recent=True)
        if not dist:
            return None

        max_action = max(dist.items(), key=lambda x: x[1])
        if max_action[1] > threshold:
            return max_action[0]

        return None

    def print_distribution(self, recent: bool = True, top_n: Optional[int] = None):
        """Print action distribution"""
        dist = self.get_distribution(recent=recent)
        diversity = self.get_diversity_score(recent=recent)

        window_desc = "recent window" if recent else "all time"
        actions_count = len(self.action_history) if recent else self.total_actions

        print("=" * 80)
        print(f"🎮 Action Distribution ({window_desc} - {actions_count} actions)")
        print("=" * 80)
        print(f"Diversity Score: {diversity:.3f} (higher = more exploration)")
        print()

        # Check if stuck
        stuck_action = self.is_stuck_on_action()
        if stuck_action:
            print(f"⚠️  WARNING: Agent appears stuck on '{stuck_action}'")
            print()

        # Sort by frequency
        sorted_actions = sorted(dist.items(), key=lambda x: x[1], reverse=True)

        if top_n:
            sorted_actions = sorted_actions[:top_n]

        print(f"{'Action':<30} {'Frequency':<12} {'Bar'}")
        print("-" * 80)

        for action, freq in sorted_actions:
            bar_length = int(freq * 50)
            bar = "█" * bar_length
            print(f"{action:<30} {freq:>10.1%}  {bar}")

        print("=" * 80)

    def get_action_trends(self, window_size: int = 100) -> Dict[str, List[float]]:
        """
        Get action frequency trends over time

        Args:
            window_size: Size of sliding window for trend calculation

        Returns:
            Dict of action -> list of frequencies over time
        """
        if len(self.episode_actions) < 2:
            return {}

        trends = defaultdict(list)

        for episode_data in self.episode_actions:
            dist = episode_data['distribution']
            # Add frequency for each action (0 if not present)
            all_actions = set()
            for ep in self.episode_actions:
                all_actions.update(ep['distribution'].keys())

            for action in all_actions:
                trends[action].append(dist.get(action, 0.0))

        return dict(trends)

    def plot_trends(self, save_path: Optional[str] = None, top_n: int = 10):
        """
        Plot action trends over episodes

        Args:
            save_path: Path to save plot (None = display)
            top_n: Number of top actions to plot
        """
        trends = self.get_action_trends()

        if not trends:
            print("❌ Not enough data to plot trends")
            return

        # Get top N most frequent actions
        avg_freqs = {action: np.mean(freqs) for action, freqs in trends.items()}
        top_actions = sorted(avg_freqs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_action_names = [action for action, _ in top_actions]

        # Plot
        plt.figure(figsize=(12, 6))
        for action in top_action_names:
            freqs = trends[action]
            plt.plot(freqs, label=action, marker='o', markersize=3)

        plt.xlabel('Episode')
        plt.ylabel('Action Frequency')
        plt.title('Action Distribution Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved action trends plot to {save_path}")
        else:
            plt.show()

        plt.close()


class ActionDistributionAnalyzer:
    """
    Advanced analyzer for action patterns

    Usage:
        analyzer = ActionDistributionAnalyzer(monitor)
        issues = analyzer.diagnose()
        analyzer.print_diagnosis()
    """

    def __init__(self, monitor: ActionMonitor):
        self.monitor = monitor

    def diagnose(self) -> Dict[str, any]:
        """
        Diagnose potential issues with action distribution

        Returns:
            Dictionary with diagnosis results
        """
        dist = self.monitor.get_distribution(recent=True)
        diversity = self.monitor.get_diversity_score(recent=True)

        issues = []
        warnings = []
        info = []

        # Check for DO_NOTHING dominance
        do_nothing_freq = dist.get('DO_NOTHING', 0.0)
        if do_nothing_freq > 0.8:
            issues.append(f"🚨 CRITICAL: DO_NOTHING at {do_nothing_freq:.1%} - Agent not learning!")
        elif do_nothing_freq > 0.5:
            warnings.append(f"⚠️  DO_NOTHING high at {do_nothing_freq:.1%} - May need more exploration")
        elif do_nothing_freq < 0.1:
            info.append(f"✅ Low DO_NOTHING ({do_nothing_freq:.1%}) - Good action diversity")

        # Check diversity
        if diversity < 1.0:
            issues.append(f"🚨 CRITICAL: Very low diversity (entropy={diversity:.2f}) - Agent stuck!")
        elif diversity < 2.0:
            warnings.append(f"⚠️  Low diversity (entropy={diversity:.2f}) - Limited exploration")
        else:
            info.append(f"✅ Good diversity (entropy={diversity:.2f})")

        # Check for action variety
        num_unique_actions = len(dist)
        if num_unique_actions < 3:
            issues.append(f"🚨 CRITICAL: Only {num_unique_actions} unique actions used")
        elif num_unique_actions < 5:
            warnings.append(f"⚠️  Limited action variety: {num_unique_actions} unique actions")
        else:
            info.append(f"✅ Good action variety: {num_unique_actions} unique actions")

        # Check for stuck patterns
        stuck_action = self.monitor.is_stuck_on_action(threshold=0.7)
        if stuck_action:
            issues.append(f"🚨 CRITICAL: Stuck on '{stuck_action}'")

        return {
            'diversity_score': diversity,
            'num_unique_actions': num_unique_actions,
            'do_nothing_frequency': do_nothing_freq,
            'issues': issues,
            'warnings': warnings,
            'info': info,
            'distribution': dist
        }

    def print_diagnosis(self):
        """Print formatted diagnosis"""
        diagnosis = self.diagnose()

        print("=" * 80)
        print("🔍 ACTION DISTRIBUTION DIAGNOSIS")
        print("=" * 80)

        # Print issues
        if diagnosis['issues']:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in diagnosis['issues']:
                print(f"   {issue}")

        # Print warnings
        if diagnosis['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in diagnosis['warnings']:
                print(f"   {warning}")

        # Print info
        if diagnosis['info']:
            print("\n✅ POSITIVE INDICATORS:")
            for item in diagnosis['info']:
                print(f"   {item}")

        # Summary stats
        print("\n📊 SUMMARY:")
        print(f"   Diversity Score:      {diagnosis['diversity_score']:.3f}")
        print(f"   Unique Actions:       {diagnosis['num_unique_actions']}")
        print(f"   DO_NOTHING Frequency: {diagnosis['do_nothing_frequency']:.1%}")

        print("\n📈 TOP 5 ACTIONS:")
        sorted_dist = sorted(diagnosis['distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
        for action, freq in sorted_dist:
            print(f"   {action:<30} {freq:>7.1%}")

        print("=" * 80)

    def suggest_fixes(self):
        """Suggest potential fixes based on diagnosis"""
        diagnosis = self.diagnose()

        print("=" * 80)
        print("💡 SUGGESTED FIXES")
        print("=" * 80)

        if diagnosis['do_nothing_frequency'] > 0.5:
            print("\n🎯 Reduce DO_NOTHING frequency:")
            print("   • Increase exploration: Set entropy_coef higher (try 0.01-0.1)")
            print("   • Add penalty for DO_NOTHING in reward function")
            print("   • Check if action masking is too restrictive")

        if diagnosis['diversity_score'] < 2.0:
            print("\n🎯 Increase action diversity:")
            print("   • Increase entropy coefficient in PPO")
            print("   • Reduce learning rate (agent may be converging too fast)")
            print("   • Add reward shaping to encourage varied actions")
            print("   • Check if some actions are never available (action masking issue)")

        if diagnosis['num_unique_actions'] < 5:
            print("\n🎯 Increase action variety:")
            print("   • Verify action masking isn't too strict")
            print("   • Add rewards for trying new actions")
            print("   • Check if some actions are impossible due to game state")
            print("   • Consider curriculum learning (start with subset of actions)")

        if not diagnosis['issues'] and not diagnosis['warnings']:
            print("\n✅ Action distribution looks healthy!")
            print("   Continue monitoring for any changes over time.")

        print("=" * 80)
