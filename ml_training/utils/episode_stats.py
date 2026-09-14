"""
Episode Statistics Tracking
Track and analyze episode-level statistics during training
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
import json
from datetime import datetime
import matplotlib.pyplot as plt


class EpisodeStatsTracker:
    """
    Track detailed statistics for each episode

    Usage:
        tracker = EpisodeStatsTracker()
        tracker.start_episode()
        tracker.record_step(action='BUILD_FARM', reward=10.5)
        tracker.end_episode(total_reward=1500, win=True)
        tracker.print_summary()
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episodes = []
        self.current_episode = None
        self.current_episode_steps = []

    def start_episode(self, episode_num: Optional[int] = None):
        """Start tracking a new episode"""
        if episode_num is None:
            episode_num = len(self.episodes)

        self.current_episode = {
            'episode_num': episode_num,
            'start_time': datetime.now().isoformat(),
            'steps': 0,
            'actions': [],
            'rewards': [],
            'states': []
        }
        self.current_episode_steps = []

    def record_step(self, action: str, reward: float, state: Optional[Dict] = None):
        """Record a step within the current episode"""
        if self.current_episode is None:
            raise RuntimeError("No episode started. Call start_episode() first.")

        self.current_episode['steps'] += 1
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)

        if state:
            self.current_episode['states'].append(state)

    def end_episode(
        self,
        total_reward: float,
        win: bool,
        final_state: Optional[Dict] = None
    ):
        """End current episode and save statistics"""
        if self.current_episode is None:
            raise RuntimeError("No episode started. Call start_episode() first.")

        self.current_episode['end_time'] = datetime.now().isoformat()
        self.current_episode['total_reward'] = total_reward
        self.current_episode['win'] = win
        self.current_episode['final_state'] = final_state or {}

        # Calculate summary stats
        self.current_episode['reward_mean'] = np.mean(self.current_episode['rewards'])
        self.current_episode['reward_std'] = np.std(self.current_episode['rewards'])
        self.current_episode['reward_min'] = np.min(self.current_episode['rewards'])
        self.current_episode['reward_max'] = np.max(self.current_episode['rewards'])

        # Action distribution
        action_counts = {}
        for action in self.current_episode['actions']:
            action_counts[action] = action_counts.get(action, 0) + 1
        self.current_episode['action_distribution'] = action_counts

        self.episodes.append(self.current_episode)
        self.current_episode = None

    def get_recent_stats(self, n: Optional[int] = None) -> Dict:
        """
        Get statistics for recent episodes

        Args:
            n: Number of recent episodes (None = use window_size)

        Returns:
            Dictionary with aggregated statistics
        """
        if n is None:
            n = self.window_size

        recent = self.episodes[-n:] if len(self.episodes) > n else self.episodes

        if not recent:
            return {'error': 'No episodes recorded'}

        # Aggregate statistics
        total_rewards = [ep['total_reward'] for ep in recent]
        episode_lengths = [ep['steps'] for ep in recent]
        wins = [ep['win'] for ep in recent]

        stats = {
            'num_episodes': len(recent),
            'total_reward': {
                'mean': np.mean(total_rewards),
                'std': np.std(total_rewards),
                'min': np.min(total_rewards),
                'max': np.max(total_rewards),
                'median': np.median(total_rewards)
            },
            'episode_length': {
                'mean': np.mean(episode_lengths),
                'std': np.std(episode_lengths),
                'min': np.min(episode_lengths),
                'max': np.max(episode_lengths)
            },
            'win_rate': np.mean(wins) if wins else 0.0,
            'total_wins': sum(wins),
            'total_losses': len(wins) - sum(wins)
        }

        # Most common actions
        all_actions = {}
        for ep in recent:
            for action, count in ep['action_distribution'].items():
                all_actions[action] = all_actions.get(action, 0) + count

        total_actions = sum(all_actions.values())
        action_frequencies = {
            action: count / total_actions
            for action, count in all_actions.items()
        }
        stats['action_frequencies'] = dict(
            sorted(action_frequencies.items(), key=lambda x: x[1], reverse=True)
        )

        return stats

    def get_trend_data(self) -> Dict[str, List[float]]:
        """Get time-series data for plotting trends"""
        if not self.episodes:
            return {}

        return {
            'episode_nums': [ep['episode_num'] for ep in self.episodes],
            'total_rewards': [ep['total_reward'] for ep in self.episodes],
            'episode_lengths': [ep['steps'] for ep in self.episodes],
            'wins': [1 if ep['win'] else 0 for ep in self.episodes]
        }

    def print_summary(self, n: Optional[int] = None):
        """Print summary of recent episodes"""
        stats = self.get_recent_stats(n)

        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return

        n_display = n or self.window_size
        print("=" * 80)
        print(f"📊 EPISODE STATISTICS (last {stats['num_episodes']} episodes)")
        print("=" * 80)

        # Rewards
        reward_stats = stats['total_reward']
        print(f"\n🎯 Total Reward per Episode:")
        print(f"   Mean:   {reward_stats['mean']:>12.2f}")
        print(f"   Median: {reward_stats['median']:>12.2f}")
        print(f"   Std:    {reward_stats['std']:>12.2f}")
        print(f"   Range:  {reward_stats['min']:>12.2f} → {reward_stats['max']:>12.2f}")

        # Episode length
        length_stats = stats['episode_length']
        print(f"\n⏱️  Episode Length (steps):")
        print(f"   Mean:   {length_stats['mean']:>12.1f}")
        print(f"   Std:    {length_stats['std']:>12.1f}")
        print(f"   Range:  {length_stats['min']:>12.0f} → {length_stats['max']:>12.0f}")

        # Win rate
        print(f"\n🏆 Win/Loss Record:")
        print(f"   Win Rate: {stats['win_rate']:>10.1%}")
        print(f"   Wins:     {stats['total_wins']:>10}")
        print(f"   Losses:   {stats['total_losses']:>10}")

        # Top actions
        print(f"\n🎮 Top 5 Actions:")
        top_actions = list(stats['action_frequencies'].items())[:5]
        for action, freq in top_actions:
            print(f"   {action:<30} {freq:>7.1%}")

        print("=" * 80)

    def plot_trends(self, save_path: Optional[str] = None, show_rolling_avg: bool = True):
        """
        Plot training trends over time

        Args:
            save_path: Path to save plot (None = display)
            show_rolling_avg: Show rolling average overlay
        """
        trend_data = self.get_trend_data()

        if not trend_data:
            print("❌ No data to plot")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        episodes = trend_data['episode_nums']
        rewards = trend_data['total_rewards']
        lengths = trend_data['episode_lengths']
        wins = trend_data['wins']

        # Plot 1: Rewards
        axes[0].plot(episodes, rewards, alpha=0.3, label='Episode Reward')
        if show_rolling_avg and len(rewards) > 10:
            window = min(20, len(rewards) // 5)
            rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            rolling_episodes = episodes[window-1:]
            axes[0].plot(rolling_episodes, rolling_avg, linewidth=2, label=f'{window}-Episode Average')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Reward Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Episode Length
        axes[1].plot(episodes, lengths, alpha=0.3, label='Episode Length')
        if show_rolling_avg and len(lengths) > 10:
            window = min(20, len(lengths) // 5)
            rolling_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            rolling_episodes = episodes[window-1:]
            axes[1].plot(rolling_episodes, rolling_avg, linewidth=2, label=f'{window}-Episode Average')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].set_title('Episode Length')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Win Rate (cumulative)
        if len(wins) > 10:
            window = min(50, len(wins) // 3)
            win_rate = np.convolve(wins, np.ones(window)/window, mode='valid')
            win_episodes = episodes[window-1:]
            axes[2].plot(win_episodes, win_rate, linewidth=2)
            axes[2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Win Rate')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Win Rate')
            axes[2].set_title(f'Win Rate ({window}-Episode Moving Average)')
            axes[2].set_ylim(0, 1)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved training trends plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def save_to_file(self, filepath: str):
        """Save all episode data to JSON file"""
        data = {
            'episodes': self.episodes,
            'summary': self.get_recent_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved episode data to {filepath}")


class EpisodeAnalyzer:
    """
    Advanced analyzer for episode patterns and issues

    Usage:
        analyzer = EpisodeAnalyzer(tracker)
        analysis = analyzer.analyze()
        analyzer.print_analysis()
    """

    def __init__(self, tracker: EpisodeStatsTracker):
        self.tracker = tracker

    def analyze(self, window: int = 100) -> Dict:
        """
        Perform comprehensive analysis of recent episodes

        Args:
            window: Number of recent episodes to analyze

        Returns:
            Dictionary with analysis results
        """
        stats = self.tracker.get_recent_stats(window)
        trend_data = self.tracker.get_trend_data()

        if 'error' in stats:
            return stats

        issues = []
        warnings = []
        insights = []

        # Check reward growth
        if len(trend_data['total_rewards']) >= 50:
            recent_50 = trend_data['total_rewards'][-50:]
            older_50 = trend_data['total_rewards'][-100:-50] if len(trend_data['total_rewards']) >= 100 else recent_50

            recent_mean = np.mean(recent_50)
            older_mean = np.mean(older_50)

            improvement = (recent_mean - older_mean) / abs(older_mean) if older_mean != 0 else 0

            if improvement < 0.01:  # Less than 1% improvement
                issues.append(f"🚨 Reward plateau detected! Improvement: {improvement:.1%}")
            elif improvement < 0.05:
                warnings.append(f"⚠️  Slow reward growth: {improvement:.1%}")
            else:
                insights.append(f"✅ Good reward growth: {improvement:.1%}")

        # Check reward variance
        reward_std = stats['total_reward']['std']
        reward_mean = stats['total_reward']['mean']
        if reward_mean > 0:
            cv = reward_std / reward_mean  # Coefficient of variation
            if cv > 1.0:
                warnings.append(f"⚠️  High reward variance (CV={cv:.2f}) - unstable training")
            elif cv < 0.2:
                insights.append(f"✅ Stable rewards (CV={cv:.2f})")

        # Check win rate
        win_rate = stats['win_rate']
        if win_rate < 0.1:
            issues.append(f"🚨 Very low win rate: {win_rate:.1%}")
        elif win_rate < 0.3:
            warnings.append(f"⚠️  Low win rate: {win_rate:.1%}")
        elif win_rate > 0.7:
            insights.append(f"✅ Strong win rate: {win_rate:.1%}")

        # Check episode length consistency
        length_std = stats['episode_length']['std']
        length_mean = stats['episode_length']['mean']
        if length_std / length_mean > 0.5:
            warnings.append(f"⚠️  Inconsistent episode lengths")

        return {
            'statistics': stats,
            'issues': issues,
            'warnings': warnings,
            'insights': insights
        }

    def print_analysis(self, window: int = 100):
        """Print formatted analysis"""
        analysis = self.analyze(window)

        if 'error' in analysis:
            print(f"❌ {analysis['error']}")
            return

        print("=" * 80)
        print(f"🔍 EPISODE ANALYSIS (last {window} episodes)")
        print("=" * 80)

        # Print issues
        if analysis['issues']:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in analysis['issues']:
                print(f"   {issue}")

        # Print warnings
        if analysis['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in analysis['warnings']:
                print(f"   {warning}")

        # Print insights
        if analysis['insights']:
            print("\n✅ POSITIVE INSIGHTS:")
            for insight in analysis['insights']:
                print(f"   {insight}")

        # Print statistics summary
        stats = analysis['statistics']
        print(f"\n📊 KEY METRICS:")
        print(f"   Avg Reward:      {stats['total_reward']['mean']:>10.1f}")
        print(f"   Avg Length:      {stats['episode_length']['mean']:>10.1f} steps")
        print(f"   Win Rate:        {stats['win_rate']:>10.1%}")

        print("=" * 80)

    def compare_periods(self, period1_range: tuple, period2_range: tuple):
        """
        Compare two training periods

        Args:
            period1_range: (start_ep, end_ep) for first period
            period2_range: (start_ep, end_ep) for second period
        """
        episodes = self.tracker.episodes

        period1 = [ep for ep in episodes if period1_range[0] <= ep['episode_num'] < period1_range[1]]
        period2 = [ep for ep in episodes if period2_range[0] <= ep['episode_num'] < period2_range[1]]

        if not period1 or not period2:
            print("❌ Not enough data in one or both periods")
            return

        def calc_stats(period):
            rewards = [ep['total_reward'] for ep in period]
            wins = [ep['win'] for ep in period]
            return {
                'mean_reward': np.mean(rewards),
                'win_rate': np.mean(wins)
            }

        stats1 = calc_stats(period1)
        stats2 = calc_stats(period2)

        print("=" * 80)
        print("📊 PERIOD COMPARISON")
        print("=" * 80)

        print(f"\nPeriod 1: Episodes {period1_range[0]}-{period1_range[1]}")
        print(f"   Mean Reward: {stats1['mean_reward']:>10.1f}")
        print(f"   Win Rate:    {stats1['win_rate']:>10.1%}")

        print(f"\nPeriod 2: Episodes {period2_range[0]}-{period2_range[1]}")
        print(f"   Mean Reward: {stats2['mean_reward']:>10.1f}")
        print(f"   Win Rate:    {stats2['win_rate']:>10.1%}")

        print(f"\nChange:")
        reward_change = stats2['mean_reward'] - stats1['mean_reward']
        reward_pct = (reward_change / stats1['mean_reward'] * 100) if stats1['mean_reward'] != 0 else 0
        print(f"   Reward: {reward_change:>+10.1f} ({reward_pct:>+6.1f}%)")

        win_change = stats2['win_rate'] - stats1['win_rate']
        print(f"   Win Rate: {win_change:>+8.1%}")

        print("=" * 80)
