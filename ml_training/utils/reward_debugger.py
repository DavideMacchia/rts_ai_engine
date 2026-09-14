"""
Reward Debugging Utilities
Track and analyze reward components to understand what the AI is learning
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json
from datetime import datetime


class RewardComponentTracker:
    """
    Tracks individual reward components to understand reward composition

    Usage:
        tracker = RewardComponentTracker()
        tracker.log('economic', 150.5, {'population': 10, 'gold': 500})
        tracker.log('military', 50.0, {'soldiers': 5})
        summary = tracker.get_summary()
    """

    def __init__(self):
        self.components = []
        self.episode_rewards = []
        self.current_episode = []

    def log(self, component_name: str, value: float, context: Optional[Dict] = None):
        """Log a reward component"""
        entry = {
            'component': component_name,
            'value': value,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        self.components.append(entry)
        self.current_episode.append(entry)

    def end_episode(self, episode_num: int):
        """Mark end of episode and save episode data"""
        if self.current_episode:
            episode_data = {
                'episode': episode_num,
                'components': self.current_episode.copy(),
                'total': sum(c['value'] for c in self.current_episode),
                'breakdown': self._get_component_breakdown(self.current_episode)
            }
            self.episode_rewards.append(episode_data)
            self.current_episode = []

    def _get_component_breakdown(self, components: List[Dict]) -> Dict[str, float]:
        """Get breakdown of rewards by component type"""
        breakdown = defaultdict(float)
        for comp in components:
            breakdown[comp['component']] += comp['value']
        return dict(breakdown)

    def get_summary(self, last_n_episodes: Optional[int] = None) -> Dict:
        """
        Get summary statistics of reward components

        Args:
            last_n_episodes: Only analyze last N episodes (None = all)

        Returns:
            Dictionary with summary statistics
        """
        episodes = self.episode_rewards
        if last_n_episodes:
            episodes = episodes[-last_n_episodes:]

        if not episodes:
            return {'error': 'No episodes recorded'}

        # Aggregate all components
        all_breakdowns = defaultdict(list)
        total_rewards = []

        for ep in episodes:
            total_rewards.append(ep['total'])
            for comp_name, value in ep['breakdown'].items():
                all_breakdowns[comp_name].append(value)

        # Calculate statistics for each component
        component_stats = {}
        for comp_name, values in all_breakdowns.items():
            component_stats[comp_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'total': np.sum(values),
                'percentage': (np.sum(values) / np.sum(total_rewards) * 100) if np.sum(total_rewards) > 0 else 0
            }

        return {
            'episodes_analyzed': len(episodes),
            'total_reward': {
                'mean': np.mean(total_rewards),
                'std': np.std(total_rewards),
                'min': np.min(total_rewards),
                'max': np.max(total_rewards)
            },
            'component_breakdown': component_stats
        }

    def print_summary(self, last_n_episodes: Optional[int] = None):
        """Print formatted summary"""
        summary = self.get_summary(last_n_episodes)

        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return

        print("=" * 80)
        print(f"📊 REWARD ANALYSIS ({summary['episodes_analyzed']} episodes)")
        print("=" * 80)

        total = summary['total_reward']
        print(f"\n🎯 Total Reward per Episode:")
        print(f"   Mean:  {total['mean']:>10.2f}")
        print(f"   Std:   {total['std']:>10.2f}")
        print(f"   Range: {total['min']:>10.2f} → {total['max']:>10.2f}")

        print(f"\n📈 Component Breakdown:")
        print(f"{'Component':<25} {'Mean':<12} {'% of Total':<12} {'Range'}")
        print("-" * 80)

        # Sort by percentage descending
        components = sorted(
            summary['component_breakdown'].items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )

        for comp_name, stats in components:
            print(f"{comp_name:<25} "
                  f"{stats['mean']:>10.2f}  "
                  f"{stats['percentage']:>10.1f}%  "
                  f"{stats['min']:>7.1f} → {stats['max']:>7.1f}")

        print("=" * 80)

    def save_to_file(self, filepath: str):
        """Save tracking data to JSON file"""
        data = {
            'episodes': self.episode_rewards,
            'summary': self.get_summary()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved reward tracking data to {filepath}")


class RewardDebugger:
    """
    Real-time reward debugger for understanding what's happening during training

    Usage:
        debugger = RewardDebugger()
        debugger.log_step(action='BUILD_FARM', reward=10.5, state={'pop': 5})
        debugger.print_recent(n=10)
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.step_history = []
        self.anomaly_threshold = 3.0  # Standard deviations for anomaly detection

    def log_step(self, action: str, reward: float, state: Optional[Dict] = None,
                 reward_components: Optional[Dict] = None):
        """Log a single step"""
        entry = {
            'step': len(self.step_history),
            'action': action,
            'reward': reward,
            'state': state or {},
            'components': reward_components or {},
            'timestamp': datetime.now().isoformat()
        }
        self.step_history.append(entry)

        # Keep buffer size manageable
        if len(self.step_history) > self.buffer_size:
            self.step_history.pop(0)

    def detect_anomalies(self, window: int = 100) -> List[Dict]:
        """Detect unusual reward patterns"""
        if len(self.step_history) < window:
            return []

        recent_rewards = [s['reward'] for s in self.step_history[-window:]]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        anomalies = []
        for step in self.step_history[-window:]:
            if abs(step['reward'] - mean_reward) > self.anomaly_threshold * std_reward:
                anomalies.append({
                    'step': step['step'],
                    'action': step['action'],
                    'reward': step['reward'],
                    'deviation': abs(step['reward'] - mean_reward) / std_reward if std_reward > 0 else 0
                })

        return anomalies

    def print_recent(self, n: int = 10):
        """Print recent reward history"""
        recent = self.step_history[-n:]

        print("=" * 80)
        print(f"📋 Recent Reward History (last {len(recent)} steps)")
        print("=" * 80)
        print(f"{'Step':<8} {'Action':<25} {'Reward':<12} {'Details'}")
        print("-" * 80)

        for entry in recent:
            components_str = ""
            if entry['components']:
                top_comp = max(entry['components'].items(), key=lambda x: abs(x[1]))
                components_str = f"{top_comp[0]}={top_comp[1]:.1f}"

            print(f"{entry['step']:<8} "
                  f"{entry['action']:<25} "
                  f"{entry['reward']:>10.2f}  "
                  f"{components_str}")

        print("=" * 80)

    def get_action_reward_correlation(self) -> Dict[str, Dict]:
        """Analyze which actions lead to highest/lowest rewards"""
        action_rewards = defaultdict(list)

        for step in self.step_history:
            action_rewards[step['action']].append(step['reward'])

        stats = {}
        for action, rewards in action_rewards.items():
            stats[action] = {
                'count': len(rewards),
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            }

        return stats

    def print_action_analysis(self):
        """Print analysis of action-reward relationships"""
        stats = self.get_action_reward_correlation()

        print("=" * 80)
        print("🎮 Action-Reward Analysis")
        print("=" * 80)
        print(f"{'Action':<25} {'Count':<8} {'Mean Reward':<15} {'Range'}")
        print("-" * 80)

        # Sort by mean reward descending
        sorted_actions = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)

        for action, data in sorted_actions:
            print(f"{action:<25} "
                  f"{data['count']:<8} "
                  f"{data['mean']:>12.2f}  "
                  f"{data['min']:>7.1f} → {data['max']:>7.1f}")

        print("=" * 80)
