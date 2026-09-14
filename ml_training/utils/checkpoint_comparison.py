"""
Checkpoint Comparison Tool
Compare different model checkpoints to track training progress
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from pathlib import Path
import json


class CheckpointComparator:
    """
    Compare multiple model checkpoints

    Usage:
        comparator = CheckpointComparator(env)
        comparator.add_checkpoint('100k', 'models/ppo_100k.zip')
        comparator.add_checkpoint('500k', 'models/ppo_500k.zip')
        results = comparator.compare(n_episodes=20)
        comparator.print_comparison()
    """

    def __init__(self, env):
        self.env = env
        self.checkpoints = {}
        self.results = {}

    def add_checkpoint(self, name: str, model_path: str):
        """
        Add a checkpoint to compare

        Args:
            name: Name/label for this checkpoint (e.g., '100k steps')
            model_path: Path to saved model
        """
        if not Path(model_path).exists():
            print(f"⚠️  Warning: Model file not found: {model_path}")
            return False

        self.checkpoints[name] = model_path
        return True

    def evaluate_checkpoint(
        self,
        model_path: str,
        n_episodes: int = 20,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate a single checkpoint

        Args:
            model_path: Path to model
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            model = PPO.load(model_path)
        except Exception as e:
            return {'error': f'Failed to load model: {str(e)}'}

        episode_rewards = []
        episode_lengths = []
        wins = []
        action_counts = {}

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Track actions
                action_counts[action] = action_counts.get(action, 0) + 1

                episode_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

            # Check if won (if info contains winner)
            if 'winner' in info:
                wins.append(info['winner'] == 0)  # Assuming agent is faction 0
            else:
                wins.append(False)

        # Calculate metrics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'win_rate': np.mean(wins) if wins else 0.0,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'action_distribution': action_counts,
            'n_episodes': n_episodes
        }

        return results

    def compare(self, n_episodes: int = 20, deterministic: bool = True):
        """
        Compare all added checkpoints

        Args:
            n_episodes: Number of episodes per checkpoint
            deterministic: Use deterministic actions

        Returns:
            Dictionary with comparison results
        """
        print("=" * 80)
        print(f"🔍 COMPARING {len(self.checkpoints)} CHECKPOINTS")
        print("=" * 80)

        for name, model_path in self.checkpoints.items():
            print(f"\nEvaluating: {name}")
            print(f"   Model: {model_path}")

            results = self.evaluate_checkpoint(model_path, n_episodes, deterministic)

            if 'error' in results:
                print(f"   ❌ {results['error']}")
                self.results[name] = results
            else:
                self.results[name] = results
                print(f"   ✅ Mean Reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
                print(f"   ✅ Win Rate: {results['win_rate']:.1%}")

        print("\n" + "=" * 80)

        return self.results

    def print_comparison(self):
        """Print detailed comparison table"""
        if not self.results:
            print("❌ No results available. Run compare() first.")
            return

        # Filter out errors
        valid_results = {name: res for name, res in self.results.items() if 'error' not in res}

        if not valid_results:
            print("❌ No valid results to compare")
            return

        print("=" * 80)
        print("📊 CHECKPOINT COMPARISON")
        print("=" * 80)

        # Header
        print(f"\n{'Checkpoint':<20} {'Mean Reward':<15} {'Win Rate':<12} {'Avg Length'}")
        print("-" * 80)

        # Sort by mean reward
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)

        for name, res in sorted_results:
            print(f"{name:<20} "
                  f"{res['mean_reward']:>10.1f} ± {res['std_reward']:<6.1f}  "
                  f"{res['win_rate']:>7.1%}     "
                  f"{res['mean_length']:>8.1f}")

        print("=" * 80)

        # Best checkpoint
        best_name, best_res = sorted_results[0]
        print(f"\n🏆 Best Checkpoint: {best_name}")
        print(f"   Mean Reward: {best_res['mean_reward']:.1f}")
        print(f"   Win Rate: {best_res['win_rate']:.1%}")

        # Improvement analysis
        if len(sorted_results) >= 2:
            worst_name, worst_res = sorted_results[-1]
            improvement = best_res['mean_reward'] - worst_res['mean_reward']
            pct_improvement = (improvement / abs(worst_res['mean_reward']) * 100) if worst_res['mean_reward'] != 0 else 0

            print(f"\n📈 Improvement ({worst_name} → {best_name}):")
            print(f"   Reward: +{improvement:.1f} ({pct_improvement:+.1f}%)")
            print(f"   Win Rate: {best_res['win_rate'] - worst_res['win_rate']:+.1%}")

        print("=" * 80)

    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Create comparison plots

        Args:
            save_path: Path to save plot (None = display)
        """
        valid_results = {name: res for name, res in self.results.items() if 'error' not in res}

        if not valid_results:
            print("❌ No valid results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        checkpoint_names = list(valid_results.keys())
        n_checkpoints = len(checkpoint_names)

        # Plot 1: Mean Rewards with error bars
        ax = axes[0, 0]
        means = [valid_results[name]['mean_reward'] for name in checkpoint_names]
        stds = [valid_results[name]['std_reward'] for name in checkpoint_names]
        x_pos = np.arange(n_checkpoints)

        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(checkpoint_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Mean Reward by Checkpoint')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Win Rates
        ax = axes[0, 1]
        win_rates = [valid_results[name]['win_rate'] for name in checkpoint_names]

        ax.bar(x_pos, win_rates, alpha=0.7, color='green')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(checkpoint_names, rotation=45, ha='right')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate by Checkpoint')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Reward distributions (box plot)
        ax = axes[1, 0]
        reward_distributions = [valid_results[name]['episode_rewards'] for name in checkpoint_names]

        bp = ax.boxplot(reward_distributions, labels=checkpoint_names)
        ax.set_xticklabels(checkpoint_names, rotation=45, ha='right')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Reward Distribution by Checkpoint')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Episode Lengths
        ax = axes[1, 1]
        mean_lengths = [valid_results[name]['mean_length'] for name in checkpoint_names]
        std_lengths = [valid_results[name]['std_length'] for name in checkpoint_names]

        ax.bar(x_pos, mean_lengths, yerr=std_lengths, capsize=5, alpha=0.7, color='orange')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(checkpoint_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Episode Length')
        ax.set_title('Episode Length by Checkpoint')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved comparison plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def export_results(self, filepath: str):
        """Export comparison results to JSON"""
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        export_data = {}
        for name, res in self.results.items():
            if 'error' not in res:
                export_data[name] = {
                    key: convert(value) for key, value in res.items()
                    if key not in ['episode_rewards', 'episode_lengths', 'action_distribution']
                }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Exported results to {filepath}")


def quick_compare(env, checkpoint_paths: Dict[str, str], n_episodes: int = 20):
    """
    Quick comparison of checkpoints

    Args:
        env: Environment
        checkpoint_paths: Dictionary of {name: path}
        n_episodes: Episodes per checkpoint

    Example:
        quick_compare(env, {
            '100k': 'models/ppo_100k.zip',
            '500k': 'models/ppo_500k.zip',
            '1M': 'models/ppo_1m.zip'
        })
    """
    comparator = CheckpointComparator(env)

    for name, path in checkpoint_paths.items():
        comparator.add_checkpoint(name, path)

    comparator.compare(n_episodes=n_episodes)
    comparator.print_comparison()

    return comparator


def find_best_checkpoint(
    env,
    checkpoint_dir: str,
    pattern: str = "*.zip",
    n_episodes: int = 10
) -> Tuple[str, float]:
    """
    Find best checkpoint in a directory

    Args:
        env: Environment
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match (default: "*.zip")
        n_episodes: Episodes per checkpoint

    Returns:
        (best_checkpoint_path, best_mean_reward)
    """
    from glob import glob

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob(pattern))

    if not checkpoint_files:
        print(f"❌ No checkpoints found in {checkpoint_dir} matching {pattern}")
        return None, 0.0

    print(f"🔍 Found {len(checkpoint_files)} checkpoints")
    print(f"   Evaluating each with {n_episodes} episodes...")

    comparator = CheckpointComparator(env)

    for checkpoint_file in checkpoint_files:
        name = checkpoint_file.stem
        comparator.add_checkpoint(name, str(checkpoint_file))

    comparator.compare(n_episodes=n_episodes, deterministic=True)

    # Find best
    valid_results = {name: res for name, res in comparator.results.items() if 'error' not in res}

    if not valid_results:
        print("❌ No valid checkpoints")
        return None, 0.0

    best_name = max(valid_results.keys(), key=lambda name: valid_results[name]['mean_reward'])
    best_reward = valid_results[best_name]['mean_reward']
    best_path = comparator.checkpoints[best_name]

    print(f"\n🏆 Best Checkpoint: {best_name}")
    print(f"   Path: {best_path}")
    print(f"   Mean Reward: {best_reward:.1f}")

    return best_path, best_reward
