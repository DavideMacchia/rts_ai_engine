"""
Session Logger - writes a human/AI-readable JSON log per training session.
Designed to be consulted by Claude Code to reason about training progress.

Output: {log_dir}/session_log.json
"""

import json
import os
import time
from datetime import datetime
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from simulator.actions import ActionType


class SessionLoggerCallback(BaseCallback):
    """
    Logs key training metrics to a JSON file at regular intervals.
    The file is designed to be read by Claude Code for analysis.

    Structure:
    {
        "session_info": { start_time, config, model_type, ... },
        "snapshots": [ { timestep, metrics... }, ... ],
        "final_summary": { ... }
    }
    """

    def __init__(self, log_dir: str, config: dict = None, snapshot_freq: int = 10000, verbose=0):
        """
        Args:
            log_dir: Directory to write session_log.json
            config: Training config dict (hyperparameters, etc.)
            snapshot_freq: Write a snapshot every N timesteps
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.snapshot_freq = snapshot_freq
        self.config_snapshot = config or {}

        # Session state
        self.session_data = {
            "session_info": {},
            "snapshots": [],
            "final_summary": None
        }

        # Tracking
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_games = 0
        self.episode_rewards = deque(maxlen=200)
        self.episode_lengths = deque(maxlen=200)
        self.current_ep_rewards = None
        self.current_ep_lengths = None
        self.episode_populations = deque(maxlen=200)
        self.episode_military = deque(maxlen=200)
        self.episode_buildings = deque(maxlen=200)
        self.action_counts = np.zeros(len(list(ActionType)), dtype=int)
        self.action_names = [a.name for a in ActionType]

        # Per-window tracking (for trend detection)
        self.window_wins = 0
        self.window_games = 0
        self.window_rewards = []

        self.start_time = None
        self.last_snapshot_step = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        os.makedirs(self.log_dir, exist_ok=True)
        n_envs = self.training_env.num_envs
        self.current_ep_rewards = [0.0] * n_envs
        self.current_ep_lengths = [0] * n_envs

        self.session_data["session_info"] = {
            "start_time": datetime.now().isoformat(),
            "n_envs": self.training_env.num_envs,
            "observation_space": list(self.training_env.observation_space.shape),
            "action_space_size": len(self.action_names),
            "config": self._extract_key_config(),
        }
        self._write_log()

    def _extract_key_config(self) -> dict:
        """Extract the most important config values for the log."""
        cfg = self.config_snapshot
        result = {}

        # Training params
        training = cfg.get('training', {})
        result['total_timesteps'] = training.get('total_timesteps', 'unknown')
        result['n_envs'] = training.get('n_envs', 'unknown')
        result['decision_interval'] = training.get('decision_interval', 'unknown')
        result['ent_coef'] = training.get('ent_coef', 'unknown')

        # PPO params
        ppo = cfg.get('ppo', {})
        result['learning_rate'] = ppo.get('learning_rate', 'unknown')
        result['batch_size'] = ppo.get('batch_size', 'unknown')
        result['n_steps_per_env'] = ppo.get('n_steps_per_env', 'unknown')

        # Policy
        policy = cfg.get('policy', {})
        result['extractor'] = policy.get('extractor', 'unknown')
        result['features_dim'] = policy.get('features_dim', 'unknown')

        return result

    def _on_step(self) -> bool:
        # Track actions
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                for a in actions.flatten():
                    idx = int(a)
                    if 0 <= idx < len(self.action_counts):
                        self.action_counts[idx] += 1

        # Track per-step rewards for manual episode tracking
        rewards = self.locals.get('rewards', [])
        for i in range(len(rewards)):
            self.current_ep_rewards[i] += rewards[i]
            self.current_ep_lengths[i] += 1

        # Track episode completions
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])

        for i, done in enumerate(dones):
            if done:
                info = infos[i] if i < len(infos) else {}
                winner = info.get('winner')

                self.total_games += 1
                self.window_games += 1

                if winner == 0:
                    self.wins += 1
                    self.window_wins += 1
                elif winner is not None:
                    self.losses += 1
                else:
                    self.draws += 1

                # Episode reward: prefer info['episode'] if available, else use manual tracking
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                else:
                    ep_reward = self.current_ep_rewards[i]
                    ep_length = self.current_ep_lengths[i]

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.window_rewards.append(ep_reward)

                self.episode_populations.append(info.get('agent_population', 0))
                self.episode_military.append(info.get('agent_military', 0))
                self.episode_buildings.append(info.get('agent_buildings', 0))

                # Reset per-env trackers
                self.current_ep_rewards[i] = 0.0
                self.current_ep_lengths[i] = 0

        # Write snapshot at regular intervals
        if self.num_timesteps - self.last_snapshot_step >= self.snapshot_freq:
            self._write_snapshot()
            self.last_snapshot_step = self.num_timesteps

        return True

    def _write_snapshot(self):
        """Write a periodic snapshot of training progress."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Action distribution
        total_actions = self.action_counts.sum()
        action_dist = {}
        if total_actions > 0:
            sorted_indices = np.argsort(self.action_counts)[::-1]
            for idx in sorted_indices:
                if self.action_counts[idx] > 0:
                    pct = round(self.action_counts[idx] / total_actions * 100, 1)
                    action_dist[self.action_names[idx]] = pct

        # Action entropy
        entropy = 0.0
        if total_actions > 0:
            probs = self.action_counts / total_actions
            nonzero = probs[probs > 0]
            entropy = float(-np.sum(nonzero * np.log(nonzero)))

        # Window win rate (recent performance)
        window_win_rate = self.window_wins / max(1, self.window_games)

        snapshot = {
            "timestep": int(self.num_timesteps),
            "elapsed_seconds": round(elapsed, 1),
            "total_games": self.total_games,
            "overall_win_rate": round(self.wins / max(1, self.total_games), 3),
            "recent_win_rate": round(window_win_rate, 3),
            "recent_games": self.window_games,
            "reward_mean": round(float(np.mean(self.episode_rewards)), 2) if self.episode_rewards else 0,
            "reward_std": round(float(np.std(self.episode_rewards)), 2) if self.episode_rewards else 0,
            "episode_length_mean": round(float(np.mean(self.episode_lengths)), 1) if self.episode_lengths else 0,
            "game_metrics": {
                "population_mean": round(float(np.mean(self.episode_populations)), 1) if self.episode_populations else 0,
                "military_mean": round(float(np.mean(self.episode_military)), 1) if self.episode_military else 0,
                "buildings_mean": round(float(np.mean(self.episode_buildings)), 1) if self.episode_buildings else 0,
            },
            "action_entropy": round(entropy, 3),
            "unique_actions_used": int(np.count_nonzero(self.action_counts)),
            "top_actions": dict(list(action_dist.items())[:5]),
        }

        self.session_data["snapshots"].append(snapshot)

        # Reset window counters
        self.window_wins = 0
        self.window_games = 0
        self.window_rewards = []

        self._write_log()

    def _on_training_end(self) -> None:
        """Write final summary when training completes."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Detect trends from snapshots
        trends = self._detect_trends()

        self.session_data["final_summary"] = {
            "end_time": datetime.now().isoformat(),
            "total_elapsed_seconds": round(elapsed, 1),
            "total_timesteps": int(self.num_timesteps),
            "total_games": self.total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "final_win_rate": round(self.wins / max(1, self.total_games), 3),
            "final_reward_mean": round(float(np.mean(self.episode_rewards)), 2) if self.episode_rewards else 0,
            "trends": trends,
            "diagnosis": self._diagnose(),
        }

        self._write_log()

        if self.verbose > 0:
            print(f"\n  Session log written to: {self._log_path()}")

    def _detect_trends(self) -> dict:
        """Analyze snapshots to detect learning trends."""
        snapshots = self.session_data["snapshots"]
        if len(snapshots) < 3:
            return {"note": "too few snapshots for trend analysis"}

        win_rates = [s["overall_win_rate"] for s in snapshots]
        rewards = [s["reward_mean"] for s in snapshots]
        entropies = [s["action_entropy"] for s in snapshots]

        def trend_direction(values):
            if len(values) < 2:
                return "unknown"
            first_half = np.mean(values[:len(values)//2])
            second_half = np.mean(values[len(values)//2:])
            diff = second_half - first_half
            if abs(diff) < 0.01:
                return "flat"
            return "improving" if diff > 0 else "declining"

        return {
            "win_rate_trend": trend_direction(win_rates),
            "reward_trend": trend_direction(rewards),
            "entropy_trend": trend_direction(entropies),
            "win_rate_first": round(win_rates[0], 3) if win_rates else 0,
            "win_rate_last": round(win_rates[-1], 3) if win_rates else 0,
            "reward_first": round(rewards[0], 2) if rewards else 0,
            "reward_last": round(rewards[-1], 2) if rewards else 0,
        }

    def _diagnose(self) -> list:
        """Generate diagnostic messages based on training data."""
        issues = []
        snapshots = self.session_data["snapshots"]

        if not snapshots:
            return ["no data collected"]

        last = snapshots[-1]

        # Win rate issues
        if self.total_games > 50 and self.wins / max(1, self.total_games) < 0.1:
            issues.append("CRITICAL: win rate below 10% - agent is not learning to win")

        # Action collapse
        if last["unique_actions_used"] <= 3:
            issues.append("WARNING: action collapse detected - only using {} actions".format(
                last["unique_actions_used"]))

        if last["action_entropy"] < 0.5:
            issues.append("WARNING: low action entropy ({:.2f}) - agent may be stuck".format(
                last["action_entropy"]))

        # Stagnation
        if len(snapshots) >= 4:
            recent_rewards = [s["reward_mean"] for s in snapshots[-4:]]
            if max(recent_rewards) - min(recent_rewards) < 1.0:
                issues.append("WARNING: reward stagnation - no improvement in recent snapshots")

        # Military
        if last["game_metrics"]["military_mean"] < 0.5 and self.total_games > 20:
            issues.append("INFO: agent rarely builds military - may need reward tuning")

        # Population
        if last["game_metrics"]["population_mean"] < 5 and self.total_games > 20:
            issues.append("INFO: low population at end of games - economy may be failing")

        if not issues:
            issues.append("OK: no issues detected")

        return issues

    def _log_path(self) -> str:
        return os.path.join(self.log_dir, "session_log.json")

    def _write_log(self):
        """Write current session data to disk."""
        try:
            with open(self._log_path(), 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            print(f"Failed to write session log: {e}")
