"""
Real-Time RTS Training Script - Config-driven
Trains RL agent using JSON configuration files.
Uses MLP extractor by default (configurable via policy.extractor in config).
"""

import os
import sys
import numpy as np
from collections import deque
import torch
import argparse
from typing import Optional

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.district.env import RealTimeRTSEnv
from agents.common.extractors import MLPExtractor, TransformerExtractor
from simulator.actions import ActionType
from config.config_loader import load_training_config
from agents.common.callbacks import EpisodeLoggerCallback
from agents.common.callbacks import SessionLoggerCallback


class ActionDiversityCallback(BaseCallback):
    """Monitor action diversity during training."""

    def __init__(self, config: dict, verbose=0):
        super().__init__(verbose)
        self.check_freq = config.get('diversity_check_freq', 2048)
        buffer_size = config.get('action_buffer_size', 10000)
        self.action_buffer = deque(maxlen=buffer_size)
        self.action_space_size = len(list(ActionType))

    def _on_step(self) -> bool:
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                self.action_buffer.extend(actions.flatten().tolist())

        if self.n_calls % self.check_freq == 0 and len(self.action_buffer) > 100:
            actions_array = np.array(list(self.action_buffer))
            action_counts = np.bincount(
                actions_array.astype(int),
                minlength=self.action_space_size
            )

            total = action_counts.sum()
            probs = action_counts / (total + 1e-8)
            nonzero = probs[probs > 0]
            entropy = -np.sum(nonzero * np.log(nonzero))
            max_entropy = np.log(self.action_space_size)
            diversity_ratio = entropy / max_entropy

            self.logger.record("action/entropy", entropy)
            self.logger.record("action/diversity_ratio", diversity_ratio)
            self.logger.record("action/unique_actions", np.count_nonzero(action_counts))

            # Log top actions
            action_types = list(ActionType)
            top_indices = np.argsort(action_counts)[::-1][:5]
            for rank, idx in enumerate(top_indices):
                if action_counts[idx] > 0:
                    pct = action_counts[idx] / total * 100
                    self.logger.record(f"action/top{rank+1}_{action_types[idx].name}", pct)

            if self.verbose > 0:
                print(f"\n  Action Diversity at step {self.num_timesteps}:")
                print(f"   Entropy: {entropy:.2f} / {max_entropy:.2f}")
                print(f"   Diversity: {diversity_ratio:.1%}")
                print(f"   Unique actions: {np.count_nonzero(action_counts)}/{self.action_space_size}")

        return True


class WinRateCallback(BaseCallback):
    """Track win rate during training."""

    def __init__(self, check_freq=100, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.wins = 0
        self.losses = 0
        self.games = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])

        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                info = infos[i] if i < len(infos) else {}
                winner = info.get('winner')

                if winner == 0:
                    self.wins += 1
                elif winner is not None:
                    self.losses += 1
                self.games += 1

        if self.episode_count >= self.check_freq and self.games > 0:
            win_rate = self.wins / max(1, self.games)
            self.logger.record("training/win_rate", win_rate)
            self.logger.record("training/games", self.games)

            if self.verbose > 0:
                print(f"\n  Win Rate: {win_rate:.1%} ({self.wins}/{self.games} games)")

            self.wins = 0
            self.losses = 0
            self.games = 0
            self.episode_count = 0

        return True


def create_env(rank: int, config: dict) -> callable:
    def _init():
        decision_interval = config['training']['decision_interval']
        reward_config = config.get('reward_config_path')
        return RealTimeRTSEnv(
            decision_interval=decision_interval,
            config_path=reward_config
        )
    return _init


def get_extractor_class(config: dict):
    """Get feature extractor class based on config."""
    extractor_type = config.get('policy', {}).get('extractor', 'mlp')
    if extractor_type == 'transformer':
        return TransformerExtractor
    return MLPExtractor


def get_extractor_kwargs(config: dict):
    """Get feature extractor kwargs based on config."""
    policy_cfg = config['policy']
    extractor_type = policy_cfg.get('extractor', 'mlp')

    if extractor_type == 'transformer':
        return dict(features_dim=policy_cfg['features_dim'])
    else:
        return dict(
            features_dim=policy_cfg['features_dim'],
            hidden_dims=tuple(policy_cfg.get('hidden_dims', [256, 256])),
        )


def train_realtime_rts(
    training_config_path: Optional[str] = None,
    reward_config_path: Optional[str] = None,
    override_params: Optional[dict] = None
):
    # Load configuration
    config = load_training_config(training_config_path)

    if override_params:
        for key, value in override_params.items():
            if '.' in key:
                parts = key.split('.')
                current = config
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                config[key] = value

    if reward_config_path:
        config['reward_config_path'] = reward_config_path

    training_cfg = config['training']
    ppo_cfg = config['ppo']
    policy_cfg = config['policy']
    norm_cfg = config['normalization']
    callback_cfg = config['callbacks']
    paths_cfg = config['paths']

    log_dir = paths_cfg['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    total_timesteps = training_cfg['total_timesteps']
    n_envs = training_cfg['n_envs']
    ent_coef = training_cfg['ent_coef']
    action_space_size = len(list(ActionType))

    extractor_type = policy_cfg.get('extractor', 'mlp')

    print("=" * 70)
    print("REAL-TIME RTS TRAINING")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Decision interval: {training_cfg['decision_interval']}s")
    print(f"Action space: {action_space_size} actions")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Feature extractor: {extractor_type}")
    print(f"Log directory: {log_dir}")
    print("=" * 70)

    # Create environments
    env = SubprocVecEnv([create_env(i, config) for i in range(n_envs)])

    env = VecNormalize(
        env,
        norm_obs=norm_cfg['norm_obs'],
        norm_reward=norm_cfg['norm_reward'],
        clip_obs=norm_cfg['clip_obs'],
        clip_reward=norm_cfg['clip_reward'],
        gamma=norm_cfg['gamma']
    )

    # Create policy configuration
    activation_fn = getattr(torch.nn, policy_cfg['activation'])
    extractor_class = get_extractor_class(config)
    extractor_kwargs = get_extractor_kwargs(config)

    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=dict(
            pi=policy_cfg['pi_layers'],
            vf=policy_cfg['vf_layers']
        ),
        activation_fn=activation_fn,
        ortho_init=policy_cfg['ortho_init'],
        # Separate actor/critic extractors (matches the BC warm-start architecture)
        share_features_extractor=False,
    )

    # MaskablePPO (not plain PPO): plain PPO ignores the env's action_masks(),
    # letting the policy pick impossible actions and collapse onto ATTACK.
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=ppo_cfg['learning_rate'],
        n_steps=ppo_cfg['n_steps_per_env'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        gamma=ppo_cfg['gamma'],
        gae_lambda=ppo_cfg['gae_lambda'],
        clip_range=ppo_cfg['clip_range'],
        ent_coef=ent_coef,
        vf_coef=ppo_cfg['vf_coef'],
        max_grad_norm=ppo_cfg['max_grad_norm'],
        verbose=1,
        tensorboard_log=log_dir
    )

    print(f"\nPPO Config:")
    print(f"   LR: {ppo_cfg['learning_rate']}, Batch: {ppo_cfg['batch_size']}")
    print(f"   N_steps: {ppo_cfg['n_steps_per_env']}, Epochs: {ppo_cfg['n_epochs']}")
    print(f"   Device: {model.device}")
    print(f"\nStarting training...")
    print(f"   Monitor: tensorboard --logdir={log_dir}\n")

    # Callbacks
    diversity_callback = ActionDiversityCallback(config=callback_cfg, verbose=1)
    episode_callback = EpisodeLoggerCallback(verbose=1)
    win_rate_callback = WinRateCallback(check_freq=100, verbose=1)
    session_logger = SessionLoggerCallback(
        log_dir=log_dir,
        config=config,
        snapshot_freq=10000,
        verbose=1
    )

    callback_list = CallbackList([diversity_callback, episode_callback, win_rate_callback, session_logger])

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )

        model_path = os.path.join(log_dir, paths_cfg['model_name'])
        vec_norm_path = os.path.join(log_dir, paths_cfg['vec_normalize_name'])

        model.save(model_path)
        env.save(vec_norm_path)

        print(f"\nTraining complete!")
        print(f"Model saved: {model_path}")
        print(f"Normalization: {vec_norm_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

        interrupted_model = os.path.join(log_dir, paths_cfg['interrupted_model_name'])
        interrupted_vec_norm = os.path.join(log_dir, paths_cfg['interrupted_vec_normalize_name'])

        model.save(interrupted_model)
        env.save(interrupted_vec_norm)

        print(f"   Model saved: {interrupted_model}")
        print(f"   Normalization: {interrupted_vec_norm}")

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train RTS AI agent using configuration from JSON files'
    )
    parser.add_argument('--training-config', type=str, default=None)
    parser.add_argument('--reward-config', type=str, default=None)
    parser.add_argument('--ent-coef', type=float, default=None)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--n-envs', type=int, default=None)

    args = parser.parse_args()

    overrides = {}
    if args.ent_coef is not None:
        overrides['training.ent_coef'] = args.ent_coef
    if args.timesteps is not None:
        overrides['training.total_timesteps'] = args.timesteps
    if args.n_envs is not None:
        overrides['training.n_envs'] = args.n_envs

    train_realtime_rts(
        training_config_path=args.training_config,
        reward_config_path=args.reward_config,
        override_params=overrides if overrides else None
    )
