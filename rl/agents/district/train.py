"""
Train against Normal Bot - Optimized with Vectorized Environments
Uses MLP extractor (better for flat state vectors) and balanced hyperparameters.
"""

import os
import sys
import argparse
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.district.env import RealTimeRTSEnv
from simulator.opponents import get_opponent_info
from simulator.actions import ActionType
from agents.common.extractors import MLPExtractor
from agents.common.callbacks import SessionLoggerCallback


class WinRateCallback(BaseCallback):
    """Track win rate and game stats during training."""

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
            loss_rate = self.losses / max(1, self.games)

            self.logger.record("training/win_rate", win_rate)
            self.logger.record("training/loss_rate", loss_rate)
            self.logger.record("training/games", self.games)

            if self.verbose > 0:
                print(f"\n  Win Rate: {win_rate:.1%} ({self.wins}/{self.games} games)")

            self.wins = 0
            self.losses = 0
            self.games = 0
            self.episode_count = 0

        return True


class ActionDistributionCallback(BaseCallback):
    """Track action distribution to detect action collapse."""

    def __init__(self, check_freq=2048, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.action_counts = np.zeros(len(list(ActionType)), dtype=int)

    def _on_step(self) -> bool:
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                for a in actions.flatten():
                    if 0 <= int(a) < len(self.action_counts):
                        self.action_counts[int(a)] += 1

        if self.n_calls % self.check_freq == 0 and self.action_counts.sum() > 0:
            total = self.action_counts.sum()
            probs = self.action_counts / total

            # Entropy
            nonzero = probs[probs > 0]
            entropy = -np.sum(nonzero * np.log(nonzero))
            max_entropy = np.log(len(self.action_counts))

            self.logger.record("action/entropy", entropy)
            self.logger.record("action/diversity_ratio", entropy / max_entropy)
            self.logger.record("action/unique_actions", np.count_nonzero(self.action_counts))

            # Log top 5 actions
            action_types = list(ActionType)
            top_indices = np.argsort(self.action_counts)[::-1][:5]
            for rank, idx in enumerate(top_indices):
                if self.action_counts[idx] > 0:
                    pct = self.action_counts[idx] / total * 100
                    self.logger.record(f"action/top{rank+1}_{action_types[idx].name}", pct)

            # Reset
            self.action_counts = np.zeros(len(action_types), dtype=int)

        return True


def make_env(rank, seed=0, opponent_type='easy', decision_interval=None):
    def _init():
        env = RealTimeRTSEnv(
            opponent_type=opponent_type,
            decision_interval=decision_interval
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train_normal_bot(
    total_timesteps: int = 500_000,
    learning_rate: float = 3e-4,
    log_dir: str = "./logs_normal_bot",
    checkpoint_freq: int = 50_000,
    resume_from: str = None,
    n_envs: int = 8,
    batch_size: int = 256,
    n_steps: int = 2048,
    opponent_type: str = 'easy',
    ent_coef: float = 0.05,
    decision_interval: float = None,   # None -> env_config.json (TIME SCALE)
    gamma: float = 0.9995,             # tracks the 1440-step episode (D13)
):
    print("\n" + "=" * 70)
    print("TRAINING AGAINST NORMAL BOT")
    print("=" * 70)
    print(f"Target timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Parallel environments: {n_envs}")
    print(f"Batch size: {batch_size}")
    print(f"Rollout buffer (per env): {n_steps}")
    print(f"Total samples per update: {n_steps * n_envs:,}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("\nWARNING: No GPU detected!")

    print(f"\nCreating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(i, opponent_type=opponent_type, decision_interval=decision_interval) for i in range(n_envs)])

    opponent_info = get_opponent_info(opponent_type)
    print(f"Opponent: {opponent_info['name']}")
    print("=" * 70)

    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        model = MaskablePPO.load(resume_from, env=env)
        # Override discount/lr/entropy for fine-tuning (loaded model keeps its own otherwise)
        model.gamma = gamma
        model.learning_rate = learning_rate
        model.ent_coef = ent_coef
        print(f"  fine-tune overrides: gamma={gamma}, lr={learning_rate}, ent_coef={ent_coef}")
    else:
        policy_kwargs = dict(
            features_extractor_class=MLPExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                hidden_dims=(256, 256),
            ),
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            ),
            activation_fn=torch.nn.ReLU,
            # Separate actor/critic extractors (matches the BC warm-start architecture)
            share_features_extractor=False,
        )

        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir
        )
        print(f"\nModel created (device: {model.device}, ent_coef={ent_coef})")

    # Callbacks
    win_rate_callback = WinRateCallback(check_freq=100, verbose=1)
    action_dist_callback = ActionDistributionCallback(check_freq=2048, verbose=0)

    session_logger = SessionLoggerCallback(
        log_dir=log_dir,
        config={
            'training': {
                'total_timesteps': total_timesteps,
                'n_envs': n_envs,
                'decision_interval': decision_interval,
                'ent_coef': ent_coef,
                'opponent': opponent_type,
            },
            'ppo': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'n_steps_per_env': n_steps,
                'gamma': gamma,
            },
            'policy': {
                'extractor': 'mlp',
                'features_dim': 256,
            },
        },
        snapshot_freq=10000,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="normal_bot_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print(f"\nStarting training...")
    print(f"Tensorboard: tensorboard --logdir {log_dir}")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[win_rate_callback, action_dist_callback, session_logger, checkpoint_callback],
            progress_bar=True
        )

        os.makedirs(log_dir, exist_ok=True)
        final_path = f"{log_dir}/final_model"
        model.save(final_path)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Final model saved: {final_path}.zip")

    finally:
        env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description='Train against Normal bot')
    parser.add_argument('--timesteps', type=int, default=2_000_000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--log-dir', type=str, default='./logs_normal_bot')
    parser.add_argument('--checkpoint-freq', type=int, default=500_000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-steps', type=int, default=2048)
    parser.add_argument('--opponent', type=str, default='easy',
                        choices=['testing', 'easy', 'medium', 'hard'])
    parser.add_argument('--ent-coef', type=float, default=0.05)
    parser.add_argument('--decision-interval', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.9995)

    args = parser.parse_args()

    train_normal_bot(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        log_dir=args.log_dir,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        opponent_type=args.opponent,
        ent_coef=args.ent_coef,
        decision_interval=args.decision_interval,
        gamma=args.gamma,
    )


if __name__ == '__main__':
    main()
