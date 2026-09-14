"""
Curriculum Training Script
Trains AI progressively through scripted opponents
"""

import os
import sys
import argparse
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.district.env import RealTimeRTSEnv
from simulator.opponents import get_curriculum_sequence, get_opponent_info
from simulator.actions import ActionType
from agents.common.extractors import MLPExtractor


class WinRateCallback(BaseCallback):
    """Track win rate and manage curriculum progression."""

    def __init__(self, check_freq=100, target_win_rate=0.80, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.target_win_rate = target_win_rate
        self.wins = 0
        self.games = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            info = self.locals.get('infos', [{}])[0]
            winner = info.get('winner')

            if winner == 0:  # AI won
                self.wins += 1
            self.games += 1

            # Check win rate periodically
            if self.episode_count % self.check_freq == 0:
                win_rate = self.wins / max(1, self.games)

                self.logger.record("curriculum/win_rate", win_rate)
                self.logger.record("curriculum/games", self.games)

                if self.verbose > 0:
                    print(f"\n📊 Win Rate: {win_rate:.1%} ({self.wins}/{self.games})")
                    if win_rate >= self.target_win_rate:
                        print(f"   ✓ Target {self.target_win_rate:.0%} reached!")

                # Reset counters
                self.wins = 0
                self.games = 0

        return True


def train_curriculum_stage(
    difficulty: str,
    target_win_rate: float,
    max_timesteps: int,
    model=None,
    log_dir="./curriculum_logs"
):
    """
    Train one curriculum stage.

    Args:
        difficulty: Opponent difficulty level
        target_win_rate: Win rate to achieve before moving on
        max_timesteps: Maximum training steps for this stage
        model: Existing model to continue training (or None for new)
        log_dir: Directory for logs

    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print(f"CURRICULUM STAGE: {difficulty.upper()}")
    print("=" * 70)
    print(f"Target win rate: {target_win_rate:.0%}")
    print(f"Max timesteps: {max_timesteps:,}")

    # Create environment with scripted opponent
    # Using refactored RealTimeRTSEnv with opponent_type parameter
    env = RealTimeRTSEnv(
        opponent_type=difficulty,
        decision_interval=None
    )

    # Get opponent info for logging
    from simulator.opponents import get_opponent_info
    opponent_info = get_opponent_info(difficulty)
    print(f"   Opponent: {opponent_info['name']} ({difficulty})")

    # Create or reuse model
    if model is None:
        policy_kwargs = dict(
            features_extractor_class=MLPExtractor,
            features_extractor_kwargs=dict(features_dim=256, hidden_dims=(256, 256)),
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            ),
            activation_fn=torch.nn.ReLU,
            share_features_extractor=False,
        )

        # MaskablePPO: plain PPO ignores action_masks() and the policy collapses.
        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.9995,
            ent_coef=0.05,
            verbose=1,
            tensorboard_log=f"{log_dir}/{difficulty}"
        )
        print(f"✅ Created new model (device: {model.device})")
    else:
        # Update environment for new opponent
        model.set_env(env)
        print(f"✅ Reusing existing model (device: {model.device})")

    # Create callback
    callback = WinRateCallback(
        check_freq=100,
        target_win_rate=target_win_rate,
        verbose=1
    )

    # Train
    print(f"\n🚀 Training against {difficulty}...")
    model.learn(
        total_timesteps=max_timesteps,
        callback=callback,
        reset_num_timesteps=False,  # Keep counting across stages
        progress_bar=True
    )

    # Save checkpoint
    os.makedirs(log_dir, exist_ok=True)
    model.save(f"{log_dir}/model_{difficulty}")
    print(f"✅ Saved checkpoint: model_{difficulty}")

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description='Curriculum training with scripted opponents')
    parser.add_argument('--log-dir', type=str, default='./curriculum_logs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--start-stage', type=int, default=0,
                        help='Start from curriculum stage (0=normal)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model checkpoint to resume from')

    args = parser.parse_args()

    # Curriculum sequence
    curriculum = get_curriculum_sequence()

    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING (2-LEVEL SYSTEM)")
    print("=" * 70)
    print("\nCurriculum stages:")
    for i, (diff, win_rate, steps) in enumerate(curriculum):
        marker = "→" if i == args.start_stage else " "
        print(f"{marker} Stage {i+1}: {diff:12s} | Target: {win_rate:.0%} | Steps: {steps:,}")
    print("=" * 70)

    # Load existing model if resuming
    model = None
    if args.resume:
        print(f"\n📂 Loading model from {args.resume}")
        model = MaskablePPO.load(args.resume)

    # Train through curriculum stages
    for i in range(args.start_stage, len(curriculum)):
        difficulty, target_win_rate, max_steps = curriculum[i]

        model = train_curriculum_stage(
            difficulty=difficulty,
            target_win_rate=target_win_rate,
            max_timesteps=max_steps,
            model=model,
            log_dir=args.log_dir
        )

        print(f"\n✓ Stage {i+1} complete!")

    # Save final model
    final_path = f"{args.log_dir}/model_final"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("✅ CURRICULUM TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved: {final_path}")
    print("\nYour AI has mastered:")
    for diff, _, _ in curriculum:
        print(f"  ✓ {diff}")
    print("\nReady for self-play or advanced training! 🚀")


if __name__ == '__main__':
    main()
