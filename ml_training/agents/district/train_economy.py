"""RL for the CIVIL district (D35): grow a settlement, no war, no teacher.

    python -m agents.district.train_economy --timesteps 2000000

Why this is trained with plain RL, and not cloned from a bot first
------------------------------------------------------------------
Behavioural cloning exists in this project for one reason (D4): the reward was SPARSE. A
conquest arrived thousands of steps after the decisions that earned it, and the sequence that
got there — barracks, train, army, attack — was far too long for random exploration to stumble
into. Without a teacher, RL found nothing.

**That reason is gone.** The civil objective is DENSE: every person born, every trade a sim
finally becomes qualified to practise, every new good the district can make, pays on the step
it happens (`env._economy_reward`). There is no long silence to cross.

And cloning has a ceiling that is structural, not incidental: a clone reproduces its teacher
and cannot exceed it — we measured that twice (D28: the spatial clone matched the auto-placer
to 97% and could not do better BY CONSTRUCTION). The SettlerBot scores 91. A clone of it scores
91. If we want more than the best behaviour tree I can write by hand, RL has to find it itself.

So the bot is not the teacher here. It is the BASELINE: the number to beat.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from agents.district.env import RealTimeRTSEnv
from agents.common.extractors import MLPExtractor


def make_env(rank: int, seed: int = 0, spatial: bool = False):
    def _init():
        env = RealTimeRTSEnv(objective='economy', spatial=spatial)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def settler_baseline(n_episodes: int = 8) -> float:
    """What the scripted settlement builder scores. The bar, not the teacher."""
    from simulator.opponents import get_opponent
    from simulator.actions import action_to_index

    env = RealTimeRTSEnv(objective='economy')
    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        bot = get_opponent('settler', faction_id=0)
        done = False
        while not done:
            mask = env.action_masks()
            action = bot.act(env.sim.state, env.sim.game_time)
            idx = action_to_index(action.action_type)
            if not mask[idx]:
                idx = int(np.where(mask)[0][-1])
            obs, _r, terminated, truncated, _info = env.step(idx)
            done = terminated or truncated
        scores.append(env.score())
    env.close()
    return float(np.mean(scores))


def score_policy(model, n_episodes: int = 8) -> float:
    """Always DETERMINISTIC. A stochastic policy's average flatters it, and we have been
    fooled by that before."""
    env = RealTimeRTSEnv(objective='economy')
    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _r, terminated, truncated, _info = env.step(int(action))
            done = terminated or truncated
        scores.append(env.score())
    env.close()
    return float(np.mean(scores))


def main():
    p = argparse.ArgumentParser(description='Train the civil district with RL (no teacher)')
    p.add_argument('--timesteps', type=int, default=2_000_000)
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--out', type=str,
                   default='./checkpoints/district/district_economy_rl_v1.zip')
    # The episode is 1440 decisions and the reward is dense, so the discount only has to see
    # far enough for an investment to pay back — a child takes a childhood to become a worker,
    # a career takes half a life to ripen. 0.999 keeps a reward 1440 steps away worth ~24%.
    p.add_argument('--gamma', type=float, default=0.999)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--eval-episodes', type=int, default=8)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline = settler_baseline(args.eval_episodes)

    print("=" * 70)
    print("CIVIL DISTRICT — RL, no teacher")
    print("=" * 70)
    print(f"device: {device}   timesteps: {args.timesteps:,}   envs: {args.n_envs}")
    print(f"THE BAR — SettlerBot scores {baseline:.0f}. Beat it, or the bot stays the answer.")
    print()

    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])

    model = MaskablePPO(
        "MlpPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=MLPExtractor,
            features_extractor_kwargs=dict(features_dim=256, hidden_dims=(256, 256)),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
            share_features_extractor=False,
        ),
        learning_rate=3e-4,
        n_steps=512,
        batch_size=512,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        device=device,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    env.close()

    scored = score_policy(model, args.eval_episodes)
    print()
    print("=" * 70)
    print(f"RL (deterministic): {scored:.0f}     SettlerBot: {baseline:.0f}")
    if scored > baseline:
        print(f"RL BEATS THE BOT by {scored - baseline:.0f}. No teacher was needed.")
    else:
        print(f"RL does NOT beat the bot ({baseline - scored:.0f} short). The dense reward was "
              f"not enough on its own — behavioural cloning from the settler is the fallback, "
              f"and now there is a measured reason to want it.")
    print(f"saved: {args.out}")


if __name__ == '__main__':
    main()
