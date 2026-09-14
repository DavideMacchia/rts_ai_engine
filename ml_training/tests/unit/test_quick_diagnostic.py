"""
Quick Diagnostic Test - With Timeouts and Progress
"""

import sys
import os

# Add ml_training directory to path (go up 3 levels: unit -> tests -> ml_training)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from training.realtime_rts_env import RealTimeRTSEnv
import training.transformer_extractor as transformer_extractor
sys.modules['transformer_extractor'] = transformer_extractor
from simulator.actions import ActionType, index_to_action
import numpy as np
import time
from collections import Counter

MODEL_PATH = "./ppo_model/rts_realtime_final.zip"
VEC_NORMALIZE_PATH = "./ppo_model/vec_normalize.pkl"
NUM_GAMES = 5
MAX_STEPS = 10000


def test_quick():
    """Quick test with timeouts"""

    print("=" * 70)
    print("🎮 QUICK DIAGNOSTIC TEST")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print(f"📁 Loading model...")

    # Create environment (uses default decision_interval from config: 1.0s)
    env = DummyVecEnv([lambda: RealTimeRTSEnv()])

    # Load normalization
    if os.path.exists(VEC_NORMALIZE_PATH):
        env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
        env.training = False
        env.norm_reward = False

    # Load model
    model = PPO.load(MODEL_PATH, env=env, device='cpu')
    print("✅ Model loaded!\n")

    print(f"🎲 Playing {NUM_GAMES} games (max {MAX_STEPS} steps each)...\n")

    for game_num in range(NUM_GAMES):
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        episode_actions = []

        game_start = time.time()

        while not done and step_count < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=False)
            action_val = int(action[0]) if hasattr(action, '__len__') else int(action)
            episode_actions.append(action_val)

            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
            step_count += 1

        game_elapsed = time.time() - game_start

        # Get result
        game_info = info[0] if isinstance(info, list) else info
        winner = game_info.get('winner', None)

        if winner == 0:
            result = "WIN"
        elif winner == 1:
            result = "LOSS"
        else:
            result = "TIMEOUT"

        # Final state
        sim = env.envs[0].sim
        agent = sim.state.factions[0]

        print(f"\nGame {game_num + 1}: {result}")
        print(f"  Steps: {step_count} | Reward: {total_reward:.0f}")
        print(f"  Final Pop: {agent.population}/{agent.population_capacity}")
        print(f"  Buildings: Farms={agent.get_building_count('farm')}, "
              f"Houses={agent.get_building_count('house')}, "
              f"Lumberyards={agent.get_building_count('lumberyard')}")
        print(f"  All Buildings: {dict(agent.buildings)}")
        print(f"  Resources: Wood={agent.get_resource('wood'):.0f}, "
              f"Stone={agent.get_resource('stone'):.0f}, "
              f"Food={agent.get_resource('grain') + agent.get_resource('bread'):.0f}")

        # Action distribution
        action_counts = Counter(episode_actions)
        total = len(episode_actions)
        action_percentages = {index_to_action(idx).name: round((count/total)*100, 3)
                              for idx, count in action_counts.items()}
        print(f"  Action Distribution: {action_percentages}")

    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    test_quick()