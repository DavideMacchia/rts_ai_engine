"""
Test Training Environment Health
Validates that the training environment is properly configured and not too predictable
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'training'))

from realtime_rts_env import RealTimeRTSEnv
from simulator.actions import ActionType
import numpy as np


def test_environment_initialization():
    """Test that environment initializes correctly"""
    print("=" * 70)
    print("TEST 1: Environment Initialization")
    print("=" * 70)

    env = RealTimeRTSEnv()

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    print(f"Decision interval: {env.decision_interval}s")
    print(f"Max game time: {env.max_game_time}s ({env.max_game_time/3600:.1f} hours)")
    print(f"Max possible steps: {env.max_game_time / env.decision_interval:.0f}")

    # Test reset
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"

    env.close()
    print("✅ PASS: Environment initializes correctly\n")
    return True


def test_game_length_variance():
    """Test that games have varying lengths (not too predictable)"""
    print("=" * 70)
    print("TEST 2: Game Length Variance")
    print("=" * 70)

    env = RealTimeRTSEnv()
    num_games = 10
    game_lengths = []
    game_times = []
    total_rewards = []
    winners = []

    print(f"Running {num_games} games with random actions...\n")

    for game_num in range(num_games):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 8000:  # 8000 steps > 7200s max game time
            action = env.action_space.sample()
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            total_reward += reward
            steps += 1

        game_time = env.sim.game_time
        winner = env.sim.state.winner

        game_lengths.append(steps)
        game_times.append(game_time)
        total_rewards.append(total_reward)
        winners.append(winner)

        print(f"Game {game_num + 1}: {steps} steps, {game_time:.1f}s, "
              f"reward={total_reward:.2f}, winner={winner}")

    # Calculate statistics
    avg_length = np.mean(game_lengths)
    std_length = np.std(game_lengths)
    min_length = np.min(game_lengths)
    max_length = np.max(game_lengths)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    unique_winners = len(set(winners))

    # Count winner distribution manually (np.unique doesn't like None)
    winner_counts = {}
    for w in winners:
        winner_counts[w] = winner_counts.get(w, 0) + 1

    print(f"\n--- Statistics ---")
    print(f"Game lengths: {avg_length:.1f} ± {std_length:.1f} steps (min={min_length}, max={max_length})")
    print(f"Game times: {np.mean(game_times):.1f}s ± {np.std(game_times):.1f}s")
    print(f"Total rewards: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Unique winners: {unique_winners} (0=agent, 1=opponent, None=draw)")
    print(f"Winner distribution: {winner_counts}")

    # Validation
    issues = []

    # Critical check: Are all games hitting the step limit?
    if min_length == max_length == 8000:
        issues.append("🚨 CRITICAL: All games hit 8,000 step limit - games never finish naturally!")
        issues.append("   This means the game time is TOO LONG or win conditions aren't working")

    # Check 1: Games should vary in length
    elif std_length < 5:
        issues.append("⚠️  Game length variance too low (std < 5) - too predictable!")
    else:
        print("✅ Game lengths have good variance")

    # Check 2: Games shouldn't all end the same way
    if unique_winners == 1:
        issues.append("⚠️  All games have same winner - too predictable!")
    else:
        print("✅ Games have different outcomes")

    # Check 3: Rewards should vary
    if std_reward < 10:
        issues.append("⚠️  Reward variance too low (std < 10) - reward signal too uniform!")
    else:
        print("✅ Rewards have good variance")

    # Check 4: Games shouldn't be too short
    if avg_length < 50:
        issues.append(f"⚠️  Games too short (avg={avg_length:.1f} steps) - might timeout before buildings complete!")
    else:
        print("✅ Game lengths are reasonable")

    env.close()

    if issues:
        print(f"\n❌ FAIL: Environment has predictability issues:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ PASS: Environment has healthy variance\n")
        return True


def test_action_acceptance():
    """Test that environment accepts diverse actions"""
    print("=" * 70)
    print("TEST 3: Action Acceptance")
    print("=" * 70)

    env = RealTimeRTSEnv()
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result

    # Try each action type
    action_results = {}
    all_actions = list(ActionType)

    for action_idx in range(len(all_actions)):
        action_name = all_actions[action_idx].name

        # Reset for clean slate
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result

        # Try action
        try:
            step_result = env.step(action_idx)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
            action_results[action_name] = "✓ Accepted"
        except Exception as e:
            action_results[action_name] = f"✗ Error: {e}"

    print(f"Testing {len(all_actions)} action types:\n")

    accepted = 0
    for action_name, result in action_results.items():
        if "✓" in result:
            accepted += 1
        if "✗" in result:
            print(f"  {action_name}: {result}")

    print(f"\nAccepted: {accepted}/{len(all_actions)} actions")

    env.close()

    if accepted == len(all_actions):
        print("✅ PASS: All actions accepted\n")
        return True
    else:
        print(f"❌ FAIL: Some actions rejected\n")
        return False


def test_reward_signal_quality():
    """Test that reward signals are informative and not too sparse"""
    print("=" * 70)
    print("TEST 4: Reward Signal Quality")
    print("=" * 70)

    env = RealTimeRTSEnv()

    # Play one game and track when rewards are given
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    done = False
    steps = 0

    rewards_received = []
    non_zero_rewards = 0

    while not done and steps < 1000:
        action = env.action_space.sample()
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        rewards_received.append(reward)
        if abs(reward) > 0.01:
            non_zero_rewards += 1

        steps += 1

    # Calculate reward statistics
    total_reward = sum(rewards_received)
    avg_reward = np.mean(rewards_received)
    reward_sparsity = non_zero_rewards / steps

    print(f"Steps taken: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {avg_reward:.4f}")
    print(f"Non-zero rewards: {non_zero_rewards}/{steps} ({reward_sparsity:.1%})")

    # Validation
    issues = []

    if reward_sparsity < 0.01:
        issues.append("⚠️  Rewards too sparse (<1% of steps) - agent gets almost no feedback!")
    else:
        print("✅ Reward density is reasonable")

    if abs(avg_reward) < 0.001:
        issues.append("⚠️  Average reward near zero - might be hard to learn!")
    else:
        print("✅ Rewards have meaningful magnitude")

    env.close()

    if issues:
        print(f"\n❌ FAIL: Reward signal issues:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ PASS: Reward signal quality is good\n")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRAINING ENVIRONMENT HEALTH CHECK")
    print("=" * 70 + "\n")

    results = []

    # Run all tests
    results.append(("Environment Initialization", test_environment_initialization()))
    results.append(("Game Length Variance", test_game_length_variance()))
    results.append(("Action Acceptance", test_action_acceptance()))
    results.append(("Reward Signal Quality", test_reward_signal_quality()))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Environment is healthy for training.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - environment may have issues!")
        print("\nPossible causes of high explained variance during training:")
        print("  1. Games ending too quickly (timeout)")
        print("  2. Rewards too sparse (only at game end)")
        print("  3. Always same winner (too predictable)")
        print("  4. Need higher exploration (increase ent_coef)")
