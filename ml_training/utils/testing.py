"""
ML Testing Utilities
Unit tests and validation for ML training components
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
import time


class RewardTester:
    """
    Test reward functions for correctness and consistency

    Usage:
        tester = RewardTester(reward_fn)
        tester.test_determinism()
        tester.test_bounds()
        tester.print_results()
    """

    def __init__(self, reward_function: Callable):
        """
        Initialize tester

        Args:
            reward_function: Function that takes (state, prev_state, action) -> reward
        """
        self.reward_fn = reward_function
        self.test_results = {}

    def test_determinism(self, test_state: Any, prev_state: Any, action: str, n_trials: int = 10) -> bool:
        """
        Test if reward function is deterministic

        Args:
            test_state: Test game state
            prev_state: Previous state
            action: Action to test
            n_trials: Number of trials

        Returns:
            True if deterministic
        """
        results = []
        for _ in range(n_trials):
            reward = self.reward_fn(test_state, prev_state, action)
            results.append(reward)

        is_deterministic = len(set(results)) == 1

        self.test_results['determinism'] = {
            'passed': is_deterministic,
            'values': results,
            'message': 'PASS: Reward is deterministic' if is_deterministic else 'FAIL: Reward is non-deterministic'
        }

        return is_deterministic

    def test_bounds(
        self,
        test_cases: List[tuple],
        expected_min: float = -1000,
        expected_max: float = 10000
    ) -> bool:
        """
        Test if rewards stay within expected bounds

        Args:
            test_cases: List of (state, prev_state, action) tuples
            expected_min: Expected minimum reward
            expected_max: Expected maximum reward

        Returns:
            True if all rewards within bounds
        """
        rewards = []
        out_of_bounds = []

        for state, prev_state, action in test_cases:
            reward = self.reward_fn(state, prev_state, action)
            rewards.append(reward)

            if reward < expected_min or reward > expected_max:
                out_of_bounds.append({
                    'action': action,
                    'reward': reward
                })

        all_in_bounds = len(out_of_bounds) == 0

        self.test_results['bounds'] = {
            'passed': all_in_bounds,
            'min': min(rewards) if rewards else None,
            'max': max(rewards) if rewards else None,
            'expected_min': expected_min,
            'expected_max': expected_max,
            'out_of_bounds': out_of_bounds,
            'message': f'PASS: All rewards in bounds [{expected_min}, {expected_max}]' if all_in_bounds
                      else f'FAIL: {len(out_of_bounds)} rewards out of bounds'
        }

        return all_in_bounds

    def test_no_nan_inf(self, test_cases: List[tuple]) -> bool:
        """
        Test that rewards don't produce NaN or Inf

        Args:
            test_cases: List of (state, prev_state, action) tuples

        Returns:
            True if no NaN/Inf found
        """
        invalid_rewards = []

        for state, prev_state, action in test_cases:
            reward = self.reward_fn(state, prev_state, action)

            if np.isnan(reward) or np.isinf(reward):
                invalid_rewards.append({
                    'action': action,
                    'reward': reward
                })

        all_valid = len(invalid_rewards) == 0

        self.test_results['nan_inf'] = {
            'passed': all_valid,
            'invalid_count': len(invalid_rewards),
            'invalid_rewards': invalid_rewards,
            'message': 'PASS: No NaN/Inf values' if all_valid else f'FAIL: {len(invalid_rewards)} NaN/Inf values found'
        }

        return all_valid

    def test_action_differentiation(self, test_state: Any, prev_state: Any, actions: List[str]) -> bool:
        """
        Test that different actions produce different rewards

        Args:
            test_state: Test game state
            prev_state: Previous state
            actions: List of actions to test

        Returns:
            True if actions produce varied rewards
        """
        rewards = {}
        for action in actions:
            rewards[action] = self.reward_fn(test_state, prev_state, action)

        unique_rewards = len(set(rewards.values()))
        has_variety = unique_rewards > 1

        self.test_results['action_differentiation'] = {
            'passed': has_variety,
            'num_unique': unique_rewards,
            'total_actions': len(actions),
            'rewards': rewards,
            'message': f'PASS: {unique_rewards}/{len(actions)} unique reward values' if has_variety
                      else 'FAIL: All actions produce same reward'
        }

        return has_variety

    def print_results(self):
        """Print all test results"""
        print("=" * 80)
        print("🧪 REWARD FUNCTION TEST RESULTS")
        print("=" * 80)

        for test_name, result in self.test_results.items():
            status = "✅" if result['passed'] else "❌"
            print(f"\n{status} {test_name.upper()}:")
            print(f"   {result['message']}")

            # Print additional details for failed tests
            if not result['passed']:
                if 'out_of_bounds' in result and result['out_of_bounds']:
                    print(f"   Out of bounds examples:")
                    for item in result['out_of_bounds'][:3]:
                        print(f"      {item['action']}: {item['reward']}")

                if 'invalid_rewards' in result and result['invalid_rewards']:
                    print(f"   Invalid reward examples:")
                    for item in result['invalid_rewards'][:3]:
                        print(f"      {item['action']}: {item['reward']}")

        print("=" * 80)

        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['passed'])

        print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
        print("=" * 80)


class EnvironmentTester:
    """
    Test environment for correct behavior

    Usage:
        tester = EnvironmentTester(env)
        tester.run_all_tests()
    """

    def __init__(self, env):
        self.env = env
        self.test_results = {}

    def test_reset(self, n_trials: int = 5) -> bool:
        """Test environment reset works correctly"""
        try:
            for _ in range(n_trials):
                obs, info = self.env.reset()

                # Check observation shape
                if obs.shape != self.env.observation_space.shape:
                    self.test_results['reset'] = {
                        'passed': False,
                        'message': f'Observation shape mismatch: {obs.shape} vs {self.env.observation_space.shape}'
                    }
                    return False

            self.test_results['reset'] = {
                'passed': True,
                'message': f'Reset works correctly ({n_trials} trials)'
            }
            return True

        except Exception as e:
            self.test_results['reset'] = {
                'passed': False,
                'message': f'Reset failed with error: {str(e)}'
            }
            return False

    def test_step(self, n_steps: int = 100) -> bool:
        """Test environment step function"""
        try:
            self.env.reset()

            for _ in range(n_steps):
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Check observation shape
                if obs.shape != self.env.observation_space.shape:
                    self.test_results['step'] = {
                        'passed': False,
                        'message': f'Observation shape mismatch during step'
                    }
                    return False

                # Check reward is finite
                if np.isnan(reward) or np.isinf(reward):
                    self.test_results['step'] = {
                        'passed': False,
                        'message': f'Invalid reward: {reward}'
                    }
                    return False

                if terminated or truncated:
                    break

            self.test_results['step'] = {
                'passed': True,
                'message': f'Step works correctly ({n_steps} steps)'
            }
            return True

        except Exception as e:
            self.test_results['step'] = {
                'passed': False,
                'message': f'Step failed with error: {str(e)}'
            }
            return False

    def test_episode_completion(self, timeout_steps: int = 10000) -> bool:
        """Test that episodes can complete"""
        try:
            self.env.reset()

            for step in range(timeout_steps):
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    self.test_results['episode_completion'] = {
                        'passed': True,
                        'message': f'Episode completed in {step + 1} steps'
                    }
                    return True

            self.test_results['episode_completion'] = {
                'passed': False,
                'message': f'Episode did not complete in {timeout_steps} steps'
            }
            return False

        except Exception as e:
            self.test_results['episode_completion'] = {
                'passed': False,
                'message': f'Episode failed with error: {str(e)}'
            }
            return False

    def test_action_masking(self) -> bool:
        """Test that action masking works if available"""
        try:
            self.env.reset()

            if not hasattr(self.env, 'action_masks'):
                self.test_results['action_masking'] = {
                    'passed': True,
                    'message': 'No action masking available (skipped)'
                }
                return True

            mask = self.env.action_masks()

            # Check mask shape
            if len(mask) != self.env.action_space.n:
                self.test_results['action_masking'] = {
                    'passed': False,
                    'message': f'Action mask size mismatch: {len(mask)} vs {self.env.action_space.n}'
                }
                return False

            # Check at least one action is valid
            if not any(mask):
                self.test_results['action_masking'] = {
                    'passed': False,
                    'message': 'No valid actions in mask'
                }
                return False

            self.test_results['action_masking'] = {
                'passed': True,
                'message': f'Action masking works ({sum(mask)}/{len(mask)} valid actions)'
            }
            return True

        except Exception as e:
            self.test_results['action_masking'] = {
                'passed': False,
                'message': f'Action masking failed: {str(e)}'
            }
            return False

    def run_all_tests(self):
        """Run all environment tests"""
        print("=" * 80)
        print("🧪 ENVIRONMENT TESTS")
        print("=" * 80)

        tests = [
            ('Reset', self.test_reset),
            ('Step', self.test_step),
            ('Episode Completion', self.test_episode_completion),
            ('Action Masking', self.test_action_masking)
        ]

        for test_name, test_fn in tests:
            print(f"\nRunning: {test_name}...")
            start = time.time()
            result = test_fn()
            elapsed = time.time() - start

            status = "✅" if result else "❌"
            print(f"{status} {test_name}: {self.test_results[test_name.lower().replace(' ', '_')]['message']}")
            print(f"   Time: {elapsed:.3f}s")

        print("\n" + "=" * 80)

        total = len(tests)
        passed = sum(1 for r in self.test_results.values() if r['passed'])

        print(f"Summary: {passed}/{total} tests passed")
        print("=" * 80)


def test_training_stability(
    env,
    model,
    n_episodes: int = 10,
    reward_threshold: Optional[float] = None
) -> Dict:
    """
    Test if a trained model performs reasonably

    Args:
        env: Environment
        model: Trained model
        n_episodes: Number of test episodes
        reward_threshold: Expected minimum average reward

    Returns:
        Dictionary with test results
    """
    print("=" * 80)
    print("🧪 TRAINING STABILITY TEST")
    print("=" * 80)

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        print(f"Episode {episode + 1}/{n_episodes}: Reward={episode_reward:.1f}, Length={steps}")

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)

    print(f"\n📊 Results:")
    print(f"   Average Reward: {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"   Average Length: {avg_length:.1f}")

    passed = True
    if reward_threshold is not None:
        if avg_reward < reward_threshold:
            print(f"\n❌ FAIL: Average reward {avg_reward:.1f} below threshold {reward_threshold:.1f}")
            passed = False
        else:
            print(f"\n✅ PASS: Average reward above threshold")

    print("=" * 80)

    return {
        'passed': passed,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
