"""
Comprehensive Demo of ML Training Utilities
Shows how to use all the refactored utilities
"""

import sys
import os

# Add ml_training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Dict

# Import utilities
from utils import (
    # Debugging
    RewardDebugger,
    RewardComponentTracker,

    # Monitoring
    ActionMonitor,
    ActionDistributionAnalyzer,
    EpisodeStatsTracker,
    EpisodeAnalyzer,

    # Curriculum
    DifficultyScheduler,
    CurriculumManager,

    # Reward Components
    ModularRewardCalculator,
    create_default_reward_calculator,

    # Testing
    EnvironmentTester,

    # Checkpoint Comparison
    CheckpointComparator,
)


def demo_reward_tracking():
    """Demo: Reward Component Tracking"""
    print("\n" + "=" * 80)
    print("DEMO 1: REWARD COMPONENT TRACKING")
    print("=" * 80)

    tracker = RewardComponentTracker()

    # Simulate 3 episodes
    for episode in range(3):
        tracker.log('survival', 100.0, {'time': 30})
        tracker.log('economic', 250.5, {'resources': 1500})
        tracker.log('population', 150.0, {'population': 15})
        tracker.log('military', 75.0, {'soldiers': 5})
        tracker.log('building', 180.0, {'buildings': 12})

        tracker.end_episode(episode)

    # Print summary
    tracker.print_summary()

    # Save to file
    tracker.save_to_file('reward_tracking_demo.json')
    print("\n✅ Demo 1 complete!")


def demo_action_monitoring():
    """Demo: Action Distribution Monitoring"""
    print("\n" + "=" * 80)
    print("DEMO 2: ACTION DISTRIBUTION MONITORING")
    print("=" * 80)

    monitor = ActionMonitor(window_size=100)

    # Simulate actions
    actions = [
        'BUILD_FARM', 'BUILD_FARM', 'BUILD_HOUSE', 'DO_NOTHING',
        'BUILD_LUMBERYARD', 'BUILD_FARM', 'TRAIN_SOLDIER', 'DO_NOTHING',
        'BUILD_BARRACKS', 'TRAIN_SOLDIER', 'ATTACK'
    ] * 10  # Repeat to get more data

    for i, action in enumerate(actions):
        monitor.record(action)
        if (i + 1) % 30 == 0:
            monitor.end_episode(i // 30)

    # Print distribution
    monitor.print_distribution()

    # Analyze with diagnostics
    analyzer = ActionDistributionAnalyzer(monitor)
    analyzer.print_diagnosis()
    analyzer.suggest_fixes()

    print("\n✅ Demo 2 complete!")


def demo_episode_statistics():
    """Demo: Episode Statistics Tracking"""
    print("\n" + "=" * 80)
    print("DEMO 3: EPISODE STATISTICS TRACKING")
    print("=" * 80)

    tracker = EpisodeStatsTracker(window_size=50)

    # Simulate 10 episodes
    for ep in range(10):
        tracker.start_episode(ep)

        # Simulate steps
        for step in range(100):
            action = np.random.choice([
                'BUILD_FARM', 'BUILD_HOUSE', 'DO_NOTHING', 'TRAIN_SOLDIER'
            ])
            reward = np.random.randn() * 10 + 50  # Mean 50, std 10
            tracker.record_step(action, reward)

        # End episode
        total_reward = 1000 + np.random.randn() * 200
        win = np.random.random() > 0.5
        tracker.end_episode(total_reward, win)

    # Print summary
    tracker.print_summary()

    # Analyze
    analyzer = EpisodeAnalyzer(tracker)
    analyzer.print_analysis()

    # Save
    tracker.save_to_file('episode_stats_demo.json')

    print("\n✅ Demo 3 complete!")


def demo_curriculum_learning():
    """Demo: Curriculum Learning Scheduler"""
    print("\n" + "=" * 80)
    print("DEMO 4: CURRICULUM LEARNING")
    print("=" * 80)

    # Option 1: Step-based scheduler
    print("\n--- Step-Based Scheduler ---")
    scheduler = DifficultyScheduler()
    scheduler.print_schedule()

    # Simulate training
    for step in [50_000, 150_000, 600_000, 1_200_000]:
        difficulty = scheduler.update(step)
        print(f"Step {step:,}: Difficulty = {difficulty}")

    # Option 2: Performance-based curriculum
    print("\n--- Performance-Based Curriculum ---")
    curriculum = CurriculumManager(
        initial_difficulty='idle',
        advancement_threshold=0.7
    )

    # Simulate episodes with improving performance
    for ep in range(100):
        reward = 1000 + ep * 10  # Improving rewards
        win = np.random.random() < (0.3 + ep * 0.005)  # Improving win rate
        curriculum.record_episode(reward, win)

        # Check if should advance
        if curriculum.should_increase_difficulty():
            curriculum.advance_difficulty()

    curriculum.print_status()

    print("\n✅ Demo 4 complete!")


def demo_modular_rewards():
    """Demo: Modular Reward Calculator"""
    print("\n" + "=" * 80)
    print("DEMO 5: MODULAR REWARD CALCULATOR")
    print("=" * 80)

    # Create calculator with default components
    calculator = create_default_reward_calculator()
    calculator.print_configuration()

    # Mock game state
    class MockState:
        def __init__(self):
            self.resources = {'wood': 100, 'stone': 50, 'grain': 30}
            self.population = 10
            self.military_strength = 5
            self.buildings = {'farm': 2, 'house': 1, 'barracks': 1}

    current_state = MockState()
    previous_state = MockState()
    previous_state.population = 8  # Less population before

    # Calculate reward with breakdown
    total_reward, breakdown = calculator.calculate_with_breakdown(
        current_state,
        previous_state,
        action='BUILD_FARM',
        delta_time=30.0
    )

    print(f"\n🎯 Total Reward: {total_reward:.2f}")
    print(f"\n📊 Component Breakdown:")
    for component, value in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"   {component:<20} {value:>10.2f}")

    print("\n✅ Demo 5 complete!")


def demo_reward_debugger():
    """Demo: Real-time Reward Debugger"""
    print("\n" + "=" * 80)
    print("DEMO 6: REAL-TIME REWARD DEBUGGER")
    print("=" * 80)

    debugger = RewardDebugger(buffer_size=100)

    # Simulate training steps
    actions = ['BUILD_FARM', 'BUILD_HOUSE', 'DO_NOTHING', 'TRAIN_SOLDIER', 'ATTACK']

    for i in range(50):
        action = np.random.choice(actions)
        reward = np.random.randn() * 20 + 100

        # Log with components
        components = {
            'survival': np.random.uniform(0, 50),
            'economic': np.random.uniform(0, 100),
            'military': np.random.uniform(0, 50)
        }

        debugger.log_step(action, reward, reward_components=components)

    # Print recent history
    debugger.print_recent(n=10)

    # Analyze action-reward correlation
    debugger.print_action_analysis()

    print("\n✅ Demo 6 complete!")


def demo_environment_testing():
    """Demo: Environment Testing (requires actual environment)"""
    print("\n" + "=" * 80)
    print("DEMO 7: ENVIRONMENT TESTING")
    print("=" * 80)

    print("⚠️  This demo requires an actual environment instance.")
    print("    To run full tests, create your environment and run:")
    print()
    print("    from utils import EnvironmentTester")
    print("    from training.realtime_rts_env import RealTimeRTSEnv")
    print()
    print("    env = RealTimeRTSEnv()")
    print("    tester = EnvironmentTester(env)")
    print("    tester.run_all_tests()")

    print("\n✅ Demo 7 skipped (requires environment)")


def demo_checkpoint_comparison():
    """Demo: Checkpoint Comparison (requires models)"""
    print("\n" + "=" * 80)
    print("DEMO 8: CHECKPOINT COMPARISON")
    print("=" * 80)

    print("⚠️  This demo requires trained model checkpoints.")
    print("    To compare checkpoints, run:")
    print()
    print("    from utils import quick_compare")
    print("    from training.realtime_rts_env import RealTimeRTSEnv")
    print()
    print("    env = RealTimeRTSEnv()")
    print("    quick_compare(env, {")
    print("        '100k': 'models/ppo_100k.zip',")
    print("        '500k': 'models/ppo_500k.zip',")
    print("        '1M': 'models/ppo_1m.zip'")
    print("    })")

    print("\n✅ Demo 8 skipped (requires models)")


def main():
    """Run all demos"""
    print("=" * 80)
    print("🎯 ML TRAINING UTILITIES - COMPREHENSIVE DEMO")
    print("=" * 80)
    print()
    print("This demo showcases all the refactored ML utilities:")
    print("  1. Reward Component Tracking")
    print("  2. Action Distribution Monitoring")
    print("  3. Episode Statistics")
    print("  4. Curriculum Learning")
    print("  5. Modular Reward Calculator")
    print("  6. Real-time Reward Debugger")
    print("  7. Environment Testing")
    print("  8. Checkpoint Comparison")
    print()
    input("Press Enter to start...")

    try:
        demo_reward_tracking()
        input("\nPress Enter for next demo...")

        demo_action_monitoring()
        input("\nPress Enter for next demo...")

        demo_episode_statistics()
        input("\nPress Enter for next demo...")

        demo_curriculum_learning()
        input("\nPress Enter for next demo...")

        demo_modular_rewards()
        input("\nPress Enter for next demo...")

        demo_reward_debugger()
        input("\nPress Enter for next demo...")

        demo_environment_testing()
        input("\nPress Enter for next demo...")

        demo_checkpoint_comparison()

        print("\n" + "=" * 80)
        print("🎉 ALL DEMOS COMPLETE!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Check the generated JSON files for saved data")
        print("  2. Integrate these utilities into your training loop")
        print("  3. Use the debuggers to diagnose training issues")
        print("  4. Compare checkpoints to track progress")
        print()
        print("For integration examples, see: examples/training_with_utils.py")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
