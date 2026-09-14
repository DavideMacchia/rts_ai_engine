"""
Example: Training with All Refactored Utilities
Shows how to integrate all utilities into a real training loop
"""

import sys
import os

# Add ml_training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import utilities
from utils import (
    RewardComponentTracker,
    ActionMonitor,
    ActionDistributionAnalyzer,
    EpisodeStatsTracker,
    EpisodeAnalyzer,
    DifficultyScheduler,
    EnvironmentTester,
)

# Placeholder for your environment
# from training.realtime_rts_env import RealTimeRTSEnv


class ComprehensiveMonitoringCallback(BaseCallback):
    """
    Custom callback that uses all monitoring utilities
    """

    def __init__(
        self,
        reward_tracker,
        action_monitor,
        episode_tracker,
        difficulty_scheduler,
        check_freq=1000,
        verbose=0
    ):
        super().__init__(verbose)

        self.reward_tracker = reward_tracker
        self.action_monitor = action_monitor
        self.episode_tracker = episode_tracker
        self.difficulty_scheduler = difficulty_scheduler
        self.check_freq = check_freq

        self.episode_num = 0
        self.episode_rewards = []
        self.episode_actions = []

    def _on_step(self) -> bool:
        """Called at each environment step"""

        # Get current info (if available)
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]

            # Get action and reward
            action = self.locals.get('actions', [None])[0]
            reward = self.locals.get('rewards', [0])[0]

            # Track reward (would need component breakdown from env)
            # For now, log total reward
            if hasattr(self.training_env, 'envs'):
                # This is approximate - in real usage, get actual action name
                action_name = f"ACTION_{action}" if action is not None else "UNKNOWN"
                self.action_monitor.record(action_name)
                self.episode_actions.append(action_name)
                self.episode_rewards.append(reward)

            # Check if episode ended
            if self.locals.get('dones', [False])[0]:
                self._on_episode_end(info)

        # Periodic checks
        if self.n_calls % self.check_freq == 0:
            self._periodic_check()

        return True

    def _on_episode_end(self, info):
        """Handle episode end"""
        self.episode_num += 1

        # End episode for monitors
        self.action_monitor.end_episode(self.episode_num)
        self.reward_tracker.end_episode(self.episode_num)

        # Track episode stats
        total_reward = sum(self.episode_rewards)
        win = info.get('winner') == 0 if 'winner' in info else False

        # Clear episode data
        self.episode_rewards = []
        self.episode_actions = []

        # Update difficulty scheduler
        if self.difficulty_scheduler:
            new_difficulty = self.difficulty_scheduler.update(self.num_timesteps)
            # In real implementation, you'd update the environment's difficulty here

    def _periodic_check(self):
        """Periodic diagnostic checks"""
        print(f"\n{'='*80}")
        print(f"📊 TRAINING CHECKPOINT - Step {self.num_timesteps:,}")
        print(f"{'='*80}")

        # Print action distribution
        print("\n🎮 Action Distribution:")
        self.action_monitor.print_distribution(recent=True, top_n=10)

        # Diagnose action issues
        analyzer = ActionDistributionAnalyzer(self.action_monitor)
        diagnosis = analyzer.diagnose()

        if diagnosis['issues']:
            print("\n⚠️  Issues detected:")
            for issue in diagnosis['issues']:
                print(f"   {issue}")

        # Print reward summary
        print("\n💰 Reward Summary:")
        self.reward_tracker.print_summary(last_n_episodes=100)

        # Print episode stats
        if hasattr(self, 'episode_tracker'):
            print("\n📈 Episode Statistics:")
            self.episode_tracker.print_summary(n=100)

        print(f"{'='*80}\n")


def setup_monitoring():
    """Setup all monitoring utilities"""
    print("🔧 Setting up monitoring utilities...")

    reward_tracker = RewardComponentTracker()
    action_monitor = ActionMonitor(window_size=1000)
    episode_tracker = EpisodeStatsTracker(window_size=100)
    difficulty_scheduler = DifficultyScheduler()

    print("✅ Monitoring setup complete!")

    return reward_tracker, action_monitor, episode_tracker, difficulty_scheduler


def test_environment(env):
    """Test environment before training"""
    print("\n🧪 Testing environment...")

    tester = EnvironmentTester(env)
    tester.run_all_tests()

    # Check if all tests passed
    all_passed = all(r['passed'] for r in tester.test_results.values())

    if all_passed:
        print("\n✅ All environment tests passed! Ready to train.")
        return True
    else:
        print("\n❌ Some environment tests failed. Please fix before training.")
        return False


def train_with_monitoring():
    """
    Main training function with comprehensive monitoring

    This is a template - replace with your actual environment
    """

    print("=" * 80)
    print("🚀 TRAINING WITH COMPREHENSIVE MONITORING")
    print("=" * 80)

    # 1. Setup monitoring utilities
    reward_tracker, action_monitor, episode_tracker, difficulty_scheduler = setup_monitoring()

    # 2. Create environment (replace with your actual environment)
    print("\n📦 Creating environment...")
    # env = RealTimeRTSEnv()  # Your actual environment
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=False)

    print("⚠️  This is a template. Uncomment and use your actual environment.")
    print("    See ml_training/training/ for your RealTimeRTSEnv")
    return

    # 3. Test environment
    # if not test_environment(env.envs[0]):
    #     return

    # 4. Print difficulty schedule
    print("\n📅 Difficulty Schedule:")
    difficulty_scheduler.print_schedule()

    # 5. Create model
    print("\n🤖 Creating PPO model...")
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,
    #     ent_coef=0.01,
    #     verbose=1,
    #     tensorboard_log="./tensorboard_logs/"
    # )

    # 6. Create monitoring callback
    # callback = ComprehensiveMonitoringCallback(
    #     reward_tracker=reward_tracker,
    #     action_monitor=action_monitor,
    #     episode_tracker=episode_tracker,
    #     difficulty_scheduler=difficulty_scheduler,
    #     check_freq=10000,
    #     verbose=1
    # )

    # 7. Train
    print("\n🏋️  Starting training...")
    print("    (Monitoring will print updates every 10,000 steps)")

    # model.learn(
    #     total_timesteps=1_000_000,
    #     callback=callback,
    #     tb_log_name="ppo_rts_monitored"
    # )

    # 8. Save results
    print("\n💾 Saving monitoring data...")
    # reward_tracker.save_to_file("training_rewards.json")
    # episode_tracker.save_to_file("training_episodes.json")
    # action_monitor.plot_trends(save_path="action_trends.png")

    # 9. Final analysis
    print("\n📊 Final Analysis:")
    # analyzer = EpisodeAnalyzer(episode_tracker)
    # analyzer.print_analysis(window=100)

    print("\n✅ Training complete!")
    print("=" * 80)


def debug_existing_training():
    """
    Use utilities to debug an existing training run

    This shows how to diagnose issues with current training
    """

    print("=" * 80)
    print("🔍 DEBUGGING EXISTING TRAINING")
    print("=" * 80)

    print("\nThis mode helps diagnose issues with current training.")
    print("It will:")
    print("  1. Load your current model")
    print("  2. Run test episodes")
    print("  3. Analyze action distribution")
    print("  4. Identify issues")
    print("  5. Suggest fixes")

    # Setup monitoring
    action_monitor = ActionMonitor(window_size=500)
    episode_tracker = EpisodeStatsTracker(window_size=50)

    # Load environment and model
    # env = RealTimeRTSEnv()
    # model = PPO.load("your_model.zip")

    print("\n⚠️  Implement with your actual environment and model path")

    # Run test episodes with monitoring
    # n_test_episodes = 20
    # for episode in range(n_test_episodes):
    #     obs = env.reset()
    #     episode_tracker.start_episode(episode)
    #     done = False
    #
    #     while not done:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, info = env.step(action)
    #
    #         # Track
    #         action_name = ActionType(action).name
    #         action_monitor.record(action_name)
    #         episode_tracker.record_step(action_name, reward)
    #
    #     episode_tracker.end_episode(...)
    #     action_monitor.end_episode(episode)

    # Analyze
    # print("\n📊 Action Analysis:")
    # analyzer = ActionDistributionAnalyzer(action_monitor)
    # analyzer.print_diagnosis()
    # analyzer.suggest_fixes()

    # print("\n📈 Episode Analysis:")
    # ep_analyzer = EpisodeAnalyzer(episode_tracker)
    # ep_analyzer.print_analysis()

    print("\n✅ Debug analysis would appear here")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Training with comprehensive monitoring')
    parser.add_argument(
        '--mode',
        choices=['train', 'debug'],
        default='train',
        help='Mode: train (new training) or debug (analyze existing)'
    )

    args = parser.parse_args()

    if args.mode == 'train':
        train_with_monitoring()
    elif args.mode == 'debug':
        debug_existing_training()


if __name__ == '__main__':
    main()
