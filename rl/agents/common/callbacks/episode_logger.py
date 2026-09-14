"""
Custom callback to log episode statistics.
Tracks episode rewards, lengths, and game-specific metrics.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque


class EpisodeLoggerCallback(BaseCallback):
    """
    Logs episode statistics when episodes complete.
    Tracks rewards, lengths, population, military, and buildings.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_populations = deque(maxlen=100)
        self.episode_military = deque(maxlen=100)
        self.episode_buildings = deque(maxlen=100)
        self.episode_count = 0

        self.current_rewards = None
        self.current_lengths = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.current_rewards = [0.0] * n_envs
        self.current_lengths = [0] * n_envs

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])

        for i in range(len(rewards)):
            self.current_rewards[i] += rewards[i]
            self.current_lengths[i] += 1

            if i < len(dones) and dones[i]:
                self.episode_count += 1

                info = infos[i] if i < len(infos) else {}
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                else:
                    ep_reward = self.current_rewards[i]
                    ep_length = self.current_lengths[i]

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Track game-specific metrics
                self.episode_populations.append(info.get('agent_population', 0))
                self.episode_military.append(info.get('agent_military', 0))
                self.episode_buildings.append(info.get('agent_buildings', 0))

                self.current_rewards[i] = 0.0
                self.current_lengths[i] = 0

        if self.n_calls % 2048 == 0 and len(self.episode_rewards) > 0:
            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards))
            self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths))
            self.logger.record("rollout/episodes", self.episode_count)

            if len(self.episode_populations) > 0:
                self.logger.record("game/mean_population", np.mean(self.episode_populations))
                self.logger.record("game/mean_military", np.mean(self.episode_military))
                self.logger.record("game/mean_buildings", np.mean(self.episode_buildings))

            if self.verbose > 0:
                print(f"\n  Episode Stats (last {len(self.episode_rewards)}):")
                print(f"   Reward: {np.mean(self.episode_rewards):.2f}")
                print(f"   Length: {np.mean(self.episode_lengths):.0f}")
                if len(self.episode_populations) > 0:
                    print(f"   Pop: {np.mean(self.episode_populations):.1f}"
                          f"  Mil: {np.mean(self.episode_military):.1f}"
                          f"  Build: {np.mean(self.episode_buildings):.1f}")

        return True
