import gymnasium as gym
import numpy as np


class EnvPoolEpisodeStats(gym.Wrapper):
    def __init__(self, env, n_envs: int):
        super().__init__(env)
        self.episode_returns = np.zeros(n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        dones = np.logical_or(terminated, truncated)

        self.episode_returns += rewards
        self.episode_lengths += 1

        if dones.any():
            info["stats"] = {
                "done_mask": dones,
                "returns": self.episode_returns.copy(),
                "lens": self.episode_lengths.copy(),
            }
            self.episode_returns[dones] = 0.0
            self.episode_lengths[dones] = 0

        return obs, rewards, terminated, truncated, info