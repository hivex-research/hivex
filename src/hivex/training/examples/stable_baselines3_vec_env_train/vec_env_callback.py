# Copyright 2022 The HIVEX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""HIVEX stablebaselines3 VecEnv training Callback."""

from typing import Union, Tuple
from pathlib import Path
import torch
import gym
import numpy as np
from tqdm import tqdm

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.base_class import BaseAlgorithm

DEVICE = torch.device("cpu")


class EvaluationCallback(BaseCallback):
    def __init__(
        self,
        eval_env: Union[VecEnv, gym.Env],
        eval_freq: int,
        n_eval_episodes: int,
        n_agents: int,
        eval_path: Path,
        normalization: bool,
        verbose=2,
    ):
        super(EvaluationCallback, self).__init__(verbose)
        self.env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = max(1, int(n_eval_episodes / n_agents))
        self.eval_path = eval_path
        self.best_average_reward = -float("inf")
        self.normalization = normalization
        self.logger = None
        self.last_callback_step = 0

    def _init_callback(self) -> None:
        print("initializeing logger")
        self.logger = Logger("/logger_path/", ["stdout", "csv", "tensorboard"])

    def _on_step(self) -> bool:
        if 0 < self.eval_freq <= (self.num_timesteps - self.last_callback_step):
            agent_rewards, episode_lengths = evaluate(
                self.model, self.env, self.n_eval_episodes
            )

            mean_reward, std_reward = agent_rewards.mean(), agent_rewards.std()
            mean_ep_length, std_ep_length = (
                episode_lengths.mean(),
                episode_lengths.std(),
            )
            self.last_mean_reward = mean_reward

            print(
                f"Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            self.logger.record(key="eval/mean_reward", value=float(mean_reward))
            self.logger.record(key="eval/mean_ep_length", value=float(mean_ep_length))

            if mean_reward > self.best_average_reward:
                print(
                    f"New best mean reward: {self.best_average_reward} -> {mean_reward}"
                )

                self.model.save(self.eval_path + "/best_model")
                if self.normalization:
                    normalization_path = str(self.eval_path / "vecnormalize.pkl")
                    self.model.get_vec_normalize_env().save(normalization_path)
                self.best_average_reward = mean_reward

            self.last_callback_step = self.num_timesteps

        return True


def evaluate(
    model: BaseAlgorithm, env: Union[VecEnv, gym.Env], number_of_episodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate for a given number of episodes."""
    rewards = []
    episode_lengths = []
    for _ in tqdm(list(range(number_of_episodes))):
        state = env.reset()
        reward_cum = 0
        steps = 0
        while True:
            actions = model.predict(state)[0]
            state, reward, done, _ = env.step(actions)
            reward_cum += reward
            steps += 1
            if np.any(done):
                break
        rewards.append(reward_cum)
        episode_lengths.append(steps)

    return np.array(rewards), np.array(episode_lengths)
