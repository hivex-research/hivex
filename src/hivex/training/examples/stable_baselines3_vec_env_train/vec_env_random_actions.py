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
"""HIVEX stablebaselines3 VecEnv random actions example."""

from hivex.training.framework_wrappers.unity_stable_baselines3.unity_vec_env_wrapper import (
    HivexVecEnvWrapper,
)
from stable_baselines3.common.vec_env import VecNormalize

from hivex.training.framework_wrappers.wrapper_utils import initialize_unity_environment

ENV_PATH = "tests/environments/hivex_test_env_rolling_ball_headless_win/ML-Rolling-Ball_Unity.exe"
WORKER_ID = 0 + 1012


def train():
    unity_env = initialize_unity_environment(0, ENV_PATH, WORKER_ID)

    vec_env_normalized = HivexVecEnvWrapper(unity_env)

    print_per_step_results = True
    for episode in range(9):
        vec_env_normalized.reset()
        tracked_agent = 0
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        step = 0
        while not done:
            # Generate an action for all agents
            actions = vec_env_normalized.generate_rnd_actions()

            # Get the new simulation results
            obs, rewards, dones, info = vec_env_normalized.step(actions)
            done = any(dones)

            print(dones)

            episode_rewards += rewards

            # Per step print
            if print_per_step_results:
                step += 1
                print(
                    f"step: {step}; "
                    + f"agent id: {tracked_agent}; "
                    + f"action: {actions[tracked_agent]}; "
                    + f"observations: {obs[tracked_agent]}; "
                    + f"reward: {rewards[tracked_agent]}; "
                    + f"done: {done};"
                )
        print(f"Total rewards for episode {episode} is {episode_rewards}")

    vec_env_normalized.close()
    print("Closed environment")


if __name__ == "__main__":
    train()
