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
"""HIVEX dm_env training example."""

from hivex.training.framework_wrappers.unity_dm_env.unity_dm_env_wrapper import (
    dm_env,
    HivexDmEnvWrapper,
)
from hivex.training.framework_wrappers.wrapper_utils import initialize_unity_environment

ENV_PATH = "tests/environments/hivex_test_env_rolling_ball_headless_win/ML-Rolling-Ball_Unity.exe"
WORKER_ID = 0


def train():
    unity_env = initialize_unity_environment(0, ENV_PATH, WORKER_ID)

    hivex_dm_env = HivexDmEnvWrapper(unity_env=unity_env)
    StepType = dm_env._environment.StepType

    print_per_step_results = True
    for episode in range(9):
        hivex_dm_env.reset()
        tracked_agent = 0
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        step = 0
        while not done:
            # Generate an action for all agents
            actions = hivex_dm_env._action_space.generate_value()

            # Get the new simulation results
            time_step = hivex_dm_env.step(actions=actions)

            obs = time_step.observation
            rew = time_step.reward
            done = time_step.step_type == StepType.LAST

            episode_rewards += rew

            # Per step print
            if print_per_step_results and not done:
                step += 1
                print(
                    f"step: {step}; "
                    + f"agent id: {tracked_agent}; "
                    + f"action: {actions[tracked_agent]}; "
                    + f"observations: {obs[tracked_agent]}; "
                    + f"reward: {rew[tracked_agent]}; "
                    + f"done: {done};"
                )
        print(f"Total rewards for episode {episode} is {episode_rewards}")

    hivex_dm_env.env.close()
    print("Closed environment")


if __name__ == "__main__":
    train()
