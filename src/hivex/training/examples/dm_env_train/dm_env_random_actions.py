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
from hivex.training.framework_wrappers.wrapper_utils import (
    EnvironmentParametersChannel,
    UnityEnvironment,
)

ENV_PATH = (
    "./tests/environment/hivex_test_env_rolling_ball_win/ML-Rolling-Ball_Unity.exe"
)


def train():
    channel = EnvironmentParametersChannel()
    channel.set_float_parameter("agent_type", 0)
    unity_env = UnityEnvironment(
        file_name=ENV_PATH, worker_id=0, no_graphics=False, side_channels=[channel]
    )
    unity_env.reset()

    hivex_dm_env = HivexDmEnvWrapper(unity_env=unity_env)
    StepType = dm_env._environment.StepType

    print_per_step_results = False
    for episode in range(9):
        hivex_dm_env.reset()
        tracked_agent = 0
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        step = 0
        while not done:
            # Generate an action for all agents
            actions = hivex_dm_env.generate_rnd_actions()

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
