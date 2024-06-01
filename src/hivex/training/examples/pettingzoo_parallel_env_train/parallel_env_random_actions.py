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
"""HIVEX pettingzoo ParallelEnv training example."""

from hivex.training.framework_wrappers.unity_pettingzoo.unity_parallel_env_wrapper import (
    HivexParallelEnvWrapper,
)

from hivex.training.framework_wrappers.wrapper_utils import (
    EnvironmentParametersChannel,
    UnityEnvironment,
)

ENV_PATH = "tests/environments/hivex_test_env_rolling_ball_headless_win/ML-Rolling-Ball_Unity.exe"


def test_parallel_env_random_actions():
    channel = EnvironmentParametersChannel()
    channel.set_float_parameter("agent_type", 0)
    unity_env = UnityEnvironment(
        file_name=ENV_PATH,
        worker_id=2,
        base_port=6000,
        no_graphics=False,
        side_channels=[channel],
    )

    parallel_env = HivexParallelEnvWrapper(unity_env=unity_env)

    print_per_step_results = True
    for episode in range(2):
        tracked_agent = parallel_env.agents[0]
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        step = 0
        parallel_env.reset()
        while not done:
            actions = {
                agent: parallel_env.action_space(agent).sample()
                for agent in parallel_env.agents
            }
            observations, rewards, terminations, truncations, info = parallel_env.step(
                actions
            )
            done = any(terminations)

            episode_rewards += rewards[tracked_agent]

            # Per step print
            if print_per_step_results:
                step += 1
                print(
                    f"step: {step}; "
                    + f"agent id: {tracked_agent}; "
                    + f"action: {actions[tracked_agent]}; "
                    + f"observations: {observations[tracked_agent]}; "
                    + f"reward: {rewards[tracked_agent]}; "
                    + f"done: {done};"
                )
        print(f"Total rewards for episode {episode} is {episode_rewards}")

    parallel_env.close()
    print("Closed environment")
