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
"""HIVEX UnityEnvironment training example."""

from hivex.training.framework_wrappers.wrapper_utils import (
    EnvironmentParametersChannel,
    UnityEnvironment,
    check_unity_environment,
)

ENV_PATH = (
    "./tests/environments/hivex_test_env_rolling_ball_win/ML-Rolling-Ball_Unity.exe"
)


def test_unity_environment_random_actions():
    channel = EnvironmentParametersChannel()
    agentType = 0
    channel.set_float_parameter("agent_type", agentType)
    unity_env = UnityEnvironment(
        file_name=ENV_PATH, worker_id=0, no_graphics=False, side_channels=[channel]
    )
    unity_env.reset()
    behavior_name, spec = check_unity_environment(env=unity_env)
    print_per_step_results = True

    for episode in range(3):
        unity_env.reset()
        decision_steps, terminal_steps = unity_env.get_steps(behavior_name)
        tracked_agent = -1  # -1 indicates not yet tracking
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        step = 0
        while not done:
            # Track the first agent we see if not tracking
            # Note : len(decision_steps) = [number of agents that requested a decision]
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # Generate an action for all agents
            actions = spec.action_spec.random_action(len(decision_steps))

            # Set the actions
            print(actions)
            unity_env.set_actions(behavior_name, actions)

            # Move the simulation forward
            unity_env.step()

            # Get the new simulation results
            decision_steps, terminal_steps = unity_env.get_steps(behavior_name)
            print(f"decision_steps {decision_steps.obs[0].shape}")
            print(f"terminal_steps {terminal_steps.obs[0].shape}")

            print("")

            # TODO: check if decision_steps and terminal steps both give rewards, observations and dones
            if tracked_agent in decision_steps:  # The agent requested a decision
                episode_rewards += decision_steps[tracked_agent].reward
                rew = decision_steps[tracked_agent].reward
                obs = decision_steps[tracked_agent].obs
            if tracked_agent in terminal_steps:  # The agent terminated its episode
                episode_rewards += terminal_steps[tracked_agent].reward
                rew = terminal_steps[tracked_agent].reward
                obs = terminal_steps[tracked_agent].obs
                done = True

            # Per step print
            if print_per_step_results:
                # TODO: we don't have a step info from unity env?
                step += 1
                print(
                    f"step: {step}; "
                    + f"agent id: {tracked_agent}; "
                    + f"action: {actions[tracked_agent]}; "
                    + f"observations: {obs}; "
                    + f"reward: {rew}; "
                    + f"done: {done};"
                )
        print(f"Total rewards for episode {episode} is {episode_rewards}")

    unity_env.close()
    print("Closed environment")
