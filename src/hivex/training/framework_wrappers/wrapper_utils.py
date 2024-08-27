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
"""HIVEX wrapper utilities."""

from typing import List, Tuple, Union, Optional
import numpy as np

# hivex
from hivex.environment_registry import HivexEnvironmentRegistry

# ML-Agents
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_envs.registry import UnityEnvRegistry


def initialize_unity_environment(
    worker_id: int,
    time_scale: Optional[float] = 20.0,
    env_path: Optional[str] = None,
    hivex_env_tag: Optional[str] = "WindFarmControl",
    no_graphics: Optional[bool] = True,
    **kwargs,
) -> UnityEnvironment:
    worker_id += 1
    env_parameter_channel = EnvironmentParametersChannel()
    for key, value in kwargs.items():
        env_parameter_channel.set_float_parameter(key, value)

    if env_path:
        engine_config_channel = EngineConfigurationChannel()
        engine_config_channel.set_configuration_parameters(time_scale=time_scale)
        hivex_unity_env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            no_graphics=no_graphics,
            side_channels=[env_parameter_channel, engine_config_channel],
        )
    else:
        stats_channel = StatsSideChannel()
        hivex_registry = HivexEnvironmentRegistry()
        hivex_unity_env = hivex_registry.make_env(
            environment_tag=hivex_env_tag,
            no_graphics=no_graphics,
            worker_id=worker_id,
            side_channels=[env_parameter_channel, stats_channel],
        )

    hivex_unity_env.reset()

    return hivex_unity_env


def check_unity_environment(env: UnityEnvironment) -> tuple([str, dict()]):
    """Check if unity environment is built correctly.

    Args:
        env (UnityEnvironment): Unity environment.
    Returns:
        behavior name and specs (tuple([str, dict()]): Returns environment behavior namd and specs.
    """

    # We will only consider the first Behavior
    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior: {behavior_name}")
    spec = env.behavior_specs[behavior_name]

    # Examine the number of observations per Agent
    obs = spec.observation_specs
    print("Number of observations: ", len(obs))

    # Is there a visual observation ?
    # Visual observation have 3 dimensions: Height, Width and number of channels
    vis_obs = any(len(spec.shape) == 3 for spec in obs)
    print("Is there a visual observation?", vis_obs)

    # Is the Action continuous or discrete?
    actions = 0
    if spec.action_spec.continuous_size > 0:
        print(f"There are {spec.action_spec.continuous_size} continuous actions")
        actions += spec.action_spec.continuous_size
    if spec.action_spec.is_discrete():
        print(f"There are {spec.action_spec.discrete_size} discrete actions")
        actions += spec.action_spec.discrete_size

    # For discrete actions only : How many different options does each action have?
    if spec.action_spec.discrete_size > 0:
        for action, branch_size in enumerate(spec.action_spec.discrete_branches):
            print(f"Action number {action} has {branch_size} different options")

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    env.set_actions(behavior_name, spec.action_spec.empty_action(len(decision_steps)))

    print(
        f"Check complete. Environment sample specs: "
        + f"actions: {actions}; "
        + f"observations: {obs}; "
        + f"rewards: {decision_steps.reward}; "
        + f"done: {terminal_steps.interrupted};"
    )

    return (behavior_name, spec)


def get_vector_obs(step_result: Union[DecisionSteps, TerminalSteps]) -> np.ndarray:
    """Get vector observation.

    Args:
        step_result (DecisionSteps/TerminalSteps): Decision or terminal step tuple (observation, reward, done, info).
    Returns:
        vector observations (np.ndarray): Vector observation in the form of `np.ndarray`.
    """
    result: List[np.ndarray] = []
    for obs in step_result.obs:
        if len(obs) != 0 and len(obs.shape) == 2:
            result.append(obs)
    return np.concatenate(result, axis=1) if len(result) != 0 else result


def get_vec_obs_size(group_spec) -> int:
    """Get vector observation size.

    Returns:
        vector observations size (int): Vector observation size in the form of `int`.
    """
    result = 0
    for obs_spec in group_spec.observation_specs:
        if len(obs_spec.shape) == 1:
            result += obs_spec.shape[0]
    return result


def get_vis_obs_count(group_spec) -> int:
    """Get the number of visual observations.

    Returns:
        visual observation count (int): Visual observation count value as `int`.
    """
    result = 0
    for obs_spec in group_spec.observation_specs:
        if len(obs_spec.shape) == 3:
            result += 1
    return result


def get_vis_obs_shape(group_spec) -> List[Tuple]:
    """Get the shape of the visual observation.

    Returns:
        visual observation shape (list(tuple)): Visual observation shape as a `List(Tuple)`.
    """
    result: List[Tuple] = []
    for obs_spec in group_spec.observation_specs:
        if len(obs_spec.shape) == 3:
            result.append(obs_spec.shape)
    return result


def get_vis_obs_list(
    step_result: Union[DecisionSteps, TerminalSteps]
) -> List[np.ndarray]:
    """Get visual observation list.

    Args:
        step_result (DecisionSteps/TerminalSteps): Decision or terminal step tuple (observation, reward, done, info).
    Returns:
        list of visual observations (list(np.ndarray)): List of visual observation in the form of `List(np.ndarray)`.
    """
    result: List[np.ndarray] = []
    for obs in step_result.obs:
        if len(obs) != 0 and len(obs.shape) == 4:
            result.append(obs)
    return result


def action_is_continuous(group_spec):
    """Check if environment includes continous actions."""
    return group_spec.action_spec.continuous_size > 0


def action_is_discrete(group_spec):
    """Check if environment includes discrete actions."""
    return group_spec.action_spec.discrete_size > 0


def update_observations(
    info: Union[DecisionSteps, TerminalSteps], multi_agent_observations, n_agents
):
    agents_to_be_updated = info.agent_id
    observations = construct_observations(info=info, n_agents=n_agents)

    for agent in agents_to_be_updated:
        multi_agent_observations[agent] = observations[agent]

    return multi_agent_observations


def construct_observations(info, n_agents):
    agents_to_be_updated = info.agent_id

    observations = [None] * n_agents

    visual_obs = get_vis_obs_list(step_result=info)
    vector_obs = get_vector_obs(step_result=info)

    for index, agent_id in enumerate(agents_to_be_updated):
        # visual observation
        if len(visual_obs) > 0:
            vis_obs = [obs[index] for obs in visual_obs]
        else:
            vis_obs = []
        # vector observation
        if len(vector_obs) > 0:
            vec_obs = vector_obs[index]

        if len(vis_obs) == 0:
            observations[agent_id] = [vec_obs]
        elif len(vec_obs) == 0:
            observations[agent_id] = [vis_obs]
        else:
            observations[agent_id] = [vis_obs, vec_obs]

    return observations


def remap(value, from1, to1, from2, to2):
    return (value - from1) / (to1 - from1) * (to2 - from2) + from2
