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
#
"""HIVEX UnityEnvironment to pettingzoo ParallelEnv wrapper."""

import numpy as np
from typing import Union, Dict, List, Optional, Tuple, TypeVar

from gym import spaces, Space

# ML-Agents
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple
from mlagents_envs import logging_util
from pettingzoo.utils.env import ParallelEnv

# hivex
from hivex.training.framework_wrappers.wrapper_utils import (
    UnityEnvironment,
    action_is_continuous,
    action_is_discrete,
    get_vis_obs_count,
    get_vec_obs_size,
    get_vis_obs_shape,
    construct_observations,
)

logger = logging_util.get_logger(__name__)
"""Logger."""

ObsType = TypeVar("ObsType")
"""Observation type."""

ActionType = TypeVar("ActionType")
"""Action type."""

AgentID = str
"""Agent id."""

ObsDict = Dict[AgentID, ObsType]
"""Observation shape."""

ActionDict = Dict[AgentID, ActionType]
"""Action shape."""

PLAYER_STR_FORMAT = "player_{index}"
"""Agent player-id format."""


class HivexParallelEnvWrapper(ParallelEnv):
    """UnityEnvironment to pettingzoo ParallelEnv wrapper class."""

    # TODO:
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, unity_env: UnityEnvironment):
        """Construct a HivexParallelEnvWrapper environment.

        The Parallel environment steps every live agent at once. If you are unsure if you
        have implemented a ParallelEnv correctly, try running the `parallel_api_test` in
        the Developer documentation on the website.

        Attributes:
            unity_env (UnityEnvironment): Unity environment.
        """

        self.env = unity_env
        """Unity environment."""

        self.env.step()
        """Step the environment once to get specs."""

        # Get name of behavior and environment specs
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        """Behavior name."""

        self.group_spec = self.env.behavior_specs[self.behavior_name]
        """Environment specs."""

        # Check for number of agents in scene.
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        """Next non-terminal step(s)."""

        self.num_players = len(decision_steps.agent_id)
        """Number of agent(s)."""

        self.possible_agents = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(self.num_players)
        ]
        """Available agent(s)."""
        self.agents = self.possible_agents
        """All agent(s)."""

        # --- Set action space of a single Agent ---
        # Set continous action space
        continous_action_range_min = []
        continous_action_range_max = []
        self.continous_action_count = 0
        if action_is_continuous(group_spec=self.group_spec):
            self.continous_action_count = self.group_spec.action_spec.continuous_size
            continous_action_range_min = [-1] * self.continous_action_count
            continous_action_range_max = [1] * self.continous_action_count
            """Continous actions."""

        # Set discrete action space
        discrete_action_range_min = []
        discrete_action_range_max = []
        self.discrete_action_count = 0
        if action_is_discrete(group_spec=self.group_spec):
            self.discrete_action_count = len(
                self.group_spec.action_spec.discrete_branches
            )
            discrete_action_range_min = [0] * self.discrete_action_count
            discrete_action_range_max = [
                action_size
                for action_size in self.group_spec.action_spec.discrete_branches
            ]
            """Discrete actions."""

        self.single_action_space = []
        self.single_action_space = spaces.Box(
            low=np.array(continous_action_range_min + discrete_action_range_min),
            high=np.array(continous_action_range_max + discrete_action_range_max),
        )

        self.action_spaces = {
            self.possible_agents[i]: self.single_action_space
            for i in range(self.num_players)
        }

        # --- Set observations space of a single Agent ---
        # get all visual observations
        visual_obs_range_min = []
        visual_obs_range_max = []
        if get_vis_obs_count(self.group_spec) > 0:
            shapes = get_vis_obs_shape(group_spec=self.group_spec)
            for shape in shapes:
                visual_obs_range_min = [0] * shape[0] * shape[1]
                visual_obs_range_max = [1] * shape[0] * shape[1]
                pass
        """Visual observations."""

        vec_obs_range_min = []
        vec_obs_range_max = []
        if get_vec_obs_size(group_spec=self.group_spec) > 0:
            # for vector observation
            vec_obs_range_min = [np.inf] * get_vec_obs_size(group_spec=self.group_spec)
            vec_obs_range_max = [-np.inf] * get_vec_obs_size(group_spec=self.group_spec)
        """Vector observations."""

        self.single_observation_space = spaces.Box(
            low=np.array(visual_obs_range_min + vec_obs_range_min),
            high=np.array(visual_obs_range_max + vec_obs_range_max),
            dtype=np.float32,
        )

        self.observation_spaces = {
            self.possible_agents[i]: self.single_observation_space
            for i in range(self.num_players)
        }

        # dynamic agent fields
        self.multi_agent_observations = {}
        self.multi_agent_rewards = {}
        self.multi_agent_infos = {}
        self.multi_agent_dones = {}

    # reset(seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) → Dict[str, ObsType]

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ObsDict:
        """
        Resets the environment and returns a dictionary of observations (keyed by the agent name)

        Args:
            seed (int): Seed for the environment.
        Returns:
            observation dictionary (ObsDict): A dictionary of all agent(s)' observations.
        """
        # Reset unity environment
        self.env.reset()

        # ObsDict = Dict[AgentID, ObsType]
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        observations = {}
        for index, observation in enumerate(decision_steps.obs[0]):
            observations[PLAYER_STR_FORMAT.format(index=index)] = observation

        return observations

    def step(
        self, actions: ActionDict
    ) -> Tuple[ObsDict, Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        """

        Args:
            actions (ActionDict): A dictionary of actions keyed by the agent name.
        Returns:
            observation dictionary (ObsDict): Observation dictionary.
            reward dictionary (Dict[str, float]): Reward dictionary.
            done dictionary (Dict[str, bool]): Done dictionary.
            info dictionary (Dict[str, dict]): Info dicitonary.
        """

        # Return new state(s)
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        # Observations
        if len(decision_steps) > 0:
            observations = construct_observations(
                info=decision_steps, n_agents=len(self.agents)
            )

            agents_to_be_updated = decision_steps.agent_id
            for agent in agents_to_be_updated:
                self.multi_agent_observations[PLAYER_STR_FORMAT.format(index=agent)] = (
                    observations[agent]
                )
        if len(terminal_steps) > 0:
            observations = construct_observations(
                info=terminal_steps, n_agents=len(self.agents)
            )

            agents_to_be_updated = terminal_steps.agent_id
            for agent in agents_to_be_updated:
                self.multi_agent_observations[PLAYER_STR_FORMAT.format(index=agent)] = (
                    observations[agent]
                )

        # Rewards
        if len(decision_steps) > 0:
            for index, reward in enumerate(decision_steps.reward):
                self.multi_agent_rewards[
                    PLAYER_STR_FORMAT.format(index=decision_steps.agent_id[index])
                ] = reward
        if len(terminal_steps) > 0:
            for index, reward in enumerate(terminal_steps.reward):
                self.multi_agent_rewards[
                    PLAYER_STR_FORMAT.format(index=terminal_steps.agent_id[index])
                ] = reward

        # Dones
        for index in range(len(self.agents)):
            self.multi_agent_dones[PLAYER_STR_FORMAT.format(index=index)] = False
        if len(terminal_steps.interrupted) > 0:
            for index, done in enumerate(terminal_steps.interrupted):
                self.multi_agent_dones[
                    PLAYER_STR_FORMAT.format(index=terminal_steps.agent_id[index])
                ] = done

        # Construct actions
        action_tuple = ActionTuple()

        if action_is_continuous(group_spec=self.group_spec):
            continous_actions = np.array(
                [actions[agent][: self.continous_action_count] for agent in actions]
            )
            action_tuple.add_continuous(continous_actions)
        if action_is_discrete(group_spec=self.group_spec):
            discrete_action = np.array(
                [actions[agent][self.continous_action_count :] for agent in actions]
            )
            action_tuple.add_discrete(discrete_action)

        # Set action(s)
        if len(self.env._env_state[self.behavior_name][0]) != 0:
            self.env.set_actions(self.behavior_name, action_tuple)

        # Step environment with new action(s)
        self.env.step()

        # Returns the
        # observation dictionary,
        # reward dictionary,
        # terminated dictionary,
        # truncated dictionary and
        # info dictionary,
        # where each dictionary is keyed by the agent.

        truncated = {}
        infos = {}

        return (
            self.multi_agent_observations,
            self.multi_agent_rewards,
            self.multi_agent_dones,
            truncated,
            infos,
        )

    def render(self, mode="human") -> None:
        """Displays a rendered frame from the environment. Not used."""
        pass
        raise NotImplementedError

    def close(self):
        """Closes the environment."""
        # Close the Unity environment
        self.env.close()

    def state(self) -> np.ndarray:
        """
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    def observation_space(self, agent: AgentID = "player_0") -> Space:
        """
        Takes in agent and returns the observation space for that agent.
        MUST return the same value for the same agent name
        Default implementation is to return the observation_spaces dict

        Args:
            agent (AgentID): Agent id.
        Returns:
            observation space specs (gym.Space): Returns the observation space specs per agent id.
        """
        return self.observation_spaces[agent]
        # return self.observation_spaces

    def action_space(self, agent: AgentID) -> Space:
        """
        Takes in agent and returns the action space for that agent.
        MUST return the same value for the same agent name
        Default implementation is to return the action_spaces dict

        Args:
            agent (AgentID): Agent id.
        Returns:
            action space specs (gym.Space): Returns the action space specs per agent id.
        """
        # warnings.warn("Your environment should override the action_space function.
        # Attempting to use the action_spaces dict attribute.")
        return self.action_spaces[agent]

    @property
    def num_agents(self) -> int:
        """
        Returns number of agent(s).

        Returns:
            num_agents (int): Number of agent(s).
        """
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """
        Returns maximum number of agent(s).

        Returns:
            num_agents (int): Maximum number of agent(s).
        """
        return len(self.possible_agents)

    def __str__(self) -> str:
        """
        Returns a name which looks like: "space_invaders_v1" by default

        Returns:
            name (str): Name of environment.
        """
        if hasattr(self, "metadata"):
            return self.metadata.get("name", self.__class__.__name__)
        else:
            return self.__class__.__name__

    @property
    def unwrapped(self) -> ParallelEnv:
        return self
