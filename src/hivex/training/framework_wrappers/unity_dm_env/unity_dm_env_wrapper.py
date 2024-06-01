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
# Based on the dm_env repository catch example.
# Docstrings have been copied for ease of development.
# https://github.com/deepmind/dm_env/blob/master/examples/catch.py
# https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py
"""HIVEX UnityEnvironment to dm_env wrapper."""

import numpy as np

import dm_env
from dm_env import specs

# ML-Agents
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps

# hivex
from hivex.training.wrapper_utils import (
    UnityEnvironment,
    action_is_continuous,
    action_is_discrete,
    get_vis_obs_count,
    get_vec_obs_size,
    get_vis_obs_shape,
    update_observations,
)

from typing import Dict, Union


class HivexDmEnvMultiAgentActionSpec:
    """
    Action spec for multi-agent environment
    consisting of discrete and continous actions
    """

    def __init__(self, agent_count: int):
        self.agent_count = agent_count
        self.ContinousActions: list[specs.BoundedArray] = []
        self.DiscreteActions: list[specs.BoundedArray] = []

    def construct_action_spec(self):
        self.multi_agent_action_spec = []
        for agent in range(self.agent_count):
            actions = []
            if len(self.ContinousActions) > 0:
                actions.append(self.ContinousActions[agent])
            if len(self.DiscreteActions) > 0:
                actions.append(self.DiscreteActions[agent])
            self.multi_agent_action_spec.append(actions)

    def generate_value(self):
        mutli_agent_actions = []
        for agent in range(self.agent_count):
            actions = []
            if len(self.ContinousActions) > 0:
                actions.append(self.ContinousActions[agent].generate_value())
            if len(self.DiscreteActions) > 0:
                actions.append(self.DiscreteActions[agent].generate_value())
            mutli_agent_actions.append(actions)

        return mutli_agent_actions


class HivexDmEnvMultiAgentObservationSpec:
    """
    Obervation spec for multi-agent environment
    consisting of vector and visual observation(s)
    """

    def __init__(self, agent_count: int):
        self.agent_count = agent_count
        self.visual_observation: list[specs.BoundedArray] = []
        self.vector_observation: list[specs.BoundedArray] = []

    def construct_observation_spec(self):
        self.obs_spec = []
        for agent in range(self.agent_count):
            self.obs_spec.append(
                [
                    self.visual_observation[agent],
                    self.vector_observation[agent],
                ]
            )


class HivexDmEnvWrapper(dm_env.Environment):
    """UnityEnvironment to dm_env wrapper class."""

    def __init__(self, unity_env: UnityEnvironment):
        """Construct a HivexDmEnvWrapper environment.

        Attributes:
            unity_env (UnityEnvironment): Unity environment.
        """
        self.env = unity_env
        """Get Unity environment."""

        self.env.reset()

        self.env.step()
        """Step the environment once to get specs."""

        # Get name of behavior and environment specs
        self.name = list(self.env.behavior_specs.keys())[0]
        """Get behavior name."""
        self.group_spec = self.env.behavior_specs[self.name]
        """Get environment specs."""

        # Check for number of agents in scene.
        decision_steps, _ = self.env.get_steps(self.name)
        """Get next non-terminal step(s)."""

        self.n_agents = len(decision_steps)
        """Get number of agent(s)."""

        # Continous 0 - Discrete 1
        self._action_space = HivexDmEnvMultiAgentActionSpec(agent_count=self.n_agents)

        # --- Set action spaces ---
        # Set continous action space
        if action_is_continuous(self.group_spec):
            continous_action_space = specs.BoundedArray(
                (self.group_spec.action_spec.continuous_size,),
                dtype=np.float32,
                minimum=-1.0,
                maximum=1.0,
                name="continous_action_space",
            )
            continous_action_space_all_agents = [
                continous_action_space for agent in range(self.n_agents)
            ]
            self._action_space.ContinousActions = continous_action_space_all_agents
            """Continous action space size."""

        # Set discrete action space
        if action_is_discrete(self.group_spec):
            branches = self.group_spec.action_spec.discrete_branches
            """Discrete action branches."""

            discrete_action_space = specs.BoundedArray(
                (self.group_spec.action_spec.discrete_size,),
                dtype=np.int,
                minimum=0,
                maximum=[branch for branch in branches],
                name="discrete_action_space",
            )

            discrete_action_space_all_agents = [
                discrete_action_space for agent in range(self.n_agents)
            ]

            self._action_space.DiscreteActions = discrete_action_space_all_agents
            """Discrete actions."""

        self._action_space.construct_action_spec()

        # --- Set observations space ---
        # get all visual observations
        visual_obs = list([])
        if get_vis_obs_count(self.group_spec) > 0:
            visual_obs = []
            shapes = get_vis_obs_shape(self.group_spec)
            for shape in shapes:
                visual_obs.append(
                    specs.BoundedArray(
                        shape=shape,
                        dtype=np.float32,
                        minimum=0,
                        maximum=1,
                        name="visual_obs",
                    )
                )
        """Visual observations."""

        vector_obs = list([])
        if get_vec_obs_size(self.group_spec) > 0:
            # for vector observation
            vec_obs_size = get_vec_obs_size(self.group_spec)
            high = np.array([np.inf] * vec_obs_size)
            vector_obs = specs.BoundedArray(
                shape=(vec_obs_size,),
                dtype=np.float32,
                minimum=-1,
                maximum=1,
                name="vector_obs",
            )
        """Vector observations."""

        # Visual Observations 0 - Vector Observations 1
        self.observation_space = HivexDmEnvMultiAgentObservationSpec(
            agent_count=self.n_agents
        )
        for _ in range(self.n_agents):
            # visual observation
            self.observation_space.visual_observation.append(visual_obs)
            # vector observation
            self.observation_space.vector_observation.append(vector_obs)

        self.observation_space.construct_observation_spec()

        # Initialize environment
        self.reset_next_step = True
        """Indicate initial step."""

        # dynamic agent fields
        self.multi_agent_observations = [None] * self.n_agents

    def reset(self) -> dm_env.TimeStep:
        """Starts a new sequence and returns the first `TimeStep` of this sequence.

        Returns:
            A `TimeStep` namedtuple containing:
            step_type (StepType): A `StepType` of `FIRST`.
            reward (None): `None`, indicating the reward is undefined.
            discount (None): `None`, indicating the discount is undefined.
            observation (np.array): A NumPy array, or a nested dict, list or tuple of arrays.
            Scalar values that can be cast to NumPy arrays (e.g. Python floats) are also
            valid in place of a scalar array. Must conform to the specification returned by
            `observation_spec()`.
        """

        self.env.reset()
        self.env.step()
        self.reset_next_step = False
        return dm_env.restart(self.observation())

    def step(self, actions: dict) -> dm_env.TimeStep:
        """Updates the environment according to the action and returns a `TimeStep`.

        Args:
            actions (np.array): A NumPy array, or a nested dict, list or tuple of arrays corresponding to `action_spec()`.

        Returns:
            A `TimeStep` namedtuple containing:
            step_type (StepType): A `StepType` value.
            reward (np.array): Reward at this timestep, or None if step_type is `StepType.FIRST`.
            Must conform to the specification returned by `reward_spec()`.
            discount (np.array): A discount in the range [0, 1], or None if step_type is `StepType.FIRST`.
            Must conform to the specification returned by `discount_spec()`.
            observation (np.ndarray): A NumPy array, or a nested dict, list or tuple of arrays.
            Scalar values that can be cast to NumPy arrays (e.g. Python floats) are also valid in place
            of a scalar array. Must conform to the specification returned by `observation_spec()`.
        """
        if self.reset_next_step:
            return self.reset()

        # Construct an `ActionTuple` as this is the action format for the `UnityEnvironment` environment.
        # Dimensions are of (n_agents, continuous_size) and (n_agents, discrete_size)
        action_tuple = ActionTuple()

        if action_is_continuous(group_spec=self.group_spec):
            continous_action = np.array([agent[0] for agent in actions])
            action_tuple.add_continuous(continous_action)
            if action_is_discrete(group_spec=self.group_spec):
                discrete_action = np.array([agent[1] for agent in actions])
                action_tuple.add_discrete(discrete_action)
        elif action_is_discrete(group_spec=self.group_spec):
            discrete_action = np.array([agent[0] for agent in actions])
            action_tuple.add_discrete(discrete_action)

        # Set action(s)
        if len(self.env._env_state[self.name][0]) != 0:
            self.env.set_actions(self.name, action_tuple)

        # Step environment
        self.env.step()
        # Return new state(s)
        decision_steps, terminal_steps = self.env.get_steps(self.name)

        rewards = np.zeros(self.n_agents, dtype=np.float32)

        # Check for termination.
        if len(terminal_steps) != 0:
            # The agent is done
            self.reset_next_step = True
            for index, reward in enumerate(terminal_steps.reward):
                rewards[terminal_steps.agent_id[index]] = reward

            return dm_env.termination(reward=rewards, observation=self.observation())
        else:
            rewards = decision_steps.reward
            for index, reward in enumerate(decision_steps.reward):
                rewards[decision_steps.agent_id[index]] = reward

            return dm_env.transition(reward=rewards, observation=self.observation())

    def observation_spec(self) -> Dict[str, list]:
        """Returns the observation spec.

        Returns:
            observation spec (specs.BoundedArray): Returns the observation spec in the form of a `specs.BoundedArray`.
        """
        """
        observation_spec = specs.BoundedArray(
            shape=(self.n_agents, self.group_spec.observation_specs[0].shape[0]),
            dtype=self.observation_space.dtype,
            minimum=-np.inf,
            maximum=np.inf,
        )
        """

        return self.observation_space.obs_spec

    def action_spec(self) -> HivexDmEnvMultiAgentActionSpec:
        """Returns the action spec.

        Returns:
            action spec (specs.BoundedArray): Returns the action spec in the form of a `specs.BoundedArray`.
        """

        return self._action_space

    def observation(self) -> np.array:
        """Returns the observation of the current state of the environment.

        Returns:
            observations (np.array): Returns observations of the current state of the environment in the form of a `np.array`.
        """

        # decision and terminal (if existing) steps for current state of environment
        decision_step, terminal_step = self.env.get_steps(behavior_name=self.name)

        if len(decision_step) > 0:
            update_observations(
                info=decision_step,
                multi_agent_observations=self.multi_agent_observations,
                n_agents=self.n_agents,
            )
        if len(terminal_step) > 0:
            update_observations(
                info=terminal_step,
                multi_agent_observations=self.multi_agent_observations,
                n_agents=self.n_agents,
            )

        return self.multi_agent_observations

    def reward_spec(self) -> specs.Array:
        """Returns the reward spec returned by the environment.

        Returns:
            reward spec (specs.Array): Returns the reward spec in the form of a `specs.Array`.
        """

        reward_spec = specs.Array(
            shape=(self.n_agents,),
            dtype=np.float32,
            name="reward",
        )

        return reward_spec
