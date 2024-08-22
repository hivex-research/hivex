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
"""HIVEX UnityEnvironment to baselines3 VecEnv wrapper"""

import numpy as np
from typing import Any, Tuple, Dict, Optional, List, Union, Iterable, Type, Sequence
import random
import logging

# from gym import error, spaces, Wrapper
from gymnasium import error, spaces, Wrapper
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps
from stable_baselines3.common.vec_env import VecEnv

from hivex.training.framework_wrappers.wrapper_utils import (
    UnityEnvironment,
    action_is_continuous,
    action_is_discrete,
    get_vis_obs_count,
    get_vec_obs_size,
    update_observations,
    get_vis_obs_shape,
)


# Define type aliases here to avoid circular import
# Used when we want to access one or more VecEnv
VecEnvIndices = Union[None, int, Iterable[int]]
# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = Tuple[np.ndarray, np.array, np.array, Dict]
"""VecEnv step result shape"""


class HivexException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents
    """

    pass


class HivexVecEnvWrapper(VecEnv):
    """UnityEnvironment to baselines3 VecEnv wrapper class"""

    def __init__(self, unity_env: UnityEnvironment, train_mode: bool = True):
        """Construct a HivexVecEnvWrapper environment

        Attributes:
            unity_env (UnityEnvironment): Unity environment
        """

        self.train_mode = train_mode

        self.env = unity_env
        """Unity environment"""

        self.env.step()
        """Step the environment once to get specs"""

        # Get name of behavior and environment specs
        self.name = list(self.env.behavior_specs.keys())[0]
        """Behavior name"""

        self.group_spec = self.env.behavior_specs[self.name]
        """Environment specs"""

        # Check for number of agents in scene.
        decision_steps, _ = self.env.get_steps(self.name)
        """Next non-terminal step(s)"""

        self.num_agents = len(decision_steps)
        """Number of agent(s)"""

        # --- Set action space of a single Agent ---
        # Set continous action space
        continous_action_range_min = []
        continous_action_range_max = []
        self.continous_action_count = 0
        if action_is_continuous(group_spec=self.group_spec):
            self.continous_action_count = self.group_spec.action_spec.continuous_size
            continous_action_range_min = [-1] * self.continous_action_count
            continous_action_range_max = [1] * self.continous_action_count
            """Continous actions"""

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
            """Discrete actions"""

        self.action_space = []
        self.action_space = spaces.Box(
            low=np.array(continous_action_range_min + discrete_action_range_min),
            high=np.array(continous_action_range_max + discrete_action_range_max),
        )

        # --- Set observations space of a single Agent ---
        # get all visual observations
        visual_obs_range_min = []
        visual_obs_range_max = []
        if get_vis_obs_count(group_spec=self.group_spec) > 0:
            shapes = get_vis_obs_shape(group_spec=self.group_spec)
            for shape in shapes:
                visual_obs_range_min = [0] * shape[0] * shape[1]
                visual_obs_range_max = [1] * shape[0] * shape[1]
                pass
        """Visual observations"""

        vec_obs_range_min = []
        vec_obs_range_max = []
        if get_vec_obs_size(group_spec=self.group_spec) > 0:
            # for vector observation
            vec_obs_range_min = [np.inf] * get_vec_obs_size(group_spec=self.group_spec)
            vec_obs_range_max = [-np.inf] * get_vec_obs_size(group_spec=self.group_spec)
        """Vector observations"""

        self.observation_space = spaces.Box(
            low=np.array(visual_obs_range_min + vec_obs_range_min),
            high=np.array(visual_obs_range_max + vec_obs_range_max),
            dtype=np.float32,
        )

        super().__init__(self.num_agents, self.observation_space, self.action_space)

        self.episode_steps = 0
        """Episode step count"""

        self.episode = 0
        """Episode count"""

        self.episode_rewards = np.zeros(self.num_agents)
        """Episode rewards"""

        self.actions = None
        """Actions"""

        self.multi_agent_observations = [None] * self.num_agents
        """Observations"""

        self.multi_agent_rewards = np.zeros(self.num_agents)
        """Rewards"""

    def reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        Returns:
            A `VecEnvObs` namedtuple containing:
            observation (VecEnvObs): Observations of all agent(s) in the form of `np.ndarray`,
                `Dict[str, np.ndarray]` or `Tuple[np.ndarray, ...]`.
        """

        self.episode_steps = 0
        observations, rewards, dones, info = self.parse_brain_info()

        if all(dones):
            self.env.reset()

        return self.parse_brain_info()[0]

    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.

        Args:
            actions (np.ndarray): Actions.
        """

        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        Returns:
            A `VecEnvStepReturn` namedtuple containing:
            observation (VecEnvObs): Observations of all agent(s).
            reward (np.ndarray): Rewards of all agent(s).
            done (np.ndarray): Wheter agent(s) are done or not in the form of `np.ndarray` of `bool`.
            information (List[Dict]): Furhter information for each agent.
        """

        # Construct an `ActionTuple` as this is the action format for the `UnityEnvironment` environment.
        # Dimensions are of (num_agents, continuous_size) and (num_agents, discrete_size)
        action_tuple = ActionTuple()

        if action_is_continuous(group_spec=self.group_spec):
            continous_actions = np.array(
                [action[: self.continous_action_count] for action in self.actions]
            )
            action_tuple.add_continuous(continous_actions)
        if action_is_discrete(group_spec=self.group_spec):
            discrete_action = np.array(
                [action[self.continous_action_count :] for action in self.actions]
            )
            action_tuple.add_discrete(discrete_action)

        # Set action(s)
        if len(self.env._env_state[self.name][0]) != 0:
            self.env.set_actions(self.name, action_tuple)

        # Step environment
        self.env.step()
        # Return new state(s)
        observations, rewards, dones, info = self.parse_brain_info()

        # print(self.episode_rewards)

        self.episode_rewards += rewards
        self.episode_steps += 1

        if any(dones):
            # print(self.episode_rewards)

            # At least one agent is done
            self.episode += 1
            self.reset()
            observations, rewards, dones, info = self.parse_brain_info()

        else:
            info = [dict() for _ in dones]

        return observations, rewards, dones, info

    def parse_brain_info(self):
        """
        Perform a single step for all agent(s).

        Args:
            info (int): The random seed. May be None for completely random seeding.
            terminal_steps (List[bool]): Bool list whether terminal step is reached.
        """

        decision_steps, terminal_steps = self.env.get_steps(self.name)

        # check if environment is done
        dones = [False] * self.num_agents
        for agent_id in terminal_steps.agent_id:
            dones[agent_id] = True

        if len(decision_steps) > 0:
            self.multi_agent_observations = update_observations(
                info=decision_steps,
                multi_agent_observations=self.multi_agent_observations,
                n_agents=self.num_agents,
            )
        if len(terminal_steps) > 0:
            self.multi_agent_observations = update_observations(
                info=terminal_steps,
                multi_agent_observations=self.multi_agent_observations,
                n_agents=self.num_agents,
            )

        if len(decision_steps.reward) > 0:
            for i, agent_id in enumerate(decision_steps.agent_id):
                self.multi_agent_rewards[agent_id] = decision_steps.reward[i]

        if len(terminal_steps.reward) > 0:
            for i, agent_id in enumerate(terminal_steps.agent_id):
                self.multi_agent_rewards[agent_id] = terminal_steps.reward[i]

        if any(dones):
            info_dict = [
                dict(episode=dict(r=self.episode_rewards[i], l=self.episode_steps))
                for i in range(self.num_agents)
            ]
        else:
            info_dict = [dict()] * self.num_agents

        dones = np.array(dones)
        multi_agent_observations = np.array(self.multi_agent_observations)
        multi_agent_observations = np.squeeze(multi_agent_observations)

        return (
            multi_agent_observations,
            self.multi_agent_rewards,
            dones,
            info_dict,
        )

    def close(self) -> None:
        """Closes the environment."""
        self.env.close()
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """
        Return attribute from vectorized environment.

        Args:
            attr_name (str): The name of the attribute whose value to return.
            indices (int): Indices of envs to get attribute from.
        Returns:
            attributes (List[Any]): List of values of 'attr_name' in all environments.
        """
        pass

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """
        Set attribute inside vectorized environments.

        Args:
            attr_name (str): The name of attribute to assign new value.
            value (Any): Value to assign to `attr_name`.
            indices (VecEnvIndices): Indices of envs to assign value.
        """
        pass

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        """
        Call instance methods of vectorized environments.

        Args:
            method_name (str): The name of the environment method to invoke.
            method_args: Any positional arguments to provide in the call.
            indices (VecEnvIndices): Indices of envs whose method to call.
            method_kwargs: Any keyword arguments to provide in the call.
        Returns:
            item list (List[Any]): List of items returned by the environment's method call.
        """
        pass

    def env_is_wrapped(
        self, wrapper_class: Type[Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper.

        Args:
            wrapper_class (Type[gym.Wrapper]): Wrapper class.
            indices (VecEnvIndices): Indices of envs whose method to call.
        Returns:
            bool list (List[bool]): True if the env is wrapped, False otherwise, for each env queried.
        """
        return [True] * self.num_agents

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        Note that all list elements may be None, if the env does not return anything when being seeded.

        Args:
            seed (int): The random seed. May be None for completely random seeding.
        Returns:
            seed list (List[int]): Returns a list containing the seeds for each individual env.
        """
        pass

    def preprocess_single_vis_obs(self, single_visual_obs: np.ndarray) -> np.ndarray:
        """Preprocess single visual observation.

        Args:
            single_visual_obs (np.ndarray): Single visual observation in the form of `np.ndarray`.
        Returns:
            single_visual_obs (np.ndarray): Processed single visual observation in the form of `np.ndarray`.
        """
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

    def generate_rnd_actions(self) -> np.array:
        """Generate random actions for testing."""
        self.action_space.sample()
        actions = np.stack([self.action_space.sample() for _ in range(self.num_agents)])
        return actions
