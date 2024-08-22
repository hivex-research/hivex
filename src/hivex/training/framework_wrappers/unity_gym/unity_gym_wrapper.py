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
# Based on the ml-agents gym-unity wrapper:
# https://github.com/Unity-Technologies/ml-agents/blob/main/gym-unity/gym_unity/envs/__init__.py
"""HIVEX UnityEnvironment to gym wrapper."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from gym import error, Env, spaces, Space

# ML-Agents
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps
from mlagents_envs import logging_util

# hivex
from hivex.training.framework_wrappers.wrapper_utils import (
    UnityEnvironment,
    get_vis_obs_list,
    get_vector_obs,
    update_observations,
)


logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)
"""Logger."""

GymStepResult = Tuple[np.ndarray, float, bool, Dict]
"""Gym step result shape."""


class HivexUnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """

    pass


class HivexGymWrapper(Env):
    """UnityEnvironment to gym wrapper class."""

    def __init__(self, unity_env: UnityEnvironment):
        """Construct a HivexGymWrapper environment.

        Attributes:
            unity_env (UnityEnvironment): Unity environment.
        """
        self.env = unity_env
        """Unity environment."""

        self.env.step()
        """Step the environment once to get specs."""

        self.name = list(self.env.behavior_specs.keys())[0]
        """Behavior name."""

        self.group_spec = self.env.behavior_specs[self.name]
        """Environment specs."""

        # Check for number of agents in scene.
        decision_steps, _ = self.env.get_steps(self.name)
        """Next non-terminal step(s)."""

        self.n_agents = len(decision_steps)
        """Number of agent(s)."""

        # Continous 0 - Discrete 1
        self._action_space = np.ndarray(shape=(self.n_agents, 2), dtype=object)

        # --- Set action spaces ---
        # Set continous action space
        if self.action_is_continuous():
            high = np.array([1] * self.group_spec.action_spec.continuous_size)
            continous_action_space = spaces.Box(-high, high, dtype=np.float32)
            for agent in range(self.n_agents):
                self._action_space[agent][0] = continous_action_space
            """Continous actions."""

        # Set discrete action space
        if self.action_is_discrete():
            branches = self.group_spec.action_spec.discrete_branches
            """Discrete action branches."""
            discrete_action_space = spaces.MultiDiscrete(
                [branch for branch in branches]
            )

            for agent in range(self.n_agents):
                self._action_space[agent][1] = discrete_action_space
            """Discrete actions."""

        # --- Set observations space ---
        # get all visual observations
        visual_obs = None
        if self.get_vis_obs_count() > 0:
            visual_obs: List[Space] = []
            shapes = self.get_vis_obs_shape()
            for shape in shapes:
                visual_obs.append(
                    spaces.Box(low=0, high=1, dtype=np.float32, shape=shape)
                )
        """Visual observations."""

        if self.get_vec_obs_size() > 0:
            # for vector observation
            high = np.array([np.inf] * self.get_vec_obs_size())
            vector_obs = spaces.Box(-high, high, dtype=np.float32)
        """Vector observations."""

        # Visual Observations 0 - Vector Observations 1
        self._observation_space = np.ndarray(shape=(self.n_agents, 2), dtype=object)
        for agent in range(self.n_agents):
            # visual observation
            self._observation_space[agent][0] = visual_obs
            # vector observation
            self._observation_space[agent][1] = vector_obs

        # dynamic agent fields
        self.multi_agent_observations = [None] * self.n_agents
        self.multi_agent_rewards = np.zeros(self.n_agents, dtype=np.float32)

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (np.ndarray/list(np.ndarray)): The initial observation of the space.
        """
        # Reset unity environment
        self.env.reset()
        # Non-terminal step(s).
        decision_step, _ = self.env.get_steps(self.name)
        # Step environment
        res: GymStepResult = self.single_step(decision_step)
        return res[0]

    def step(self, actions: np.ndarray) -> GymStepResult:
        """Run one timestep of the environment's dynamics. Accepts an action and returns a tuple (observation, reward, done, info).

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (list(object)): An action provided by the environment.
        Returns:
            observation (object/list(object)): Agent's observation of the current environment.
            reward (float/list(float)) : Amount of reward returned after previous action.
            done (bool/list(bool)): Whether the episode has ended.
            info (dict): Contains auxiliary diagnostic information.
        """

        # Construct an `ActionTuple` as this is the action format for the `UnityEnvironment` environment.
        # Dimensions are of (n_agents, continuous_size) and (n_agents, discrete_size)
        action_tuple = ActionTuple()

        # actions[agent][0] continous
        # actions[agent][1] discrete
        if self.action_is_continuous():
            continous_action = np.array([action[0] for action in actions])
            action_tuple.add_continuous(continous_action)
        if self.action_is_discrete():
            discrete_action = np.array([action[1] for action in actions])
            action_tuple.add_discrete(discrete_action)

        # Set action(s)
        if len(self.env._env_state[self.name][0]) != 0:
            self.env.set_actions(self.name, action_tuple)

        # Step environment with new action(s)
        self.env.step()
        # Return new state(s)
        decision_steps, terminal_steps = self.env.get_steps(self.name)

        self.multi_agent_rewards = np.zeros(self.n_agents, dtype=np.float32)

        # Check for termination.
        if len(terminal_steps) > 0:
            # The agent is done
            self.game_over = True
            for index, reward in enumerate(terminal_steps.reward):
                self.multi_agent_rewards[terminal_steps.agent_id[index]] = reward
            return self.single_step(terminal_steps)
        if len(decision_steps) > 0:
            for index, reward in enumerate(decision_steps.reward):
                self.multi_agent_rewards[decision_steps.agent_id[index]] = reward
            return self.single_step(decision_steps)

    def single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        """Performs a single step.

        Accepts an info and returns a tuple (observation, reward, done, info).

        Args:
            info (list(object)): Current step info tuple provided by the environment.
        Returns:
            A `GymStepResult` namedtuple containing:
            observation (Tuple(agent, observation)): Agent's observation of the current environment,
                where observation consists of a list of Tuple([visual observations], vector observations).
            reward (float/list(float)) : Amount of reward returned after previous action.
            done (bool/list(bool)): Whether the episode has ended.
            info (dict): Contains auxiliary diagnostic information.
        """

        # check if environment is done
        dones = [False] * self.n_agents
        if isinstance(info, TerminalSteps):
            for agent in info:
                dones[agent] = True

        decision_steps, terminal_steps = self.env.get_steps(behavior_name=self.name)

        if len(decision_steps) > 0:
            update_observations(
                info=decision_steps,
                multi_agent_observations=self.multi_agent_observations,
                n_agents=self.n_agents,
            )
        if len(terminal_steps) > 0:
            update_observations(
                info=terminal_steps,
                multi_agent_observations=self.multi_agent_observations,
                n_agents=self.n_agents,
            )

        return (
            self.multi_agent_observations,
            self.multi_agent_rewards,
            dones,
            {"step": info},
        )

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

    def get_vis_obs_count(self) -> int:
        """Get the number of visual observations.

        Returns:
            visual observation count (int): Visual observation count value as `int`.
        """
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result += 1
        return result

    def get_vis_obs_shape(self) -> List[Tuple]:
        """Get the shape of the visual observation.

        Returns:
            visual observation shape (list(tuple)): Visual observation shape as a `List(Tuple)`.
        """
        result: List[Tuple] = []
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result.append(obs_spec.shape)
        return result

    def get_vis_obs_list(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        """Get visual observation list.

        Args:
            step_result (DecisionSteps/TerminalSteps): Decision or terminal step tuple (observation, reward, done, info).
        Returns:
            list of visual observations (list(np.ndarray)): List of visual observation in the form of `List(np.ndarray)`.
        """
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def get_vector_obs(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> np.ndarray:
        """Get vector observation.

        Args:
            step_result (DecisionSteps/TerminalSteps): Decision or terminal step tuple (observation, reward, done, info).
        Returns:
            vector observations (np.ndarray): Vector observation in the form of `np.ndarray`.
        """
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def get_vec_obs_size(self) -> int:
        """Get vector observation size.

        Returns:
            vector observations size (int): Vector observation size in the form of `int`.
        """
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                result += obs_spec.shape[0]
        return result

    def render(self, mode="rgb_array") -> List[np.ndarray]:
        """Latest visual observation.

        Note that it will not render a new frame of the environment.

        Args:
            mode (str): Rendering mode as a `str`.
        Returns:
            visual observation (list(np.ndarray)): The latest visual observations.
        """
        return self.visual_obs

    def close(self, timeout: Optional[int] = None) -> None:
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when garbage collected or when the program exits.
        """
        # Close the Unity environment
        if timeout is not None:
            self.env._close(timeout=timeout)
        else:
            self.env.close()

    def seed(self, seed: Any = None) -> None:
        # TODO: Implement seed.
        """Sets the seed for this env's random number generator(s). Currently not implemented.

        Args:
            seed (int): seed for the environments random.
        """
        logger.warning("Could not seed environment %s", self.name)
        return

    def action_is_continuous(self):
        """Check if environment includes continous actions."""
        return self.group_spec.action_spec.continuous_size > 0

    def action_is_discrete(self):
        """Check if environment includes discrete actions."""
        return self.group_spec.action_spec.discrete_size > 0

    def generate_rnd_actions(self) -> np.array:
        actions = np.array(
            [
                [
                    (action_space[0].sample() if action_space[0] is not None else None),
                    (action_space[1].sample() if action_space[1] is not None else None),
                ]
                for action_space in self.action_space
            ]
        )
        return actions

    @property
    def metadata(self):
        """Returns render mode.

        Returns:
            render mode (dict): Render mode `rgb_array`.
        """
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        """Returns reward range.

        Returns:
            reward range (Tuple(float, float)): Reward range `Tuple(float, float)`.
        """
        return -float("inf"), float("inf")

    @property
    def action_space(self) -> Space:
        """Returns action space.

        Returns:
            action space (gym.Space): Action space `gym.Space`.
        """
        return self._action_space

    @property
    def observation_space(self):
        """Returns observation space.

        Returns:
            observation space (gym.Space): Observation space `list(tuple)` or `list(Space)`.
        """
        return self._observation_space


def check_unity_gym_environment(env: HivexGymWrapper) -> None:
    """Check UnityEnvironment to gym wrapper is working.

    Args:
        env (HivexGymWrapper): Hivex UnityEnvironment to gym wrapper `HivexGymWrapper`.
    """
    # Get observation from reset
    obs = env.reset()
    # Check observation shape
    assert isinstance(obs, np.ndarray)
    print(f"Name of the behavior : {env.name}")

    # Generate a random action sample
    actions = np.array(
        [
            [
                (action_space[0].sample() if action_space[0] is not None else None),
                (action_space[1].sample() if action_space[1] is not None else None),
            ]
            for action_space in env.action_space
        ]
    )

    # actions = np.array([env.action_space[i].sample() for i in range(env.n_agents)])
    # Use generated action and step environment
    obs, rew, done, info = env.step(actions)

    # Check observation shape
    assert isinstance(obs, np.ndarray)
    print("Number of observations : ", len(obs))

    # Check reward shape
    assert isinstance(rew, np.ndarray)
    # Check done shape
    assert isinstance(done, (bool, np.bool_))
    # Check info shape
    assert isinstance(info, dict)

    # TODO: visual observations?
    print(f"There are {len(actions)} actions")

    print(
        f"Check complete. Environment sample specs: actions: {actions};  observations: {obs}; rewards: {rew}, done: {done}"
    )
