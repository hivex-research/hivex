import numpy as np
import random
import time
from typing import Callable, Optional, Tuple, List
import random

# RLlib
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple as TupleSpace
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv

# ML-Agents
from mlagents_envs.side_channel.side_channel import SideChannel
import mlagents_envs
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


class HivexBaseEnv(Unity3DEnv):
    def __init__(
        self,
        file_name: str = None,
        port: Optional[int] = None,
        seed: int = 0,
        timeout_wait: int = 300,
        episode_horizon: int = 3000,
        stop_time_steps: int = 2000000,
        side_channel: List[SideChannel] = None,
        no_graphics: bool = True,
    ):
        self.stats_channel = side_channel[1]

        # Skip env checking as the nature of the agent IDs depends on the game
        # running in the connected Unity editor.
        self._skip_env_checking = True

        MultiAgentEnv.__init__(self)

        if file_name is None:
            print(
                "No game binary provided, will use a running Unity editor "
                "instead.\nMake sure you are pressing the Play (|>) button in "
                "your editor to start."
            )

        # Try connecting to the Unity3D game instance. If a port is blocked
        port_ = None
        while True:
            # Sleep for random time to allow for concurrent startup of many
            # environments (num_workers >> 1). Otherwise, would lead to port
            # conflicts sometimes.
            if port_ is not None:
                time.sleep(random.randint(1, 10))
            port_ = port or (
                self._BASE_PORT_ENVIRONMENT if file_name else self._BASE_PORT_EDITOR
            )
            # cache the worker_id and
            # increase it for the next environment
            worker_id_ = Unity3DEnv._WORKER_ID if file_name else 0
            Unity3DEnv._WORKER_ID += 1
            try:
                self.unity_env = UnityEnvironment(
                    file_name=file_name,
                    worker_id=worker_id_,
                    base_port=port_,
                    seed=seed,
                    no_graphics=no_graphics,
                    timeout_wait=timeout_wait,
                    side_channels=side_channel,
                )

                print("Created UnityEnvironment for port {}".format(port_ + worker_id_))
            except mlagents_envs.exception.UnityWorkerInUseException:
                pass
            else:
                break

        # ML-Agents API version.
        self.api_version = self.unity_env.API_VERSION.split(".")
        self.api_version = [int(s) for s in self.api_version]

        # Reset entire env every this number of step calls.
        self.episode_horizon = episode_horizon
        self.stop_time_steps = stop_time_steps
        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0
        self.total_time_steps = 0

        # self agent tracker: is agent following LLM instructions or receives actions from NN
        # self.unity_env.reset()
        self.behavior_name = list(self.unity_env.behavior_specs)[0]
        agent_ids = list(self.unity_env.get_steps(self.behavior_name)[0].keys())
        self.all_agent_keys = [
            f"{self.behavior_name}_{agent_id}" for agent_id in agent_ids
        ]

        print(f"AGENT COUNT: {len(self.all_agent_keys)}")
        print("###################################################################")
        print("####################### ENVIRONMENT CREATED #######################")
        print("###################################################################")

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Performs one multi-agent step through the game.

        Args:
            action_dict: Multi-agent action dict with:
                keys=agent identifier consisting of
                [MLagents behavior name, e.g. "Goalie?team=1"] + "_" +
                [Agent index, a unique MLAgent-assigned index per single agent]

        Returns:
            tuple:
                - obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                - rewards: Rewards dict matching `obs`.
                - dones: Done dict with only an __all__ multi-agent entry in
                    it. __all__=True, if episode is done for all agents.
                - infos: An (empty) info dict.
        """

        # print(f"--- STEP {self.episode_timesteps} ---")

        obs, _, _, _, _ = self._get_step_results()

        infos = {}

        # Set only the required actions (from the DecisionSteps) in Unity3D.
        # brain name: Agent

        actions = []
        for agent_id in self.all_agent_keys:
            # agent_id = Agent?team=0_0

            if agent_id not in action_dict:
                continue

            actions.append(action_dict[agent_id])

        if len(actions) > 0:
            if isinstance(actions[0], tuple):
                action_tuple = ActionTuple(
                    continuous=np.array([c[0] for c in actions]),
                    discrete=np.array([c[1] for c in actions]),
                )
            else:
                if actions[0].dtype == np.float32:
                    action_tuple = ActionTuple(continuous=np.array(actions))
                else:
                    action_tuple = ActionTuple(discrete=np.array(actions))
            self.unity_env.set_actions(self.behavior_name, action_tuple)

        # Do the step.
        self.unity_env.step()
        obs, rewards, terminateds, truncateds, _ = self._get_step_results()
        self.episode_timesteps += 1
        self.total_time_steps += 1

        return obs, rewards, terminateds, truncateds, infos

    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        print("### RESET ###")

        """Resets the entire Unity3D scene (a single multi-agent episode)."""
        self.episode_timesteps = 0
        self.unity_env.reset()
        obs, _, _, _, infos = self._get_step_results()
        return obs, infos

    # override
    def _get_step_results(self):
        """Collects those agents' obs/rewards that have to act in next `step`.

        Returns:
            Tuple:
                obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                rewards: Rewards dict matching `obs`.
                dones: Done dict with only an __all__ multi-agent entry in it.
                    __all__=True, if episode is done for all agents.
                infos: An (empty) info dict.
        """
        obs = {}
        rewards = {}
        infos = {}

        # for behavior_name in self.unity_env.behavior_specs:
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        # ({'__all__': True, 'Agent?team=0_0': True, 'Agent?team=0_1': True, 'Agent?team=0_2': True},)
        # dones = dict({"__all__": False})
        for agent_id, idx in decision_steps.agent_id_to_index.items():
            key = self.behavior_name + "_{}".format(agent_id)
            # dones[key] = False
            os = tuple(o[idx] for o in decision_steps.obs)
            os = os[0] if len(os) == 1 else os
            obs[key] = os
            rewards[key] = decision_steps.reward[idx] + decision_steps.group_reward[idx]

        for agent_id, idx in terminal_steps.agent_id_to_index.items():
            key = self.behavior_name + "_{}".format(agent_id)
            # dones[key] = True
            # Only overwrite rewards (last reward in episode), b/c obs
            # here is the last obs (which doesn't matter anyways).
            # Unless key does not exist in obs.
            if key not in obs:
                os = tuple(o[idx] for o in terminal_steps.obs)
                os = os[0] if len(os) == 1 else os
                obs[key] = os
            rewards[key] = terminal_steps.reward[idx] + terminal_steps.group_reward[idx]

        if len(terminal_steps.interrupted) > 0:
            dones = dict(
                {"__all__": True},
                **{agent_id: True for agent_id in self.all_agent_keys},
            )
            print(
                f"total steps taken: {self.total_time_steps} - last episode total steps taken: {self.episode_timesteps} - {len(terminal_steps.interrupted)}/{len(self.all_agent_keys)} agents done"
            )
        else:
            dones = dict(
                {"__all__": False},
                **{agent_id: False for agent_id in self.all_agent_keys},
            )

        # Only use dones if all agents are done, then we should do a reset.
        return obs, rewards, dones, dones, infos

    # override
    @staticmethod
    def get_policy_configs_for_game(
        game_name: str,
    ) -> Tuple[dict, Callable[[AgentID], PolicyID]]:
        # The RLlib server must know about the Spaces that the Client will be
        # using inside Unity3D, up-front.
        obs_spaces = {
            "WindFarmControl": Box(float("-inf"), float("inf"), (6,), dtype=np.float32),
            "WildfireResourceManagement": Box(
                float("-inf"), float("inf"), (7,), dtype=np.float32
            ),
            "DroneBasedReforestation": TupleSpace(
                [
                    Box(float(0), float(1), (16, 16, 1)),
                    Box(float("-inf"), float("inf"), (20,), dtype=np.float32),
                ]
            ),
            "OceanPlasticCollection": TupleSpace(
                [
                    Box(float(0), float(1), (25, 25, 2)),
                    Box(float("-inf"), float("inf"), (8,), dtype=np.float32),
                ]
            ),
            "AerialWildfireSuppression": TupleSpace(
                [
                    Box(float(0), float(1), (42, 42, 3)),
                    Box(float("-inf"), float("inf"), (8,), dtype=np.float32),
                ]
            ),
        }
        action_spaces = {
            "WindFarmControl": MultiDiscrete([3]),
            "WildfireResourceManagement": MultiDiscrete([3, 3, 3, 3]),
            "DroneBasedReforestation": TupleSpace(
                [
                    Box(float(-1), float(1), (3,), dtype=np.float32),
                    MultiDiscrete([2]),
                ]
            ),
            "OceanPlasticCollection": MultiDiscrete([2, 3]),
            "AerialWildfireSuppression": TupleSpace(
                [
                    Box(float(-1), float(1), (1,), dtype=np.float32),
                    MultiDiscrete([2]),
                ]
            ),
        }

        policies = {
            game_name: PolicySpec(
                observation_space=obs_spaces[game_name],
                action_space=action_spaces[game_name],
            ),
        }

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return game_name

        return policies, policy_mapping_fn

    def remap(value, from1, to1, from2, to2):
        return (value - from1) / (to1 - from1) * (to2 - from2) + from2
