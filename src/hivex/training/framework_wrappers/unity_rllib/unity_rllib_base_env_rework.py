from gymnasium.spaces import Box, MultiDiscrete, Tuple as TupleSpace
from typing import Callable, Tuple, List, Optional
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.rllib.policy.policy import PolicySpec
import numpy as np
from mlagents_envs.side_channel.side_channel import SideChannel
import random
import time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID


class HivexBaseEnv(Unity3DEnv):
    def __init__(
        self,
        file_name: str = None,
        port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = True,
        timeout_wait: int = 300,
        episode_horizon: int = 1000,
        side_channel: List[SideChannel] = None,
    ):
        """Initializes a Unity3DEnv object.

        Args:
            file_name (Optional[str]): Name of the Unity game binary.
                If None, will assume a locally running Unity3D editor
                to be used, instead.
            port (Optional[int]): Port number to connect to Unity environment.
            seed: A random seed value to use for the Unity3D game.
            no_graphics: Whether to run the Unity3D simulator in
                no-graphics mode. Default: False.
            timeout_wait: Time (in seconds) to wait for connection from
                the Unity3D instance.
            episode_horizon: A hard horizon to abide to. After at most
                this many steps (per-agent episode `step()` calls), the
                Unity3D game is reset and will start again (finishing the
                multi-agent episode that the game represents).
                Note: The game itself may contain its own episode length
                limits, which are always obeyed (on top of this value here).
        """
        # Skip env checking as the nature of the agent IDs depends on the game
        # running in the connected Unity editor.
        self._skip_env_checking = True

        self.stats_channel = side_channel[1]

        MultiAgentEnv.__init__(self)

        if file_name is None:
            print(
                "No game binary provided, will use a running Unity editor "
                "instead.\nMake sure you are pressing the Play (|>) button in "
                "your editor to start."
            )

        import mlagents_envs
        from mlagents_envs.environment import UnityEnvironment

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
        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0

    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Resets the entire Unity3D scene (a single multi-agent episode)."""
        self.episode_timesteps = 0
        # self.unity_env.reset()
        obs, _, _, _, infos = self._get_step_results()
        return obs, infos

    @staticmethod
    def get_policy_configs_for_game(
        game_name: str,
    ) -> Tuple[dict, Callable[[AgentID], PolicyID]]:

        # The RLlib server must know about the Spaces that the Client will be
        # using inside Unity3D, up-front.
        obs_spaces = {
            "WindFarmControl": Box(float("-inf"), float("inf"), (6,)),
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
