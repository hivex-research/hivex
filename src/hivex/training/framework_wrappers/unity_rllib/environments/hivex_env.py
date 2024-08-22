import numpy as np
from typing import Optional, Tuple, List
from abc import ABC

from hivex.training.framework_wrappers.unity_rllib.environments.hivex_base_env import (
    HivexBaseEnv,
)

# RLlib
from ray.rllib.utils.typing import MultiAgentDict

# ML-Agents
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.base_env import ActionTuple


class HivexEnv(HivexBaseEnv, ABC):
    def __init__(
        self,
        run_config,
        file_name: str = None,
        port: Optional[int] = None,
        seed: int = 0,
        timeout_wait: int = 300,
        episode_horizon: int = 3000,
        stop_time_steps: int = 2000000,
        side_channel: List[SideChannel] = None,
        no_graphics: bool = None,
    ):
        super().__init__(
            run_config,
            file_name,
            port,
            seed,
            timeout_wait,
            episode_horizon,
            stop_time_steps,
            side_channel,
            no_graphics,
        )

    # override
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

        obs, _, _, _, _ = self._get_step_results()

        infos = {}
        actions = []
        for agent_id in self.all_agent_keys:
            infos[agent_id] = {"NN": action_dict[agent_id]}
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
