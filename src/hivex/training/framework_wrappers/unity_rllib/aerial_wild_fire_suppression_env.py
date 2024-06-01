import numpy as np
from typing import Callable, Optional, Tuple, List
import random

from environments.hivex_base_env import HivexBaseEnv

from LLM_interpreters.aerial_wildfire_suppression_interpreters import (
    AerialWildfireSuppressionAgentInterpreter,
)

# RLlib
from ray.rllib.utils.typing import MultiAgentDict

# ML-Agents
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.base_env import ActionTuple


class AerialWildFireSuppressionEnv(HivexBaseEnv):
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
        NN_only: bool = None,
        agent_interpreter_object: object = AerialWildfireSuppressionAgentInterpreter,
        task_decay: int = 500,
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
            NN_only,
            agent_interpreter_object,
            task_decay,
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

        # print(f"--- STEP {self.episode_timesteps} ---")

        obs, _, _, _, _ = self._get_step_results()

        infos = {}

        # check keys_to_exclude_from_LLM
        keys_to_exclude_from_LLM = [
            agent_id
            for agent_id, values in self.agent_policy_state.items()
            if values["following"] == "LLM"
        ]

        # build task list if not NN only activated
        if self.NN_only == False:
            tasks = {}
            if self.human_intervention_channel.command != "":
                print(
                    f"received human command: {self.human_intervention_channel.command}"
                )
                response = (
                    self.agent_interpreter.Human_intervention_to_task_interpreter(
                        self.human_intervention_channel.get_and_reset_commands()
                    )
                )
                print(f"response: {response}")
                tasks = self.agent_interpreter.parse_LLM_tasks(
                    response, human_task=True
                )
                print(f"tasks: {tasks}")
            else:
                if len(keys_to_exclude_from_LLM) < len(self.agent_policy_state):
                    # LLM INTERPRETER
                    self.agent_interpreter.interprete_observations(obs)

                    # Found fire?
                    if len(self.agent_interpreter.all_agents_last_seen_fire) > 0:
                        print(f"--- STEP {self.episode_timesteps} ---")
                        print(
                            f"Fire detected: {self.agent_interpreter.all_agents_last_seen_fire}"
                        )
                        self.agent_interpreter.observations_to_text(
                            keys_to_exclude_from_LLM
                        )
                        if self.agent_interpreter.all_agents_fire_info != "":
                            response = (
                                self.agent_interpreter.LLM_info_to_task_interpreter()
                            )
                            print(f"response: {response}")
                            tasks = self.agent_interpreter.parse_LLM_tasks(response)
                            print(f"tasks: {tasks}")
                    # else:
                    #     print("No fire detected")

            self.task_count += len(tasks)
            self.total_task_count += len(tasks)

            for agent_id, task in tasks.items():
                if agent_id in self.all_agent_keys:
                    self.agent_policy_state[agent_id]["following"] = "LLM"
                    # go to tasked location
                    self.agent_policy_state[agent_id]["tasks"].append(task)
                    print(
                        f"{agent_id} holding water: {self.agent_interpreter.all_agents_holding_water[agent_id]}"
                    )
                    if self.agent_interpreter.all_agents_holding_water[agent_id]:
                        # get water
                        self.agent_policy_state[agent_id]["tasks"].append("water")
                        # and go back to tasked location
                        self.agent_policy_state[agent_id]["tasks"].append(task)

        # Set only the required actions (from the DecisionSteps) in Unity3D.
        # brain name: Agent

        actions = []
        for agent_id in self.all_agent_keys:
            # agent_id = Agent?team=0_0
            # action_dict[agent_id] = (array([0.21004367], dtype=float32), array([0], dtype=int64))

            if agent_id not in action_dict:
                continue

            if (
                self.agent_policy_state[agent_id]["following"] == "LLM"
                and len(self.agent_policy_state[agent_id]["tasks"]) > 0
                and self.get_LLM_task_threshold > random.uniform(0, 1)
            ):
                # print(
                #     f"Agent: {agent_id} is following LLM. Previews action was: {action_dict[agent_id]}"
                # )

                # TASK: PICK WATER
                if self.agent_policy_state[agent_id]["tasks"][0] == "water":
                    # check if task is completed
                    picked_water = (
                        self.agent_interpreter.all_agents_holding_water[agent_id] == 1
                    )
                    if picked_water == False:
                        action_to_pick_up_water_from_closest_source = (
                            self.get_action_to_pick_up_water_from_closest_source(
                                obs, agent_id
                            )
                        )

                        action_dict[agent_id] = (
                            np.array(
                                [action_to_pick_up_water_from_closest_source],
                                dtype=np.float32,
                            ),
                            tuple([0]),
                        )
                    else:
                        self.agent_policy_state[agent_id]["tasks"].pop(0)
                # TASK: GO TO LOCATION
                else:
                    got_to_location = self.agent_policy_state[agent_id]["tasks"][0]
                    go_to_x = got_to_location[0]
                    go_to_y = got_to_location[1]
                    target = [go_to_x, go_to_y]

                    action_to_go_to_target = self.get_action_to_go_to_target(
                        obs, agent_id, target
                    )
                    action_dict[agent_id] = (
                        np.array([action_to_go_to_target], dtype=np.float32),
                        tuple([0]),
                    )

                    # print(f"go to location - action: {action_dict[agent_id]}")
                    distance_to_target = self.distance_to_target(obs, agent_id, target)
                    if distance_to_target < 5:
                        print(
                            f"agent {agent_id} finished task {self.agent_policy_state[agent_id]['tasks'][0]}"
                        )
                        self.agent_policy_state[agent_id]["tasks"].pop(0)

                # if action_dict[agent_id][0][0] < 1.0 and action_dict[agent_id][0][0] > -1.0:
                #     print(
                #         f"Agent: {agent_id} is following LLM. LLM suggested action is: {action_dict[agent_id]}"
                #     )

                # print(
                #     f"Agent: {agent_id} is following LLM. LLM suggested action is: {action_dict[agent_id]}"
                # )
                infos[agent_id] = {"LLM": action_dict[agent_id]}
            else:
                #     print(
                #         f"Agent: {agent_id} using actions from NN: {action_dict[agent_id]}"
                #     )
                infos[agent_id] = {"NN": action_dict[agent_id]}

            # DECAY TASK
            self.agent_policy_state[agent_id]["task_decay"] -= 1
            # print(
            #     f"{agent_id} task decay is: {self.agent_policy_state[agent_id]['task_decay']}"
            # )
            if self.decay:
                self.get_LLM_task_threshold -= 1.0 / self.stop_time_steps
                # print(f"Current LLM decay: {self.get_LLM_task_threshold}")
            # print(f"{agent_id} is taking action: {action_dict[agent_id]}")
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

        # check agents on LLM dutys decay or terminated and reset
        for agent_id, values in self.agent_policy_state.items():
            if terminateds["__all__"]:
                print(f"all agents are done")
                pass
            elif values["task_decay"] < 0:
                print(f"task_decay is below 0")
                pass
            if terminateds["__all__"] or values["task_decay"] < 0:
                print(f"agent {agent_id} is done, resetting agent_policy_state")
                self.agent_policy_state[agent_id]["following"] = "NN"
                self.agent_policy_state[agent_id]["task_decay"] = self.task_decay
                self.agent_policy_state[agent_id]["tasks"] = []

        self.episode_timesteps += 1
        self.total_time_steps += 1

        return obs, rewards, terminateds, truncateds, infos

    def signed_angle_between_vectors(self, vector_a, vector_b):
        dot_product = np.dot(vector_a, vector_b)
        determinant = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]

        angle_rad = np.arctan2(determinant, dot_product)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def get_action_to_pick_up_water_from_closest_source(self, obs, agent_key):
        # float pos_x = Remap(this.transform.localPosition.x, -750f, 750f, -1f, 1f);
        # float pos_z = Remap(this.transform.localPosition.z, -750f, 750f, -1f, 1f);
        agent_location_x = obs[agent_key][1][0] * 750
        agent_location_y = obs[agent_key][1][1] * 750

        agent_location_x_abs = abs(agent_location_x)
        agent_location_y_abs = abs(agent_location_y)

        agent_dir_x = obs[agent_key][1][2]
        agent_dir_y = obs[agent_key][1][3]
        agent_dir = np.array([agent_dir_x, agent_dir_y])

        if agent_location_x_abs < agent_location_y_abs:
            # water closer in y
            if agent_location_y < 0:
                water_dir = np.array([0, -1])
            else:
                water_dir = np.array([0, 1])
        else:
            # water closer in x
            if agent_location_x < 0:
                water_dir = np.array([-1, 0])
            else:
                water_dir = np.array([1, 0])

        rotation = self.signed_angle_between_vectors(agent_dir, water_dir)

        # angle degree multiplier = 15
        action_multiplier = np.clip(rotation, -15, 15)
        action = self.remap(action_multiplier, -15, 15, -1, 1)

        return action

    def get_action_to_go_to_target(self, obs, agent_key, target):
        agent_location_x = obs[agent_key][1][0] * 750
        agent_location_y = obs[agent_key][1][1] * 750

        target_x = target[0] * 300
        target_y = target[1] * 300

        agent_to_target = np.array(
            [target_x - agent_location_x, target_y - agent_location_y]
        )
        agent_dir_x = obs[agent_key][1][2]
        agent_dir_y = obs[agent_key][1][3]
        agent_dir = np.array([agent_dir_x, agent_dir_y])

        rotation = self.signed_angle_between_vectors(agent_dir, agent_to_target)

        # angle degree multiplier = 15
        action_multiplier = np.clip(rotation, -15, 15)
        action = self.remap(action_multiplier, -15, 15, -1, 1)

        return action

    def distance_to_target(self, obs, agent_key, target):
        agent_location_x = obs[agent_key][1][0] * 750
        agent_location_y = obs[agent_key][1][1] * 750

        target_x = target[0] * 300
        target_y = target[1] * 300

        agent_to_target = np.array(
            [target_x - agent_location_x, target_y - agent_location_y]
        )

        length = np.linalg.norm(agent_to_target)

        return length
