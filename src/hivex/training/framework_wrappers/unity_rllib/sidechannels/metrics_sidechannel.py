from ray.rllib.algorithms.callbacks import DefaultCallbacks

###
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.env.base_env import BaseEnv
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.rllib.policy import Policy
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2


class CustomMetricsCallback(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
        self.last_episode_steps = 0

    def on_episode_step(self, worker, base_env, policies, episode, **kwargs) -> None:
        dif = episode.total_agent_steps - self.last_episode_steps
        self.last_episode_steps = episode.total_agent_steps
        self.step_count += dif
        if self.step_count >= 9000:
            # Custom logic to record stats
            # print("")
            self.step_count = 0

    # def on_train_result(
    #     self, *, algorithm: "Algorithm", result: dict, **kwargs
    # ) -> None:
    #     self.step_count += result["timesteps_this_iter"]
    #     if self.step_count >= 3000:
    #         # Reset step count
    #         self.step_count = 0
    #         # Custom logic to record stats
    #         print("")

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):

        self.last_episode_steps = 0

        unity_env = base_env.get_sub_environments()[0]
        stats_side_channel = unity_env.stats_channel.stats

        for key, metric in stats_side_channel.items():
            episode.custom_metrics[key] = sum([value[0] for value in metric]) / len(
                metric
            )

        print(episode.custom_metrics)
