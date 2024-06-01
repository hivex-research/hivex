from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        unity_env = base_env.get_sub_environments()[0]
        stats_side_channel = unity_env.stats_channel.stats

        agent_count = len(episode.agent_rewards)
        agent_reward_sum = sum(
            [value for key, value in list(episode.agent_rewards.items())]
        )
        print(f"total reward: {agent_reward_sum}")
        print(f"mean reward: {agent_reward_sum / agent_count}")
        if agent_count != 8:
            print("")

        if "total reward" not in episode.hist_data:
            episode.hist_data["total reward"] = []
        episode.hist_data["total reward"].append(agent_reward_sum)

        if "mean reward" not in episode.hist_data:
            episode.hist_data["mean reward"] = []
        episode.hist_data["mean reward"].append(agent_reward_sum / agent_count)

        for key, metric in stats_side_channel.items():
            metric_sum = sum([value[0] for value in metric[-agent_count:]])
            print(f"logging {key}: {metric_sum / agent_count}")
            if key not in episode.hist_data:
                episode.hist_data[key] = []
            episode.hist_data[key].append(metric_sum / agent_count)
