import os
import yaml
from pathlib import Path
from typing import Callable, Tuple
from gymnasium.spaces import Box, Tuple as TupleSpace, MultiDiscrete
import numpy as np

# RLlib
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from ray.rllib.algorithms.impala import ImpalaConfig, ImpalaTorchPolicy
from ray import train
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import PolicyID, AgentID

# ML-Agents
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

# Hivex
from hivex.training.framework_wrappers.unity_rllib.environments.hivex_env import (
    HivexEnv,
)
from hivex.training.framework_wrappers.unity_rllib.sidechannels.metrics_sidechannel import (
    CustomMetricsCallback,
)


# The max. number of `step()`s for any episode (per agent) before it'll be reset again automatically.
EPISODE_HORIZON = 3000
FRAMEWORK = "torch"
NUM_ROLLOUT_WORKERS = 1
# Number of iterations to train.
STOP_ITERS = 9999
# Number of timesteps to train.
STOP_TIMESTEPS = 1800000
# Reward at which we stop training.
STOP_REWARD = 9999.0
DATA_PATH = Path("src/data")
LOCAL_MODE = True


def get_policy_configs_for_game(
    name: str, policy: PolicySpec
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
        name: PolicySpec(
            policy_class=policy,
            observation_space=obs_spaces[name],
            action_space=action_spaces[name],
        ),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return name

    return policies, policy_mapping_fn


def train(experiment_config: dict):
    channel = EnvironmentParametersChannel()
    for key, env_parameter in experiment_config["env_parameters"].items():
        channel.set_float_parameter(key, env_parameter)

    stats_channel = StatsSideChannel()
    if experiment_config["policy"] == "PPO":
        policy = PPOTorchPolicy
        policy_config = PPOConfig
    elif experiment_config["policy"] == "IMPALA":
        policy = ImpalaTorchPolicy
        policy_config = ImpalaConfig
    else:
        raise ValueError(f"Policy {experiment_config['policy']} not supported.")
    policies, policy_mapping_fn = get_policy_configs_for_game(
        experiment_config["tag"], policy
    )

    param_space = policy_config()

    param_space.environment(
        "unity3d",
        env_config={
            "file_name": experiment_config["file_name"],
            "episode_horizon": EPISODE_HORIZON,
        },
    )
    param_space.framework(FRAMEWORK)
    param_space.rollouts(
        num_rollout_workers=NUM_ROLLOUT_WORKERS,
        rollout_fragment_length=9000,  # 128,  # in ml agents: time_horizon
    )
    if experiment_config["policy"] == "PPO":
        param_space.training(
            lr=0.0003,
            lambda_=0.95,
            gamma=0.995,
            sgd_minibatch_size=256,  # in ml agents: batch_size
            train_batch_size=9000,  # 2048,  # 4096  # in ml agents: buffer_size
            num_sgd_iter=3,
            clip_param=0.2,
            model={
                "fcnet_hiddens": [256, 256],
            },
        )
    elif experiment_config["policy"] == "IMPALA":
        param_space.training(
            lr=0.0003,
            gamma=0.995,
            train_batch_size=9000,  # 2048,  # 4096  # in ml agents: buffer_size
            model={
                "fcnet_hiddens": [256, 256],
            },
        )
    else:
        raise ValueError(f"Policy {experiment_config['policy']} not supported.")
    param_space.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    param_space.resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0" if LOCAL_MODE else "1")),
    )
    param_space.callbacks(CustomMetricsCallback)

    param_space = param_space.to_dict()

    stop = {
        "training_iteration": STOP_ITERS,
        "timesteps_total": STOP_TIMESTEPS,
    }

    run_config_dict = {
        "name": experiment_config["name"],
        "checkpoint_config": CheckpointConfig(
            checkpoint_frequency=50,
            checkpoint_at_end=True,
        ),
        "verbose": 2,
        "stop": stop,
    }

    run_config_dict.update(experiment_config)

    run_config = RunConfig(
        name=run_config_dict["name"],
        checkpoint_config=run_config_dict["checkpoint_config"],
        verbose=run_config_dict["verbose"],
        stop=run_config_dict["stop"],
        local_dir="C:/ray",
    )

    tune.register_env(
        "unity3d",
        lambda c: HivexEnv(
            run_config=experiment_config,
            file_name=experiment_config["file_name"],
            episode_horizon=EPISODE_HORIZON,
            stop_time_steps=STOP_TIMESTEPS,
            side_channel=[channel, stats_channel],
        ),
    )

    # Run the experiment
    tuner = tune.Tuner(
        experiment_config["policy"],
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=1 if LOCAL_MODE else 10),
        run_config=run_config,
    )
    results = tuner.fit()


if __name__ == "__main__":
    experiment_config_file = "src/hivex/training/examples/rllib_train/configs/aerial_wildfire_suppression.yml"

    ray.init(local_mode=LOCAL_MODE)
    with open(experiment_config_file, "r") as file:
        experiment_config = yaml.safe_load(file)

    train(experiment_config)

    ray.shutdown()
