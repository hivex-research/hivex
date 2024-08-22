import yaml

# RLlib
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved
from ray import train
from ray.air import CheckpointConfig, RunConfig

# ML-Agents
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

# Hivex
from environments.aerial_wild_fire_suppression_env import AerialWildFireSuppressionEnv
from sidechannels.metrics_sidechannel import CustomMetricsCallback

ENV_PATH = (
    "./tests/environment/hivex_test_env_rolling_ball_win/ML-Rolling-Ball_Unity.exe"
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

AS_TEST = False

LOCAL_MODE = True


def train(experiment_config, env):
    channel = EnvironmentParametersChannel()
    for key, env_parameter in experiment_config["env_parameters"].items():
        channel.set_float_parameter(key, env_parameter)

    stats_channel = StatsSideChannel()

    # TODO:
    policies, policy_mapping_fn = (
        env.get_policy_configs_for_game()
    )

    param_space = (
        PPOConfig()
        .environment(
            "unity3d",
            env_config={
                "file_name": ENV_PATH,
                "episode_horizon": EPISODE_HORIZON,
            },
        )
        .framework(FRAMEWORK)
        .rollouts(
            num_rollout_workers=NUM_ROLLOUT_WORKERS,
            rollout_fragment_length=9000,  # in ml agents: time_horizon
        )
        .training(
            lr=0.0003,
            lambda_=0.95,
            gamma=0.995,
            sgd_minibatch_size=256,  # in ml agents: batch_size
            train_batch_size=9000,  # in ml agents: buffer_size
            num_sgd_iter=3,
            clip_param=0.2,
            model={
                "fcnet_hiddens": [256, 256],
            },
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(
            # num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0" if LOCAL_MODE else "1")),
            # num_gpus_per_worker=0 if LOCAL_MODE else 1,
        )
        .callbacks(CustomMetricsCallback)
    ).to_dict()

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
        callbacks=[],
        local_dir="C:/ray",
    )

    tune.register_env(
        "unity3d",
        lambda c: env(
            run_config=experiment_config,
            file_name=ENV_PATH,
            episode_horizon=EPISODE_HORIZON,
            stop_time_steps=STOP_TIMESTEPS,
            side_channel=[channel, stats_channel],
        ),
    )

    # Run the experiment
    tuner = tune.Tuner(
        "PPO",
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=run_config,
    )
    results = tuner.fit()

    # And check the results
    if AS_TEST:
        check_learning_achieved(results, STOP_REWARD)

experiment_config_file = "configs/aerial_wildfire_suppression_training.yml"
env = AerialWildFireSuppressionEnv

if __name__ == "__main__":

    ray.init(local_mode=LOCAL_MODE)
    with open(experiment_config_file, "r") as file:
        experiment_config = yaml.safe_load(file)

    train(experiment_config, env)

    ray.shutdown()
