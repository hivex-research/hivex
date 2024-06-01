import os
import yaml
from dotenv import load_dotenv
import os
from pathlib import Path
import wandb

# RLlib
import ray
from ray.tune import Tuner, TuneConfig, register_env
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray import train
from ray.air.integrations.wandb import WandbLoggerCallback

# ML-Agents
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

# Hivex
from hivex.training.examples.rllib_train.sidechannels.metrics_sidechannel import (
    CustomMetricsCallback,
)
from hivex.training.framework_wrappers.unity_rllib.unity_rllib_base_env import (
    HivexBaseEnv,
)

# Load the .env file
load_dotenv()

# Retrieve the WANDB API key
wandb_api_key = os.getenv("WANDB_API_KEY")
FRAMEWORK = "torch"
NUM_ROLLOUT_WORKERS = 4
STOP_TIMESTEPS = 40000  # 0
DATA_PATH = Path("src/data")
LOCAL_MODE = True


def train(experiment_config, task_index: int, difficulty_level: tuple[str, int]):
    if not LOCAL_MODE:
        wandb.login()

    training_config = experiment_config["training_config"]

    environment_parameters_channel = EnvironmentParametersChannel()
    for key, env_parameter in experiment_config["env_parameters"].items():
        environment_parameters_channel.set_float_parameter(key, env_parameter)

    environment_parameters_channel.set_float_parameter("task", task_index)
    environment_parameters_channel.set_float_parameter(
        difficulty_level[0], difficulty_level[1]
    )

    stats_channel = StatsSideChannel()

    policies, policy_mapping_fn = HivexBaseEnv.get_policy_configs_for_game(
        game_name=experiment_config["tag"]
    )

    param_space = (
        PPOConfig()
        .environment(
            "unity3d",
            env_config={
                "file_name": experiment_config["file_name"],
                "episode_horizon": experiment_config["episode_horizon"],
            },
        )
        .framework(FRAMEWORK)
        .rollouts(
            num_rollout_workers=NUM_ROLLOUT_WORKERS,
            rollout_fragment_length="auto",
            num_envs_per_worker=1,
        )
        .training(
            lr=training_config["lr"],
            lambda_=training_config["lambda"],
            gamma=training_config["gamma"],
            sgd_minibatch_size=training_config["sgd_minibatch_size"],
            train_batch_size=training_config["train_batch_size"],
            num_sgd_iter=training_config["num_sgd_iter"],
            clip_param=training_config["clip_param"],
            model=training_config["model"],
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            count_steps_by="env_steps",
        )
        .resources(
            num_gpus=1,
            num_gpus_per_worker=1.0 / NUM_ROLLOUT_WORKERS,
            num_cpus_per_worker=7,
        )
        .callbacks(CustomMetricsCallback)
    ).to_dict()

    stop = {
        "timesteps_total": STOP_TIMESTEPS,
    }

    run_config = {
        "name": experiment_config["name"],
        "checkpoint_config": CheckpointConfig(
            checkpoint_frequency=50,
            checkpoint_at_end=True,
        ),
        "verbose": 2,
        "stop": stop,
    }

    run_config.update(experiment_config)

    wandb_callback = WandbLoggerCallback(
        project="hivex_test",
        api_key=wandb_api_key,
        upload_checkpoints=True,
        save_checkpoints=True,
        group=f"{run_config['name']}_task_{task_index}",
        log_config=run_config,
    )

    air_run_config = RunConfig(
        name=run_config["name"],
        checkpoint_config=run_config["checkpoint_config"],
        verbose=run_config["verbose"],
        stop=run_config["stop"],
        callbacks=[wandb_callback] if not LOCAL_MODE else [],
        local_dir="C:/ray",
    )

    register_env(
        "unity3d",
        lambda env_config: HivexBaseEnv(
            file_name=experiment_config["file_name"],
            episode_horizon=experiment_config["episode_horizon"],
            stop_time_steps=STOP_TIMESTEPS,
            side_channel=[
                environment_parameters_channel,
                stats_channel,
            ],
        ),
    )

    # Run the experiment
    tuner = Tuner(
        "PPO",
        tune_config=TuneConfig(num_samples=1 if LOCAL_MODE else 3),
        run_config=air_run_config,
        param_space=param_space,
    )
    results = tuner.fit()

    if not LOCAL_MODE:
        wandb.finish()


WFC = "wind_farm_control.yml"

if __name__ == "__main__":
    config_file = WFC
    configs_dir = "src/hivex/training/examples/rllib_train/configs/"
    experiment_config_dir = configs_dir + config_file

    with open(experiment_config_dir, "r") as file:
        experiment_config = yaml.safe_load(file)

    difficulty_pattern_key = "difficulty" if config_file is not WFC else "pattern"
    ray.init(local_mode=LOCAL_MODE)
    for task_name, task_index in experiment_config["tasks"].items():
        for difficulty_pattern_level in experiment_config[difficulty_pattern_key]:
            level = (
                difficulty_pattern_level
                if config_file is not WFC
                else experiment_config["pattern_map"][difficulty_pattern_level]
            )
            print(
                f"Running Task: {task_name} at {difficulty_pattern_key}-level: {level}"
            )
            train(
                experiment_config=experiment_config,
                task_index=task_index,
                difficulty_level=(difficulty_pattern_key, difficulty_pattern_level),
            )

    ray.shutdown()
