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
"""HIVEX stablebaselines3 VecEnv training example."""

import argparse

from torch import nn
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecNormalize

from hivex.training.examples.stable_baselines3_vec_env_train.vec_env_callback import (
    EvaluationCallback,
)

# hivex
from hivex.training.framework_wrappers.unity_stable_baselines3.unity_vec_env_wrapper import (
    HivexVecEnvWrapper,
)

from hivex.training.framework_wrappers.wrapper_utils import initialize_unity_environment

ENV_PATH = "tests/environments/hivex_test_env_rolling_ball_headless_win/ML-Rolling-Ball_Unity.exe"
WORKER_ID = 0


def initialize_model(vec_env: HivexVecEnvWrapper, algorithm: str = "ppo"):
    def remove_none_entries(d):
        return {k: v for k, v in list(d.items()) if v is not None}

    tensorboard_log_path = "/tensorboard_logs"

    policy_layers_comma_sep: str = "128,128,128"
    value_layers_comma_sep: str = "128,128,128"

    policy_layers = [
        int(layer_width) for layer_width in policy_layers_comma_sep.split(",")
    ]
    value_layers = [
        int(layer_width) for layer_width in value_layers_comma_sep.split(",")
    ]

    net_arch = [dict(vf=value_layers, pi=policy_layers)]

    activation_function = None
    log_std_init = None
    ppo_a2c_ortho_init = None
    policy_kwargs = remove_none_entries(
        dict(
            activation_fn=nn.ReLU if activation_function == "ReLU" else None,
            net_arch=net_arch,
            log_std_init=log_std_init,
            ortho_init=ppo_a2c_ortho_init,
        )
    )

    if algorithm == "ppo":
        algorithm_specific_parameters = remove_none_entries(
            dict(
                target_kl=0.1,
                gae_lambda=0.95,
                n_epochs=1,
                clip_range=0.2,
            )
        )

        model_optional_parameters = remove_none_entries(
            dict(
                batch_size=10,
                n_steps=100,
                use_sde=None,
                sde_sample_freq=None,
            )
        )
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=2,
            tensorboard_log=str(tensorboard_log_path),
            device="cuda",
            gamma=0.9,
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,
            **model_optional_parameters,
            **algorithm_specific_parameters,
        )
    elif algorithm == "a2c":
        algorithm_specific_parameters = remove_none_entries(dict(gae_lambda=0.95))
        model_optional_parameters = remove_none_entries(dict(n_steps=100))

        model = A2C(
            policy="MlpPolicy",
            env=vec_env,
            verbose=2,
            tensorboard_log=str(tensorboard_log_path),
            device="cuda",
            gamma=0.9,
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,
            **model_optional_parameters,
            **algorithm_specific_parameters,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model


def train(algorithm: str = "ppo"):
    unity_env = initialize_unity_environment(
        worker_id=WORKER_ID + 1, hivex_env_tag="WindFarmControl", no_graphics=True
    )

    vec_env = HivexVecEnvWrapper(unity_env)

    vec_env_normalized = VecNormalize(vec_env, norm_reward=True)

    model = initialize_model(vec_env=vec_env_normalized, algorithm=algorithm)

    eval_callback = EvaluationCallback(
        eval_env=vec_env_normalized,
        eval_freq=5000,
        n_eval_episodes=1,
        n_agents=8,
        eval_path="/eval",
        normalization=False,
    )

    model.learn(total_timesteps=1000, callback=[eval_callback], progress_bar=True)

    vec_env_normalized.close()
    print("Closed environment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "algorithm",
        type=str,
        help="Name of the algorithm to use for training (e.g., ppo or a2c)",
    )

    args = parser.parse_args()
    train(args.algorithm)

    train("ppo")
