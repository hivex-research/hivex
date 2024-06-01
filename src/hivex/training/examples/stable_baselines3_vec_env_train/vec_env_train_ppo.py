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

from torch import nn
from stable_baselines3 import PPO

# TODO
from .vec_env_callback import (
    EvaluationCallback,
)

# hivex
from hivex.training.framework_wrappers.unity_stable_baselines3.unity_vec_env_wrapper import (
    HivexVecEnvWrapper,
)
from hivex.training.framework_wrappers.wrapper_utils import (
    EnvironmentParametersChannel,
    EngineConfigurationChannel,
    UnityEnvironment,
)


ENV_PATH = (
    "./tests/environment/hivex_test_env_rolling_ball_win/ML-Rolling-Ball_Unity.exe"
)


def initialize_model(vec_env: HivexVecEnvWrapper):
    algorithm_specific_parameters = dict(
        target_kl=0.1,
        gae_lambda=0.95,
        n_epochs=10,
        clip_range=0.2,
    )

    def remove_none_entries(d):
        return {k: v for k, v in list(d.items()) if v is not None}

    model_optional_parameters = remove_none_entries(
        dict(
            batch_size=300,
            n_steps=30000,
            use_sde=None,
            sde_sample_freq=None,
        )
    )

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
        **remove_none_entries(algorithm_specific_parameters),
    )

    return model


def test_vec_env_train_ppo():
    envParameterChannel = EnvironmentParametersChannel()
    envParameterChannel.set_float_parameter("agent_type", 0)
    engineConfigChannel = EngineConfigurationChannel()
    engineConfigChannel.set_configuration_parameters(time_scale=100.0)
    unity_env = UnityEnvironment(
        file_name=ENV_PATH,
        worker_id=0,
        no_graphics=True,
        side_channels=[envParameterChannel, engineConfigChannel],
    )
    unity_env.reset()

    vec_env = HivexVecEnvWrapper(unity_env=unity_env)

    model = initialize_model(vec_env=vec_env)

    eval_callback = EvaluationCallback(
        eval_env=vec_env,
        eval_freq=100000,
        n_eval_episodes=10,
        n_agents=3,
        eval_path="/eval",
        normalization=False,
    )

    model.learn(total_timesteps=300000, callback=[eval_callback])

    vec_env.close()
    print("Closed environment")
