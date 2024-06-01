import pytest

# from pytest import raises
from sys import platform

from hivex.training.framework_wrappers.unity_dm_env.unity_dm_env_wrapper import (
    dm_env,
    HivexDmEnvWrapper,
)

from hivex.training.framework_wrappers.unity_gym.unity_gym_wrapper import (
    HivexGymWrapper,
)

from hivex.training.framework_wrappers.unity_stable_baselines3.unity_vec_env_wrapper import (
    HivexVecEnvWrapper,
)

from hivex.training.framework_wrappers.unity_pettingzoo.unity_parallel_env_wrapper import (
    HivexParallelEnvWrapper,
)

from hivex.training.framework_wrappers.wrapper_utils import (
    EnvironmentParametersChannel,
    EngineConfigurationChannel,
    UnityEnvironment,
)

ENV_PATH = ""

if platform == "linux" or platform == "linux2":
    # Linux
    ENV_PATH = "./tests/environments/hivex_test_env_rolling_ball_headless_linux/ML-Rolling-Ball_Unity.x86_64"
elif platform == "win32":
    # Windows
    ENV_PATH = "./tests/environments/hivex_test_env_rolling_ball_headless_win/ML-Rolling-Ball_Unity.exe"

# 0 - RollerAgent w/ continous and discrete actions, vector observations, multi-agent
# 1 - RollerAgent w/ continous actions, vector observations, multi-agent
# 2 - RollerAgent w/ continous actions, vector observations w/ 100 max steps, multi-agent
# 3 - RollerAgent w/ continous actions, vector and visual observations, multi-agent
# 4 - RollerAgent w/ continous actions, visual observations, multi-agent
# 5 - RollerAgent w/ discrete actions, vector observations, multi-agent
# 6 - RollerAgent w/ continous actions, vector observations, single-agent

AGENT_TYPE_PARAMETERS = [0, 1, 2, 3, 4, 5, 6]

WORKER_ID = 0


def initialize_unity_environment(agentType: int) -> UnityEnvironment:
    global WORKER_ID
    WORKER_ID += 1
    envParameterChannel = EnvironmentParametersChannel()
    envParameterChannel.set_float_parameter("agent_type", agentType)
    engineConfigChannel = EngineConfigurationChannel()
    engineConfigChannel.set_configuration_parameters(time_scale=20.0)

    unity_env = UnityEnvironment(
        file_name=ENV_PATH,
        worker_id=WORKER_ID,
        no_graphics=True,
        side_channels=[envParameterChannel, engineConfigChannel],
    )

    unity_env.reset()

    return unity_env


@pytest.mark.parametrize("agentType", AGENT_TYPE_PARAMETERS)
def test_dm_env(agentType):
    unity_env = initialize_unity_environment(agentType)

    dm_env_env = HivexDmEnvWrapper(unity_env=unity_env)
    StepType = dm_env._environment.StepType

    print_per_step_results = False
    for episode in range(2):
        dm_env_env.reset()
        tracked_agent = 0
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        step = 0
        while not done:
            # Generate an action for all agents
            actions = dm_env_env._action_space.generate_value()

            # Get the new simulation results
            time_step = dm_env_env.step(actions=actions)

            observations = time_step.observation
            rewards = time_step.reward
            done = time_step.step_type == StepType.LAST

            episode_rewards += rewards

            step += 1
            if step == 2000:
                done = True

            # Per step print
            if print_per_step_results:
                print(
                    f"step: {step}; "
                    + f"agent id: {tracked_agent}; "
                    + f"action: {actions[tracked_agent]}; "
                    + f"observations: {observations[tracked_agent]}; "
                    + f"reward: {rewards[tracked_agent]}; "
                    + f"done: {done};"
                )
        print(f"Total rewards for episode {episode} is {episode_rewards}")

    dm_env_env.env.close()
    print("Closed environment")


@pytest.mark.parametrize("agentType", AGENT_TYPE_PARAMETERS)
def test_gym(agentType):
    # dm_env_test.check_unity_dm_environment()
    unity_env = initialize_unity_environment(agentType)

    gym_env = HivexGymWrapper(unity_env=unity_env)

    print_per_step_results = False
    for episode in range(2):
        gym_env.reset()
        tracked_agent = 0
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        step = 0
        while not done:
            # Generate an action for all agents
            actions = gym_env.generate_rnd_actions()

            # Get the new simulation results
            observations, rewards, done, info = gym_env.step(actions)

            episode_rewards += rewards

            step += 1
            if step == 2000:
                done = True

            # Per step print
            if print_per_step_results:
                print(
                    f"step: {step}; "
                    + f"agent id: {tracked_agent}; "
                    + f"action: {actions[tracked_agent]}; "
                    + f"observations: {observations[tracked_agent]}; "
                    + f"reward: {rewards[tracked_agent]}; "
                    + f"done: {done};"
                )
        print(f"Total rewards for episode {episode} is {episode_rewards}")

    gym_env.close(timeout=0)
    print("Closed environment")


@pytest.mark.parametrize("agentType", AGENT_TYPE_PARAMETERS)
def test_pettingzoo(agentType):

    unity_env = initialize_unity_environment(agentType)

    parallel_env = HivexParallelEnvWrapper(unity_env=unity_env)

    print_per_step_results = True
    for episode in range(2):
        tracked_agent = parallel_env.agents[0]
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        step = 0
        parallel_env.reset()
        while not done:
            actions = {
                agent: parallel_env.action_space(agent).sample()
                for agent in parallel_env.agents
            }
            observations, rewards, terminations, truncations, info = parallel_env.step(
                actions
            )
            done = any(terminations)

            episode_rewards += rewards[tracked_agent]

            step += 1
            if step == 2000:
                done = True

            # Per step print
            if print_per_step_results:
                print(
                    f"step: {step}; "
                    + f"agent id: {tracked_agent}; "
                    + f"action: {actions[tracked_agent]}; "
                    + f"observations: {observations[tracked_agent]}; "
                    + f"reward: {rewards[tracked_agent]}; "
                    + f"done: {done};"
                )
        print(f"Total rewards for episode {episode} is {episode_rewards}")

    parallel_env.close()
    print("Closed environment")


@pytest.mark.parametrize("agentType", AGENT_TYPE_PARAMETERS)
def test_stable_baselines3(agentType):
    unity_env = initialize_unity_environment(agentType)

    vec_env = HivexVecEnvWrapper(unity_env=unity_env)

    print_per_step_results = False
    for episode in range(2):
        vec_env.reset()
        tracked_agent = 0
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        step = 0
        while not done:
            # Generate an action for all agents
            actions = vec_env.generate_rnd_actions()

            # Get the new simulation results
            observations, rewards, dones, info = vec_env.step(actions)
            done = any(dones)

            episode_rewards += rewards

            step += 1
            if step == 2000:
                done = True

            # Per step print
            if print_per_step_results:
                print(
                    f"step: {step}; "
                    + f"agent id: {tracked_agent}; "
                    + f"action: {actions[tracked_agent]}; "
                    + f"observations: {observations[tracked_agent]}; "
                    + f"reward: {rewards[tracked_agent]}; "
                    + f"done: {done};"
                )
        print(f"Total rewards for episode {episode} is {episode_rewards}")

    vec_env.close()
    print("Closed environment")
