import pytest

# from pytest import raises
from sys import platform

from hivex.training.framework_wrappers.unity_pettingzoo.unity_parallel_env_wrapper import (
    HivexParallelEnvWrapper,
)

from hivex.training.framework_wrappers.wrapper_utils import initialize_unity_environment

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
WORKER_ID = 5032


@pytest.mark.parametrize("agentType", AGENT_TYPE_PARAMETERS)
def test_pettingzoo(agentType):
    unity_env = initialize_unity_environment(agentType, ENV_PATH, WORKER_ID)

    parallel_env = HivexParallelEnvWrapper(unity_env=unity_env)

    print_per_step_results = True
    for episode in range(9):
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
