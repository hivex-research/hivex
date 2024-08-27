from mlagents_envs.registry import UnityEnvRegistry
from mlagents_envs.environment import UnityEnvironment


class HivexEnvironmentRegistry:
    def __init__(self) -> None:
        self.registry = UnityEnvRegistry()
        self.registry.register_from_yaml(
            "https://raw.githubusercontent.com/hivex-research/hivex-environments/main/hivex_environment_registry.yaml"
        )

    def make_env(self, environment_tag: str, **kwargs) -> UnityEnvironment:
        return self.registry[environment_tag].make(**kwargs)

    def get_env_tags(self) -> list[str]:
        return list(self.registry.keys())
