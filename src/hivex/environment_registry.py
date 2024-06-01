from mlagents_envs.registry import UnityEnvRegistry


class HivexEnvironmentRegistry:
    def __init__(self) -> None:
        self.registry = UnityEnvRegistry()
        self.registry.register_from_yaml(
            "https://raw.githubusercontent.com/hivex-research/hivex-environments/master/hivex_environment_registry.yaml"
        )

    def make_env(self, environment_tag: str):
        return self.registry[environment_tag].make()

    def get_env_tags(self):
        return list(self.registry.keys())
