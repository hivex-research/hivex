import wandb
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.experiment import Trial


class CustomWandbLoggerCallback(WandbLoggerCallback):
    def __init__(
        self,
        api_key,
        upload_checkpoints=True,
        save_checkpoints=True,
        project="your_project",
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            upload_checkpoints=upload_checkpoints,
            save_checkpoints=save_checkpoints,
            project=project,
            **kwargs
        )

    def log_custom_artifact(
        self, artifact_name, type, artifact_description, artifact_path
    ):
        """
        Method to log an artifact to W&B.
        """
        artifact = wandb.Artifact(
            artifact_name, type=type, description=artifact_description
        )
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)
