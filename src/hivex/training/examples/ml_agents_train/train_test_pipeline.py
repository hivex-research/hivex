from pathlib import Path
import subprocess
from tqdm import tqdm
from pathlib import Path
from hivex.training.examples.ml_agents_train.utils import (
    load_hivex_config,
    clean_temp_configs,
    construct_command,
    create_training_config_files,
    create_test_config_files,
)


def train(train_config_path: Path, experiment_name: str, train_run_count: int = 1):
    batch_config_files = create_training_config_files(
        experiment_name=experiment_name,
        train_config_path=train_config_path,
        train_run_count=train_run_count,
    )

    for batch_config_file in tqdm(batch_config_files, desc="train"):
        cmd = construct_command(
            config_path=batch_config_file, force=False, experiment_name=experiment_name
        )
        subprocess.run(cmd, shell=True)


def test(test_config_path: Path, experiment_name: str, test_run_count: int = 1):
    batch_config_files = create_test_config_files(
        test_config_path=test_config_path,
        experiment_name=experiment_name,
        test_run_count=test_run_count,
    )

    for batch_config_file in tqdm(batch_config_files, desc="test"):
        cmd = construct_command(
            config_path=batch_config_file,
            experiment_name=experiment_name,
            force=False,
            train=False,
        )
        subprocess.run(cmd, shell=True)


def run_pipeline(config: Path):
    hivex_config = load_hivex_config(config)
    train(
        train_config_path=hivex_config["train_config_path"],
        experiment_name=hivex_config["experiment_name"],
        train_run_count=hivex_config["train_run_count"],
    )
    test(
        test_config_path=hivex_config["test_config_path"],
        experiment_name=hivex_config["experiment_name"],
        test_run_count=hivex_config["test_run_count"],
    )
    clean_temp_configs()


if __name__ == "__main__":
    # WindFarm
    # run_pipeline(config=Path("configs/hivex/WindFarm_hivex.yaml"))
    # Wildfire
    # run_pipeline(config=Path("configs/hivex/Wildfire_hivex.yaml"))
    # Reforestation
    # run_pipeline(config=Path("configs/hivex/Reforestation_hivex.yaml"))
    # OceanPlasticCollector
    # run_pipeline(config=Path("configs/hivex/OceanPlasticCollector_hivex.yaml"))
    # AerialWildfireSuppression
    run_pipeline(config=Path("configs/hivex/AerialWildfireSuppression_hivex.yaml"))
