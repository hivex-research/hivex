from pathlib import Path
import subprocess
from tqdm import tqdm
from pathlib import Path
from hivex.training.baseline.ml_agents.utils import (
    load_hivex_config,
    clean_temp_configs,
    construct_command,
    create_training_config_files,
    create_test_config_files,
)


def train(
    train_config_path: Path,
    experiment_name: str,
    train_run_count: int = 1,
    port: str = "5005",
):
    batch_config_files = create_training_config_files(
        experiment_name=experiment_name,
        train_config_path=train_config_path,
        train_run_count=train_run_count,
    )

    for batch_config_file in tqdm(batch_config_files, desc="train"):
        cmd = construct_command(
            config_path=batch_config_file, force=False, experiment_name=experiment_name
        )
        cmd += "--base-port " + port
        subprocess.run(cmd, shell=True)


def test(
    test_config_path: Path,
    experiment_name: str,
    test_run_count: int = 1,
    port: str = "5005",
):
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
        cmd += "--base-port " + port
        subprocess.run(cmd, shell=True)


def run_pipeline(config: Path, port: str):
    hivex_config = load_hivex_config(config)
    train(
        train_config_path=hivex_config["train_config_path"],
        experiment_name=hivex_config["experiment_name"],
        train_run_count=hivex_config["train_run_count"],
        port=port,
    )
    test(
        test_config_path=hivex_config["test_config_path"],
        experiment_name=hivex_config["experiment_name"],
        test_run_count=hivex_config["test_run_count"],
        port=port,
    )
    clean_temp_configs()


if __name__ == "__main__":
    ### BASELINE
    # Wind Farm Control
    # run_pipeline(
    #     config=Path(
    #         "src/hivex/training/baseline/ml_agents/configs/experiments/WindFarmControl_hivex.yaml"
    #     ),
    #     port="5005",
    # )
    # Wildfire Resource Management
    # run_pipeline(
    #     config=Path(
    #         "src/hivex/training/baseline/ml_agents/configs/experiments/WildfireResourceManagement_hivex.yaml"
    #     ),
    #     port="5006",
    # )
    # Drone-Based Reforestation
    # run_pipeline(
    #     config=Path(
    #         "src/hivex/training/baseline/ml_agents/configs/experiments/DroneBasedReforestation_hivex.yaml"
    #     ),
    #     port="5007",
    # )
    # OceanPlasticCollector
    # run_pipeline(
    #     config=Path(
    #         "src/hivex/training/baseline/ml_agents/configs/experiments/OceanPlasticCollection_hivex.yaml"
    #     ),
    #     port="5008",
    # )
    # Aerial Wildfire Suppression
    run_pipeline(
        config=Path(
            "src/hivex/training/baseline/ml_agents/configs/experiments/AerialWildfireSuppression_hivex.yaml"
        ),
        port="5009",
    )

    ### AGENT NUMBER SCALE
    # Wind Farm Control
    # run_pipeline(
    #     config=Path(
    #         "src/hivex/training/baseline/ml_agents/configs/experiments/WindFarmControl_agent_number_hivex.yaml"
    #     ),
    #     port="5010",
    # )
    # Drone-Based Reforestation
    # run_pipeline(
    #     config=Path(
    #         "src/hivex/training/baseline/ml_agents/configs/experiments/DroneBasedReforestation_agent_number_hivex.yaml"
    #     ),
    #     port="5011",
    # )
    # Aerial Wildfire Suppression
    # run_pipeline(
    #     config=Path(
    #         "src/hivex/training/baseline/ml_agents/configs/experiments/AerialWildfireSuppression_agent_number_hivex.yaml"
    #     ),
    #     port="5012",
    # )
