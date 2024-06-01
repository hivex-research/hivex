import yaml
import operator
import itertools
from functools import reduce
from pathlib import Path
from tqdm import tqdm
import os

ROOT = Path(__file__).parent


def construct_command(config_path, experiment_name="", force=True, train=True):
    base_cmd = "mlagents-learn"
    midfix = "train" if train else "test"
    run_id = "--run-id=" + (
        f"{experiment_name}/{midfix}/{config_path.stem}"
        if experiment_name != ""
        else config_path.stem
    )
    force = "--force" if force else ""
    all_cmds = [base_cmd, config_path.as_posix(), run_id, force]

    cmd = " ".join(all_cmds)
    print(f"running cmd: {cmd}")
    return cmd


def nested_dict_pairs_iterator(dict_obj):
    """This function accepts a nested dictionary as argument
    and iterate over all values of nested dictionaries
    """
    # Iterate over all key-value pairs of dict argument
    for key, value in dict_obj.items():
        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for pair in nested_dict_pairs_iterator(value):
                yield (key, *pair)
        else:
            # If value is not dict type then yield the value
            yield (key, value)


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def create_config_files(
    experiment_name: str,
    config_path: Path,
    output_folder_path: Path,
    run_count: int = 1,
):

    tmp_file_paths = []
    with open(ROOT.as_posix() + config_path.as_posix(), "r") as f:
        configuration = yaml.safe_load(f)

        configuration["env_settings"]["env_path"] = (
            ROOT.as_posix() + configuration["env_settings"]["env_path"]
        )

        all_range_parameter_keys = []
        all_range_parameter_values = []

        for item in nested_dict_pairs_iterator(configuration):
            # if the leafe item is a list
            if type(item[-1]) == list:
                all_range_parameter_keys.append(item[:-1])
                all_range_parameter_values.append(item[-1])

        parameter_permutations = list(itertools.product(*all_range_parameter_values))

        for test_run_id in range(run_count):
            for _, parameter_permutation in enumerate(parameter_permutations):
                new_configuration = configuration
                name_suffix = ""
                for key, value in zip(all_range_parameter_keys, parameter_permutation):
                    setInDict(new_configuration, key, value)
                    name_suffix += (
                        "_" + str(key[-1]) + "_" + str(value).replace(".", "")
                    )

                tmp_file_name = f"{experiment_name}{name_suffix}_run_id_{test_run_id}_{output_folder_path.stem}.yaml"

                tmp_file_path = output_folder_path / tmp_file_name
                tmp_file_paths.append(tmp_file_path)
                with open(tmp_file_path, "w") as outfile:
                    yaml.dump(new_configuration, outfile, default_flow_style=False)

    return tmp_file_paths


def create_training_config_files(
    experiment_name: str, train_config_path: Path, train_run_count: int
):

    output_folder_path = ROOT / Path("configs/mlagents/tmp/train")
    output_folder_path.mkdir(parents=True, exist_ok=True)

    return create_config_files(
        experiment_name=experiment_name,
        config_path=train_config_path,
        output_folder_path=output_folder_path,
        run_count=train_run_count,
    )


def create_test_config_files(
    test_config_path: Path, experiment_name: str, test_run_count: int
):

    output_folder_path = ROOT / Path("configs/mlagents/tmp/test")
    output_folder_path.mkdir(parents=True, exist_ok=True)

    train_results_dir = Path(f"results/{experiment_name}/train")

    temp_test_config_file_paths = create_config_files(
        experiment_name=experiment_name,
        config_path=test_config_path,
        output_folder_path=output_folder_path,
        run_count=test_run_count,
    )

    for test_config_path in temp_test_config_file_paths:
        test_params_split = (
            test_config_path.stem.replace(f"{experiment_name}_", "")
            .replace("_test", "")
            .split("_")
        )
        test_params = [
            test_params_split[index - 1] + "_" + test_params_split[index]
            for index in range(len(test_params_split))
            if test_params_split[index].isnumeric()
        ]
        # remove id
        # test_params = [test_param for test_param in test_params if "id" not in test_param]

        for train_folder_name in os.listdir(train_results_dir):
            # if "run_id_0" not in train_folder_name:
            #     continue
            train_params_split = (
                train_folder_name.replace(f"{experiment_name}_", "")
                .replace("_train", "")
                .split("_")
            )
            train_params = [
                train_params_split[index - 1] + "_" + train_params_split[index]
                for index in range(len(train_params_split))
                if train_params_split[index].isnumeric()
            ]
            # remove id
            # train_params = [train_param for train_param in train_params if "id" not in train_param]

            match_count = 0
            for train_param in train_params:
                if train_param in test_params:
                    match_count += 1

            if match_count == len(train_params):
                with open(test_config_path, "r") as f:
                    configuration = yaml.safe_load(f)
                    configuration["behaviors"]["Agent"]["init_path"] = str(
                        (
                            train_results_dir
                            / train_folder_name
                            / "Agent/checkpoint.pt"
                        ).absolute()
                    )
                f.close()
                with open(test_config_path, "w") as f:
                    yaml.dump(configuration, f)
                f.close()
                break

    return temp_test_config_file_paths


def delete_folder(folder_path):
    path = Path(folder_path)
    try:
        for file in path.glob("**/*"):
            if file.is_file():
                file.unlink()
        path.rmdir()
        print(f"Successfully deleted folder: {folder_path}")
    except OSError as e:
        print(f"Error deleting folder: {folder_path} - {e}")


def clean_temp_configs():
    delete_folder(folder_path="configs/mlagents/tmp/train")
    delete_folder(folder_path="configs/mlagents/tmp/test")


def load_hivex_config(config_file_path: Path):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    return {
        "experiment_name": config["experiment_name"],
        "train_config_path": Path(config["train_config_path"]),
        "test_config_path": Path(config["test_config_path"]),
        "test_run_count": config.get("test_run_count", 1),
        "train_run_count": config.get("train_run_count", 1),
    }
