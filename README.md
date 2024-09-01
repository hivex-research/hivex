<div align="center">
  <img src="docs/images/hivex_thumb.png"
      style="border-radius:20px"
      alt="hivex header image"/>
</div>

# HIVEX

_Multi-Agent Reinforcement Learning environment suite with a focus on critical ecological challenges._

## About

The motivation of the hivex suite is to provide advanced reinforcement learning benchmarking environments with an emphasis on: (1) **_real-world_** scenarios, (2) **_multi-agent_** systems, (3) investigating problems and solutions with **_high impact on society_**.

<!-- , (4) **_cooperation and communication_** mechanisms. -->

## ⚡ Quick Overview (TL;DR)

- Download available [HIVEX Environments](https://github.com/hivex-research/hivex-environments)
- Reproducing HIVEX baselines results: [Train-Test-Pipeline Script](https://github.com/hivex-research/hivex/blob/main/src/hivex/training/baseline/ml_agents/train_test_pipeline.py)
- Additional frameworks: [Training Examples](https://github.com/hivex-research/hivex/tree/main/src/hivex/training/examples)
- [HIVEX Leaderboard](https://huggingface.co/spaces/hivex-research/hivex-leaderboard) on Huggingface 🤗
- [HIVEX result plots](https://github.com/hivex-research/hivex-results) on GitHub :octocat:

## Available Environments

| Thumbnail                                                                                                                                                                                                                                                                                               | Title                        | Tag                                     | Tasks          | Difficulty Levels |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | --------------------------------------- | -------------- | ----------------- |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_WindFarmControl/"><img src="https://github.com/hivex-research/hivex/blob/main/docs/images/WFC_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Wind Farm Control"></a>                      | Wind Farm Control            | <code>WindFarmControl</code>            | <code>2</code> | <code>9</code>    |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_WilfireResourceManagement/"><img src="https://github.com/hivex-research/hivex/blob/main/docs/images/WRM_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Wildfire Resource Management"></a> | Wildfire Resource Management | <code>WildfireResourceManagement</code> | <code>3</code> | <code>10</code>   |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_DroneBasedReforestation/"><img src="https://github.com/hivex-research/hivex/blob/main/docs/images/DBR_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Drone-Based Reforestation"></a>      | Drone-Based Reforestation    | <code>DroneBasedReforestation</code>    | <code>7</code> | <code>10</code>   |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_OceanPlasticCollection/"><img src="https://github.com/hivex-research/hivex/blob/main/docs/images/OPC_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Ocean Plastic Collection"></a>        | Ocean Plastic Collection     | <code>OceanPlasticCollection</code>     | <code>4</code> | <code>-</code>    |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_AerialWildfireSuppression/"><img src="https://github.com/hivex-research/hivex/blob/main/docs/images/AWS_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Aerial Wildfire Suppression"></a>  | Aerial Wildfire Suppression  | <code>AerialWildFireSuppression</code>  | <code>9</code> | <code>10</code>   |

<br>

## 🐍 Installation using Conda Virtual Environment (Recommended)

The installation steps are
as follows:

1. Create and activate a virtual environment, e.g.:

   ```shell
   conda create -n hivex python=3.9 -y
   conda activate hivex
   ```

2. Install `ml-agents`:

   ```shell
   pip install git+https://github.com/Unity-Technologies/ml-agents.git@release_20#subdirectory=ml-agents
   ```

3. Install `hivex`:

   ```shell
   git clone git@github.com:hivex-research/hivex.git
   cd hivex
   pip install -e .
   ```

## 🌍 Download HIVEX Environments

### Option 1: Download / Clone binaries locally

1. Download the HIVEX environment binaries for your operating system from the [hivex-environments](https://github.com/hivex-research/hivex-environments) repository:

   ```shell
   git clone git@github.com:hivex-research/hivex-environments.git
   ```

2. Please make sure to un-zip the environment folders.

   This is what the environment paths need to look like (windows):

   - `hivex-environments/Hivex_WindFarmControl_win/Hivex_WindfarmControl.exe`
   - `hivex-environments/Hivex_WildfireResourceManagement_win/Hivex_WildfireResourceManagement.exe`
   - `hivex-environments/Hivex_DroneBasedReforestation_win/Hivex_DroneBasedReforestation.exe`
   - `hivex-environments/Hivex_OceanPlasticCollection_win/Hivex_OceanPlasticCollection.exe`
   - `hivex-environments/Hivex_AerialWildfireSuppression_win/Hivex_AerialWildfireSuppression.exe`

Note: If you want to use a custom directory for your environments and use the `train_test_pipeline.py` script to reproduce results from the paper, adjust the env_path in the config files here `src\hivex\training\baseline\ml_agents\configs\mlagents`.

### Option 2: Use the UnityEnvRegistry

There is an option to use the `UnityEnvRegistry`, so you don't have to download the environments manually. This is how you can include automatic environment download and un-zipping using code:

```python
# import dependencies
from mlagents_envs.registry import UnityEnvRegistry
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

# initialize side channels
env_parameter_channel = EnvironmentParametersChannel()
stats_channel = StatsSideChannel()

# register hivex environments
registry = UnityEnvRegistry()
registry.register_from_yaml(
   "https://raw.githubusercontent.com/hivex-research/hivex-environments/main/hivex_environment_registry.yaml"
)

# initialize environment
# "WindFarmControl", "DroneBasedReforestation", "WildfireResourceManagement", "OceanPlasticCollector", "AerialWildfireSuppression"
unity_env = registry["WindFarmControl"].make(
   no_graphics=no_graphics,
   worker_id=0,
   side_channels=[env_parameter_channel, stats_channel],
)
```

## 🧪 Reproducing Paper Results

Download the HIVEX environment binaries localy as described in [Download HIVEX Environments](#download-hivex-environments).

### Install dependencies:

1. Install dependencies for **ML-Agents**

   <pre><code>pip install hivex[<span style="color: #ff5733;">ml_agents</span>]</code></pre>

### Option 1: Train and Test using ML-Agents locally

Start train and test pipeline:

```shell
python src/hivex/training/baseline/ml_agents/train_test_pipeline.py
```

### Option 2: Train and Test using ML-Agents in Jupyter Notebook

[![Jupyter](https://img.shields.io/badge/ML%20Agents%20Reproducing%20Paper%20Results-Notebook-orange?style=flat-square&logo=jupyter)](https://github.com/hivex-research/hivex/blob/main/notebooks/ml_agents_reproducing_paper_results.ipynb)

```shell
notebooks/ml_agents_reproducing_paper_results.ipynb
```

### 📊 Baseline Results

All results can be found in the [hivex-results](https://github.com/hivex-research/hivex-results) repository.

## 📚 Additional Environments and Training Frameworks

### Install Dependencies

Please make sure to follow the steps in [Installation using conda virtual environment (recommended)](#installation-using-conda-virtual-environment-recommended) section before installing additional dependencies.

You might need to create individual conda environments for each training framework as their dependencies might have conflicts.

### [dm_env](https://github.com/google-deepmind/dm_env) | Random Actions

1. Install dependencies for `dm_env`:

   <pre><code>pip install hivex[<span style="color: #ff5733;">dm_env</span>]</code></pre>

2. Run Example using **Random Actions**:

   ```shell
   python src/hivex/training/examples/dm_env_train/dm_env_random_actions.py
   ```

### [Gym](https://github.com/openai/gym) | Random Actions

The dependencies that come with `pip install git+https://github.com/Unity-Technologies/ml-agents.git@release_20#subdirectory=ml-agents` are sufficient to also use `gym` environments.

1. Run Example using **Random Actions**:

   ```shell
   python src/hivex/training/examples/gym_train/gym_random_actions.py
   ```

### [PettingZoo](https://pettingzoo.farama.org/index.html) | Random Actions

1. Install dependencies for `pettingzoo`:

   <pre><code>pip install hivex[<span style="color: #ff5733;">pettingzoo</span>]</code></pre>

2. Run Example using **Random Actions**:

   ```shell
   python src/hivex/training/examples/pettingzoo_parallel_env_train/parallel_env_random_actions.py
   ```

### [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) | Random Actions | PPO | A2C

1. Install dependencies for `stable_baselines3`:

   <pre><code>pip install hivex[<span style="color: #ff5733;">stable_baselines3</span>]</code></pre>

2. Run Example using **Random Actions**:

   ```shell
   python src/hivex/training/examples/stable_baselines3_vec_env_train/vec_env_random_actions.py
   ```

3. Run Example using [Stable Baslines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html):

   ```shell
   python src/hivex/training/examples/stable_baselines3_vec_env_train/vec_env_train.py ppo
   ```

Or

3. Run Example using [Stable Baslines3 A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html):

   ```shell
   python src/hivex/training/examples/stable_baselines3_vec_env_train/vec_env_train.py a2c
   ```

#### Alternatively you can run training using PPO or A2C using Stable Baselines3 in a Jupyter Notebook:

[![Jupyter](https://img.shields.io/badge/Stable%20Baselines3%20PPO%20&%20A2C-Notebook-orange?style=flat-square&logo=jupyter)](https://github.com/hivex-research/hivex/blob/main/notebooks/vec_env_training_example.ipynb)

```shell
notebooks/vec_env_training_example.ipynb
```

### [RLlib](https://docs.ray.io/en/latest/rllib/index.html) | PPO | IMPALA

1. Install dependencies for `rllib`:

   <pre><code>pip install hivex[<span style="color: #ff5733;">rllib</span>]</code></pre>

2. Run Example using [RLlib PPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo) or [RLlib IMPALA](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#impala)

   ```shell
   python src/hivex/training/examples/rllib_train/rllib_train.py
   ```

If you want to run any other [algorithm](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html) from RLlib, you have to adjust the `policy` field in the corresponding [config file](src\hivex\training\examples\rllib_train\configs\aerial_wildfire_suppression.yml). Currently you can use `PPO` and `IMPALA`.

#### Alternatively you can run training using PPO or IMPALA using RLlib in a Jupyter Notebook:

[![Jupyter](https://img.shields.io/badge/RLlib%20PPO%20&%20IMPALA-Notebook-orange?style=flat-square&logo=jupyter)](https://github.com/hivex-research/hivex/blob/main/notebooks/rllib_training_example.ipynb)

```shell
notebooks/rllib_training_example.ipynb
```

## Adding/Updating dependencies

To add further dependencies, add them to the corresponding `*.in` file in the `./requirements` folder and re-compile using `pip-compile-multi`:

```shell
pip install pip-compile-multi
pip-compile-multi --autoresolve
```

### [ML-Agents](https://github.com/Unity-Technologies/ml-agents)

Apart from the [🧪 Reproducing Paper Results](#reproducing-paper-results) section, we also provide a notebook showing how to use ML-Agents and `hivex-environments` using a from scratch implemented `MADDPG`. This also acts as an example of how you can implement your own algorithms and use our environments. A Jupyter Notebook can be found here:

[![Jupyter](https://img.shields.io/badge/ML%20Agents%20MADDPG-Notebook-orange?style=flat-square&logo=jupyter)](https://github.com/hivex-research/hivex/blob/main/notebooks/ml_agents_training_example_MADDPG.ipynb)

```shell
notebooks/ml_agents_training_example_MADDPG.ipynb
```
## ✨ Submit your own Results to the [HIVEX Leaderboard](https://huggingface.co/spaces/hivex-research/hivex-leaderboard) on Huggingface 🤗

You can follow the steps in the [hivex-results repository](https://github.com/hivex-research/hivex-results?tab=readme-ov-file#submit-your-own-results-to-the-hivex-leaderboard-on-huggingface-) or stay here and follow these steps:

1. Install all dependencies as described [above](https://github.com/hivex-research/hivex?tab=readme-ov-file#-installation-using-conda-virtual-environment-recommended).

2. Run the Train and Test Pipeline, either using [ML-Agents](https://github.com/hivex-research/hivex/tree/main?tab=readme-ov-file#-reproducing-paper-results) or with your [favorite framework](https://github.com/hivex-research/hivex/tree/main?tab=readme-ov-file#-additional-environments-and-training-frameworks).

3. Clone the [hivex-results repository](https://github.com/hivex-research/hivex-results/tree/master).

4. In your local [hivex-results repository](https://github.com/hivex-research/hivex-results/tree/master), add your results to the respective environment/train and environment/test folders. We have provided a `train_dummy_folder` and `test_dummy_folder` with results for training and testing on the Wind Farm Control environment.

5. Run `find_best_models.py`

This script generates data from your results.

```shell
python tools/huggingface/find_best_models.py
```

6. Run `generate_hf_yaml.py`

Uncomment the environment data parser you need for your data. For example, for our dummy data, we need `generate_yaml_WFC(data['WindFarmControl'], key)`. This script takes the data generated in the previous step and turns it into folders including the checkpoint etc. of your training run and a `README.md`, which serves as the model card including important meta-data that is needed for the automatic fetching of the leaderboard of your model.

```shell
python tools/huggingface/generate_hf_yaml.py
```

7. Finally, upload the content of the generated folder(s) to Huggingface 🤗 as a new model.

8. Every 24 hours, the [HIVEX Leaderboard](https://huggingface.co/spaces/hivex-research/hivex-leaderboard) is fetching new models. We will review your model as soon as possible and add it to the verified list of models as soon as possible. If you have any questions, please feel free to reach out to p.d.siedler@gmail.com.

**Congratulations, you did it 🚀!**

## 🤝 Contributing

We welcome contributions! There are several ways you can help improve this project:

### 1. Add an Environment Wrapper or Training Script

If you have an idea for a new environment wrapper or training script, feel free to contribute! Please create a pull request with your code, and our team will review it as soon as possible.

### 2. Enhance Visualization with Plotting Scripts

Another great way to contribute is by adding scripts that generate insightful plots. Visualization is key to understanding model performance, so if you have plotting scripts that can help users analyze their results better, we'd love to see them!

### Pull Request Guidelines

- Ensure that your code follows the existing style and conventions of the project.
- Include detailed comments and documentation where necessary.
- Test your code before submitting the pull request.
- Once your pull request is submitted, please wait for the review process. We will provide feedback and guidance if any changes are needed.

Thank you for contributing to the HIVEX project!

## 📝 Citing HIVEX

If you are using hivex in your work, please cite:

```bibtex
@software{siedler_hivex_2024,
   author={Philipp D. Siedler},
   title={HIVEX: A High-Impact Environment Suite for Multi-Agent Research},
   year={2024},
   month={8},
   url={https://github.com/hivex-research/hivex},
}
```
