<div align="center">
  <img src="docs/images/hivex_thumbnail.png"
      style="border-radius:20px"
      alt="hivex header image"/>
</div>

# HIVEX

_Multi-Agent Reinforcement Learning environment suite with a focus on real-world, climate related domains._

## About

The motivation of the hivex suite is to provide advanced reinforcement learning benchmarking environments with an emphasis on: (1) **_real-world_** scenarios, (2) **_multi-agent_** systems, (3) investigating problems and solutions with **_high impact on society_**.

<!-- , (4) **_cooperation and communication_** mechanisms. -->

## Available Environments

| Thumbnail | Title | Tag | Tasks | Difficulty Levels |
| --- | --- | --- | --- | --- |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_WindFarmControl/"><img src="https://github.com/hivex-research/hivex/master/docs/images/WindFarmControl_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Wind Farm Control"></a> | Wind Farm Control | <code>WindFarmControl</code> | 2 | 9 |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_WilfireResourceManagement/"><img src="https://github.com/hivex-research/hivex/master/docs/images/WildfireResourceManagement_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Wildfire Resource Management"></a> | Wildfire Resource Management | <code>WildfireResourceManagement</code> | 3 | 10 |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_DroneBasedReforestation/"><img src="https://github.com/hivex-research/hivex/master/docs/images/DroneBasedReforestation_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Drone-Based Reforestation"></a> | Drone-Based Reforestation |<code>DroneBasedReforestation</code> | 7 | 10 |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_OceanPlasticCollection/"><img src="https://github.com/hivex-research/hivex/master/docs/images/OceanPlasticCollection_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Ocean Plastic Collection"></a> | Ocean Plastic Collection | <code>OceanPlasticCollection</code> | 4 | - |
| <a href="https://github.com/hivex-research/hivex-environments/master/environments/Hivex_AerialWildfireSuppression/"><img src="https://github.com/hivex-research/hivex/master/docs/images/AerialWildfireSuppression_thumb.jpg" height="auto" width="300" style="border-radius:10px" alt="Aerial Wildfire Suppression"></a> | Aerial Wildfire Suppression | <code>AerialWildFireSuppression</code> | 9 | 10 |

<br>

## Installation using conda virtual environment (recommendet)

The installation steps are
as follows:

1. Create and activate a virtual environment, e.g.:

   ```shell
   conda create -n hivex python=3.9 -y
   conda activate hivex
   ```

2. Install `ml-agents`:

   ```shell
   pip install git+https://github.com/Unity-Technologies/ml-agents.git@release_20
   ```

3. Install `hivex`:

   ```shell
   git clone git@github.com:hivex-research/hivex.git
   cd hivex
   pip install -e . --no-deps
   ```

4. Test the `hivex` installation using `pytest`:
   ```shell
   pip install pytest
   pytest tests/test_hivex.py
   ```

## Train and Test using ML-Agents

1. Download the HIVEX environment binaries for your operating system from the [hivex-environments](https://github.com/hivex-research/hivex-environments) repository:

   ```shell
   git clone git@github.com:hivex-research/hivex-environments.git
   ```

2. Copy environment folders to this direction: `src/hivex/training/baseline/ml_agents/dev_environments`.

   This is what the environment paths need to look like (windows):

   - `src/hivex/training/baseline/ml_agents/dev_environments/Hivex_WindFarmControl_win/Hivex_WindfarmControl.exe`
   - `src/hivex/training/baseline/ml_agents/dev_environments/Hivex_WildfireResourceManagement_win/Hivex_WildfireResourceManagement.exe`
   - `src/hivex/training/baseline/ml_agents/dev_environments/Hivex_DroneBasedReforestation_win/Hivex_DroneBasedReforestation.exe`
   - `src/hivex/training/baseline/ml_agents/dev_environments/Hivex_OceanPlasticCollection_win/Hivex_OceanPlasticCollection.exe`
   - `src/hivex/training/baseline/ml_agents/dev_environments/Hivex_AerialWildfireSuppression_win/Hivex_AerialWildfireSuppression.exe`

3. Start train and test pipeline:

   ```shell
   python src/hivex/training/baseline/ml_agents/train_test_pipeline.py
   ```

## Results Baseline

All results can be found in the [hivex-results](https://github.com/hivex-research/hivex-results) repository.

## Citing HIVEX

If you use hivex in your work, please cite:

```bibtex
@inproceedings{siedler_hivex_2024,
    title={HIVEX: A High-Impact Environment Suite for Multi-Agent Research},
    author={Philipp D. Siedler},
    year={2024},
    journal={},
    organization={}
}
```
