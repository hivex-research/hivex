# Training with ML-Agents

Unity's ML-Agents provides us with a set of algorithms we can use to train via CLI. The only thing we need is a config file in `.yaml` format.
In depth guide on training a custom unity environment using ML Agents can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/release_19_docs/docs/Training-ML-Agents.md).

## ML-Agents Training Introduction

### Example Config File

Basic [config file](https://github.com/Unity-Technologies/ml-agents/blob/main/config/ppo/Basic.yaml):

```yaml
behaviors:
  Basic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 32
      buffer_size: 256
      learning_rate: 0.0003
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 20
      num_layers: 1
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.9
        strength: 1.0
    keep_checkpoints: 5
    max_steps: 500000
    time_horizon: 3
    summary_freq: 2000
```

### Training

Training command:

```shell
mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>
```

More information can be found using `--help`:

```shell
mlagents-learn --help
```

Further useful flags for training:

```shell
--force # override if results with run-id exists
--no-graphics # don't use the unity environment graphics layer - this sometimes decreases training time
--resume # continue training after i.e. a crash
```

### Inference

Inference command:

```shell
mlagents-learn <trainer-config-file> --env=<env_name> --initialize-from=<run-identifier> --run-id=<run-identifier> --resume --inference
```

For inference we still have to provide a config file. Unfortunately inference does not take the `max_steps` into account, so either we have to kill the run manually or set the `learning_rate` to `0`.

## Training Example for the Windfarm environment

### Windfarm Config File:

```yaml
behaviors:
  Basic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 32
      buffer_size: 256
      learning_rate: 0.0003
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 20
      num_layers: 1
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.9
        strength: 1.0
    keep_checkpoints: 5
    max_steps: 500000
    time_horizon: 3
    summary_freq: 2000
```

### Windfarm Training>

Training command:

```shell
mlagents-learn hivex/wrappers/examples/ml_agents_train/config/WindFarm.yaml --env=environments/WindFarm/hivex_WindFarm_x86_64 --run-id="Windfarm_training_01" --no_graphics
```