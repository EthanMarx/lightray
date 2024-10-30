# lightray
A CLI for easily integrating `LightningCLI` with `RayTune` hyperparameter optimization

## Getting Started
Imagine you have the following lightning `DataModule`, `LightningModule` and `LightningCLI`

```python
from lightning.pytorch as pl

class DataModule(pl.LightningDataModule):
    def __init__(self, hidden_dim: int, learning_rate: float, parameter: int):
        self.parameter = parameter
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
    
    def train_dataloader(self):
        ...

class LightningModule(pl.LightningModule):
    def __init__(self, parameter: int):
        self.parameter = parameter
    
    def training_step(self):
        ...

class CustomCLI(pl.cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.init_args.parameter", 
            "model.init_args.parameter", 
            apply_on="parse"
        )

```

To launching a hyperparameter tuning job with `RayTune` using this `LightningCLI` can be done by configuring
a `yaml` that looks like the following

```yaml
# tune.yaml

# ray.tune.TuneConfig
tune_config:
  mode: "min"
  metric: "val_loss"
  scheduler: 
    class_path: ray.tune.schedulers.ASHAScheduler
    init_args:
      max_t: 200
      grace_period: 21
      reduction_factor: 2
  num_samples: 1
  reuse_actors: true

# ray.train.RunConfig
run_config:
  name: "my-first-run"
  storage_path: s3://aframe-test/new-tune/
  failure_config:
    class_path: ray.train.FailureConfig
    init_args:
      max_failures: 1
  checkpoint_config:
    class_path: ray.train.CheckpointConfig
    init_args:
      num_to_keep: 5
      checkpoint_score_attribute: "val_loss"
      checkpoint_score_order: "min"
  verbose: null

# ray.train.SyncConfig
sync_config:
  sync_period: 1000

# ray.init
ray_init:
  address: null
  
# tune.Tune.param_space
param_space:
  model.learning_rate: tune.loguniform(1e-3, 4)

# ray.tune.TuneCallback
tune_callback:
  class_path: lightray.callbacks.LightRayReportCheckpointCallback
  init_args:
    'on': "validation_end"
    checkpoint_every: 10

# resources per trial
cpus_per_trial: 2
gpus_per_trial: 1

# lightning cli class to tune
lightning_cli_cls: example.lightning.Cli
```

Then, launch the tuning job.

```console
lightray --config tune.yaml -- --config cli.yaml
```

Any arguments passed after "--" will automatically be forwarded to the specified `LightningCLI` class as if it were being run from the command line. This way, tuning configuration and training configuration can be kept separate. Also, you can easily utilize existing configuration files compatible with your `LightningCLI`.


## S3 Support
In addition, there is automatic support for `s3` storage. Make sure you have set the `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` environment variables. 
Then, simply pass the path to your bucket with the `s3` prefix (e.g. `s3://{bucket}/{folder}`) to the `run_config.storage_path` argument.

```console
lightray --config tune.yaml --run_config.storage_path s3://{bucket}/{folder}
```

There is also a wrapper around the `ray.tune.integration.pytorch_lightning.TuneReportCheckpointCallback` that will do checkpoint reporting with retries to handle transient s3 errors.
This is provided at `lightray.callbacks.LightRayReportCheckpointCallback`


## Remote cluster
To connect to a remote ray cluster, pass the ip address (using port `10001`) to `ray_init.address`:

```console
lightray --config tune.yaml --ray_init.address ray://{ip}:10001 -- --config cli.yaml
```
