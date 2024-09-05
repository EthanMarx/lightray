# lightray
Easily integrate a `LightningCLI` with `RayTune` hyperparameter optimization

## Getting Started
Extend your custom `LightningCLI` to a `Trainable` compatible with `RayTune`.

Imagine you have a `LightningCLI` parser for a 

```python
from lightning.pytorch as pl

class CustomCLI(pl.cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.init_args.parameter", 
            "model.init_args.parameter", 
            apply_on="parse"
        )

class DataModule(pl.LightningDataModule):
    def __init__(self, hidden_dim: int, learning_rate: float, parameter: int):
        self.parameter = parameter
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
    
    def train_dataloader(self):
        ...

class DataModule(pl.LightningModule):
    def __init__(self, parameter: int):
        self.parameter = parameter
    
    def training_step(self):
        ...

```

Launching a hyperparameter tuning job with `RayTune` using this `LightningCLI` is as simple as

```python
from ray import tune
from raylightning import run

# define search space
search_space = {
    "model.init_args.hidden_dim": tune.choice([32, 64])
    "model.init_args.learning_rate": tune.loguniform(1e-4, 1e-2)
}

# define scheduler
scheduler = ASHAScheduler(max_t=50, grace_period=1, reduction_factor=2)

# pass command line style arguments as if you were
# running your `LightningCLI` from the command line
args = ["--config", "/path/to/config.yaml"]
results = run(
    args=args
    cli_cls=simple_cli,
    name="tune-test",
    metric="val_loss",
    objective="min",
    search_space=search_space,
    scheduler=scheduler,
    storage_dir=storage_dir,
    address=None,
    num_samples=num_samples,
    workers_per_trial=1,
    gpus_per_worker=1.0,
    cpus_per_gpu=4.0,
    temp_dir=None,
)
```

s3 storage works out of the box. Make sure you have set the `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` environment variables set. Then, simply pass an s3 path (e.g. `s3://{bucket}/{folder}` to the `storage_dir` argument.
