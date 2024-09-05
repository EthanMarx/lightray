"""
Hyperparameter tuning utilities based largely on this tutorial
https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
"""


import importlib
import math
import os
from tempfile import NamedTemporaryFile
from typing import Any, List, Optional, Type, Union

import lightning.pytorch as pl
import pyarrow.fs
import yaml
from lightning.pytorch.cli import LightningCLI
from ray import train
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer

from lightray.callbacks import RayTrainReportCallback


def get_host_cli(cli: Type[LightningCLI]):
    """
    Return a LightningCLI class that will utilize
    the `cli` class passed as a parent class
    for parsing arguments.

    Since this is run on the client,
    we don't actually want to do anything with the arguments we parse,
    just record them, so override the couple parent
    methods responsible for instantiating classes and running
    subcommands.
    """

    class HostCLI(cli):
        def instantiate_classes(self):
            return

        def _run_subcommand(self):
            return

    return HostCLI


def get_worker_cli(
    cli: Type[LightningCLI],
    callbacks: Optional[List[pl.callbacks.Callback]] = None,
):
    """
    Return a LightningCLI class that will actually execute
    training runs on worker nodes
    """

    callbacks = callbacks or [RayTrainReportCallback()]

    class WorkerCLI(cli):
        def instantiate_trainer(self, **kwargs):
            kwargs = kwargs | dict(
                enable_progress_bar=False,
                devices="auto",
                accelerator="auto",
                strategy=RayDDPStrategy(),
                callbacks=callbacks,
                plugins=[RayLightningEnvironment()],
            )
            return super().instantiate_trainer(**kwargs)

    return WorkerCLI


def get_search_space(search_space: Union[str, dict]) -> dict[str, callable]:
    if isinstance(search_space, dict):
        return search_space

    # determine if the path is a file path or a module path
    if os.path.isfile(search_space):
        # load the module from the file
        module_name = os.path.splitext(os.path.basename(search_space))[0]
        spec = importlib.util.spec_from_file_location(
            module_name, search_space
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        # load the module using importlib
        module = importlib.import_module(search_space)

    # try to get the 'space' attribute from the module
    try:
        space = module.space
    except AttributeError:
        raise ValueError(f"Module {module.__name__} has no space dictionary")

    if not isinstance(space, dict):
        raise TypeError(
            "Expected search space in module {} to be "
            "a dictionary, found {}".format(module.__name__, type(space))
        )

    return space


def stop_on_nan(trial_id: str, result: dict) -> bool:
    return math.isnan(result["train_loss"])


class TrainFunc:
    """
    Callable wrapper that takes a `LightningCLI` and executes
    it with the both the `config` passed here at initialization
    time as well as the arguments supplied by a particular
    hyperparameter config. Meant for execution on workers during
    tuning run, which expect a callable that a particular
    hyperparameter config as its only argument.

    All runs of the function will be given the same
    Weights & Biases group name of `name` for tracking.
    The names of individual runs in this group will be
    randomly chosen by W&B.
    """

    def __init__(
        self,
        cli: Type[LightningCLI],
        name: str,
        config: dict,
        callbacks: Optional[list[pl.callbacks.Callback]] = None,
    ) -> None:
        self.cli = cli
        self.name = name
        self.config = config
        self.callbacks = callbacks

    def validate_logger(self, args):
        try:
            logger = self.config["trainer"]["logger"]
        except KeyError:
            logger = None

        if logger is not None:
            if "WandbLogger" not in logger:
                raise ValueError(
                    "Only W&B logging is supported for distributed tuning"
                )
            args.append(f"--trainer.logger.group={self.name}")
        return args

    def __call__(self, hparams: dict[str, Any]):
        """
        Dump the config to a file, then parse it
        along with the hyperparameter configuration
        passed here using our CLI.
        """

        with NamedTemporaryFile(mode="w") as f:
            # dump the core config,
            # then add the hyperparameters
            yaml.dump(self.config, f)
            args = ["-c", f.name]
            for key, value in hparams.items():
                args.append(f"--{key}={value}")

            args = self.validate_logger(args)

            cli_cls = get_worker_cli(self.cli, self.callbacks)
            cli = cli_cls(
                run=False, args=args, save_config_kwargs={"overwrite": True}
            )

        log_dir = cli.trainer.logger.log_dir or cli.trainer.logger.save_dir
        if not log_dir.startswith("s3://"):
            ckpt_prefix = ""
        else:
            ckpt_prefix = "s3://"

        # restore from checkpoint if available
        checkpoint = train.get_checkpoint()
        ckpt_path = None
        if checkpoint:
            ckpt_path = os.path.join(
                ckpt_prefix, checkpoint.path, "checkpoint.ckpt"
            )

        trainer = prepare_trainer(cli.trainer)
        trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt_path)


def configure_deployment(
    train_func: TrainFunc,
    metric_name: str,
    workers_per_trial: int,
    gpus_per_worker: int,
    cpus_per_gpu: int,
    objective: str = "max",
    storage_dir: Optional[str] = None,
    fs: Optional[pyarrow.fs.FileSystem] = None,
) -> TorchTrainer:
    """
    Set up a training function that can be distributed
    among the workers in a ray cluster.

    Args:
        train_func:
            Function that each worker will execute
            with a config specifying the hyperparameter
            configuration for that trial.
        metric_name:
            Name of the metric that will be optimized
            during the hyperparameter search
        workers_per_trial:
            Number of training workers to deploy
        gpus_per_worker:
            Number of GPUs to train over within each worker
        cpus_per_gpu:
            Number of CPUs to attach to each GPU. If gpus_per_worker
            is 0, this is interpretated as the number of CPUs per worker
        objective:
            `"max"` or `"min"`, indicating how the indicated
            metric ought to be optimized
        storage_dir:
            Directory to save ray checkpoints and logs
            during training.
        fs: Filesystem to use for storage
    """

    if gpus_per_worker == 0:
        cpus_per_worker = cpus_per_gpu
    else:
        cpus_per_worker = cpus_per_gpu * gpus_per_worker

    use_gpu = gpus_per_worker > 0
    scaling_config = ScalingConfig(
        trainer_resources={"CPU": 0},
        resources_per_worker={"CPU": cpus_per_worker, "GPU": gpus_per_worker},
        num_workers=workers_per_trial,
        use_gpu=use_gpu,
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute=metric_name,
            checkpoint_score_order=objective,
        ),
        failure_config=FailureConfig(
            max_failures=5,
        ),
        storage_filesystem=fs,
        storage_path=storage_dir,
        name=train_func.name,
        stop=stop_on_nan,
    )
    return TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )


def parse_args(
    cli_cls: Type[LightningCLI], args: Optional[list[str]] = None
) -> dict:
    """
    Use a `LightningCLI` class to parse command line arguments
    """
    host_cli = get_host_cli(cli_cls)
    host_cli = host_cli(run=False, args=args)
    config = host_cli.parser.dump(host_cli.config, format="yaml")
    config = yaml.safe_load(config)
    return config
