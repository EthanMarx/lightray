"""
Hyperparameter tuning utilities based largely on this tutorial
https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html

since the APIs used in this tutorial
https://docs.ray.io/en/latest/tune/examples/tune-vanilla-pytorch-lightning.html

are out of date with the latest version of lightning.
(They import `from pytorch_lightning`, which doesn't
play well with the `from lightning import pytorch`, the
modern syntax that we use. This is insane, of course, but
that's just the way it is.)

The downside is that I can't figure out how to get local
tune jobs to use the correct `gpus_per_worker`. In my local
tests, one job just claims all the available GPUs no matter
what. This doesn't seem to be a problem for remote, which
is the use case we're largely targeting anyway, so I'm not
freaking out about it. But it would be nice to figure this out.
Unfortunately it does not seem as if this is a heavily trafficked
API, at least not the latest version, and so I've had some
difficulty finding resources from other people dealing with
this issue.
"""

import importlib
import math
import os
from tempfile import NamedTemporaryFile
from typing import List, Optional

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

from raylightning.callbacks import RayTrainReportCallback


def get_host_cli(cli: LightningCLI):
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
    cli: LightningCLI, callbacks: Optional[List[pl.callbacks.Callbakcs]] = None
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


def get_search_space(search_space: str):
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

    def __init__(self, cli: LightningCLI, name: str, config: dict) -> None:
        self.cli = cli
        self.name = name
        self.config = config

    def __call__(self, config):
        """
        Dump the config to a file, then parse it
        along with the hyperparameter configuration
        passed here using our CLI.
        """

        with NamedTemporaryFile(mode="w") as f:
            yaml.dump(self.config, f)
            args = ["-c", f.name]
            for key, value in config.items():
                args.append(f"--{key}={value}")

            # TODO: this is technically W&B specific,
            # but if we're distributed tuning I don't
            # really know what other logger we would use
            args.append(f"--trainer.logger.group={self.name}")
            cli_cls = get_worker_cli(self.cli)
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

        # I have no idea what this `prepare_trainer`
        # ray method does but they say to do it so :shrug:
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
            Number of CPUs to attach to each GPU
        objective:
            `"max"` or `"min"`, indicating how the indicated
            metric ought to be optimized
        storage_dir:
            Directory to save ray checkpoints and logs
            during training.
        fs: Filesystem to use for storage
    """

    cpus_per_worker = cpus_per_gpu * gpus_per_worker
    scaling_config = ScalingConfig(
        trainer_resources={"CPU": 0},
        resources_per_worker={"CPU": cpus_per_worker, "GPU": gpus_per_worker},
        num_workers=workers_per_trial,
        use_gpu=True,
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
