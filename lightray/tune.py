import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Type, Union

import lightning.pytorch as pl
import ray
from ray import tune
from ray.tune.schedulers import TrialScheduler

from lightray import utils
from lightray.fs import setup_filesystem

if TYPE_CHECKING:
    from lightning.pytorch.cli import LightningCLI


def run(
    cli_cls: Type["LightningCLI"],
    name: str,
    metric_name: str,
    objective: Literal["min", "max"],
    search_space: Union[dict, Path, str],
    scheduler: TrialScheduler,
    storage_dir: Optional[str] = None,
    address: Optional[str] = None,
    num_samples: int = 10,
    workers_per_trial: int = 1,
    gpus_per_worker: float = 1.0,
    cpus_per_gpu: float = 1.0,
    callbacks: Optional[Type[pl.callbacks.Callback]] = None,
    temp_dir: Optional[str] = None,
    args: Optional[list[str]] = None,
) -> tune.ResultGrid:
    """
    Hyperparameter tune a LightningCLI with Ray Tune

    Args:
        cli_cls:
            LightningCLI subclass to use for each trials training
        name:
            Name of the tuning run
        metric_name:
            Name of the metric to optimize
        objective:
            Either "min" or "max", indicating whether
            the metric should be minimized or maximized
        search_space:
            A dictionary, path to a python file containing
            a dictionary named `space`, or a python module path
            to import that contains a dictionary named `space`.
            This dictionary should map the names of the arguments
            of the `cli_cls` to the search space for that argument.
        scheduler:
            Ray Tune scheduler to use
        storage_dir:
            Directory to store Ray Tune logs and checkpoints
        address:
            Address of the Ray cluster to connect to. If None,
            Ray will launch a local cluster.
        num_samples:
            Number of hyperparameter configurations to try
        workers_per_trial:
            Number of workers to deploy for each trial
        gpus_per_worker:
            Number of GPUs to attach to each worker
        cpus_per_gpu:
            Number of CPUs to attach to each GPU.
            If `gpus_per_worker` is 0, this is interpreted
            as the number of CPUs per worker.
        callbacks:
            Lightning `Callback` classes to attach
            to each trials training loop.
            These should be __classess__, not instances. They will
            be instantiated during runtime for each trials so that
            trial specific information can be used. A callback that
            reports trial information to Ray Tune is automatically
            added.
        temp_dir:
            Temporary directory to use for Ray Tune
        args:
            Arguments to pass to the LightningCLI. This should be a list
            of command line style arguments,
            e.g. `["--config", "/path/to/config.yaml"]`.
    """

    # parse the training configuration file, and
    # any argument overrides
    # using the user passed LightningCLI class;
    config = utils.parse_args(cli_cls, args)

    # if specified, connect to a running ray cluster
    # otherwise, ray will assume one is running locally
    logging.info("Initializing Ray")
    if address is not None:
        logging.info(f"Connecting to Ray cluster at {address}")
        # ensure the ray adress starts with "ray://"
        if not address.startswith("ray://"):
            raise ValueError(
                f"Address must start with 'ray://', got {address}"
            )
    ray.init(address, _temp_dir=temp_dir)

    logging.info("Initializing checkpoint storage filesystems")
    internal_fs, external_fs, storage_dir = setup_filesystem(str(storage_dir))

    # construct the function that will actually
    # execute the training loop, and then set it
    # up for Ray to distribute it over our cluster,
    # with the desired number of resources allocated
    # to each running version of the job

    train_func = utils.configure_deployment(
        utils.TrainFunc(cli_cls, name, config, callbacks),
        metric_name=metric_name,
        workers_per_trial=workers_per_trial,
        gpus_per_worker=gpus_per_worker,
        cpus_per_gpu=cpus_per_gpu,
        objective=objective,
        storage_dir=storage_dir or None,
        fs=internal_fs,
    )

    search_space = utils.get_search_space(search_space)

    # restore from a previous tuning run
    path = os.path.join(storage_dir, name)
    if tune.Tuner.can_restore(path, storage_filesystem=external_fs):
        logging.info(f"Restoring from previous tuning run at {path}")
        tuner = tune.Tuner.restore(
            path,
            train_func,
            resume_errored=True,
            storage_filesystem=external_fs,
        )

    else:
        tuner = tune.Tuner(
            train_func,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric=metric_name,
                mode=objective,
                num_samples=num_samples,
                scheduler=scheduler,
                reuse_actors=True,
                trial_name_creator=lambda trial: f"{trial.trial_id}",
                trial_dirname_creator=lambda trial: f"{trial.trial_id}",
            ),
        )

    logging.info("Starting tune job")
    results = tuner.fit()
    ray.shutdown()
    return results


def cli():
    pass


if __name__ == "__main__":
    cli()
