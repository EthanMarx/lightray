import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import ray
import yaml
from ray import tune
from ray.tune.schedulers import TrialScheduler

from raylightning import utils

if TYPE_CHECKING:
    from lightning.pytorch.cli import LightningCLI


def main(
    config: Path,
    cli: "LightningCLI",
    name: str,
    metric_name: str,
    objective: Literal["min", "max"],
    search_space: Union[dict, Path],
    scheduler: TrialScheduler,
    storage_dir: Optional[str] = None,
    address: Optional[str] = None,
    num_samples: int = 10,
    workers_per_trial: int = 1,
    gpus_per_worker: float = 1.0,
    cpus_per_gpu: float = 1.0,
    temp_dir: Optional[str] = None,
) -> tune.ResultGrid:
    # parse the training configuration file
    # using the user passed LightningCLI class;
    host_cli = utils.get_host_cli(cli)
    cli = host_cli(run=False)
    config = cli.parser.dump(cli.config, format="yaml")
    config = yaml.safe_load(config)

    # if specified, connect to a running ray cluster
    # otherwise, ray will assume one is running locally
    if address is not None:
        # ensure the ray adress starts with "ray://"
        if not address.startswith("ray://"):
            raise ValueError(
                f"Address must start with 'ray://', got {address}"
            )
    ray.init(address, _temp_dir=temp_dir)

    internal_fs, external_fs = utils.setup_filesystem(storage_dir)

    # construct the function that will actually
    # execute the training loop, and then set it
    # up for Ray to distribute it over our cluster,
    # with the desired number of resources allocated
    # to each running version of the job
    train_func = utils.configure_deployment(
        utils.TrainFunc(cli, name, config),
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
    results = tuner.fit()

    return results


def cli():
    pass


if __name__ == "__main__":
    cli()
