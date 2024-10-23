"""
A large portion of this was inspired by
https://github.com/mauvilsa/ray-tune-cli/
"""

import logging
import os
import sys
from typing import Optional, Type

import pyarrow
import ray
from jsonargparse import ActionConfigFile, ArgumentParser, capture_parser
from lightning.pytorch.cli import LightningCLI
from ray import train, tune
from ray.tune.integration.pytorch_lightning import TuneCallback

from lightray import fs, utils


def cli():
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_subclass_arguments(TuneCallback, "tune_callback")
    parser.add_class_arguments(tune.TuneConfig, "tune_config")
    parser.add_class_arguments(
        train.RunConfig, "run_config", skip="storage_filesystem"
    )
    parser.add_function_arguments(ray.init, "ray_init")
    parser.add_argument(
        "lightning_cli_cls",
        type=Type[LightningCLI],
        help="Lightning CLI class",
    )
    parser.add_argument("--param_space", type=dict)
    parser.add_argument(
        "--external_fs",
        type=Optional[pyarrow.fs.FileSystem],
        help="External filesystem to use",
    )
    parser.add_argument("--gpus_per_trial", type=int, default=0)
    parser.add_argument("--cpus_per_trial", type=int, default=1)
    parser.add_argument(
        "lightning_config",
        type=str,
        help="Path to the lightning config file",
    )

    cfg = parser.parse_args()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        stream=sys.stdout,
    )

    # get lightning cli parser and parse passed config
    lightning_cli_cls = cfg.lightning_cli_cls
    lightning_parser = capture_parser(lightning_cli_cls)
    if lightning_parser._subcommands_action:
        lightning_parser = (
            lightning_parser._subcommands_action._name_parser_map["fit"]
        )

    lightning_cfg = lightning_parser.parse_args(
        ["--config", cfg.lightning_config]
    )

    callbacks = [cfg.tune_callback]
    callbacks += lightning_cfg.get("trainer.callbacks") or []

    lightning_cfg["trainer.callbacks"] = callbacks
    fit_dump = lightning_parser.dump(lightning_cfg)

    # instantiate tune related classes from config
    # and parse the parameter space
    cfg_init = parser.instantiate_classes(cfg)
    utils.eval_tune_run_config(cfg_init.param_space)

    # initialize ray
    ray.init(**cfg_init.ray_init)

    # set up the storage path and filesystem
    storage_path = cfg_init.run_config.storage_path
    internal_fs = fs.setup_filesystem(storage_path)

    trainable = utils.get_trainable(
        storage_path,
        lightning_cli_cls,
        fit_dump,
        cfg.cpus_per_trial,
        cfg.gpus_per_trial,
    )

    if cfg_init.external_fs is None:
        cfg_init.external_fs = internal_fs

    # if this is an s3 path, strip out the prefix
    storage_path = storage_path.removeprefix("s3://")
    storage_path = os.path.join(storage_path, cfg_init.run_config.name)

    if tune.Tuner.can_restore(
        storage_path, storage_filesystem=cfg_init.external_fs
    ):
        # if we can restore from a previous tuning run
        # instantiate tuner from the stored state
        logging.info(f"Restoring from previous tuning run at {storage_path}")
        tuner = tune.Tuner.restore(
            storage_path,
            trainable,
            resume_errored=True,
            storage_filesystem=cfg_init.external_fs,
        )
    else:
        # otherwise, instantiate a new tuner from config
        tuner = tune.Tuner(
            trainable,
            param_space=cfg_init.param_space,
            tune_config=cfg_init.tune_config,
            run_config=cfg_init.run_config,
        )

    # othereise, instantiate a new tuner from config
    results = tuner.fit()

    ray.shutdown()
    return results


if __name__ == "__main__":
    cli()
