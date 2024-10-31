"""
Heavily inspired by https://github.com/mauvilsa/ray-tune-cli/
"""

import logging
import os
from typing import Any, Optional, Type

import ray
from jsonargparse import (
    REMAINDER,
    ActionConfigFile,
    ArgumentParser,
    Namespace,
    capture_parser,
)
from lightning.pytorch.cli import LightningCLI
from ray import tune

from lightray import fs as fs_utils
from lightray import utils

ArgsType = Optional[list[str | dict[str, Any], Namespace]]


class RayTuneCli:
    """
    A customizable CLI to run a LightningCLI-based function with Ray Tune.
    """

    def __init__(
        self,
        lightning_cli_cls: Type[LightningCLI] = LightningCLI,
        parser_kwargs: Optional[dict] = None,
        args: ArgsType = None,
    ):
        """
        Args:
            lightning_cli_cls:
                The `LightningCLI` class to tune
            parser_kwargs:
                Additional arguments to pass to the ArgumentParser
            args:
                Arguments to parse. If None, `sys.argv` is used
        """
        self.lightning_cli_cls = lightning_cli_cls
        self.parser = self.build_parser(parser_kwargs)
        self.add_arguments_to_parser(self.parser)
        self.config = self.parser.parse_args(args)

    def build_parser(self, parser_kwargs):
        parser = ArgumentParser(**parser_kwargs)
        parser.add_argument("--config", action=ActionConfigFile)
        parser.add_class_arguments(tune.Tuner, "tuner")
        parser.add_function_arguments(ray.init, "ray_init")

        parser.add_argument(
            "--gpus_per_trial",
            type=int,
            default=0,
            help="Number of GPUs to allocate per trial. "
            "Will be passed to `tune.with_resources` "
            "when wrapping the LightningCLI as a trainable.",
        )
        parser.add_argument(
            "--cpus_per_trial",
            type=int,
            default=1,
            help="Number of CPU's to allocate per trial. "
            "Will be passed to `tune.with_resources` "
            "when wrapping the LightningCLI as a trainable.",
        )

        parser.add_argument(
            "lightning_args",
            nargs=REMAINDER,
            help='All arguments after the double dash "--"'
            "are forwarded to the `lightning_cli_cls`",
        )
        return parser

    def add_arguments_to_parser(self, parser: ArgumentParser) -> None:
        """Implement to add extra arguments to the parser or link arguments.

        Args:
            parser: The parser object to which arguments can be added

        """

    def build_trainable(self):
        """
        Construct a trainable function from the `LightningCLI`
        class and arguments that is compatible with Ray Tune.
        """
        lightning_cli_cls = self.config.lightning_cli_cls
        lightning_parser = capture_parser(lightning_cli_cls)
        if lightning_parser._subcommands_action:
            lightning_parser = (
                lightning_parser._subcommands_action._name_parser_map["fit"]
            )

        if len(self.config.lightning_args) > 1:
            lightning_cfg = lightning_parser.parse_args(
                self.config.lightning_args[1:]
            )
        else:
            lightning_cfg = lightning_parser.get_defaults()

        callbacks = [self.config.tune_callback]
        callbacks += lightning_cfg.get("trainer.callbacks") or []

        lightning_cfg["trainer.callbacks"] = callbacks
        fit_dump = lightning_parser.dump(lightning_cfg)

        trainable = utils.get_trainable(
            self.config.run_config.storage_path,
            lightning_cli_cls,
            fit_dump,
            self.config.cpus_per_trial,
            self.config.gpus_per_trial,
        )
        return trainable

    def run(self):
        trainable = self.build_trainable()

        # instantiate tune related classes from config
        # and parse the parameter space
        cfg_init = self.parser.instantiate_classes(self.config)
        utils.eval_tune_run_config(cfg_init.param_space)

        # set up the storage path and filesystem
        storage_path = self.config.run_config.storage_path
        fs = fs_utils.setup_filesystem(storage_path)

        # if this is an s3 path, strip out the prefix
        storage_path = storage_path.removeprefix("s3://")
        storage_path = os.path.join(storage_path, cfg_init.run_config.name)

        # initialize ray
        ray.init(**cfg_init.ray_init)

        if tune.Tuner.can_restore(storage_path, storage_filesystem=fs):
            # if we can restore from a previous tuning run
            # instantiate tuner from the stored state
            logging.info(
                f"Restoring from previous tuning run at {storage_path}"
            )
            tuner = tune.Tuner.restore(
                storage_path,
                trainable,
                resume_errored=True,
                storage_filesystem=fs,
            )
        else:
            # otherwise, instantiate a new tuner from config
            tuner = tune.Tuner(
                trainable,
                param_space=cfg_init.tuner.param_space,
                tune_config=cfg_init.tuner.tune_config,
                run_config=cfg_init.tuner.run_config,
            )

        results = tuner.fit()

        ray.shutdown()
        return results
