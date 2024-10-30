import os
import re
from ast import literal_eval
from typing import Type
from unittest.mock import patch

from lightning.pytorch.cli import LightningCLI
from ray import train, tune

re_tune_func = re.compile(r"^tune.(\w+)\((.+)\)$")


def eval_tune_run_config(config):
    """
    Evaluates a string into a python statement,
    but only allowing tune.* functions.
    """
    if not config:
        return
    for key, val in config.items():
        val_match = re_tune_func.match(val)
        assert val_match and hasattr(tune, val_match.group(1))
        func = getattr(tune, val_match.group(1))
        args = literal_eval(val_match.group(2))
        if not isinstance(args, tuple):
            args = (args,)
        config[key] = func(*args)


def get_trainable(
    storage_path: str,
    cli_cls: Type[LightningCLI],
    cfg: str,
    cpus_per_trial: int,
    gpus_per_trial: int,
) -> callable:
    def trainable(config):
        args = [f"--config={cfg}"] if cfg else []
        for key, val in config.items():
            args.append(f"--{key}={val}")

        # setup lightning logger path to point to ray trial dir
        log_dir = train.get_context().get_trial_dir()
        args.append(f"--trainer.logger.save_dir={log_dir}")

        # get ckpt prefix based on fs
        if not storage_path.startswith("s3://"):
            ckpt_prefix = ""
        else:
            ckpt_prefix = "s3://"

        # restore from checkpoint if available
        checkpoint = train.get_checkpoint()
        ckpt_path = None
        if checkpoint:
            ckpt_path = os.path.join(
                ckpt_prefix, checkpoint.path, "checkpoint"
            )
            args.append(f"--ckpt_path={ckpt_path}")

        with patch("sys.argv", [""] + ["fit"] + args):
            cli_cls()

    resources = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    return tune.with_resources(trainable, resources)
