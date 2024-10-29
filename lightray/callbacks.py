import logging
import time
from typing import Literal

from botocore.exceptions import ClientError, ConnectTimeoutError
from ray import train
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

BOTO_RETRY_EXCEPTIONS = (ClientError, ConnectTimeoutError)


def report_with_retries(metrics, checkpoint=None, retries: int = 10):
    """
    Call `train.report`, which will persist checkpoints to s3,
    retrying after any possible errors
    """
    for _ in range(retries):
        try:
            train.report(metrics=metrics, checkpoint=checkpoint)
            break
        except BOTO_RETRY_EXCEPTIONS:
            time.sleep(5)
            continue


class LightRayReportCheckpointCallback(TuneReportCheckpointCallback):
    """
    Subclass of Rays TuneReportCheckpointCallback
    with addition of a retry mechanism to the `train.report`
    call to handle transient s3 errors.

    Adds a `checkpoint_every` parameter to the constructor
    that will only checkpoint every `checkpoint_every` epochs

    Adds a `best` parameter to the constructor that will
    checkpoint if the current metric is better than
    the previous best metric.

    Somehow integrating with `ModelCheckpoint` would be nice
    """

    def __init__(
        self,
        *args,
        checkpoint_every: int = 1,
        metric: str = None,
        mode: Literal["max", "min"] = "min",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.checkpoint_every = checkpoint_every
        self.mode = mode
        self.step = 0
        self.best = None

    def is_best(self, metric):
        if self.mode == "max":
            return metric > self.best
        elif self.mode == "min":
            return metric < self.best
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _handle(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        report_dict = self._get_report_dict(trainer, pl_module)
        if not report_dict:
            return

        report_checkpoint = False

        # report checkpoint if step is a multiple of checkpoint_every
        if not self.step % self.checkpoint_every:
            report_checkpoint = True

        # or if the metric is better than the previous best
        if self.metric is not None:
            step_metric = report_dict[self.metric]
            if self.best is None or self.is_best(step_metric):
                self.best = step_metric
                report_checkpoint = True

        if report_checkpoint:
            with self._get_checkpoint(trainer) as checkpoint:
                logging.debug(f"Reporting checkpoint on epoch {self.step}")
                report_with_retries(report_dict, checkpoint=checkpoint)
        else:
            train.report(metrics=report_dict)

        self.step += 1
