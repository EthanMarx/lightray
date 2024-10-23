import time

from botocore.exceptions import ClientError, ConnectTimeoutError
from ray import train
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

BOTO_RETRY_EXCEPTIONS = (ClientError, ConnectTimeoutError)


def report_with_retries(metrics, checkpoint, retries: int = 10):
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
    Equivalent of Rays TuneReportCheckpointCallback,

    with addition of a retry mechanism to the `train.report`
    call to handle transient s3 errors
    """

    def _handle(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        report_dict = self._get_report_dict(trainer, pl_module)
        if not report_dict:
            return

        with self._get_checkpoint(trainer) as checkpoint:
            report_with_retries(report_dict, checkpoint=checkpoint)
