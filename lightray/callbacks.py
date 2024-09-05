import os
import shutil
import time
from pathlib import Path

from botocore.exceptions import ClientError, ConnectTimeoutError
from ray import train
from ray.train import Checkpoint
from ray.train.lightning import RayTrainReportCallback

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


class TrainReportCallback(RayTrainReportCallback):
    """
    Equivalent of the RayTrainReportCallback
    (https://docs.ray.io/en/latest/train/api/doc/ray.train.lightning.RayTrainReportCallback.html)
    except adds a retry mechanism to the `train.report` call
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Creates a checkpoint dir with fixed name
        tmpdir = Path(
            self.tmpdir_prefix, str(trainer.current_epoch)
        ).as_posix()
        os.makedirs(tmpdir, exist_ok=True)

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        # Save checkpoint to local
        ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
        trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Report to train session
        checkpoint = Checkpoint.from_directory(tmpdir)
        report_with_retries(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        trainer.strategy.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)
            shutil.rmtree(tmpdir)
