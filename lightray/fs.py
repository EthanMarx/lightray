import os
from typing import Optional

import pyarrow
import s3fs

# s3 retry configuration
retry_config = {"retries": {"total_max_attempts": 10, "mode": "adaptive"}}


def setup_filesystem(
    storage_dir: Optional[str] = None,
) -> tuple[
    Optional[pyarrow.fs.PyFileSystem], Optional[pyarrow.fs.PyFileSystem]
]:
    internal_fs = external_fs = None
    if storage_dir and storage_dir.startswith("s3://"):
        storage_dir = storage_dir.removeprefix("s3://")
        endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        # directly use s3 instead of rays pyarrow s3 default due to
        # this issue https://github.com/ray-project/ray/issues/41137
        internal_fs = s3fs.S3FileSystem(
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=endpoint_url,
            config_kwargs=retry_config,
        )
        internal_fs = pyarrow.fs.PyFileSystem(
            pyarrow.fs.FSSpecHandler(internal_fs)
        )

        external_fs = s3fs.S3FileSystem(
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv(
                "AWS_EXTERNAL_ENDPOINT_URL", os.getenv("AWS_ENDPOINT_URL")
            ),
            config_kwargs=retry_config,
        )
        external_fs = pyarrow.fs.PyFileSystem(
            pyarrow.fs.FSSpecHandler(external_fs)
        )
    return internal_fs, external_fs
