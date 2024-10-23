from typing import Any, Optional

import pyarrow
import s3fs

# s3 retry configuration
retry_config = {"retries": {"total_max_attempts": 10, "mode": "adaptive"}}


def setup_filesystem(
    storage_path: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    config_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[pyarrow.fs.PyFileSystem]:
    if storage_path.startswith("s3://"):
        storage_path = storage_path.removeprefix("s3://")
        # directly use s3 instead of rays pyarrow s3 default due to
        # this issue https://github.com/ray-project/ray/issues/41137
        fs = s3fs.S3FileSystem(
            key=key,
            secret=secret,
            endpoint_url=endpoint_url,
            config_kwargs=config_kwargs,
        )
        fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs))
        return fs
    return None
