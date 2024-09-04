from pathlib import Path

import pytest
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from raylightning.tune import run


@pytest.fixture
def config():
    return Path(__file__).parent / "config.yaml"


@pytest.fixture
def scheduler():
    return ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)


@pytest.fixture
def storage_dir(tmp_path):
    # Create a temporary directory
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    return storage_dir


def test_run(scheduler, config, storage_dir, simple_cli):
    search_space = {
        "model.init_args.hidden_size": tune.randint(2, 8),
        "model.init_args.learning_rate": tune.loguniform(1e-4, 1e-1),
    }

    num_samples = 2
    results = run(
        config,
        simple_cli,
        "tune-test",
        "val_loss",
        "min",
        search_space,
        scheduler,
        storage_dir,
        address=None,
        num_samples=num_samples,
        workers_per_trial=1,
        gpus_per_worker=0.0,
        cpus_per_gpu=1.0,
        temp_dir=None,
    )

    assert len(results) == num_samples
