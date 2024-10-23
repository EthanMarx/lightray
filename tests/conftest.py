from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader, TensorDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, argument: int):
        self.argument = argument


class Model(pl.LightningModule):
    def __init__(self, argument: int):
        self.argument = argument


@pytest.fixture
def dummy_cli():
    class Cli(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments(
                "data.init_args.argument",
                "model.init_args.argument",
                apply_on="parse",
            )

    return Cli


space = {"a": "b"}


class SimpleDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dim: int, num_samples: int = 64, batch_size: int = 32
    ):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.data_dim = data_dim

    def setup(self, stage):
        self.train_data = torch.randn(self.num_samples, self.data_dim)
        self.train_labels = torch.randint(0, 2, (self.num_samples,))
        self.val_data = torch.randn(self.num_samples // 5, self.data_dim)
        self.val_labels = torch.randint(0, 2, (self.num_samples // 5,))
        self.setup_called = True

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.train_data, self.train_labels),
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.val_data, self.val_labels),
            batch_size=self.batch_size,
        )


class SimpleModel(pl.LightningModule):
    def __init__(
        self, data_dim: int, hidden_size: int = 32, learning_rate: float = 0.01
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.layer1 = torch.nn.Linear(data_dim, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.sigmoid(self.layer2(x))

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(
            y_hat.squeeze(), y.float()
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(
            y_hat.squeeze(), y.float()
        )
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


@pytest.fixture
def config():
    return Path(__file__).parent / "config.yaml"


def simple_cli() -> LightningCLI:
    class Cli(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments(
                "data.init_args.data_dim",
                "model.init_args.data_dim",
                apply_on="parse",
            )

    return Cli
