import tempfile
from pathlib import Path

import yaml

from raylightning import utils


def test_get_host_cli(dummy_cli):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    content = {
        "data": {
            "class_path": "tests.conftest.DataModule",
            "init_args": {"argument": 1},
        },
        "model": {"class_path": "tests.conftest.Model"},
    }
    with open(temp.name, "w") as file:
        yaml.dump(content, file)
        args = ["--config", temp.name]
        cli = utils.get_host_cli(dummy_cli)
        cli = cli(run=False, args=args)
        assert cli.config["model"]["init_args"]["argument"] == 1


def test_get_search_space():
    # test loading from a file
    path = Path(__file__).parent / "conftest.py"
    space = utils.get_search_space(path)
    assert space["a"] == "b"

    # test passing a module
    space = utils.get_search_space("tests.conftest")
    assert space["a"] == "b"

    # test passing a dictionary
    space = utils.get_search_space({"a": "b"})
    assert space["a"] == "b"


def test_parse_args(simple_cli, config):
    # test args from config file are parsed correctly
    yaml = utils.parse_args(simple_cli, config)
    assert yaml["model"]["init_args"]["hidden_size"] == 32
    assert yaml["model"]["init_args"]["learning_rate"] == 0.01
    assert yaml["data"]["init_args"]["data_dim"] == 10

    # now test ability to override args
    args = ["--model.init_args.hidden_size", "64"]
    yaml = utils.parse_args(simple_cli, config, args)
    assert yaml["model"]["init_args"]["hidden_size"] == 64
    assert yaml["model"]["init_args"]["learning_rate"] == 0.01
