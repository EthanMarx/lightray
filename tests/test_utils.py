import tempfile
from pathlib import Path

import yaml

from raylightning import utils


def test_get_host_cli(cli):
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
        cli = utils.get_host_cli(cli)
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
