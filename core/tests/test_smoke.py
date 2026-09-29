import importlib

import pytest

from spacemidi.cli import main

SUBPACKAGES = [
    "dsp", "algos", "nn", "refs", "notes", "midi", "export", "pipeline", "eval", "eval.datasets",
    "algos.separation", "algos.pitch", "algos.poly", "algos.drums",
    "algos.rhythm", "algos.instruments", "algos.post",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"spacemidi.{name}")


def test_cli_info(capsys):
    assert main(["info"]) == 0
    assert "spacemidi" in capsys.readouterr().out
