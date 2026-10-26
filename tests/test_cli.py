"""Kommandozeile."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from gradweight_orchestration import __version__
from gradweight_orchestration.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_sandbox_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sandbox-info"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["platform"] == sys.platform


def test_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", "sandbox_tools:beleg_parsen"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["function"]["name"] == "beleg_parsen"
    assert main(["schema", "sandbox_tools"]) == 2
    assert main(["schema", "sandbox_tools:add"]) == 2


def test_module_entry_point() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "gradweight_orchestration", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in completed.stdout
