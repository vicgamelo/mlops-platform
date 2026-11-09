import json

import pytest

from mlops_platform.cli import main


def test_cli_stdout_is_pure_json(tracking: str, capsys: pytest.CaptureFixture[str]) -> None:
    """MLflow prints run URLs to stdout by default, which breaks `mlops train | jq`."""
    assert main(["--model-name", "cli", "train", "--synthetic", "--rows", "4000"]) == 0
    trained = json.loads(capsys.readouterr().out)
    assert trained["version"] == "1"

    assert main(["--model-name", "cli", "promote", "--synthetic", "--rows", "4000", "--version", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["promoted"] is True

    # retraining on identical data cannot beat the champion: exit code 2 for pipelines
    main(["--model-name", "cli", "train", "--synthetic", "--rows", "4000"])
    capsys.readouterr()
    assert main(["--model-name", "cli", "promote", "--synthetic", "--rows", "4000", "--version", "2"]) == 2
