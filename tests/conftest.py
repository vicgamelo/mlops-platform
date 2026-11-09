import os
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from mlops_platform.data import split_eval, synthetic_credit

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Throwaway MLflow tracking + registry (SQLite) with artifacts under tmp."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    yield uri
    mlflow.set_tracking_uri(None)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_eval(synthetic_credit(n=3000, seed=11))
