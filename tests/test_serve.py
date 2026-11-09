import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from mlops_platform.data import FEATURES
from mlops_platform.promote import CHAMPION
from mlops_platform.serve import Scorer, create_app
from mlops_platform.train import train


def test_serves_champion_and_hot_swaps(
    tracking: str, data: tuple[pd.DataFrame, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    pool, evaluation = data
    v1, v2 = (
        train(pool, model_name="credit").version,
        train(pool.sample(frac=0.6, random_state=1), model_name="credit").version,
    )
    client = MlflowClient()
    client.set_registered_model_alias("credit", CHAMPION, v1)
    monkeypatch.setenv("ADMIN_TOKEN", "t0ken")

    applicants = evaluation[FEATURES].head(3).to_dict(orient="records")
    with TestClient(create_app(Scorer("credit"), poll_seconds=0)) as c:
        assert c.get("/health").json() == {"status": "ok", "model_version": v1}
        body = c.post("/predict", json={"applicants": applicants}).json()
        assert body["model_version"] == v1 and len(body["predictions"]) == 3
        assert all(0 <= p["default_probability"] <= 1 for p in body["predictions"])

        client.set_registered_model_alias("credit", CHAMPION, v2)
        assert c.post("/admin/reload").status_code == 403
        assert c.post("/admin/reload", headers={"X-Admin-Token": "t0ken"}).json() == {"swapped": True, "version": v2}
        assert c.post("/predict", json={"applicants": applicants}).json()["model_version"] == v2


def test_input_validation(tracking: str, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    pool, evaluation = data
    v1 = train(pool, model_name="credit").version
    MlflowClient().set_registered_model_alias("credit", CHAMPION, v1)
    good = evaluation[FEATURES].head(1).to_dict(orient="records")[0]
    with TestClient(create_app(Scorer("credit"), poll_seconds=0)) as c:
        assert c.post("/predict", json={"applicants": []}).status_code == 422
        assert c.post("/predict", json={"applicants": [{**good, "age": 7}]}).status_code == 422
        assert c.post("/predict", json={"applicants": [{**good, "checking_status": "rich"}]}).status_code == 422
