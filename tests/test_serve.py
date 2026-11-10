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


# ---------------------------------------------------------------- live pipeline and batch


import json  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

from mlops_platform.data import CATEGORICAL  # noqa: E402

EXAMPLES = Path(__file__).parent.parent / "examples"
CSV = {"content-type": "text/csv"}


@pytest.fixture
def served(tracking: str, data: tuple[pd.DataFrame, pd.DataFrame]) -> Iterator[tuple[TestClient, dict]]:
    pool, evaluation = data
    v1 = train(pool, model_name="credit").version
    MlflowClient().set_registered_model_alias("credit", CHAMPION, v1)
    good = evaluation[FEATURES].head(1).to_dict(orient="records")[0]
    with TestClient(create_app(Scorer("credit"), poll_seconds=0)) as c:
        yield c, good


def sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_predict_refuses_categories_outside_credit_g(served: tuple[TestClient, dict]) -> None:
    """Bug: any text was accepted and silently scored as the encoder's "infrequent" bucket."""
    c, good = served
    for col in CATEGORICAL:
        r = c.post("/predict", json={"applicants": [{**good, col: "banana"}]})
        assert r.status_code == 422, col
    assert c.post("/predict", json={"applicants": [{**good, "purpose": "domestic appliance"}]}).status_code == 200


def test_pipeline_streams_every_stage(served: tuple[TestClient, dict]) -> None:
    c, good = served
    with c.stream("POST", "/pipeline?lang=en", json={"applicant": {**good, "age": None, "duration": "24"}}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-store"
        events = sse_events(r.read().decode())
    assert [d.get("id", e) for e, d in events] == [
        "raw", "transform", "normalize", "fit", "model", "explain", "result", "done"
    ]  # fmt: skip
    stage = {d["id"]: d for e, d in events if e == "stage"}
    assert {"name": "age", "value": 33, "strategy": "median"} in stage["transform"]["imputed"]
    result = events[-2][1]
    single = c.post("/predict", json={"applicants": [{**good, "age": 33, "duration": 24}]}).json()["predictions"][0]
    assert result["default_probability"] == pytest.approx(single["default_probability"], abs=1e-4)
    assert result["decision"] == single["decision"]


@pytest.mark.parametrize(
    ("applicant", "code"),
    [
        ({"age": 7}, "out_of_range"),
        ({"purpose": "carro"}, "out_of_domain"),
        ({"foreign_worker": "yes"}, "protected_column"),
    ],
)
def test_pipeline_invalid_is_422_before_streaming(served: tuple[TestClient, dict], applicant: dict, code: str) -> None:
    c, good = served
    r = c.post("/pipeline?lang=pt", json={"applicant": {**good, **applicant}})
    assert r.status_code == 422
    assert [p["code"] for p in r.json()["detail"]["problems"]] == [code]


def test_pipeline_is_busy_when_the_concurrency_cap_is_reached(
    tracking: str, data: tuple[pd.DataFrame, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    pool, evaluation = data
    MlflowClient().set_registered_model_alias("credit", CHAMPION, train(pool, model_name="credit").version)
    monkeypatch.setenv("PIPELINE_CONCURRENCY", "1")
    good = evaluation[FEATURES].head(1).to_dict(orient="records")[0]
    with TestClient(create_app(Scorer("credit"), poll_seconds=0)) as c:
        with c.stream("POST", "/pipeline", json={"applicant": good}) as first:
            assert first.status_code == 200
            assert c.post("/pipeline", json={"applicant": good}).status_code == 429
            first.read()
        assert c.post("/pipeline", json={"applicant": good}).status_code == 200  # released after the stream


def test_batch_validate_reports_without_scoring(served: tuple[TestClient, dict]) -> None:
    c, _ = served
    r = c.post("/batch/validate?lang=en", content=(EXAMPLES / "batch-errors.csv").read_bytes(), headers=CSV)
    assert r.status_code == 200
    body = r.json()
    assert not body["ok"] and body["valid_rows"] == 4 and body["errors_total"] == 8
    assert "results" not in body and "csv" not in body
    assert any("did you mean 'own'" in p["message"] for p in body["problems"])


def test_batch_score_returns_results_and_a_safe_csv(served: tuple[TestClient, dict]) -> None:
    c, _ = served
    r = c.post("/batch/score", content=(EXAMPLES / "batch-errors.csv").read_bytes(), headers=CSV)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report"]["valid_rows"] == 4 and len(body["results"]) == 12
    assert sum(1 for x in body["results"] if x["status"] == "scored") == 4
    assert "'=HYPERLINK" in body["csv"] and ",=HYPERLINK" not in body["csv"]
    ok = c.post("/batch/score", content=(EXAMPLES / "batch-valid.csv").read_bytes(), headers=CSV).json()
    assert all(0 <= x["default_probability"] <= 1 for x in ok["results"])


def test_batch_score_with_no_valid_rows_is_422(served: tuple[TestClient, dict]) -> None:
    c, _ = served
    r = c.post("/batch/score", content=b"id,age\n1,30\n", headers=CSV)
    assert r.status_code == 422 and r.json()["detail"]["reason"] == "no_valid_rows"
    assert r.json()["detail"]["report"]["file_errors"][0]["code"] == "missing_columns"


def test_batch_content_type_and_size(
    tracking: str, data: tuple[pd.DataFrame, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    pool, _ = data
    MlflowClient().set_registered_model_alias("credit", CHAMPION, train(pool, model_name="credit").version)
    monkeypatch.setenv("BATCH_MAX_BYTES", "1000")
    body = (EXAMPLES / "batch-valid.csv").read_bytes()
    with TestClient(create_app(Scorer("credit"), poll_seconds=0)) as c:
        assert c.post("/batch/validate", json={"csv": "x"}).status_code == 415
        r = c.post("/batch/validate", content=body, headers=CSV)
        assert r.status_code == 413 and r.json()["detail"] == {"reason": "too_large", "max_bytes": 1000}
        assert c.post("/batch/score", content=body, headers=CSV).status_code == 413
