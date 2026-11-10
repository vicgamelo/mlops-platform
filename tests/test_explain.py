from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mlops_platform import explain
from mlops_platform.data import FEATURES, NUMERIC, TARGET
from mlops_platform.train import candidates

STAGES = ["raw", "transform", "normalize", "fit", "model", "explain"]


@pytest.fixture(scope="module")
def models(data: tuple[pd.DataFrame, pd.DataFrame]) -> dict[str, Any]:
    pool, _ = data
    return {name: clone(est).fit(pool[FEATURES], pool[TARGET]) for name, est in candidates().items()}


@pytest.fixture
def applicant(data: tuple[pd.DataFrame, pd.DataFrame]) -> dict[str, Any]:
    _, evaluation = data
    row = evaluation[FEATURES].iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def run(model: Any, applicant: dict[str, Any], **kw: Any) -> list[tuple[str, dict[str, Any]]]:
    prepared, problems = explain.prepare(applicant)
    assert prepared is not None, problems
    return list(explain.run(model, "7", prepared, **kw))


def test_reference_is_credit_g() -> None:
    ref = explain.reference()
    assert len(ref.frame) == 1000 and round(ref.frame[TARGET].mean(), 2) == 0.30
    assert ref.median["age"] == 33 and ref.mode["checking_status"] == "no checking"


def test_prepare_casts_text_and_imputes_missing_with_median_or_mode(applicant: dict[str, Any]) -> None:
    raw = {**applicant, "duration": "48", "age": None, "housing": ""}
    del raw["job"]  # absent = missing
    prepared, problems = explain.prepare(raw)
    assert problems == [] and prepared is not None
    assert prepared.clean["duration"] == 48 and prepared.clean["age"] == 33
    assert prepared.clean["housing"] == "own" and prepared.clean["job"] == "skilled"
    assert {i["name"]: i["strategy"] for i in prepared.imputed} == {"age": "median", "housing": "mode", "job": "mode"}
    duration = next(c for c in prepared.casts if c["name"] == "duration")
    assert duration == {"name": "duration", "from": "string", "to": "int", "raw": "48", "value": 48, "coerced": True}


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"age": 7}, "out_of_range"),
        ({"purpose": "carro"}, "out_of_domain"),
        ({"duration": "doze"}, "not_integer"),
        ({"personal_status": "male single"}, "protected_column"),
        ({"salary": 1}, "unknown_column"),
        ({"age": True}, "wrong_type"),
        ({"age": [1]}, "wrong_type"),
    ],
)
def test_prepare_refuses_what_the_model_cannot_use(applicant: dict[str, Any], patch: dict[str, Any], code: str) -> None:
    prepared, problems = explain.prepare({**applicant, **patch})
    assert prepared is None and [p.code for p in problems] == [code]


def test_stages_come_in_order_and_match_the_model(models: dict[str, Any], applicant: dict[str, Any]) -> None:
    model = models["gradient_boosting"]
    events = run(model, applicant, threshold=0.5)
    assert [d["id"] for e, d in events if e == "stage"] == STAGES and events[-1][0] == "result"
    stage = {d["id"]: d for e, d in events if e == "stage"}
    frame = pd.DataFrame([applicant])[FEATURES]
    p = float(model.predict_proba(frame)[0, 1])

    assert [f["name"] for f in stage["raw"]["fields"]] == FEATURES
    assert stage["raw"]["model"] == {"name": "credit-risk", "alias": "champion", "version": "7"}

    prep = model.named_steps["prep"]
    width = prep.transform(frame).shape[1]
    assert stage["transform"]["shape"] == [1, width]
    assert sum(e["n_columns"] for e in stage["transform"]["encoding"]) + len(NUMERIC) == width
    # active_index rebuilds exactly the one-hot vector the encoder produced (the site draws it).
    ohe = prep.named_transformers_["cat"]
    onehot = ohe.transform(frame[list(ohe.feature_names_in_)])
    onehot = onehot.toarray() if hasattr(onehot, "toarray") else onehot
    rebuilt: list[float] = []
    for e in stage["transform"]["encoding"]:
        cells = [0.0] * e["n_columns"]
        if e["active_index"] is not None:
            cells[e["active_index"]] = 1.0
        rebuilt += cells
    assert rebuilt == [float(v) for v in onehot[0]]

    scaler = prep.named_transformers_["num"]
    for i, row in enumerate(stage["normalize"]["rows"]):
        assert row["name"] == NUMERIC[i]
        assert row["after"] == pytest.approx((applicant[row["name"]] - scaler.mean_[i]) / scaler.scale_[i], abs=1e-3)

    assert stage["model"]["probability"] == pytest.approx(p, abs=1e-4)
    assert stage["model"]["threshold"] == 0.5
    assert stage["model"]["trees"] == model.named_steps["clf"].n_iter_
    assert 1 / (1 + np.exp(-stage["model"]["log_odds"])) == pytest.approx(p, abs=1e-3)
    assert stage["fit"]["rows_train"] == 750 and 0.6 < stage["fit"]["holdout_auc"] < 0.95
    assert all(d["ms"] >= 0 for _, d in events if "ms" in d)

    result = events[-1][1]
    assert result["default_probability"] == pytest.approx(p, abs=1e-4)
    assert result["decision"] == ("review" if p >= 0.5 else "approve")


def test_impacts_are_the_probability_change_when_swapping_for_the_typical_value(
    models: dict[str, Any], applicant: dict[str, Any]
) -> None:
    model = models["gradient_boosting"]
    ref = explain.reference()
    stage = {d["id"]: d for e, d in run(model, applicant, live_fit=False) if e == "stage"}
    assert "fit" not in stage
    base = float(model.predict_proba(pd.DataFrame([applicant])[FEATURES])[0, 1])
    impacts = stage["explain"]["features"]
    assert impacts and [abs(f["impact"]) for f in impacts] == sorted((abs(f["impact"]) for f in impacts), reverse=True)
    for f in impacts:
        swapped = {**applicant, f["name"]: ref.typical(f["name"])}
        expected = base - float(model.predict_proba(pd.DataFrame([swapped])[FEATURES])[0, 1])
        assert f["impact"] == pytest.approx(expected, abs=1e-4) and f["reference"] == ref.typical(f["name"])


def test_encoding_shows_infrequent_and_unseen_categories(models: dict[str, Any], applicant: dict[str, Any]) -> None:
    # The synthetic training data never has "domestic appliance": the encoder never saw it.
    model = models["logistic_regression"]
    events = run(model, {**applicant, "purpose": "domestic appliance"}, live_fit=False)
    stage = {d["id"]: d for e, d in events if e == "stage"}
    purpose = next(e for e in stage["transform"]["encoding"] if e["name"] == "purpose")
    assert purpose["seen_in_training"] is False
    assert purpose["column"] in {None, "purpose=infrequent_sklearn"}
    housing = next(e for e in stage["transform"]["encoding"] if e["name"] == "housing")
    assert housing == {
        "name": "housing",
        "value": applicant["housing"],
        "column": f"housing={applicant['housing']}",
        "infrequent": False,
        "seen_in_training": True,
        "n_columns": 3,
        "active_index": ["free", "own", "rent"].index(applicant["housing"]),
    }
    assert stage["model"]["estimator"] == "LogisticRegression" and "coefficients" in stage["model"]


def test_works_with_a_model_that_is_not_our_pipeline(applicant: dict[str, Any]) -> None:
    class Opaque:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return np.tile([0.2, 0.8], (len(frame), 1))

    events = run(Opaque(), applicant, live_fit=False)
    stage = {d["id"]: d for e, d in events if e == "stage"}
    assert stage["transform"]["encoding"] == [] and stage["normalize"]["rows"] == []
    assert stage["explain"]["features"] == [] and events[-1][1]["decision"] == "review"
