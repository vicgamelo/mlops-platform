import json

import pandas as pd
from mlflow import MlflowClient

from mlops_platform.data import TARGET
from mlops_platform.promote import CHAMPION, PREVIOUS, Gates, evaluate_and_promote, rollback
from mlops_platform.train import train


def test_train_registers_candidate_with_lineage(tracking: str, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    pool, _ = data
    result = train(pool, model_name="credit")
    client = MlflowClient()
    mv = client.get_model_version("credit", result.version)
    assert mv.tags["status"] == "candidate"
    assert mv.tags["estimator"] == result.best
    assert json.loads(mv.tags["holdout"])["roc_auc"] == result.holdout["roc_auc"]
    run = client.get_run(result.run_id)
    assert run.data.metrics["holdout_roc_auc"] > 0.7
    assert set(result.cv_auc) == {"logistic_regression", "gradient_boosting"}
    artifacts = {a.path for a in client.list_artifacts(result.run_id, "evaluation")}
    assert artifacts == {"evaluation/summary.json", "evaluation/calibration.json"}


def test_promotion_gates_and_rollback(tracking: str, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    pool, evaluation = data
    client = MlflowClient()

    v1 = train(pool, model_name="credit").version
    first = evaluate_and_promote("credit", v1, evaluation)
    assert first.promoted and first.champion is None
    assert evaluate_and_promote("credit", v1, evaluation).reasons == ["challenger is already the champion"]
    assert str(client.get_model_version_by_alias("credit", CHAMPION).version) == v1

    # Same data, same model: no meaningful gain, so the champion stays.
    v2 = train(pool, model_name="credit").version
    second = evaluate_and_promote("credit", v2, evaluation)
    assert not second.promoted
    assert any("does not beat champion" in r for r in second.reasons)
    assert client.get_model_version("credit", v2).tags["status"] == "rejected"

    # Garbage labels: fails the absolute AUC floor.
    noisy = pool.copy()
    noisy[TARGET] = noisy[TARGET].sample(frac=1, random_state=0).to_numpy()
    v3 = train(noisy, model_name="credit").version
    third = evaluate_and_promote("credit", v3, evaluation, Gates(min_auc_gain=-1))
    assert not third.promoted and any("below floor" in r for r in third.reasons)

    # Relaxed gain gate promotes v2, keeping v1 as previous; then roll back.
    fourth = evaluate_and_promote("credit", v2, evaluation, Gates(min_auc_gain=-1))
    assert fourth.promoted
    assert str(client.get_model_version_by_alias("credit", PREVIOUS).version) == v1
    assert rollback("credit") == (v1, v2)
    assert str(client.get_model_version_by_alias("credit", CHAMPION).version) == v1


def test_dry_run_changes_nothing(tracking: str, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    pool, evaluation = data
    v1 = train(pool, model_name="credit").version
    decision = evaluate_and_promote("credit", v1, evaluation, dry_run=True)
    assert decision.promoted
    assert MlflowClient().get_registered_model("credit").aliases == {}
