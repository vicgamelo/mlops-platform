"""Train candidates, log everything to MLflow, register the best one.

One parent run per training job, one nested run per candidate. The best candidate
(by cross-validated ROC AUC on the training split) is evaluated on the holdout,
logged with signature + input example, and registered as a new model version
tagged ``status=candidate``. Promotion is a separate, gated step (see promote.py).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.data.pandas_dataset
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, TARGET

# MLflow serializes sklearn models with skops, which refuses to load types that
# can be abused by a crafted file. We only trust what this pipeline itself
# produces, and the registry is write-restricted to the training job.
SKOPS_TRUSTED = {
    "gradient_boosting": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
}


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5), CATEGORICAL),
        ]
    )


def candidates(seed: int = 42) -> dict[str, BaseEstimator]:
    return {
        "logistic_regression": Pipeline(
            [("prep", preprocessor()), ("clf", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", preprocessor()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05, max_depth=3, max_iter=300, l2_regularization=1.0, random_state=seed
                    ),
                ),
            ]
        ),
    }


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1": float(f1_score(y_true, proba >= threshold)),
    }


def age_auc_gap(df: pd.DataFrame, proba: np.ndarray, cutoff: int = 25) -> float | None:
    """|AUC(age < cutoff) - AUC(age >= cutoff)|: a simple fairness signal to monitor."""
    young = (df["age"] < cutoff).to_numpy()
    y = df[TARGET].to_numpy()
    groups = [young, ~young]
    if any(len(np.unique(y[g])) < 2 for g in groups):
        return None
    return float(abs(roc_auc_score(y[young], proba[young]) - roc_auc_score(y[~young], proba[~young])))


@dataclass
class TrainResult:
    run_id: str
    model_name: str
    version: str
    best: str
    holdout: dict[str, float]
    cv_auc: dict[str, float]


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(
    df: pd.DataFrame,
    *,
    experiment: str = "credit-risk",
    model_name: str = "credit-risk-brl",
    seed: int = 42,
    test_size: float = 0.25,
    dataset_name: str = "credit-g-brl",
) -> TrainResult:
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    X_train, y_train = train_df[FEATURES], train_df[TARGET].to_numpy()
    X_test, y_test = test_df[FEATURES], test_df[TARGET].to_numpy()

    mlflow.set_experiment(experiment)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    with mlflow.start_run(run_name="train") as parent:
        mlflow.set_tags({"git_sha": _git_sha() or "unknown", "dataset": dataset_name})
        mlflow.log_params(
            {
                "rows": len(df),
                "default_rate": round(float(df[TARGET].mean()), 4),
                "test_size": test_size,
                "seed": seed,
                "excluded_features": ",".join(EXCLUDED),
            }
        )
        mlflow.log_input(
            mlflow.data.pandas_dataset.from_pandas(train_df, name=f"{dataset_name}-train", targets=TARGET), "training"
        )

        cv_auc: dict[str, float] = {}
        for name, estimator in candidates(seed).items():
            with mlflow.start_run(run_name=name, nested=True):
                scores = cross_val_score(clone(estimator), X_train, y_train, cv=cv, scoring="roc_auc")
                cv_auc[name] = float(scores.mean())
                mlflow.log_params({"estimator": name, **_flat_params(estimator)})
                mlflow.log_metrics({"cv_roc_auc": cv_auc[name], "cv_roc_auc_std": float(scores.std())})

        best = max(cv_auc, key=lambda k: cv_auc[k])
        with mlflow.start_run(run_name=f"{best}-final", nested=True) as final:
            model = clone(candidates(seed)[best]).fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            holdout = classification_metrics(y_test, proba)
            gap = age_auc_gap(test_df, proba)
            mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout.items()})
            if gap is not None:
                mlflow.log_metric("holdout_age_auc_gap", gap)
            mlflow.log_dict({"cv_roc_auc": cv_auc, "holdout": holdout, "features": FEATURES}, "evaluation/summary.json")
            mlflow.log_dict(_reliability_table(y_test, proba), "evaluation/calibration.json")
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=infer_signature(X_test, proba),
                input_example=X_test.head(3),
                pyfunc_predict_fn="predict_proba",
                skops_trusted_types=SKOPS_TRUSTED.get(best),
            )

        version = mlflow.register_model(info.model_uri, model_name)
        client = MlflowClient()
        client.set_model_version_tag(model_name, str(version.version), "status", "candidate")
        client.set_model_version_tag(model_name, str(version.version), "estimator", best)
        client.set_model_version_tag(model_name, str(version.version), "training_run", parent.info.run_id)
        client.set_model_version_tag(model_name, str(version.version), "holdout", json.dumps(holdout))
        return TrainResult(final.info.run_id, model_name, str(version.version), best, holdout, cv_auc)


def _flat_params(estimator: BaseEstimator) -> dict[str, Any]:
    params = estimator.get_params(deep=True)
    return {k: v for k, v in params.items() if k.startswith("clf__") and isinstance(v, int | float | str | bool)}


def _reliability_table(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> dict[str, list[float] | list[int]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(proba, edges) - 1, 0, bins - 1)
    mean_pred, frac_pos, count = [], [], []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            mean_pred.append(round(float(proba[mask].mean()), 4))
            frac_pos.append(round(float(y[mask].mean()), 4))
            count.append(int(mask.sum()))
    return {"mean_predicted": mean_pred, "fraction_positive": frac_pos, "count": count}
