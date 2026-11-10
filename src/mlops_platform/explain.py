"""The live pipeline behind the site demo: every step the champion runs on one applicant, made visible.

Steps, each one computed when it is emitted (``ms`` is the real time it took):

1. ``raw``: the fields as they arrived, with their JSON types.
2. ``transform``: text → number casts, missing values filled with the credit-g median/mode, and the
   one-hot encoding the champion's own ``OneHotEncoder`` produces (including the "infrequent" bucket).
3. ``normalize``: the champion's ``StandardScaler``: value before, the mean/std it learned in training,
   value after.
4. ``fit``: a live fit of a small baseline (the logistic regression candidate from ``train.py``) on
   credit-g. It is not the model that decides: it is there to show a real fit, which takes a few ms.
5. ``model``: inference with the ``@champion`` (log-odds and probability).
6. ``explain``: which features moved the probability the most, by swapping each one for the credit-g
   median/mode and scoring again (one batched ``predict_proba`` call).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .data import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, TARGET
from .schema import BOUNDS, Lang, Problem, Value, check_value, errors

Event = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class Reference:
    """credit-g (1000 rows) shipped with the package: the scoring container has no internet."""

    frame: pd.DataFrame
    median: dict[str, float]
    mode: dict[str, str]

    def typical(self, column: str) -> Value:
        return self.median[column] if column in NUMERIC else self.mode[column]

    def percentile(self, column: str, value: float) -> float:
        return float((self.frame[column] <= value).mean())


@cache
def reference() -> Reference:
    with resources.files("mlops_platform").joinpath("reference/credit_g.csv.gz").open("rb") as fh:
        frame = pd.read_csv(fh, compression="gzip")
    for col in CATEGORICAL:
        frame[col] = frame[col].astype(str)
    median = {c: float(frame[c].median()) for c in NUMERIC}
    for c in NUMERIC:
        if BOUNDS[c].kind == "int":
            median[c] = float(round(median[c]))
    mode = {c: str(frame[c].mode().iloc[0]) for c in CATEGORICAL}
    return Reference(frame, median, mode)


@dataclass
class Prepared:
    raw: dict[str, Any]
    clean: dict[str, Value]
    casts: list[dict[str, Any]] = field(default_factory=list)
    imputed: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[Problem] = field(default_factory=list)


def _json_type(v: object) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int | float):
        return "number"
    return "string" if isinstance(v, str) else type(v).__name__


def prepare(
    raw: Mapping[str, Any], lang: Lang = "pt", ref: Reference | None = None
) -> tuple[Prepared | None, list[Problem]]:
    """Validates one applicant. Missing values (null / absent / blank) are allowed and imputed.
    Returns (None, errors) when something can't be used."""
    ref = ref or reference()
    problems: list[Problem] = []
    unknown = sorted(set(raw) - set(FEATURES))
    for col in unknown:
        code = "protected_column" if col in EXCLUDED else "unknown_column"
        text = {
            ("protected_column", "pt"): "atributo protegido: o modelo não usa",
            ("protected_column", "en"): "protected attribute: the model does not use it",
            ("unknown_column", "pt"): "campo desconhecido",
            ("unknown_column", "en"): "unknown field",
        }[(code, lang)]
        problems.append(Problem(col[:64], code, text))
    prepared = Prepared(raw={c: raw.get(c) for c in FEATURES}, clean={})
    for col in FEATURES:
        value = raw.get(col)
        if not (value is None or isinstance(value, str | int | float)) or isinstance(value, bool):
            problems.append(Problem(col, "wrong_type", "tipo não suportado" if lang == "pt" else "unsupported type"))
            continue
        clean, found = check_value(col, value, lang)
        if found and found[0].code == "missing":
            typical = ref.typical(col)
            if col in NUMERIC and BOUNDS[col].kind == "int":
                typical = int(typical)
            prepared.clean[col] = typical
            prepared.imputed.append({"name": col, "value": typical, "strategy": "median" if col in NUMERIC else "mode"})
            continue
        problems.extend(errors(found))
        prepared.warnings.extend(p for p in found if p.severity == "warning" and p.code != "type_coerced")
        if clean is None:
            continue
        prepared.clean[col] = clean
        if col in NUMERIC:
            prepared.casts.append(
                {
                    "name": col,
                    "from": _json_type(value),
                    "to": BOUNDS[col].kind,
                    "raw": value,
                    "value": clean,
                    "coerced": isinstance(value, str) or (BOUNDS[col].kind == "int" and isinstance(value, float)),
                }
            )
    if problems:
        return None, problems
    return prepared, []


def _num(x: Any, digits: int = 6) -> float:
    return round(float(x), digits)


def _parts(model: Any) -> tuple[Any, Any] | None:
    """(ColumnTransformer, classifier) for the pipelines train.py produces; None for anything else."""
    steps = getattr(model, "named_steps", None)
    if not steps or "prep" not in steps or "clf" not in steps:
        return None
    return steps["prep"], steps["clf"]


def _dense(x: Any) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=float)


_fit_lock = threading.Lock()


def fit_baseline(ref: Reference, seed: int = 42) -> tuple[Any, dict[str, Any]]:
    """Fits the logistic regression candidate on 75% of credit-g, scores the other 25%.
    One fit at a time: it is cheap (tens of ms) but the VPS is shared."""
    from .train import candidates

    with _fit_lock:
        started = time.perf_counter()
        train_df, test_df = train_test_split(ref.frame, test_size=0.25, stratify=ref.frame[TARGET], random_state=seed)
        model = clone(candidates(seed)["logistic_regression"]).fit(train_df[FEATURES], train_df[TARGET])
        auc = roc_auc_score(test_df[TARGET], model.predict_proba(test_df[FEATURES])[:, 1])
        ms = (time.perf_counter() - started) * 1000
    clf = model.named_steps["clf"]
    return model, {
        "estimator": "LogisticRegression",
        "params": {"C": clf.C, "class_weight": clf.class_weight},
        "rows_train": len(train_df),
        "rows_test": len(test_df),
        "features_out": int(clf.coef_.shape[1]),
        "iterations": int(np.max(clf.n_iter_)),
        "holdout_auc": _num(auc, 4),
        "fit_ms": round(ms, 1),
    }


def run(
    model: Any,
    version: str,
    prepared: Prepared,
    *,
    model_name: str = "credit-risk",
    threshold: float = 0.5,
    ref: Reference | None = None,
    live_fit: bool = True,
    top: int = 6,
) -> Iterator[Event]:
    ref = ref or reference()
    clock = time.perf_counter

    t = clock()
    raw_fields = [{"name": c, "value": prepared.raw[c], "type": _json_type(prepared.raw[c])} for c in FEATURES]
    yield (
        "stage",
        {
            "id": "raw",
            "fields": raw_fields,
            "excluded": EXCLUDED,
            "model": {"name": model_name, "alias": "champion", "version": version},
            "ms": round((clock() - t) * 1000, 2),
        },
    )

    t = clock()
    frame = pd.DataFrame([prepared.clean])[FEATURES]
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col])
    parts = _parts(model)
    encoding: list[dict[str, Any]] = []
    shape: list[int] | None = None
    if parts:
        prep, _ = parts
        matrix = _dense(prep.transform(frame))
        shape = [int(matrix.shape[0]), int(matrix.shape[1])]
        ohe = prep.named_transformers_.get("cat")
        if ohe is not None and hasattr(ohe, "categories_"):
            # Read the active column off the encoder's own output, so what is shown is what the model gets.
            names = [str(n) for n in ohe.get_feature_names_out()]
            onehot = _dense(ohe.transform(frame[list(ohe.feature_names_in_)]))[0]
            for i, col in enumerate(ohe.feature_names_in_):
                value = str(prepared.clean[col])
                own = [j for j, n in enumerate(names) if n.startswith(f"{col}_")]
                active = [names[j][len(col) + 1 :] for j in own if onehot[j] == 1]
                label = active[0] if active else None
                encoding.append(
                    {
                        "name": str(col),
                        "value": value,
                        # None: a category the encoder never saw in training (all zeros).
                        "column": None if label is None else f"{col}={label}",
                        "infrequent": label == "infrequent_sklearn",
                        "seen_in_training": value in set(map(str, ohe.categories_[i])),
                        "n_columns": len(own),
                    }
                )
    yield (
        "stage",
        {
            "id": "transform",
            "casts": prepared.casts,
            "imputed": prepared.imputed,
            "encoding": encoding,
            "shape": shape,
            "warnings": [p.as_dict() for p in prepared.warnings],
            "ms": round((clock() - t) * 1000, 2),
        },
    )

    t = clock()
    rows: list[dict[str, Any]] = []
    scaler = parts[0].named_transformers_.get("num") if parts else None
    if scaler is not None and hasattr(scaler, "mean_"):
        for i, col in enumerate(scaler.feature_names_in_):
            before = float(frame[col].iloc[0])
            mean, std = float(scaler.mean_[i]), float(scaler.scale_[i])
            rows.append(
                {
                    "name": str(col),
                    "before": before,
                    "mean": _num(mean, 3),
                    "std": _num(std, 3),
                    "after": _num((before - mean) / std, 3),
                    "percentile": _num(ref.percentile(str(col), before), 3),
                }
            )
    yield (
        "stage",
        {
            "id": "normalize",
            "scaler": type(scaler).__name__ if scaler is not None else None,
            "rows": rows,
            "ms": round((clock() - t) * 1000, 2),
        },
    )

    baseline_p: float | None = None
    if live_fit:
        t = clock()
        baseline, info = fit_baseline(ref)
        baseline_p = float(baseline.predict_proba(frame)[0, 1])
        yield "stage", {"id": "fit", **info, "probability": _num(baseline_p, 4), "ms": round((clock() - t) * 1000, 1)}

    t = clock()
    p = float(model.predict_proba(frame)[0, 1])
    info_model: dict[str, Any] = {"estimator": type(model).__name__}
    if parts:
        prep, clf = parts
        info_model["estimator"] = type(clf).__name__
        if hasattr(clf, "decision_function"):
            info_model["log_odds"] = _num(clf.decision_function(prep.transform(frame))[0], 4)
        if hasattr(clf, "n_iter_") and type(clf).__name__.startswith("HistGradientBoosting"):
            info_model["trees"] = int(clf.n_iter_)
            info_model["params"] = {"learning_rate": clf.learning_rate, "max_depth": clf.max_depth}
        elif hasattr(clf, "coef_"):
            info_model["coefficients"] = int(clf.coef_.shape[1])
    yield (
        "stage",
        {
            "id": "model",
            **info_model,
            "probability": _num(p, 4),
            "version": version,
            "ms": round((clock() - t) * 1000, 2),
        },
    )

    t = clock()
    variants = [prepared.clean]
    for col in FEATURES:
        variants.append({**prepared.clean, col: ref.typical(col)})
    batch = pd.DataFrame(variants)[FEATURES]
    probs = model.predict_proba(batch)[:, 1]
    impacts: list[dict[str, Any]] = []
    for i, col in enumerate(FEATURES, start=1):
        impact = float(probs[0] - probs[i])
        if abs(impact) >= 1e-4:
            impacts.append(
                {"name": col, "value": prepared.clean[col], "reference": ref.typical(col), "impact": _num(impact, 4)}
            )
    impacts.sort(key=lambda r: abs(r["impact"]), reverse=True)
    yield (
        "stage",
        {
            "id": "explain",
            "method": "substitution",
            "features": impacts[:top],
            "ms": round((clock() - t) * 1000, 2),
        },
    )

    yield (
        "result",
        {
            "default_probability": _num(p, 4),
            "decision": "review" if p >= threshold else "approve",
            "threshold": threshold,
            "model_version": version,
            "baseline_probability": None if baseline_p is None else _num(baseline_p, 4),
            "top_features": impacts[:3],
        },
    )
