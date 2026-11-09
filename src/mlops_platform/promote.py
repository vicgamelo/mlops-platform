"""Champion / challenger promotion with explicit gates.

The challenger and the current champion are scored on the *same* evaluation
set (fresh, labeled data), so the comparison is apples to apples. Promotion
moves the ``champion`` alias; the previous champion keeps the ``previous`` alias,
so a rollback is a single alias swap and the serving layer picks it up live.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .data import FEATURES, TARGET
from .train import age_auc_gap, classification_metrics

CHAMPION, PREVIOUS = "champion", "previous"


@dataclass(frozen=True)
class Gates:
    min_auc: float = 0.70  # absolute floor
    min_auc_gain: float = 0.005  # must beat the champion by this much
    max_brier_increase: float = 0.01  # calibration may not get meaningfully worse
    max_age_auc_gap: float = 0.15  # fairness guardrail


@dataclass
class Decision:
    promoted: bool
    challenger: str
    champion: str | None
    reasons: list[str] = field(default_factory=list)
    challenger_metrics: dict[str, float] = field(default_factory=dict)
    champion_metrics: dict[str, float] = field(default_factory=dict)


def _score(model_uri: str, eval_df: pd.DataFrame) -> dict[str, float]:
    model = mlflow.sklearn.load_model(model_uri)
    proba = model.predict_proba(eval_df[FEATURES])[:, 1]
    metrics = classification_metrics(eval_df[TARGET].to_numpy(), proba)
    gap = age_auc_gap(eval_df, proba)
    if gap is not None:
        metrics["age_auc_gap"] = gap
    return metrics


def current_champion(client: MlflowClient, name: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, CHAMPION).version)
    except MlflowException:
        return None


def evaluate_and_promote(
    name: str, challenger: str, eval_df: pd.DataFrame, gates: Gates | None = None, *, dry_run: bool = False
) -> Decision:
    gates = gates or Gates()
    challenger = str(challenger)
    client = MlflowClient()
    champion = current_champion(client, name)
    if champion == challenger:
        return Decision(False, challenger, champion, ["challenger is already the champion"])

    chal = _score(f"models:/{name}/{challenger}", eval_df)
    champ = _score(f"models:/{name}/{champion}", eval_df) if champion else {}
    reasons: list[str] = []

    if chal["roc_auc"] < gates.min_auc:
        reasons.append(f"roc_auc {chal['roc_auc']:.4f} below floor {gates.min_auc}")
    if chal.get("age_auc_gap", 0.0) > gates.max_age_auc_gap:
        reasons.append(f"age_auc_gap {chal['age_auc_gap']:.4f} above {gates.max_age_auc_gap}")
    if champ:
        if chal["roc_auc"] < champ["roc_auc"] + gates.min_auc_gain:
            reasons.append(
                f"roc_auc {chal['roc_auc']:.4f} does not beat champion {champ['roc_auc']:.4f} + {gates.min_auc_gain}"
            )
        if chal["brier"] > champ["brier"] + gates.max_brier_increase:
            reasons.append(f"brier {chal['brier']:.4f} worse than champion {champ['brier']:.4f}")

    decision = Decision(not reasons, challenger, champion, reasons or ["all gates passed"], chal, champ)
    if dry_run:
        return decision

    client.set_model_version_tag(name, challenger, "status", "champion" if decision.promoted else "rejected")
    client.set_model_version_tag(name, challenger, "gate_result", "; ".join(decision.reasons)[:5000])
    if decision.promoted:
        if champion:
            client.set_registered_model_alias(name, PREVIOUS, champion)
            client.set_model_version_tag(name, champion, "status", "retired")
        client.set_registered_model_alias(name, CHAMPION, challenger)
    return decision


def rollback(name: str) -> tuple[str, str]:
    """Swap ``champion`` and ``previous``. Returns (new_champion, demoted)."""
    client = MlflowClient()
    champion = str(client.get_model_version_by_alias(name, CHAMPION).version)
    previous = str(client.get_model_version_by_alias(name, PREVIOUS).version)
    client.set_registered_model_alias(name, CHAMPION, previous)
    client.set_registered_model_alias(name, PREVIOUS, champion)
    client.set_model_version_tag(name, previous, "status", "champion")
    client.set_model_version_tag(name, champion, "status", "rolled_back")
    return previous, champion


def as_dict(decision: Decision) -> dict[str, object]:
    return asdict(decision)
