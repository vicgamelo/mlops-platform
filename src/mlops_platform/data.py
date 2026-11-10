"""Credit risk dataset (Statlog German Credit, OpenML ``credit-g``), in Brazilian reais.

Target: ``default`` = 1 when the applicant was a bad credit risk.

credit-g comes from a 1970s German bank, in Deutsche Marks (DM). Everything money-related is converted
to reais (R$) before training, at ``BRL_PER_DM``, so the model, the API and the demo speak reais end to
end: ``credit_amount`` is multiplied by the rate and the DM brackets of ``checking_status`` and
``savings_status`` become R$ brackets ("0<=X<200" -> "0<=X<1940"). Scaling a feature does not change
what the model learns (the scaler and the trees are scale-invariant); only the units change.

``personal_status`` (which encodes sex) and ``foreign_worker`` are deliberately
left out of the features: they are protected attributes and should not drive a
credit decision. ``age`` stays, but the training pipeline reports the model's
performance gap between age groups so it can be monitored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "default"
NUMERIC = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]
CATEGORICAL = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
]
EXCLUDED = ["personal_status", "foreign_worker"]
FEATURES = NUMERIC + CATEGORICAL

# Reais per DM of 1973-75, by purchasing power: 1 EUR = 1.95583 DM (fixed parity), German prices rose
# about 3.25x since 1975 and the euro is near R$ 5.85, so a DM from back then is worth about R$ 9.70.
BRL_PER_DM = 9.70
# DM brackets of credit-g -> the same brackets in reais.
MONEY_CATEGORIES: dict[str, dict[str, str]] = {
    "checking_status": {"0<=X<200": "0<=X<1940", ">=200": ">=1940"},
    "savings_status": {
        "<100": "<970",
        "100<=X<500": "970<=X<4850",
        "500<=X<1000": "4850<=X<9700",
        ">=1000": ">=9700",
    },
}


def to_reais(df: pd.DataFrame) -> pd.DataFrame:
    """credit-g money columns, from DM to reais. Category labels outside the money brackets are kept."""
    out = df.copy()
    out["credit_amount"] = (pd.to_numeric(out["credit_amount"]) * BRL_PER_DM).round(2)
    for col, mapping in MONEY_CATEGORIES.items():
        out[col] = out[col].astype(str).replace(mapping)
    return out


def load_credit(cache_dir: str | Path = "~/.cache/mlops-platform") -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    raw = fetch_openml("credit-g", version=1, as_frame=True, data_home=str(Path(cache_dir).expanduser())).frame
    return prepare(raw)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df[TARGET] = (df.pop("class").astype(str) == "bad").astype(int)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col])
    return to_reais(df[[*FEATURES, TARGET]])


def synthetic_credit(n: int = 600, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    """Offline stand-in with the same schema (in reais), used by tests and demos. ``shift``
    moves the numeric distributions to simulate drift. Generated on the DM scale of credit-g and
    converted with ``to_reais``, like the real data."""
    rng = np.random.default_rng(seed)
    cats = {
        "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
        "credit_history": ["critical/other existing credit", "existing paid", "delayed previously", "all paid"],
        "purpose": ["new car", "used car", "furniture/equipment", "radio/tv", "education", "business"],
        "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
        "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
        "other_parties": ["none", "guarantor", "co applicant"],
        "property_magnitude": ["real estate", "life insurance", "car", "no known property"],
        "other_payment_plans": ["none", "bank", "stores"],
        "housing": ["own", "rent", "for free"],
        "job": ["skilled", "unskilled resident", "high qualif/self emp/mgmt", "unemp/unskilled non res"],
        "own_telephone": ["none", "yes"],
    }
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in cats.items()})
    df["duration"] = rng.integers(4, 72, n) + int(shift * 12)
    df["credit_amount"] = (rng.lognormal(7.9, 0.75, n) * (1 + shift)).round()
    df["installment_commitment"] = rng.integers(1, 5, n)
    df["residence_since"] = rng.integers(1, 5, n)
    df["age"] = rng.integers(19, 75, n)
    df["existing_credits"] = rng.integers(1, 4, n)
    df["num_dependents"] = rng.integers(1, 3, n)
    logit = (
        -1.3
        + 1.6 * (df["checking_status"] == "<0")
        - 1.4 * (df["checking_status"] == "no checking")
        + 0.05 * (df["duration"] - 20)
        + 0.00015 * (df["credit_amount"] - 3000)
        + 1.0 * (df["credit_history"] == "delayed previously")
        - 1.0 * (df["savings_status"] == ">=1000")
        - 0.015 * (df["age"] - 35)
    )
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return to_reais(df[[*FEATURES, TARGET]])


def split_eval(df: pd.DataFrame, fraction: float = 0.2, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve out a fixed evaluation set that training never sees. Promotion
    decisions are made on it, so every challenger is judged on the same data."""
    from sklearn.model_selection import train_test_split

    pool, evaluation = train_test_split(df, test_size=fraction, stratify=df[TARGET], random_state=seed)
    return pool.reset_index(drop=True), evaluation.reset_index(drop=True)
