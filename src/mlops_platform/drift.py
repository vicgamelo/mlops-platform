"""Data drift with the Population Stability Index (PSI).

PSI = sum((actual% - expected%) * ln(actual% / expected%)) over bins.
Rule of thumb used in credit scoring: < 0.1 stable, 0.1-0.25 moderate, > 0.25 major.
Numeric features are binned by reference quantiles; categoricals by category,
with unseen categories grouped into their own bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

EPS = 1e-4
Severity = Literal["stable", "moderate", "major"]


def severity(psi: float) -> Severity:
    if psi < 0.1:
        return "stable"
    return "moderate" if psi < 0.25 else "major"


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference.dropna(), edges)[0] / max(reference.notna().sum(), 1)
    cur = np.histogram(current.dropna(), edges)[0] / max(current.notna().sum(), 1)
    return _psi(ref, cur)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.astype(str).value_counts(normalize=True)
    cur = current.astype(str).value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    return _psi(ref.reindex(cats, fill_value=0).to_numpy(), cur.reindex(cats, fill_value=0).to_numpy())


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    severity: Severity


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> list[FeatureDrift]:
    out = [FeatureDrift(c, round(p := psi_numeric(reference[c], current[c]), 4), severity(p)) for c in numeric]
    out += [FeatureDrift(c, round(p := psi_categorical(reference[c], current[c]), 4), severity(p)) for c in categorical]
    return sorted(out, key=lambda d: -d.psi)
