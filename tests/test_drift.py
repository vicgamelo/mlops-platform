import numpy as np
import pandas as pd
import pytest

from mlops_platform.data import CATEGORICAL, NUMERIC, synthetic_credit
from mlops_platform.drift import drift_report, psi_categorical, psi_numeric, severity


def test_same_distribution_is_stable() -> None:
    a, b = synthetic_credit(2000, seed=1), synthetic_credit(2000, seed=2)
    report = drift_report(a, b, NUMERIC, CATEGORICAL)
    assert all(r.severity == "stable" for r in report), report[:3]


def test_shift_is_detected_and_ranked_first() -> None:
    report = drift_report(
        synthetic_credit(2000, seed=1), synthetic_credit(2000, seed=2, shift=1.0), NUMERIC, CATEGORICAL
    )
    assert report[0].feature in {"credit_amount", "duration"}
    assert report[0].severity == "major"


def test_psi_known_value() -> None:
    # (0.25 - 0.5) * ln(0.25 / 0.5) + (0.75 - 0.5) * ln(0.75 / 0.5) = 0.25 * ln(3)
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 25 + ["b"] * 75)
    assert psi_categorical(ref, cur) == pytest.approx(0.25 * np.log(3))


def test_unseen_category_and_constant_feature() -> None:
    assert psi_categorical(pd.Series(["a", "b"] * 50), pd.Series(["a", "b", "c"] * 50)) > 0.1
    assert psi_numeric(pd.Series([1.0] * 100), pd.Series([5.0] * 100)) == 0.0


def test_severity_thresholds() -> None:
    assert [severity(x) for x in (0.05, 0.1, 0.3)] == ["stable", "moderate", "major"]
