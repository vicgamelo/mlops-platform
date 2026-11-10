import math

import pytest

from mlops_platform.data import CATEGORICAL, FEATURES, load_credit
from mlops_platform.schema import DOMAIN, TRAIN_RANGE, check_value


def codes(column: str, raw: object) -> list[str]:
    return [p.code for p in check_value(column, raw)[1]]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("column", "raw", "expected"),
    [
        ("age", None, ["missing"]),
        ("age", "  ", ["missing"]),
        ("purpose", "", ["missing"]),
        ("age", "trinta", ["not_integer"]),
        ("age", 30.5, ["not_integer"]),
        ("age", "30.5", ["not_integer"]),
        ("credit_amount", "abc", ["not_number"]),
        ("credit_amount", "2500,50", ["decimal_comma"]),
        ("credit_amount", "1.500,50", ["decimal_comma"]),
        ("age", 7, ["out_of_range"]),
        ("age", 101, ["out_of_range"]),
        ("credit_amount", 0, ["out_of_range"]),  # exclusive lower bound, like /predict (gt=0)
        ("duration", 96, ["outside_training"]),  # valid for the API, never seen in training
        ("purpose", "carro", ["out_of_domain"]),
        ("purpose", "x" * 65, ["too_long"]),
        ("age", True, ["not_number"]),
        ("age", math.nan, ["not_integer"]),
        ("credit_amount", math.inf, ["not_number"]),
        ("credit_amount", "inf", ["not_number"]),
        ("age", "30", ["type_coerced"]),
        ("age", 30, []),
        ("age", 30.0, []),
        ("purpose", "radio/tv", []),
    ],
)
def test_check_value(column: str, raw: object, expected: list[str]) -> None:
    assert codes(column, raw) == expected


def test_casts_to_the_model_types() -> None:
    assert check_value("age", "30")[0] == 30 and isinstance(check_value("age", "30")[0], int)
    assert check_value("credit_amount", "2500.5")[0] == 2500.5
    assert check_value("purpose", " radio/tv ")[0] == "radio/tv"


def test_domain_errors_suggest_the_closest_category_in_both_languages() -> None:
    _, [pt] = check_value("housing", "Own", "pt")
    _, [en] = check_value("housing", "Own", "en")
    assert "'own'" in pt.message and "você quis dizer" in pt.message and "Aceitos" in pt.message
    assert "did you mean 'own'" in en.message


def test_messages_say_how_to_fix() -> None:
    _, [p] = check_value("credit_amount", "2500,50", "pt")
    assert "2500.50" in p.message
    _, [p] = check_value("age", 7, "en")
    assert "between 18 and 100" in p.message


def test_domain_matches_the_real_dataset() -> None:
    """Every category and numeric range in credit-g (OpenML cache) fits the declared contract."""
    try:
        df = load_credit()
    except Exception:  # offline CI without the cache
        pytest.skip("credit-g not available")
    for col in CATEGORICAL:
        assert set(df[col].unique()) <= set(DOMAIN[col]), col
    for col, (lo, hi) in TRAIN_RANGE.items():
        assert (df[col].min(), df[col].max()) == (lo, hi), col
    for _, row in df[FEATURES].head(200).iterrows():
        for col in FEATURES:
            assert [p for p in check_value(col, row[col])[1] if p.severity == "error"] == [], (col, row[col])


def test_missing_hint_shows_the_exclusive_bound() -> None:
    _, [p] = check_value("credit_amount", "", "pt")
    assert "número >0–1000000" in p.message
    _, [p] = check_value("age", None, "en")
    assert "integer 18–100" in p.message
