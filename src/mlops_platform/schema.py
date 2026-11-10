"""Input contract of the credit model: the credit-g domain for each feature, plus checks that
explain *what* is wrong with a value (wrong type, outside the domain, missing...) in PT and EN.

The categories are the ones declared in OpenML ``credit-g`` (v1). Anything else is refused instead
of being silently folded into the encoder's "infrequent" bucket.
"""

from __future__ import annotations

import difflib
import math
import numbers
from dataclasses import dataclass
from typing import Literal

from .data import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC

Lang = Literal["pt", "en"]

DOMAIN: dict[str, tuple[str, ...]] = {
    "checking_status": ("<0", "0<=X<200", ">=200", "no checking"),
    "credit_history": (
        "no credits/all paid",
        "all paid",
        "existing paid",
        "delayed previously",
        "critical/other existing credit",
    ),
    "purpose": (
        "new car",
        "used car",
        "furniture/equipment",
        "radio/tv",
        "domestic appliance",
        "repairs",
        "education",
        "retraining",
        "business",
        "other",
    ),
    "savings_status": ("<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"),
    "employment": ("unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"),
    "other_parties": ("none", "co applicant", "guarantor"),
    "property_magnitude": ("real estate", "life insurance", "car", "no known property"),
    "other_payment_plans": ("bank", "stores", "none"),
    "housing": ("rent", "own", "for free"),
    "job": ("unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"),
    "own_telephone": ("none", "yes"),
}


@dataclass(frozen=True)
class Bounds:
    kind: Literal["int", "float"]
    min: float
    max: float
    min_exclusive: bool = False


# Same bounds as the /predict API. The training data is narrower (see TRAIN_RANGE): values in between
# are accepted with a warning, because the model has never seen them.
BOUNDS: dict[str, Bounds] = {
    "duration": Bounds("int", 1, 120),
    "credit_amount": Bounds("float", 0, 1_000_000, min_exclusive=True),
    "installment_commitment": Bounds("int", 1, 4),
    "residence_since": Bounds("int", 1, 4),
    "age": Bounds("int", 18, 100),
    "existing_credits": Bounds("int", 0, 10),
    "num_dependents": Bounds("int", 0, 10),
}
TRAIN_RANGE: dict[str, tuple[float, float]] = {
    "duration": (4, 72),
    "credit_amount": (250, 18424),
    "installment_commitment": (1, 4),
    "residence_since": (1, 4),
    "age": (19, 75),
    "existing_credits": (1, 4),
    "num_dependents": (1, 2),
}
assert set(DOMAIN) == set(CATEGORICAL) and set(BOUNDS) == set(NUMERIC) == set(TRAIN_RANGE)

MAX_TEXT = 64  # longest category is 30 characters

Value = str | int | float


@dataclass(frozen=True)
class Problem:
    """One thing wrong (or worth a warning) with a field. ``code`` is stable; ``message`` is for people."""

    column: str
    code: str
    message: str
    value: str | None = None
    severity: Literal["error", "warning"] = "error"

    def as_dict(self) -> dict[str, str | None]:
        return {
            "column": self.column,
            "code": self.code,
            "message": self.message,
            "value": self.value,
            "severity": self.severity,
        }


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


MESSAGES: dict[str, dict[Lang, str]] = {
    "missing": {"pt": "valor faltando: preencha {hint}", "en": "missing value: fill in {hint}"},
    "not_integer": {
        "pt": "tipo errado: esperado número inteiro (ex.: {example}), veio {got!r}",
        "en": "wrong type: expected a whole number (e.g. {example}), got {got!r}",
    },
    "not_number": {
        "pt": "tipo errado: esperado número (ex.: {example}, com ponto decimal), veio {got!r}",
        "en": "wrong type: expected a number (e.g. {example}, dot as decimal separator), got {got!r}",
    },
    "decimal_comma": {
        "pt": "use ponto como separador decimal e nada de separador de milhar: {got!r} → {fixed}",
        "en": "use a dot as decimal separator and no thousands separator: {got!r} → {fixed}",
    },
    "out_of_range": {
        "pt": "fora do intervalo: {got} não está entre {lo} e {hi}",
        "en": "out of range: {got} is not between {lo} and {hi}",
    },
    "out_of_domain": {
        "pt": "fora do domínio: {got!r} não é uma categoria do credit-g{suggestion}. Aceitos: {allowed}",
        "en": "not in the domain: {got!r} is not a credit-g category{suggestion}. Allowed: {allowed}",
    },
    "too_long": {"pt": "valor longo demais (máx. {max} caracteres)", "en": "value too long (max {max} characters)"},
    "outside_training": {
        "pt": "aceito, mas fora do que o modelo viu no treino ({lo} a {hi}): trate o score com cautela",
        "en": "accepted, but outside what the model saw in training ({lo} to {hi}): treat the score with care",
    },
    "type_coerced": {"pt": "texto {got!r} convertido para número", "en": "text {got!r} converted to a number"},
}
SUGGESTION: dict[Lang, str] = {"pt": " (você quis dizer {0!r}?)", "en": " (did you mean {0!r}?)"}


def message(code: str, lang: Lang, **kw: object) -> str:
    return MESSAGES[code][lang].format(**kw)


def _hint(column: str, lang: Lang) -> str:
    if column in DOMAIN:
        return ("uma destas: " if lang == "pt" else "one of: ") + ", ".join(DOMAIN[column])
    b = BOUNDS[column]
    kind = ("inteiro" if lang == "pt" else "integer") if b.kind == "int" else ("número" if lang == "pt" else "number")
    low = f">{_fmt(b.min)}" if b.min_exclusive else _fmt(b.min)
    return f"{kind} {low}–{_fmt(b.max)}"


def check_value(column: str, raw: Value | None, lang: Lang = "pt") -> tuple[Value | None, list[Problem]]:
    """Casts and checks one value. Returns (clean value or None, problems).

    ``None`` or blank means missing. Numbers may come as text (CSV, forms); that is reported as a
    coercion so the live pipeline can show it, not as an error."""
    if column not in BOUNDS and column not in DOMAIN:
        raise KeyError(column)
    shown = None if raw is None else str(raw)[:MAX_TEXT]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, [Problem(column, "missing", message("missing", lang, hint=_hint(column, lang)))]
    if isinstance(raw, str) and len(raw) > MAX_TEXT:
        return None, [Problem(column, "too_long", message("too_long", lang, max=MAX_TEXT), shown)]

    if column in DOMAIN:
        text = raw if isinstance(raw, str) else _fmt(raw) if not isinstance(raw, bool) else str(raw)
        text = text.strip()
        if text in DOMAIN[column]:
            return text, []
        close = difflib.get_close_matches(text.lower(), [d.lower() for d in DOMAIN[column]], n=1, cutoff=0.6)
        suggestion = ""
        if close:
            original = next(d for d in DOMAIN[column] if d.lower() == close[0])
            suggestion = SUGGESTION[lang].format(original)
        msg = message("out_of_domain", lang, got=text, suggestion=suggestion, allowed=", ".join(DOMAIN[column]))
        return None, [Problem(column, "out_of_domain", msg, shown)]

    b = BOUNDS[column]
    problems: list[Problem] = []
    number: float
    if isinstance(raw, bool):
        return None, [Problem(column, "not_number", message("not_number", lang, example=_example(b), got=raw), shown)]
    if isinstance(raw, numbers.Real):
        number = float(raw)
    else:
        text = str(raw).strip()
        try:
            number = float(text)
        except ValueError:
            fixed = _decimal_comma(text)
            if fixed is not None:
                msg = message("decimal_comma", lang, got=text, fixed=fixed)
                return None, [Problem(column, "decimal_comma", msg, shown)]
            code = "not_integer" if b.kind == "int" else "not_number"
            return None, [Problem(column, code, message(code, lang, example=_example(b), got=text), shown)]
        problems.append(Problem(column, "type_coerced", message("type_coerced", lang, got=text), shown, "warning"))
    if not math.isfinite(number):
        code = "not_integer" if b.kind == "int" else "not_number"
        return None, [Problem(column, code, message(code, lang, example=_example(b), got=shown), shown)]
    if b.kind == "int" and not number.is_integer():
        msg = message("not_integer", lang, example=_example(b), got=shown)
        return None, [Problem(column, "not_integer", msg, shown)]
    low_bad = number <= b.min if b.min_exclusive else number < b.min
    if low_bad or number > b.max:
        lo = f">{_fmt(b.min)}" if b.min_exclusive else _fmt(b.min)
        msg = message("out_of_range", lang, got=_fmt(number), lo=lo, hi=_fmt(b.max))
        return None, [Problem(column, "out_of_range", msg, shown)]
    lo_t, hi_t = TRAIN_RANGE[column]
    if not lo_t <= number <= hi_t:
        msg = message("outside_training", lang, lo=_fmt(lo_t), hi=_fmt(hi_t))
        problems.append(Problem(column, "outside_training", msg, shown, "warning"))
    value: Value = int(number) if b.kind == "int" else number
    return value, problems


def _example(b: Bounds) -> str:
    return "24" if b.kind == "int" else "2500.50"


def _decimal_comma(text: str) -> str | None:
    """'1.500,50' or '2500,5' → the dotted form, when that is clearly what was meant."""
    if text.count(",") != 1:
        return None
    candidate = text.replace(".", "").replace(",", ".")
    try:
        float(candidate)
    except ValueError:
        return None
    return candidate


def errors(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == "error"]


__all__ = [
    "BOUNDS",
    "DOMAIN",
    "EXCLUDED",
    "FEATURES",
    "TRAIN_RANGE",
    "Lang",
    "Problem",
    "check_value",
    "errors",
    "message",
]
