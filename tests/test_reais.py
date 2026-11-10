"""O credit-g vira reais antes do treino: valor pedido vezes a taxa e as faixas de DM em faixas de R$."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from mlops_platform.data import BRL_PER_DM, CATEGORICAL, FEATURES, MONEY_CATEGORIES, TARGET, synthetic_credit, to_reais
from mlops_platform.explain import reference
from mlops_platform.schema import DOMAIN, TRAIN_RANGE, check_value
from mlops_platform.train import candidates


def dm_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "credit_amount": [1169, 250, 18424],
            "checking_status": ["<0", "0<=X<200", ">=200"],
            "savings_status": ["no known savings", "<100", ">=1000"],
        }
    )


def test_amount_and_brackets_move_to_reais() -> None:
    out = to_reais(dm_frame())
    assert BRL_PER_DM == 9.70
    assert out["credit_amount"].tolist() == [11339.3, 2425.0, 178712.8]
    assert out["checking_status"].tolist() == ["<0", "0<=X<1940", ">=1940"]
    assert out["savings_status"].tolist() == ["no known savings", "<970", ">=9700"]


def test_every_dm_bracket_has_a_reais_twin_in_the_contract() -> None:
    for col, mapping in MONEY_CATEGORIES.items():
        assert set(mapping.values()) <= set(DOMAIN[col])
        assert not set(mapping) & set(DOMAIN[col])  # nenhuma faixa em DM sobra no contrato


def test_shipped_reference_is_in_reais() -> None:
    frame = reference().frame
    for col in CATEGORICAL:
        assert set(frame[col]) <= set(DOMAIN[col]), col
    assert (frame["credit_amount"].min(), frame["credit_amount"].max()) == TRAIN_RANGE["credit_amount"]
    assert set(synthetic_credit(200)["savings_status"]) <= set(DOMAIN["savings_status"])


def test_old_dm_bracket_is_refused_and_points_to_reais() -> None:
    _, [p] = check_value("savings_status", ">=1000", "pt")
    assert p.code == "out_of_domain" and "'>=9700'" in p.message


def test_scaling_does_not_change_what_the_model_learns() -> None:
    """Re-treinar em reais não muda o AUC: mesma ordem dos valores, mesmas categorias com outro nome."""
    rng = np.random.default_rng(3)
    dm = synthetic_credit(400, seed=3)
    dm["credit_amount"] = (dm["credit_amount"] / BRL_PER_DM).round()  # volta para DM inteiro
    for col, mapping in MONEY_CATEGORIES.items():
        dm[col] = dm[col].replace({v: k for k, v in mapping.items()})
    brl = to_reais(dm)
    test = rng.random(len(dm)) < 0.3
    for name, est in candidates().items():
        p_dm = clone(est).fit(dm[~test][FEATURES], dm[~test][TARGET]).predict_proba(dm[test][FEATURES])[:, 1]
        p_brl = clone(est).fit(brl[~test][FEATURES], brl[~test][TARGET]).predict_proba(brl[test][FEATURES])[:, 1]
        assert np.allclose(p_dm, p_brl, atol=1e-6), name
