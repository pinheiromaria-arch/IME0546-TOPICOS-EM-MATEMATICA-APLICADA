"""Módulo para as funções de treinamento e validação de modelos."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from . import config
from .pipelines import _cat_encoder, _num_linear
from .utils import inv_boxcox


def extrair_expressao_pipeline(pipe: Pipeline, nome_modelo: str = "Modelo") -> str:
    """
    Extrai a equação matemática de um Pipeline scikit-learn.

    Retorna a string da equação em vez de escrevê-la em um arquivo.
    """
    preprocessor = pipe.named_steps.get("pre")
    regressor = pipe.named_steps.get("model")

    if not all([preprocessor, regressor]):
        return f"{nome_modelo}: (Não foi possível extrair a expressão)"

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback para quando get_feature_names_out não está disponível
        try:
            n_features = regressor.n_features_in_
        except AttributeError:
            n_features = regressor.coef_.shape[0]
        feature_names = [f"x_{i}" for i in range(n_features)]

    clean_names = [f.split("__")[-1] for f in feature_names]
    intercept = regressor.intercept_
    coefs = regressor.coef_

    termos = [f"{intercept:.4f}"]
    for coef, name in zip(coefs, clean_names):
        sinal = "+" if coef >= 0 else "-"
        termos.append(f"{sinal} {abs(coef):.4f} * ({name})")

    return f"{nome_modelo}: y_hat = {' '.join(termos)}"


def fit_predict_2sls(
    pipe: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    model_name: str = "Modelo_2sls",
) -> tuple[np.ndarray, list[str]]:
    """
    Ajusta e prediz usando uma abordagem de 2 estágios.

    Estágio 1: Modela a mediana do animal.
    Estágio 2: Modela o desvio do modelo empírico em relação à mediana.

    Retorna as predições e as equações dos modelos.
    """
    train = train.copy()
    test = test.copy()
    equacoes = []

    # --- Estágio 1: Predizer o patamar médio (mediana) do animal ---
    train_animal = train.drop_duplicates(subset=["ID_Observacao"]).copy()
    train_animal["mediana_animal"] = (
        train.groupby("ID_Observacao")[config.TARGET].median().loc[train_animal["ID_Observacao"]].values
    )

    x_cols_animal = [c for c in x_cols if c != "Modelo"]
    num_animal = [c for c in num_cols if c != "Modelo"]
    cat_animal = [c for c in cat_cols if c != "Modelo"]

    if not x_cols_animal:
        pred_median_test = np.full(len(test), train_animal["mediana_animal"].mean())
        equacoes.append(f"{model_name}_Stage1: y_hat = {pred_median_test[0]:.4f} (Média Global)")
    else:
        pre_animal = ColumnTransformer(
            [("num", _num_linear(), num_animal), ("cat", _cat_encoder(), cat_animal)],
            remainder="drop",
        )
        pipe_median = Pipeline([("pre", pre_animal), ("model", LinearRegression())])
        pipe_median.fit(train_animal[x_cols_animal], train_animal["mediana_animal"])
        pred_median_test = pipe_median.predict(test[x_cols_animal])
        equacoes.append(extrair_expressao_pipeline(pipe_median, f"{model_name}_Stage1"))

    # --- Estágio 2: Predizer o desvio do modelo empírico ---
    mediana_real_train = train.groupby("ID_Observacao")[config.TARGET].transform("median")
    y_train_resid = train[config.TARGET] - mediana_real_train

    x_cols_resid = ["Modelo"]
    pre_resid = ColumnTransformer([("cat", _cat_encoder(), x_cols_resid)])
    pipe_resid = Pipeline([("pre", pre_resid), ("model", LinearRegression())])

    pipe_resid.fit(train[x_cols_resid], y_train_resid)
    pred_resid_test = pipe_resid.predict(test[x_cols_resid])
    equacoes.append(extrair_expressao_pipeline(pipe_resid, f"{model_name}_Stage2"))

    # --- Combinação ---
    pred_log = pred_median_test + pred_resid_test
    pred_final = inv_boxcox(pred_log, config.get_best_lambda())

    return pred_final, equacoes


def fit_predict_simples(
    pipe: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_cols: list[str],
    model_name: str = "Modelo_Simples",
) -> tuple[np.ndarray, list[str]]:
    """
    Ajusta e prediz um modelo de regressão simples diretamente.

    Retorna as predições e a equação do modelo.
    """
    train = train.copy()
    test = test.copy()

    pipe.fit(train[x_cols], train[config.TARGET])
    pred_log = pipe.predict(test[x_cols])
    pred_final = inv_boxcox(pred_log, config.get_best_lambda())
    equacao = extrair_expressao_pipeline(pipe, model_name)

    return pred_final, [equacao]