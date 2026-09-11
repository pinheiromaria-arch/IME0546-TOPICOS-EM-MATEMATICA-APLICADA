"""Módulo de utilitários para os scripts Python do projeto RGNA."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def log(msg: str) -> None:
    """Função utilitária para centralizar e formatar os logs do console."""
    print(f"[{datetime.datetime.now():%d/%m/%Y %H:%M:%S}] {msg}", flush=True)

def _is_root(p: Path) -> bool:
    """Verifica se um diretório é a raiz do projeto."""
    return (p / "projeto-rgna.Rproj").exists() or (p / "data" / "raw" / "dados_treino.xlsx").exists()


def find_root() -> Path:
    """
    Encontra a raiz do projeto de forma simplificada.

    1. Tenta usar a variável de ambiente RGNA_ROOT.
    2. Sobe a árvore de diretórios a partir deste arquivo.
    """
    env = os.environ.get("RGNA_ROOT", "").strip()
    if env:
        p = Path(env).resolve()
        if _is_root(p):
            return p

    # Fallback: sobe a árvore a partir do local deste arquivo (__file__)
    current_dir = Path(__file__).resolve().parent
    for p in [current_dir, *current_dir.parents]:
        if _is_root(p):
            return p

    raise FileNotFoundError(
        "Raiz do projeto não encontrada. "
        "Certifique-se de que 'projeto-rgna.Rproj' existe ou defina a variável de ambiente RGNA_ROOT."
    )


def load_frame(root: Path) -> pd.DataFrame:
    """Carrega o dataframe de modelagem."""
    csv_path = root / "data" / "processed" / "03_modelagem.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Rode antes scripts/03_engenharia_features.R")
    df = pd.read_csv(csv_path)
    df["Estudo"] = df["Estudo"].astype(str)
    return df


def rmse(y_true, y_pred) -> float:
    """Calcula o Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    """Calcula o erro percentual médio ABSOLUTO da PREVISÃO em relação ao MAPE real.

    Atenção: isto NÃO é o MAPE científico do case (a variável-alvo, já chamada
    MAPE nos dados). É uma métrica de avaliação do MODELO: o erro percentual
    entre o MAPE real de cada linha e o MAPE previsto pelo seletor/regressão
    para aquela linha. O nome coincide por convenção de métrica de regressão,
    não porque sejam a mesma grandeza. Como MAPE real pode ser próximo de
    zero em algumas linhas (mínimo observado ~0,0005), o erro relativo pode
    ficar bem maior que os erros absolutos (MAE/RMSE) sugerem — isso é
    esperado da fórmula percentual, não um bug.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true)
    rel = np.divide(y_true - y_pred, denom, out=np.zeros_like(y_true, dtype=float), where=denom > 0)
    return float(np.mean(np.abs(rel)))


def regression_scores(y_true, y_pred) -> dict:
    """Calcula um dicionário de métricas de regressão, todas na mesma escala.

    y_true e y_pred devem estar ambos na escala original do MAPE (não em
    MAPE_boxcox) — quem chama esta função é responsável por reverter Box-Cox
    antes, com o MESMO lambda usado na transformação (ver config.get_best_lambda).
    "mape" aqui é a métrica de erro da previsão, ver docstring de mape().
    "r2" é o r2_score padrão do sklearn (1 - SSE/SST); pode ser negativo de
    forma legítima quando o modelo erra mais que a média do próprio conjunto
    de teste — não é clampado nem invertido em lugar nenhum do pipeline.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }


def inv_boxcox(y_trans: np.ndarray, lambda_param: float) -> np.ndarray:
    """Reverte a transformação de Box-Cox para a escala original."""
    if lambda_param == 0:
        return np.exp(y_trans)
    
    # Garante que o argumento da potência seja estritamente positivo para evitar NaNs
    inner = np.maximum(lambda_param * y_trans + 1.0, 1e-12)
    return inner ** (1.0 / lambda_param)