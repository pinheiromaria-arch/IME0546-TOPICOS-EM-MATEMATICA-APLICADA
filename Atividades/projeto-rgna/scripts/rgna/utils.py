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


def regression_scores(y_true, y_pred) -> dict:
    """Calcula um dicionário de métricas de regressão."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true))),
    }


def inv_boxcox(y_trans: np.ndarray, lambda_param: float) -> np.ndarray:
    """Reverte a transformação de Box-Cox para a escala original."""
    if lambda_param == 0:
        return np.exp(y_trans)
    
    # Garante que o argumento da potência seja estritamente positivo para evitar NaNs
    inner = np.maximum(lambda_param * y_trans + 1.0, 1e-12)
    return inner ** (1.0 / lambda_param)