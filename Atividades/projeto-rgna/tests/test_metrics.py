"""Testes de regressão para as métricas de modelagem (Fase 4).

Cobre os dois pontos que já causaram bugs reais ou dúvida no projeto:
1. O lambda de Box-Cox usado para reverter previsões precisa ser exatamente
   o mesmo usado para criar MAPE_boxcox (ver scripts/rgna/config.py).
2. R² é o r2_score padrão do sklearn (1 - SSE/SST) e pode legitimamente dar
   negativo — o pipeline não deve clampar nem "corrigir" isso.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rgna import config, utils  # noqa: E402


def _project_root() -> Path:
    return utils.find_root()


def test_boxcox_lambda_e_meta_csv_sao_a_mesma_fonte():
    """O lambda usado no pipeline deve ser lido de 03_meta.csv, não hardcoded."""
    root = _project_root()
    meta = pd.read_csv(root / "data" / "processed" / "03_meta.csv")
    lambda_meta = float(meta["lambda_boxcox"].iloc[0])

    config._best_lambda_cache = None  # força reler do disco, sem cache de outro teste
    lambda_config = config.get_best_lambda()

    assert lambda_config == pytest.approx(lambda_meta)


def test_boxcox_round_trip_com_dados_reais():
    """y -> boxcox (fórmula do 03_engenharia_features.R) -> inv_boxcox -> y."""
    root = _project_root()
    df = pd.read_csv(root / "data" / "processed" / "03_modelagem.csv")
    lam = config.get_best_lambda(root=root)

    y = df["MAPE"].to_numpy(dtype=float)
    y_boxcox_csv = df["MAPE_boxcox"].to_numpy(dtype=float)

    # Confirma que a coluna MAPE_boxcox do CSV bate com a fórmula de Box-Cox
    # padrão, usando o MESMO lambda salvo em 03_meta.csv.
    y_boxcox_manual = np.log(y) if lam == 0 else (y**lam - 1) / lam
    assert y_boxcox_manual == pytest.approx(y_boxcox_csv, abs=1e-9)

    # Reverte com o mesmo lambda e confirma que reconstrói o MAPE original.
    y_reconstruido = utils.inv_boxcox(y_boxcox_csv, lam)
    assert y_reconstruido == pytest.approx(y, abs=1e-9)


def test_boxcox_round_trip_com_lambda_errado_diverge():
    """Regressão: um lambda != o real deve distorcer o round-trip de forma visível.

    Isso documenta o bug que existia (config.BEST_LAMBDA=0.3434343 hardcoded
    vs. 0.3 real): serve de sentinela caso alguém volte a hardcodar o valor.
    """
    root = _project_root()
    df = pd.read_csv(root / "data" / "processed" / "03_modelagem.csv")
    lam_real = config.get_best_lambda(root=root)
    lam_errado = lam_real + 0.0434343

    y = df["MAPE"].to_numpy(dtype=float)
    y_boxcox_csv = df["MAPE_boxcox"].to_numpy(dtype=float)

    y_reconstruido_errado = utils.inv_boxcox(y_boxcox_csv, lam_errado)
    erro_max = np.max(np.abs(y_reconstruido_errado - y))
    assert erro_max > 0.005  # o mesmo desalinhamento observado na auditoria


def test_r2_manual_bate_com_sklearn():
    """SSE, SST e 1 - SSE/SST calculados na mão devem bater com r2_score."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.2, 1.1])
    y_pred = np.array([2.5, 0.0, 2.1, 7.8, 3.9, 1.5])

    mean_y = y_true.mean()
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - mean_y) ** 2))
    r2_manual = 1.0 - sse / sst

    assert r2_manual == pytest.approx(r2_score(y_true, y_pred))


def test_r2_negativo_e_legitimo_quando_modelo_e_pior_que_a_media():
    """R² < 0 é um resultado válido, não deve ser tratado como erro.

    Constrói um caso propositalmente ruim (previsão constante bem longe da
    média real) e confirma que regression_scores devolve o valor negativo
    sem clampar, inverter sinal ou aplicar abs().
    """
    rng = np.random.default_rng(0)
    y_true = rng.normal(loc=0.2, scale=0.05, size=50)  # baixa variância real
    y_pred_ruim = np.full_like(y_true, fill_value=0.9)  # previsão constante, longe da média

    scores = utils.regression_scores(y_true, y_pred_ruim)

    assert scores["r2"] < 0
    assert scores["r2"] == pytest.approx(r2_score(y_true, y_pred_ruim))


def test_r2_positivo_quando_modelo_acerta_bem():
    """Contraponto do teste anterior: um modelo bom deve dar R² próximo de 1."""
    rng = np.random.default_rng(1)
    y_true = rng.normal(loc=0.2, scale=0.05, size=50)
    y_pred_bom = y_true + rng.normal(loc=0.0, scale=0.001, size=50)

    scores = utils.regression_scores(y_true, y_pred_bom)
    assert scores["r2"] > 0.9
