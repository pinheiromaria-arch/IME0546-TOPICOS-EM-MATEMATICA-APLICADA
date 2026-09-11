"""Módulo de configuração para os scripts Python do projeto RGNA."""

import pandas as pd

# Constantes de Modelagem
PAISES_OCULTOS = [
    "Campo Rico",
    "Ilha Verdejante",
    "Terra Nortenha",
    "Monção Dourada",
    "Costa Austral",
]

TARGET = "MAPE_boxcox"  # Note. MAPE na escala logarítmica (Box-Cox) para regressão linear

_best_lambda_cache: float | None = None


def get_best_lambda(root=None) -> float:
    """Lê o lambda de Box-Cox salvo por scripts/03_engenharia_features.R.

    data/processed/03_meta.csv é a ÚNICA fonte de verdade para esse valor: é
    o mesmo lambda usado para criar a coluna MAPE_boxcox. Um valor hardcoded
    aqui pode divergir silenciosamente do lambda real usado na transformação
    (era o caso antes: 0.3434343 hardcoded vs. 0.3 real), o que corrompe a
    reversão de Box-Cox -> MAPE e distorce MAE/RMSE/R² calculados na escala
    original. Resultado é cacheado em processo para evitar releituras do CSV.
    """
    global _best_lambda_cache
    if _best_lambda_cache is None:
        if root is None:
            from .utils import find_root

            root = find_root()
        meta = pd.read_csv(root / "data" / "processed" / "03_meta.csv")
        _best_lambda_cache = float(meta["lambda_boxcox"].iloc[0])
    return _best_lambda_cache

# Candidatos a features para varredura completa
NUM_CANDIDATES = [
    "Peso_Corporal_kg",
    "Consumo_MS_kg",
    "Fracao_Perda_A",
]
CAT_CANDIDATES = [
    "Pais_Estudo",
    "Status_Metabolico",
    "Sexo_Animal",
    "Modelo",
]

map_columns = {
    "Peso_Corporal_kg": "peso",
    "Consumo_MS_kg": "consumo",
    "Fracao_Perda_A": "fracao",
    "Pais_Estudo": "pais",
    "Status_Metabolico": "status",
    "Sexo_Animal": "sexo",
    "Modelo": "modelo"
    }

# Configurações de Bootstrap
N_BOOT = 1000
BOOT_SEED = 42

# Estratégia de agregação dos resultados por fold
# Opções: "weighted_mean" (padrão, usa pesos pelo número de observações do fold),
# "mean" (média simples), "median" (mediana por fold).
AGGREGATION_STRATEGY = "median"

# Configurações do filtro de modelos
TOP_N_MODELOS_POR_PROTOCOLO = 10