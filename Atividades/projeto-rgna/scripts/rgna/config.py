"""Módulo de configuração para os scripts Python do projeto RGNA."""

# Constantes de Modelagem
PAISES_OCULTOS = [
    "Campo Rico",
    "Ilha Verdejante",
    "Terra Nortenha",
    "Monção Dourada",
    "Costa Austral",
]

TARGET = "MAPE_boxcox"  # Note. MAPE na escala logarítmica (Box-Cox) para regressão linear
BEST_LAMBDA = 0.3434343

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