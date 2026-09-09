"""Módulo para a definição dos pipelines de pré-processamento do scikit-learn."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def _cat_encoder() -> Pipeline:
    """Pipeline para codificação de variáveis categóricas."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )


def _num_linear() -> Pipeline:
    """Pipeline para pré-processamento de variáveis numéricas lineares."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )


def linear_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Pré-processador para modelos lineares."""
    return ColumnTransformer(
        [
            ("num", _num_linear(), num_cols),
            ("cat", _cat_encoder(), cat_cols),
        ],
        remainder="drop",
    )


def spline_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Pré-processador para modelos com splines."""
    if not num_cols:
        # Se não houver colunas numéricas, retorna um pré-processador apenas com as categóricas
        return ColumnTransformer([("cat", _cat_encoder(), cat_cols)], remainder="drop")

    weight_col = "Peso_Animal"
    num_spline = [weight_col] if weight_col in num_cols else []
    num_linear = [col for col in num_cols if col != weight_col]

    transformers = [
        ("cat", _cat_encoder(), cat_cols),
    ]

    if num_spline:
        transformers.append(
            (
                "spl",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        (
                            "bs",
                            SplineTransformer(
                                n_knots=4,
                                degree=3,
                                include_bias=False,
                                extrapolation="constant",
                            ),
                        ),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_spline,
            )
        )

    if num_linear:
        transformers.append(("num", _num_linear(), num_linear))

    return ColumnTransformer(transformers, remainder="drop")


def ols_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Cria um pipeline de regressão OLS com pré-processamento."""
    return Pipeline([("pre", linear_preprocessor(num_cols, cat_cols)), ("model", LinearRegression())])


def spline_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Cria um pipeline de regressão com spline em Peso_Animal."""
    return Pipeline([("pre", spline_preprocessor(num_cols, cat_cols)), ("model", LinearRegression())])