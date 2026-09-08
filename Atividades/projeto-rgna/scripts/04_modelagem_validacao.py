"""Fase 4 — OLS, splines, OLS simples e Splines simples; GroupKFold por Estudo; VIF e bootstrap.

Testa TODAS as combinações possíveis de variáveis numéricas e categóricas.
Prevê MAPE na escala original. Encoding de país no fold (sem Estudo como feature).
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

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

N_BOOT = 1000
BOOT_SEED = 42


def _is_root(p: Path) -> bool:
    return (p / "projeto-rgna.Rproj").exists() or (p / "data" / "raw" / "dados_treino.xlsx").exists()


def find_root() -> Path:
    env = os.environ.get("RGNA_ROOT", "").strip()
    if env:
        p = Path(env).resolve()
        if _is_root(p):
            return p
    starts = [Path(__file__).resolve().parent.parent, Path.cwd().resolve()]
    seen: set[Path] = set()
    for here in starts:
        for p in [here, *here.parents]:
            if p in seen:
                continue
            seen.add(p)
            if _is_root(p):
                return p
            for nested in (p / "projeto-rgna", p / "Atividades" / "projeto-rgna"):
                if _is_root(nested):
                    return nested.resolve()
    raise FileNotFoundError("data/raw/dados_treino.xlsx")


def load_frame(root: Path) -> pd.DataFrame:
    csv_path = root / "data" / "processed" / "03_modelagem.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Rode antes scripts/03_engenharia_features.R")
    df = pd.read_csv(csv_path)
    df["Estudo"] = df["Estudo"].astype(str)
    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_scores(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true))),
    }


def top1_accuracy(frame: pd.DataFrame, pred_col: str) -> float:
    real = frame.loc[frame.groupby("ID_Observacao")["MAPE"].idxmin(), ["ID_Observacao", "Modelo"]]
    hat = frame.loc[frame.groupby("ID_Observacao")[pred_col].idxmin(), ["ID_Observacao", "Modelo"]]
    m = real.merge(hat, on="ID_Observacao", suffixes=("_real", "_hat"))
    return float((m["Modelo_real"] == m["Modelo_hat"]).mean())


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    med = train.groupby("Modelo", observed=True)["MAPE"].median()
    fallback = float(train["MAPE"].median())
    return test["Modelo"].map(med).fillna(fallback).to_numpy(dtype=float)


def campeao_do_fold(train: pd.DataFrame) -> str:
    med = train.groupby("Modelo", observed=True)["MAPE"].median()
    return str(med.idxmin())


def generate_feature_sets() -> list[dict]:
    """Gera todas as combinações possíveis de variáveis numéricas e categóricas."""
    num_combinations = []
    for r in range(1, len(NUM_CANDIDATES) + 1):
        num_combinations.extend(list(combinations(NUM_CANDIDATES, r)))

    cat_combinations = []
    for r in range(1, len(CAT_CANDIDATES) + 1):
        cat_combinations.extend(list(combinations(CAT_CANDIDATES, r)))

    feature_sets = []
    for num_cols, cat_cols in product(num_combinations, cat_combinations):
        num_list = list(num_cols)
        cat_list = list(cat_cols)
        feature_sets.append(
            {
                "num": num_list,
                "cat": cat_list,
                "x_cols": num_list + cat_list,
                "name": f"num({'_'.join(num_list)})__cat({'_'.join(cat_list)})",
            }
        )
    # Adicionar combinações com apenas uma variável numérica e nenhuma categórica
    for num_cols in num_combinations:
        num_list = list(num_cols)
        feature_sets.append(
            {
                "num": num_list,
                "cat": [],
                "x_cols": num_list,
                "name": f"num({'_'.join(num_list)})__cat(none)",
            }
        )
    return feature_sets


def _cat_encoder() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )


def _num_linear() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )


def linear_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", _num_linear(), num_cols),
            ("cat", _cat_encoder(), cat_cols),
        ]
    )


def spline_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_spline = [num_cols[0]]
    num_linear = num_cols[1:]

    transformers = [
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
        ),
        ("cat", _cat_encoder(), cat_cols),
    ]

    if num_linear:
        transformers.append(("num", _num_linear(), num_linear))

    return ColumnTransformer(transformers)


def ols_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    return Pipeline([("pre", linear_preprocessor(num_cols, cat_cols)), ("model", LinearRegression())])


def spline_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    return Pipeline([("pre", spline_preprocessor(num_cols, cat_cols)), ("model", LinearRegression())])

def inv_boxcox(y_trans: np.ndarray, lambda_param: float) -> np.ndarray:
    """Reverte a transformação de Box-Cox para a escala original."""
    if lambda_param == 0:
        return np.exp(y_trans)
    else:
        # Garante que o argumento da potência seja estritamente positivo para evitar NaNs
        inner = np.maximum(lambda_param * y_trans + 1.0, 1e-12)
        return inner ** (1.0 / lambda_param)


def extrair_expressao_pipeline(pipe: Pipeline, nome_modelo: str = "Modelo") -> str:
    """Extrai a equação matemática de um Pipeline scikit-learn contendo
    ColumnTransformer + LinearRegression.
    """
    preprocessor = pipe.named_steps["pre"]
    regressor = pipe.named_steps["model"]

    # 1. Recupera os nomes das variáveis geradas pelo pré-processador (Ex: OHE, Splines)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"x_{i}" for i in range(len(regressor.coef_))]

    # Limpa os prefixos gerados pelo ColumnTransformer (ex: 'cat__Sexo_Animal_M' -> 'Sexo_Animal_M')
    clean_names = [f.split("__")[-1] for f in feature_names]

    intercept = regressor.intercept_
    coefs = regressor.coef_

    # 2. Monta a string da expressão matemática
    termos = [f"{intercept:.4f}"]
    for coef, name in zip(coefs, clean_names):
        sinal = "+" if coef >= 0 else "-"
        termos.append(f"{sinal} {abs(coef):.4f} * ({name})")

    expressao = " ".join(termos)
    expr = f"{nome_modelo}: y_hat = {expressao}"
    
    with open("equacao_modelo.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]\n {expr}\n")


def fit_predict_2sls(
    pipe: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    model_name: str = "Modelo_2sls",
) -> np.ndarray:
    train = train.copy()
    test = test.copy()

    # ── ESTÁGIO 1: Predizer o patamar médio (mediana) do animal
    train_animal = train.drop_duplicates(subset=["ID_Observacao"]).copy()
    train_animal["mediana_animal"] = (
        train.groupby("ID_Observacao")[TARGET].median().loc[train_animal["ID_Observacao"]].values
    )

    x_cols_animal = [c for c in x_cols if c != "Modelo"]
    num_animal = [c for c in num_cols if c != "Modelo"]
    cat_animal = [c for c in cat_cols if c != "Modelo"]

    pre_animal = ColumnTransformer(
        [
            ("num", _num_linear(), num_animal),
            ("cat", _cat_encoder(), cat_animal),
        ]
    )

    pipe_median = Pipeline([("pre", pre_animal), ("model", LinearRegression())])
    pipe_median.fit(train_animal[x_cols_animal], train_animal["mediana_animal"].to_numpy())
    pred_median_test = pipe_median.predict(test[x_cols_animal])  # Note. Previsão da mediana do animal no teste
    extrair_expressao_pipeline(pipe_median, f"{model_name}_Stage1")

    # ── ESTÁGIO 2: Predizer o desvio do modelo empírico (centralizado no animal)
    mediana_real_train = train.groupby("ID_Observacao")[TARGET].transform("median")
    y_train_resid = train[TARGET].to_numpy() - mediana_real_train.to_numpy()
    # Note. O vetor mediana_real_train está correto?

    pipe.fit(train[x_cols], y_train_resid)
    pred_resid_test = pipe.predict(test[x_cols])    # Note. Previsão do desvio do modelo empírico no teste
    extrair_expressao_pipeline(pipe_median, f"{model_name}_Stage2")

    pred_log = pred_median_test + pred_resid_test
    return inv_boxcox(pred_log, BEST_LAMBDA)

def fit_predict_simples(
    pipe: Pipeline, train: pd.DataFrame, test: pd.DataFrame, x_cols: list[str], model_name: str = "Modelo_Simples"
) -> np.ndarray:
    """Ajusta o modelo diretamente na escala original (MAPE), sem estratégia de etapas."""
    train = train.copy()
    test = test.copy()

    pipe.fit(train[x_cols], train[TARGET].to_numpy())
    extrair_expressao_pipeline(pipe, model_name)
    return inv_boxcox(pipe.predict(test[x_cols]), BEST_LAMBDA)


def collect_fold(
    name: str, protocol: str, fold: str, test: pd.DataFrame, pred: np.ndarray, campeao: str
) -> tuple[dict, pd.DataFrame]:
    tmp = test.copy()
    tmp["pred"] = pred
    real_win = tmp.loc[tmp.groupby("ID_Observacao")["MAPE"].idxmin(), "Modelo"]
    scores = regression_scores(tmp["MAPE"], pred)
    scores.update(
        {
            "modelo": name,
            "protocolo": protocol,
            "fold": fold,
            "n": int(len(tmp)),
            "top1": top1_accuracy(tmp, "pred"),
            "top1_sempre_campeao": float((real_win.to_numpy() == campeao).mean()),
        }
    )
    return scores, tmp


def model_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
    preds = {
        "baseline_mediana_modelo": baseline_predict(train, test),
        }
    
    feature_sets = generate_feature_sets()
    
    for fset in feature_sets:
        num_cols = fset["num"]
        cat_cols = fset["cat"]
        x_cols = fset["x_cols"]
        suffix = fset["name"]
        
        pipe_ols = ols_pipeline(num_cols, cat_cols)
        pipe_spl = spline_pipeline(num_cols, cat_cols)
        
        preds[f"ols__{suffix}"] = fit_predict_2sls(pipe_ols, train, test, x_cols, num_cols, cat_cols, f"ols__{suffix}")
        preds[f"splines__{suffix}"] = fit_predict_2sls(
            pipe_spl, train, test, x_cols, num_cols, cat_cols, f"splines__{suffix}"
            )
        preds[f"ols_simples__{suffix}"] = fit_predict_simples(pipe_ols, train, test, x_cols, f"ols_simples__{suffix}")
        preds[f"splines_simples__{suffix}"] = fit_predict_simples(
            pipe_spl, train, test, x_cols, f"splines_simples__{suffix}"
            )
    
    return preds


def run_group_kfold(df: pd.DataFrame, n_splits: int = 5) -> tuple[list[dict], pd.DataFrame]:
    groups = df["Estudo"].to_numpy()
    cv = GroupKFold(n_splits=min(n_splits, df["Estudo"].nunique()))
    
    # Salvar grupos em pastas separadas para debug. 1 pasta por grupo, e dado de treino e teste dentro de cada pasta.
    group_dir = Path(find_root()) / "data" / "processed" / "group_kfold_folds"
    group_dir.mkdir(exist_ok=True)
    for i, (tr, te) in enumerate(cv.split(df, df["MAPE"], groups), start=1):
        fold_dir = group_dir / f"fold_{i}"
        fold_dir.mkdir(exist_ok=True)
        train, test = df.iloc[tr].copy(), df.iloc[te].copy()
        train.to_csv(fold_dir / "train.csv", index=False)
        test.to_csv(fold_dir / "test.csv", index=False)
    
    rows = []
    oof_parts = []
    for i, (tr, te) in enumerate(cv.split(df, df["MAPE"], groups), start=1):
        # Note. estou usando MAPE_log para treinar?
        train, test = df.iloc[tr].copy(), df.iloc[te].copy()
        g_tr = train["Estudo"].to_numpy()
        campeao = campeao_do_fold(train)
        print(f"GroupKFold fold {i}", flush=True)
        preds = model_predictions(train, test)
        fold_name = f"gkf{i}"
        for name, pred in preds.items():
            scores, _tmp = collect_fold(name, "GroupKFold", fold_name, test, pred, campeao)
            rows.append(scores)
            part = test[["ID_Observacao", "Estudo", "Pais_Estudo", "Modelo", "MAPE"]].copy()
            part["pred"] = pred
            part["modelo_ml"] = name
            part["protocolo"] = "GroupKFold"
            part["fold"] = fold_name
            oof_parts.append(part)
    return rows, pd.concat(oof_parts, ignore_index=True)


def run_loso_paises(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    estudos = (
        df.loc[df["Pais_Estudo"].isin(PAISES_OCULTOS), "Estudo"].drop_duplicates().tolist()
    )
    rows = []
    oof_parts = []
    for estudo in estudos:
        test = df[df["Estudo"] == estudo].copy()
        train = df[df["Estudo"] != estudo].copy()
        if test.empty or train.empty:
            continue
        campeao = campeao_do_fold(train)
        preds = model_predictions(train, test)
        for name, pred in preds.items():
            print(f"LOSO_paises_case estudo {estudo} modelo {name}", flush=True)
            scores, _tmp = collect_fold(name, "LOSO_paises_case", str(estudo), test, pred, campeao)
            rows.append(scores)
            part = test[["ID_Observacao", "Estudo", "Pais_Estudo", "Modelo", "MAPE"]].copy()
            part["pred"] = pred
            part["modelo_ml"] = name
            part["protocolo"] = "LOSO_paises_case"
            part["fold"] = str(estudo)
            oof_parts.append(part)
    return rows, pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()


def summarize(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(rows)
    agg = (
        raw.groupby(["protocolo", "modelo"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            mape=("mape", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            top1=("top1", "mean"),
            top1_sempre_campeao=("top1_sempre_campeao", "mean"),
        )
        .sort_values(["protocolo", "mape"])
    )
    return raw, agg


def vif_from_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.astype(float).copy()
    keep = [c for c in X.columns if float(X[c].std(ddof=0)) > 1e-12]
    X = X[keep]
    rows = []
    for col in X.columns:
        y = X[col].to_numpy()
        Z = X.drop(columns=[col]).to_numpy()
        if Z.size == 0:
            vif = float("inf")
            r2 = float("nan")
        else:
            lr = LinearRegression()
            lr.fit(Z, y)
            r2 = float(lr.score(Z, y))
            vif = float("inf") if r2 >= 1.0 - 1e-12 else float(1.0 / (1.0 - r2))
        rows.append({"bloco": None, "variavel": col, "r2_auxiliar": r2, "VIF": vif})
    return pd.DataFrame(rows)


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """VIF nos numéricos (peso e fração A)."""
    vif_num = vif_from_matrix(df[NUM_CANDIDATES].copy())
    vif_num["bloco"] = "numericos_brutos"
    vif_num["alerta"] = np.where(
        vif_num["VIF"] >= 10, "VIF>=10", np.where(vif_num["VIF"] >= 5, "VIF>=5", "ok")
    )
    return vif_num.sort_values("VIF", ascending=False)


def cluster_bootstrap_error(
    oof: pd.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """IC 95% do MAE e do RMSE via bootstrap por Estudo (OOF do GroupKFold)."""
    gkf = oof[oof["protocolo"] == "GroupKFold"].copy()
    if gkf.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for name, sub in gkf.groupby("modelo_ml"):
        studies = sub["Estudo"].unique()
        buckets = {s: sub[sub["Estudo"] == s] for s in studies}
        n_g = len(studies)
        mae_b = np.empty(n_boot)
        rmse_b = np.empty(n_boot)
        for b in range(n_boot):
            draw = rng.choice(studies, size=n_g, replace=True)
            parts = [buckets[s] for s in draw]
            y = np.concatenate([p["MAPE"].to_numpy() for p in parts])
            yhat = np.concatenate([p["pred"].to_numpy() for p in parts])
            mae_b[b] = mean_absolute_error(y, yhat)
            rmse_b[b] = rmse(y, yhat)
        y_obs = sub["MAPE"].to_numpy()
        yhat_obs = sub["pred"].to_numpy()
        rows.append(
            {
                "protocolo": "GroupKFold",
                "modelo": name,
                "n_boot": n_boot,
                "agrupamento": "Estudo",
                "mape": float(np.mean(np.abs((y_obs - yhat_obs) / y_obs))),
                "mape_ic95_inf": float(np.quantile(np.abs((y_obs - yhat_obs) / y_obs), 0.025)),
                "mape_ic95_sup": float(np.quantile(np.abs((y_obs - yhat_obs) / y_obs), 0.975)),
                "mae": float(mean_absolute_error(y_obs, yhat_obs)),
                "mae_ic95_inf": float(np.quantile(mae_b, 0.025)),
                "mae_ic95_sup": float(np.quantile(mae_b, 0.975)),
                "rmse": rmse(y_obs, yhat_obs),
                "rmse_ic95_inf": float(np.quantile(rmse_b, 0.025)),
                "rmse_ic95_sup": float(np.quantile(rmse_b, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values("mae")


def filter_models(agg: pd.DataFrame, raw: pd.DataFrame, oof: pd.DataFrame, boot: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filtra modelos com top1 < 0.5 ou com MAE pior que baseline."""
    baseline_mape = agg[agg["modelo"] == "baseline_mediana_modelo"][["protocolo", "mape"]].set_index("protocolo")[
        "mape"].to_dict()
    
    baseline_mae = agg[agg["modelo"] == "baseline_mediana_modelo"][["protocolo", "mae"]].set_index("protocolo")["mae"].to_dict()
    
    agg_filtered_tmp = agg[(
        agg.apply(lambda row: row["mape"] <= baseline_mape.get(row["protocolo"], float("inf")), axis=1))].copy()
    
    models_to_keep = set(agg_filtered_tmp["modelo"])
    
    agg_filtered_tmp = agg_filtered_tmp[(
        agg_filtered_tmp.apply(lambda row: row["mae"] <= baseline_mae.get(row["protocolo"], float("inf")), axis=1))].copy()
    
    if not agg_filtered_tmp.empty:
        models_to_keep = set(agg_filtered_tmp["modelo"])
    
    agg_filtered = agg[agg["modelo"].isin(models_to_keep)].copy()
    raw_filtered = raw[raw["modelo"].isin(models_to_keep)].copy()
    oof_filtered = oof[oof["modelo_ml"].isin(models_to_keep)].copy()
    boot_filtered = boot[boot["modelo"].isin(models_to_keep)].copy()
    
    return agg_filtered, raw_filtered, oof_filtered, boot_filtered

def main() -> None:
    root = find_root()
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df = load_frame(root)
    print(f"n={len(df)} estudos={df['Estudo'].nunique()}", flush=True)

    vif = compute_vif(df)
    print("VIF ok", flush=True)
    vif.to_csv(out / "04_vif.csv", index=False)

    rows_g, oof_g = run_group_kfold(df)
    rows_l, oof_l = run_loso_paises(df)
    raw, agg = summarize(rows_g + rows_l)
    oof = pd.concat([oof_g, oof_l], ignore_index=True)

    boot = cluster_bootstrap_error(oof)
    
    agg, raw, oof, boot = filter_models(agg, raw, oof, boot)

    agg.to_csv(out / "04_metricas_resumo.csv", index=False)
    raw.to_csv(out / "04_metricas_folds.csv", index=False)
    oof.to_csv(out / "04_predicoes_oof.csv", index=False)
    boot.to_csv(out / "04_bootstrap_ic.csv", index=False)
    (out / "04_resumo.json").write_text(
        json.dumps(
            {
                "metricas": agg.to_dict(orient="records"),
                # "bootstrap": boot.to_dict(orient="records"),
                "vif_numericos": vif.loc[vif["bloco"] == "numericos_brutos"].to_dict(
                    orient="records"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("VIF (numéricos):")
    print(vif.loc[vif["bloco"] == "numericos_brutos"].to_string(index=False))
    print("\nMétricas (média dos folds):")
    print(agg.head(10).to_string(index=False))  # Exibe os 10 melhores
    print("\nBootstrap cluster (IC 95% do erro, GroupKFold):")
    print(boot.head(10).to_string(index=False))
    print(f"\nSaídas em {out}")


if __name__ == "__main__":
    main()