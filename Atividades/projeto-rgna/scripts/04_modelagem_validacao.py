"""Fase 4 — OLS, splines, OLS simples e Splines simples; GroupKFold por Estudo; VIF e bootstrap.

Testa TODAS as combinações possíveis de variáveis numéricas e categóricas.
Prevê MAPE na escala original. Encoding de país no fold (sem Estudo como feature).
"""

from __future__ import annotations

import json
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from rgna import config, modeling, pipelines, utils


def top1_accuracy(frame: pd.DataFrame, pred_col: str) -> float:
    """Calcula a acurácia top-1."""
    real = frame.loc[frame.groupby("ID_Observacao")["MAPE"].idxmin(), ["ID_Observacao", "Modelo"]]
    hat = frame.loc[frame.groupby("ID_Observacao")[pred_col].idxmin(), ["ID_Observacao", "Modelo"]]
    m = real.merge(hat, on="ID_Observacao", suffixes=("_real", "_hat"))
    return float((m["Modelo_real"] == m["Modelo_hat"]).mean())


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Gera predições da baseline (mediana por modelo)."""
    med = train.groupby("Modelo", observed=True)["MAPE"].median()
    fallback = float(train["MAPE"].median())
    return test["Modelo"].map(med).fillna(fallback).to_numpy(dtype=float)


def campeao_do_fold(train: pd.DataFrame) -> str:
    """Identifica o modelo 'campeão' (menor mediana de MAPE) em um fold de treino."""
    med = train.groupby("Modelo", observed=True)["MAPE"].median()
    return str(med.idxmin())


def generate_feature_sets() -> list[dict]:
    """Gera todas as combinações possíveis de variáveis numéricas e categóricas."""
    num_combinations = []
    for r in range(1, len(config.NUM_CANDIDATES) + 1):
        num_combinations.extend(list(combinations(config.NUM_CANDIDATES, r)))

    cat_combinations = []
    for r in range(1, len(config.CAT_CANDIDATES) + 1):
        cat_combinations.extend(list(combinations(config.CAT_CANDIDATES, r)))

    feature_sets = []
    # Combinações com variáveis numéricas e categóricas
    for num_cols, cat_cols in product(num_combinations, cat_combinations):
        num_list, cat_list = list(num_cols), list(cat_cols)
        feature_sets.append({
            "num": num_list,
            "cat": cat_list,
            "x_cols": num_list + cat_list,
            "name": f"num({'_'.join(num_list)})__cat({'_'.join(cat_list)})",
        })
    # Combinações com apenas variáveis numéricas
    for num_cols in num_combinations:
        num_list = list(num_cols)
        feature_sets.append({
            "num": num_list,
            "cat": [],
            "x_cols": num_list,
            "name": f"num({'_'.join(num_list)})__cat(none)",
        })
    return feature_sets


def collect_fold_results(
    name: str, protocol: str, fold: str, test: pd.DataFrame, pred: np.ndarray, campeao: str
) -> tuple[dict, pd.DataFrame]:
    """Coleta e calcula as métricas para um único fold."""
    tmp = test.copy()
    tmp["pred"] = pred
    real_win = tmp.loc[tmp.groupby("ID_Observacao")["MAPE"].idxmin(), "Modelo"]
    scores = utils.regression_scores(tmp["MAPE"], pred)
    scores.update({
        "modelo": name,
        "protocolo": protocol,
        "fold": fold,
        "n": int(len(tmp)),
        "top1": top1_accuracy(tmp, "pred"),
        "top1_sempre_campeao": float((real_win.to_numpy() == campeao).mean()),
    })
    return scores, tmp


def run_all_models_for_fold(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """
    Executa todos os modelos para um determinado par de treino/teste.

    Retorna um dicionário com as predições e as equações de cada modelo.
    """
    predictions = {"baseline_mediana_modelo": (baseline_predict(train, test), [])}
    feature_sets = generate_feature_sets()

    for fset in feature_sets:
        num, cat, x_cols, name = fset["num"], fset["cat"], fset["x_cols"], fset["name"]

        # Define os pipelines
        pipe_ols = pipelines.ols_pipeline(num, cat)
        pipe_spl = pipelines.spline_pipeline(num, cat)

        # Executa os modelos e coleta predições/equações
        predictions[f"ols__{name}"] = modeling.fit_predict_2sls(
            pipe_ols, train, test, x_cols, num, cat, f"ols__{name}"
        )
        predictions[f"splines__{name}"] = modeling.fit_predict_2sls(
            pipe_spl, train, test, x_cols, num, cat, f"splines__{name}"
        )
        predictions[f"ols_simples__{name}"] = modeling.fit_predict_simples(
            pipe_ols, train, test, x_cols, f"ols_simples__{name}"
        )
        predictions[f"splines_simples__{name}"] = modeling.fit_predict_simples(
            pipe_spl, train, test, x_cols, f"splines_simples__{name}"
        )

    return predictions


def run_group_kfold(df: pd.DataFrame, n_splits: int = 5) -> tuple[list[dict], pd.DataFrame, list[str]]:
    """Executa a validação cruzada GroupKFold."""
    groups = df["Estudo"].to_numpy()
    cv = GroupKFold(n_splits=min(n_splits, df["Estudo"].nunique()))
    
    rows, oof_parts, all_equations = [], [], []
    for i, (tr_idx, te_idx) in enumerate(cv.split(df, groups=groups), 1):
        train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        campeao = campeao_do_fold(train)
        fold_name = f"gkf{i}"
        utils.log(f"Executando GroupKFold Fold {i}...")

        preds_and_eqs = run_all_models_for_fold(train, test)
        
        for name, (pred, equacoes) in preds_and_eqs.items():
            scores, _ = collect_fold_results(name, "GroupKFold", fold_name, test, pred, campeao)
            rows.append(scores)
            all_equations.extend(equacoes)
            
            part = test[["ID_Observacao", "Estudo", "Pais_Estudo", "Modelo", "MAPE"]].copy()
            part["pred"] = pred
            part["modelo_ml"] = name
            part["protocolo"] = "GroupKFold"
            part["fold"] = fold_name
            oof_parts.append(part)
            
    return rows, pd.concat(oof_parts, ignore_index=True), all_equations


def run_loso_paises(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, list[str]]:
    """Executa a validação cruzada Leave-One-Study-Out para países específicos."""
    estudos = df.loc[df["Pais_Estudo"].isin(config.PAISES_OCULTOS), "Estudo"].unique()
    
    rows, oof_parts, all_equations = [], [], []
    for estudo in estudos:
        test = df[df["Estudo"] == estudo].copy()
        train = df[df["Estudo"] != estudo].copy()
        if test.empty or train.empty:
            continue
            
        campeao = campeao_do_fold(train)
        utils.log(f"Executando LOSO para Estudo: {estudo}...")

        preds_and_eqs = run_all_models_for_fold(train, test)

        for name, (pred, equacoes) in preds_and_eqs.items():
            scores, _ = collect_fold_results(name, "LOSO_paises_case", str(estudo), test, pred, campeao)
            rows.append(scores)
            all_equations.extend(equacoes)

            part = test[["ID_Observacao", "Estudo", "Pais_Estudo", "Modelo", "MAPE"]].copy()
            part["pred"] = pred
            part["modelo_ml"] = name
            part["protocolo"] = "LOSO_paises_case"
            part["fold"] = str(estudo)
            oof_parts.append(part)
            
    return rows, pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(), all_equations


def summarize_results(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega os resultados dos folds."""
    utils.log(f"Agregando resultados dos folds...")
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
    """Calcula o VIF a partir de uma matriz de features."""
    X = X.astype(float).copy()
    # Remove colunas com variância zero
    keep = [c for c in X.columns if X[c].std(ddof=0) > 1e-12]
    X = X[keep]
    
    rows = []
    for col in X.columns:
        y = X[col]
        Z = X.drop(columns=[col])
        
        if Z.empty:
            vif, r2 = 1.0, 0.0
        else:
            lr = LinearRegression().fit(Z, y)
            r2 = lr.score(Z, y)
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 - 1e-9 else float("inf")
            
        rows.append({"bloco": None, "variavel": col, "r2_auxiliar": r2, "VIF": vif})
        
    return pd.DataFrame(rows)


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o VIF para as variáveis numéricas candidatas."""
    vif_num = vif_from_matrix(df[config.NUM_CANDIDATES])
    vif_num["bloco"] = "numericos_brutos"
    vif_num["alerta"] = np.where(vif_num["VIF"] >= 10, "VIF>=10", np.where(vif_num["VIF"] >= 5, "VIF>=5", "ok"))
    return vif_num.sort_values("VIF", ascending=False)


def cluster_bootstrap_error(oof: pd.DataFrame) -> pd.DataFrame:
    """Calcula o erro via bootstrap clusterizado por 'Estudo' sem reconstruir DataFrames em cada iteração."""
    utils.log(f"Calculando erro via bootstrap clusterizado por Estudo...")
    gkf = oof[oof["protocolo"] == "GroupKFold"].copy()
    if gkf.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(config.BOOT_SEED)
    rows = []

    for name, sub in gkf.groupby("modelo_ml", sort=False):
        sub = sub.sort_values(["Estudo", "ID_Observacao"]).copy()
        grouped = sub.groupby("Estudo", sort=False)
        studies = list(grouped.groups.keys())
        n_studies = len(studies)

        y_by_study = [grouped.get_group(s)["MAPE"].to_numpy(dtype=float) for s in studies]
        yhat_by_study = [grouped.get_group(s)["pred"].to_numpy(dtype=float) for s in studies]

        boot_idx = rng.choice(n_studies, size=(config.N_BOOT, n_studies), replace=True)
        mae_b = np.empty(config.N_BOOT, dtype=float)
        rmse_b = np.empty(config.N_BOOT, dtype=float)
        mape_b = np.empty(config.N_BOOT, dtype=float)

        for b in range(config.N_BOOT):
            idx = boot_idx[b]
            y_b = np.concatenate([y_by_study[i] for i in idx])
            yhat_b = np.concatenate([yhat_by_study[i] for i in idx])

            err = y_b - yhat_b
            mae_b[b] = float(np.mean(np.abs(err)))
            rmse_b[b] = float(np.sqrt(np.mean(err ** 2)))

            denom = np.abs(y_b)
            rel = np.divide(err, y_b, out=np.zeros_like(err, dtype=float), where=denom > 0)
            mape_b[b] = float(np.mean(np.abs(rel)))

        y_obs = sub["MAPE"].to_numpy(dtype=float)
        yhat_obs = sub["pred"].to_numpy(dtype=float)
        err_obs = y_obs - yhat_obs

        denom_obs = np.abs(y_obs)
        rel_obs = np.divide(err_obs, y_obs, out=np.zeros_like(err_obs, dtype=float), where=denom_obs > 0)

        rows.append({
            "protocolo": "GroupKFold",
            "modelo": name,
            "n_boot": config.N_BOOT,
            "agrupamento": "Estudo",
            "mape": float(np.mean(np.abs(rel_obs))),
            "mape_ic95_inf": float(np.quantile(mape_b, 0.025)),
            "mape_ic95_sup": float(np.quantile(mape_b, 0.975)),
            "mae": float(np.mean(np.abs(err_obs))),
            "mae_ic95_inf": float(np.quantile(mae_b, 0.025)),
            "mae_ic95_sup": float(np.quantile(mae_b, 0.975)),
            "rmse": float(np.sqrt(np.mean(err_obs ** 2))),
            "rmse_ic95_inf": float(np.quantile(rmse_b, 0.025)),
            "rmse_ic95_sup": float(np.quantile(rmse_b, 0.975)),
        })

    return pd.DataFrame(rows).sort_values("mae")


def filter_models(agg: pd.DataFrame) -> set[str]:
    """Filtra modelos com desempenho inferior à baseline."""
    utils.log(f"Filtrando modelos com desempenho inferior à baseline...")
    baseline_metrics = agg[agg["modelo"] == "baseline_mediana_modelo"].set_index("protocolo")[["mape", "mae"]].to_dict('index')
    
    models_to_keep = set(agg["modelo"])
    
    for _, row in agg.iterrows():
        protocol = row["protocolo"]
        if protocol in baseline_metrics:
            # Se o MAE ou MAPE for pior que a baseline, marca para remoção
            if row["mape"] > baseline_metrics[protocol]["mape"] or row["mae"] > baseline_metrics[protocol]["mae"]:
                models_to_keep.discard(row["modelo"])
                
    return models_to_keep


def main() -> None:
    """Orquestra a execução da modelagem e avaliação."""
    root = utils.find_root()
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = utils.load_frame(root)
    utils.log(f"Dados carregados: n={len(df)}, estudos={df['Estudo'].nunique()}")

    vif = compute_vif(df)
    vif.to_csv(out_dir / "04_vif.csv", index=False)
    utils.log(f"VIF calculado.")

    rows_g, oof_g, eqs_g = run_group_kfold(df)
    rows_l, oof_l, eqs_l = run_loso_paises(df)
    
    raw, agg = summarize_results(rows_g + rows_l)
    oof = pd.concat([oof_g, oof_l], ignore_index=True)
    all_equations = eqs_g + eqs_l

    boot = cluster_bootstrap_error(oof)
    
    # Filtra os modelos com base no desempenho
    # models_to_keep = filter_models(agg)
    # agg_f = agg[agg["modelo"].isin(models_to_keep)].copy()
    # raw_f = raw[raw["modelo"].isin(models_to_keep)].copy()
    # oof_f = oof[oof["modelo_ml"].isin(models_to_keep)].copy()
    # boot_f = boot[boot["modelo"].isin(models_to_keep)].copy()
    
    # Salva os resultados
    agg.to_csv(out_dir / "04_metricas_resumo.csv", index=False)
    raw.to_csv(out_dir / "04_metricas_folds.csv", index=False)
    oof.to_csv(out_dir / "04_predicoes_oof.csv", index=False)
    boot.to_csv(out_dir / "04_bootstrap_ic.csv", index=False)
    (out_dir / "04_equacoes_modelos.txt").write_text("\n".join(sorted(list(set(all_equations)))), encoding="utf-8")
    
    # Cria o JSON de resumo
    summary_json = {
        "metricas": agg.to_dict(orient="records"),
        "vif_numericos": vif.to_dict(orient="records"),
    }
    (out_dir / "04_resumo.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    utils.log("\n--- Resultados ---")
    utils.log("VIF (Numéricos):")
    utils.log(vif.to_string(index=False))
    utils.log("\nMelhores Modelos (Métricas Agregadas):")
    utils.log(agg.head(10).to_string(index=False))
    utils.log("\nBootstrap (IC 95% do Erro - GroupKFold):")
    utils.log(boot.head(10).to_string(index=False))
    utils.log(f"\nResultados salvos em: {out_dir}")


if __name__ == "__main__":
    main()