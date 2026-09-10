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
from sklearn.model_selection import GroupKFold

from rgna import config, modeling, pipelines, utils


def top1_accuracy(frame: pd.DataFrame, pred_col: str) -> float:
    """Calcula a acurácia top-1."""
    real = (
        frame.sort_values(["ID_Observacao", "MAPE", "Modelo"])
        .drop_duplicates("ID_Observacao", keep="first")[["ID_Observacao", "Modelo"]]
    )
    hat = (
        frame.sort_values(["ID_Observacao", pred_col, "Modelo"])
        .drop_duplicates("ID_Observacao", keep="first")[["ID_Observacao", "Modelo"]]
    )
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
    feature_sets = []

    num_combinations = [combo for r in range(1, len(config.NUM_CANDIDATES) + 1) for combo in combinations(config.NUM_CANDIDATES, r)]
    cat_combinations = [combo for r in range(1, len(config.CAT_CANDIDATES) + 1) for combo in combinations(config.CAT_CANDIDATES, r)]

    for num_cols in num_combinations:
        num_list = list(num_cols)
        num_names = [config.map_columns.get(c, "-") for c in num_list]
        feature_sets.append(
            {
                "num": num_list,
                "cat": [],
                "x_cols": num_list,
                "name": f"({'_'.join(num_names)})",
            }
        )

    for cat_cols in cat_combinations:
        cat_list = list(cat_cols)
        cat_names = [config.map_columns.get(c, "-") for c in cat_list]
        feature_sets.append(
            {
                "num": [],
                "cat": cat_list,
                "x_cols": cat_list,
                "name": f"({'_'.join(cat_names)})",
            }
        )

    for num_cols, cat_cols in product(num_combinations, cat_combinations):
        num_list, cat_list = list(num_cols), list(cat_cols)
        num_names = [config.map_columns.get(c, "-") for c in num_list]
        cat_names = [config.map_columns.get(c, "-") for c in cat_list]
        feature_sets.append(
            {
                "num": num_list,
                "cat": cat_list,
                "x_cols": num_list + cat_list,
                "name": f"({'_'.join(num_names)})_({'_'.join(cat_names)})",
            }
        )
    return feature_sets


def collect_fold_results(
    name: str, protocol: str, fold: str, test: pd.DataFrame, pred: np.ndarray, campeao: str
) -> tuple[dict, pd.DataFrame]:
    """Coleta e calcula as métricas para um único fold."""
    tmp = test.copy()
    tmp["pred"] = pred
    real_win = (
        tmp.sort_values(["ID_Observacao", "MAPE", "Modelo"])
        .drop_duplicates("ID_Observacao", keep="first")["Modelo"]
    )
    scores = utils.regression_scores(tmp["MAPE"], pred)
    scores["mape"] = utils.mape(tmp["MAPE"], pred)
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
    weight_feature = "Peso_Corporal_kg"

    for fset in feature_sets:
        num, cat, x_cols, name = fset["num"], fset["cat"], fset["x_cols"], fset["name"]
        has_weight = weight_feature in x_cols

        # Define os pipelines
        pipe_ols = pipelines.ols_pipeline(num, cat)
        pipe_spl = pipelines.spline_pipeline(num, cat) if has_weight else None
        allow_2sls = "Modelo" in x_cols

        # Executa os modelos e coleta predições/equações
        if allow_2sls:
            model_key = f"ols_2s_{name}"
            predictions[model_key] = modeling.fit_predict_2sls(
                pipe_ols, train, test, x_cols, num, cat, model_key
            )

            if has_weight and pipe_spl is not None:
                model_key = f"spl_2s_{name}"
                predictions[model_key] = modeling.fit_predict_2sls(
                    pipe_spl, train, test, x_cols, num, cat, model_key
                )

        model_key = f"ols_{name}"
        predictions[model_key] = modeling.fit_predict_simples(
            pipe_ols, train, test, x_cols, model_key
        )

        if has_weight and pipe_spl is not None:
            model_key = f"spl_{name}"
            predictions[model_key] = modeling.fit_predict_simples(
                pipe_spl, train, test, x_cols, model_key
            )

    return predictions


def run_group_kfold(df: pd.DataFrame, n_splits: int = 5) -> tuple[list[dict], pd.DataFrame, list[str]]:
    """Executa a validação cruzada GroupKFold."""
    groups = df["Estudo"].to_numpy()
    n_groups = int(df["Estudo"].nunique())
    if n_groups < 2:
        raise ValueError("GroupKFold requer pelo menos 2 estudos distintos.")
    cv = GroupKFold(n_splits=min(n_splits, n_groups))
    test_rows = sum(len(te_idx) for _, te_idx in cv.split(df, groups=groups))
    if test_rows == 0:
        raise ValueError("GroupKFold não produziu folds de teste válidos.")
    
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
    if raw.empty:
        cols = ["protocolo", "modelo", "n_folds", "mape", "mae", "rmse", "r2", "top1", "top1_sempre_campeao"]
        return raw, pd.DataFrame(columns=cols)
    agg = (
        raw.groupby(["protocolo", "modelo"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            n=("n", "sum"),
            mape=("mape", lambda s: np.average(s, weights=raw.loc[s.index, "n"])),
            mae=("mae", lambda s: np.average(s, weights=raw.loc[s.index, "n"])),
            rmse=("rmse", lambda s: np.average(s, weights=raw.loc[s.index, "n"])),
            r2=("r2", lambda s: np.average(s, weights=raw.loc[s.index, "n"])),
            top1=("top1", lambda s: np.average(s, weights=raw.loc[s.index, "n"])),
            top1_sempre_campeao=("top1_sempre_campeao", lambda s: np.average(s, weights=raw.loc[s.index, "n"])),
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
    missing = [c for c in config.NUM_CANDIDATES if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas numéricas ausentes para VIF: {missing}")
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

            mape_b[b] = utils.mape(y_b, yhat_b)

        y_obs = sub["MAPE"].to_numpy(dtype=float)
        yhat_obs = sub["pred"].to_numpy(dtype=float)
        err_obs = y_obs - yhat_obs

        rows.append({
            "protocolo": "GroupKFold",
            "modelo": name,
            "n_boot": config.N_BOOT,
            "agrupamento": "Estudo",
            "mape": utils.mape(y_obs, yhat_obs),
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
    baseline_metrics = agg[agg["modelo"] == "baseline_mediana_modelo"].set_index("protocolo")[["mape", "mae", "rmse"]].to_dict("index")
    
    models_to_keep = set(agg["modelo"])
    
    for _, row in agg.iterrows():
        protocol = row["protocolo"]
        if protocol in baseline_metrics:
            if row["modelo"] == "baseline_mediana_modelo":
                continue
            if (
                row["mape"] > baseline_metrics[protocol]["mape"]
                # and row["mae"] > baseline_metrics[protocol]["mae"]
                # and row["rmse"] > baseline_metrics[protocol]["rmse"]
            ):
                models_to_keep.discard(row["modelo"])
                
    return models_to_keep


def top_models_union_after_median_filter(agg: pd.DataFrame) -> set[str]:
    """
    Retorna a união dos top-N modelos de LOSO e GroupKFold após o filtro por baseline.

    O N é parametrizado por config.TOP_N_MODELOS_POR_PROTOCOLO.
    A baseline mediana simples deve permanecer sempre no conjunto final para permitir
    comparação direta com os modelos candidatos.
    """
    n_top = int(getattr(config, "TOP_N_MODELOS_POR_PROTOCOLO", 5))
    if n_top <= 0 or agg.empty:
        return set()

    baseline_name = "baseline_mediana_modelo"
    selected: set[str] = set()
    if baseline_name in agg["modelo"].unique():
        selected.add(baseline_name)

    for protocolo in ("LOSO_paises_case", "GroupKFold"):
        pd_proto = agg[agg["protocolo"] == protocolo]
        if pd_proto.empty:
            continue
        if baseline_name in pd_proto["modelo"].unique():
            selected.add(baseline_name)

        top = (
            pd_proto
            .sort_values(["mape", "mae", "rmse", "modelo"], ascending=[True, True, True, True])
            .head(n_top)
        )
        selected.update(top["modelo"].tolist())
    return selected


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
    oof_parts = [part for part in [oof_g, oof_l] if not part.empty]
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    all_equations = eqs_g + eqs_l

    boot = cluster_bootstrap_error(oof) if not oof.empty else pd.DataFrame()
    
    # Filtra os modelos com base no desempenho, por protocolo
    keep_mask = pd.Series(False, index=agg.index)
    for protocol in agg["protocolo"].unique():
        proto = agg[agg["protocolo"] == protocol].copy()
        models_to_keep = filter_models(proto)
        keep_mask |= agg["protocolo"].eq(protocol) & agg["modelo"].isin(models_to_keep)

    agg_f = agg[keep_mask].copy()

    # Segundo filtro: mantém apenas a união dos top-N de LOSO e top-N de GroupKFold.
    top_union_models = top_models_union_after_median_filter(agg_f)
    if top_union_models:
        agg_f = agg_f[agg_f["modelo"].isin(top_union_models)].copy()

    raw_f = raw.merge(agg_f[["protocolo", "modelo"]].drop_duplicates(), on=["protocolo", "modelo"], how="inner")
    keep_pairs = agg_f[["protocolo", "modelo"]].drop_duplicates()
    oof_f = oof.merge(keep_pairs, left_on=["protocolo", "modelo_ml"], right_on=["protocolo", "modelo"], how="inner")
    oof_f = oof_f.drop(columns=["modelo"], errors="ignore")
    boot_f = boot.merge(keep_pairs, on=["protocolo", "modelo"], how="inner")
    
    # Salva os resultados
    agg_f.to_csv(out_dir / "04_metricas_resumo.csv", index=False)
    raw_f.to_csv(out_dir / "04_metricas_folds.csv", index=False)
    oof_f.to_csv(out_dir / "04_predicoes_oof.csv", index=False)
    boot_f.to_csv(out_dir / "04_bootstrap_ic.csv", index=False)
    (out_dir / "04_equacoes_modelos.txt").write_text("\n".join(sorted(list(set(all_equations)))), encoding="utf-8")
    
    # Cria o JSON de resumo
    summary_json = {
        "metricas": agg_f.to_dict(orient="records"),
        "vif_numericos": vif.to_dict(orient="records"),
    }
    (out_dir / "04_resumo.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    utils.log("\n--- Resultados ---")
    utils.log("VIF (Numéricos):")
    utils.log(vif.to_string(index=False))
    utils.log("\nMelhores Modelos (Métricas Agregadas):")
    utils.log(agg_f.head(10).to_string(index=False))
    utils.log("\nBootstrap (IC 95% do Erro - GroupKFold):")
    utils.log(boot_f.head(10).to_string(index=False))
    utils.log(f"\nResultados salvos em: {out_dir}")


if __name__ == "__main__":
    main()