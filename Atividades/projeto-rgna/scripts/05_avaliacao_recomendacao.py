"""Fase 5 — Consolidar métricas e decidir se o seletor de modelos supera a baseline."""

from __future__ import annotations

import json
import pandas as pd

from rgna import utils


def criar_tabela_decisao(resumo: pd.DataFrame, campeao_global: str) -> pd.DataFrame:
    """
    Cria a tabela de decisão final de forma vetorizada com pandas.

    Args:
        resumo: DataFrame com as métricas agregadas dos modelos.
        campeao_global: O nome do modelo campeão global.

    Returns:
        DataFrame com a análise de decisão por protocolo.
    """
    # Encontra o melhor modelo (menor mape) para cada protocolo
    best_models_idx = resumo.groupby("protocolo")["mape"].idxmin()
    best_models = resumo.loc[best_models_idx].set_index("protocolo")

    # Isola a baseline para cada protocolo
    baseline = resumo[resumo["modelo"] == "baseline_mediana_modelo"].set_index("protocolo")

    # Junta as informações do melhor modelo e da baseline
    decisao = best_models.join(
        baseline,
        lsuffix="_melhor",
        rsuffix="_baseline"
    )

    # Calcula as colunas de decisão
    decisao["ganho_mape_vs_baseline"] = decisao["mape_baseline"] - decisao["mape_melhor"]
    # Adiciona 1e-9 para estabilidade numérica ao comparar floats
    decisao["seletor_supera_campeao"] = decisao["top1_melhor"] > (decisao["top1_sempre_campeao_melhor"] + 1e-9)
    decisao["campeao_global_treino"] = campeao_global
    
    # Renomeia e seleciona as colunas finais
    decisao = decisao.rename(columns={"modelo_melhor": "melhor_modelo_mape"})
    colunas_finais = [
        "melhor_modelo_mape",
        "mape_melhor",
        "mape_baseline",
        "ganho_mape_vs_baseline",
        "top1_melhor",
        "top1_sempre_campeao_melhor",
        "seletor_supera_campeao",
        "campeao_global_treino",
    ]
    decisao = decisao[colunas_finais].reset_index()

    # Pivota a tabela de resumo para ter mape de cada modelo como uma coluna
    mape_pivot = resumo.pivot_table(index="protocolo", columns="modelo", values="mape").reset_index()
    
    # Junta a tabela de decisão com os mapes pivotados
    decisao_final = pd.merge(decisao, mape_pivot, on="protocolo")
    
    return decisao_final


def main() -> None:
    """Orquestra a avaliação e a criação dos artefatos de decisão."""
    root = utils.find_root()
    out_dir = root / "data" / "processed"
    
    # Carrega os artefatos da fase de modelagem
    try:
        resumo = pd.read_csv(out_dir / "04_metricas_resumo.csv")
        campeao_global = (out_dir / "01_campeao_global.txt").read_text(encoding="utf-8").strip()
        boot_path = out_dir / "04_bootstrap_ic.csv"
        boot = pd.read_csv(boot_path) if boot_path.exists() else pd.DataFrame()
    except FileNotFoundError as e:
        print(f"Erro: Arquivo necessário não encontrado. Rode o script 04 antes. Detalhe: {e}")
        return

    # Cria a tabela de decisão
    tabela_decisao = criar_tabela_decisao(resumo, campeao_global)

    # Salva os resultados
    tabela_decisao.to_csv(out_dir / "05_decisao.csv", index=False)

    # Cria o payload JSON para o resumo
    payload = {
        "decisao": tabela_decisao.to_dict(orient="records"),
    }
    if not boot.empty:
        payload["bootstrap_ic95"] = boot.to_dict(orient="records")
        
    (out_dir / "05_decisao.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Imprime os resultados no console
    print("--- Análise de Decisão ---")
    print("Campeão Global (mediana no treino completo):", campeao_global)
    print("\nTabela de Decisão por Protocolo:")
    print(tabela_decisao.to_string(index=False))
    
    if not boot.empty:
        print("\nIC 95% Bootstrap (MAPE, GroupKFold por Estudo):")
        print(boot.to_string(index=False))
        
    print("\nRegra de Decisão:")
    print("O seletor (argmin do MAPE previsto) é considerado superior se sua acurácia Top-1")
    print("for estritamente maior que a do baseline ('sempre campeão') naquele protocolo.")
    print(f"\nResultados salvos em: {out_dir}")


if __name__ == "__main__":
    main()