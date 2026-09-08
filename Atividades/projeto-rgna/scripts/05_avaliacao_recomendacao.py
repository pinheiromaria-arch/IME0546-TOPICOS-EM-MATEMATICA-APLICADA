"""Fase 5 — consolidar métricas e decidir se o seletor bate o campeão global."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


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


def main() -> None:
    root = find_root()
    out = root / "data" / "processed"
    resumo = pd.read_csv(out / "04_metricas_resumo.csv")
    campeao = (out / "01_campeao_global.txt").read_text(encoding="utf-8").strip()
    boot_path = out / "04_bootstrap_ic.csv"
    boot = pd.read_csv(boot_path) if boot_path.exists() else pd.DataFrame()

    linhas = []
    for proto in resumo["protocolo"].unique():
        sub = resumo[resumo["protocolo"] == proto]
        best = sub.loc[sub["mae"].idxmin()]
        base = sub[sub["modelo"] == "baseline_mediana_modelo"].iloc[0]
        seletor_ok = float(best["top1"]) > float(best["top1_sempre_campeao"]) + 1e-9
        linha = {
            "protocolo": proto,
            "melhor_mae": best["modelo"],
            "mae_melhor": best["mae"],
            "mae_baseline": base["mae"],
            "ganho_mae_vs_baseline": float(base["mae"]) - float(best["mae"]),
            "top1_melhor": best["top1"],
            "top1_sempre_campeao": best["top1_sempre_campeao"],
            "seletor_supera_campeao": seletor_ok,
            "campeao_global_treino": campeao,
        }
        for _, row in sub.iterrows():
            linha[f"mae_{row['modelo']}"] = row["mae"]
        linhas.append(linha)

    dec = pd.DataFrame(linhas)
    dec.to_csv(out / "05_decisao.csv", index=False)
    payload = {"decisao": linhas}
    if not boot.empty:
        payload["bootstrap_ic95"] = boot.to_dict(orient="records")
    (out / "05_decisao.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Campeao global (mediana no treino completo):", campeao)
    print(dec.to_string(index=False))
    if not boot.empty:
        print("")
        print("IC 95% bootstrap (MAE, GroupKFold por Estudo):")
        print(boot.to_string(index=False))
    print("")
    print("Regra: o seletor (argmin do MAPE previsto) so substitui")
    print("sempre o campeao se Top-1 for maior naquele protocolo.")
    print(f"Saidas em {out}")


if __name__ == "__main__":
    main()
