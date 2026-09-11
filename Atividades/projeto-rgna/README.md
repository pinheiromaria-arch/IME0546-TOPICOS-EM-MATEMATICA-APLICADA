# Projeto RGNA (IME0546)

Metodologia: [`docs/proposta.md`](docs/proposta.md). Enunciado: [`docs/case.md`](docs/case.md).

Relatório consolidado (EDA, nichos, features, validação e apostas): [`relatorio/relatorio-eda.html`](relatorio/relatorio-eda.html). PDF: rode `Rscript scripts/render_relatorio.R` (gera HTML e PDF via Chrome; precisa de `rmarkdown` e `pagedown`).

Raiz do projeto: `projeto-rgna.Rproj` (RStudio) e `.Rprofile` (opções + `RGNA_ROOT`). Os scripts sobem até essa pasta; rode a partir dela:

```
cd caminho/para/projeto-rgna
Rscript scripts/01_eda_diagnostico.R
Rscript scripts/02_analise_bivariada.R
Rscript scripts/03_engenharia_features.R
python3 scripts/04_modelagem_validacao.py
Rscript scripts/04_modelagem_validacao_monolitico.R --use-existing-phase3
python3 scripts/05_avaliacao_recomendacao.py
Rscript scripts/run_pipeline.R
```

Para orquestrar toda a pipeline principal de análise de uma vez, use:

```bash
Rscript scripts/run_pipeline.R
```

Se precisar, você pode trocar os executáveis com variáveis de ambiente, por exemplo `PYTHON_BIN=python` ou `R_SCRIPT_BIN=/opt/homebrew/bin/Rscript`.

Para gerar uma versão monolítica em R das fases 03 e 04 sem substituir o fluxo atual, use:

```bash
Rscript scripts/04_modelagem_validacao_monolitico.R --use-existing-phase3
```

Esse script preserva a execução atual da fase 04 em Python e grava artefatos paralelos com sufixo `_r_monolitico` em `data/processed/`.
