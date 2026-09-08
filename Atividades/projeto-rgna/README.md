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
python3 scripts/05_avaliacao_recomendacao.py
Rscript scripts/06_figuras_modelagem.R
Rscript scripts/render_relatorio.R
```

No Cursor, o `.Rproj` não muda o diretório sozinho: use o terminal na pasta do projeto (ou abra essa pasta como workspace).

No PyCharm, o working directory costuma ser a pasta da disciplina (`IME0546...`), não `scripts/`. Os scripts agora procuram `tema_rgna.R` e a raiz sozinhos. Se ainda falhar, na configuração de Run do R defina **Working directory** para a pasta `projeto-rgna` (a que contém `projeto-rgna.Rproj`).

Teste lacrado até as apostas em `docs/proposta.md`.
