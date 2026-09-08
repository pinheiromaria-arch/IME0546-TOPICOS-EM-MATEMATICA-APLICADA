# RGNA — estratégia de solução implementada

Estudo mascarado da **Rede Global de Nutrição Animal**. Cada animal (`ID_Observacao`) aparece em **5 linhas** (`Modelo_1` … `Modelo_5`). O alvo é o **MAPE** daquele modelo naquela observação: quanto menor, melhor.

Enunciado: [`case.md`](case.md) (se existir no repositório). Diagrama da fase 04: [`fluxo-modelagem-04.md`](fluxo-modelagem-04.md).

## Dois momentos da matéria

| Momento | O que entregar | Dados | Situação |
|---|---|---|---|
| **Case (agora)** | EDA, hipóteses e apostas para os 5 países com estudo oculto | só `dados_treino.xlsx` | **Feito** (treino). Apostas registradas abaixo. Teste **lacrado**. |
| **Competição (depois)** | modelo que **prevê o MAPE** de cada linha | treino para ajustar; teste só na avaliação oficial | Baseline por mediana de modelo escolhido na validação; teste ainda não aberto. |

Não prever metano. Prever o **erro** dos cinco modelos empíricos.

## Ideia da estratégia (o que o código faz)

O MAPE muda muito **entre estudos** (ICC ≈ 0,54) e também tem um **patamar por animal** (cinco linhas compartilham o mesmo indivíduo). A pipeline implementada:

1. **Explorar** o treino sem olhar o teste (scripts 01–02).
2. **Preparar** uma tabela de modelagem sem `Estudo` como preditor e sem variáveis constantes (script 03).
3. **Validar por estudo** (`GroupKFold` e LOSO nos 5 países do case).
4. **Comparar** um baseline simples (mediana de MAPE **por `Modelo`** no fold) com regressores lineares.
5. Os regressores **não** preveem o MAPE bruto de uma vez. Preveem, no log:
   - o **patamar do animal** (mediana de `log(MAPE)`);
   - o **desvio de cada modelo empírico** em torno desse patamar;
   - e reconstroem `exp(patamar + desvio)`.
6. **Decidir** (script 05) se o seletor (`argmin` do MAPE previsto) substitui “sempre o campeão do fold”. No GroupKFold atual, o baseline ainda ganha em MAE; o seletor **não** substitui o campeão.

## Estrutura do repositório

```
projeto-rgna/
├── README.md
├── projeto-rgna.Rproj
├── .Rprofile
├── requirements.txt
├── relatorio/
├── data/raw/          # originais (não editar)
├── data/processed/    # tabelas, JSON e figuras
├── docs/
└── scripts/
```

- **01–03 e 06 em R:** exploração, features e figuras.
- **04–05 em Python:** validação agrupada, modelo em dois estágios, VIF, bootstrap e decisão.
- **`dados_teste.xlsx`:** fora de `raw` até confrontar as apostas.

## Regras que não negociamos

1. **Cortar validação por `Estudo`**, nunca por linha.
2. **Não usar `Estudo` como feature.** País entra em one-hot **dentro do fold** (`handle_unknown=ignore`).
3. **`Sistema_Producao`:** só Confinamento no treino → **não entra**.
4. **Baseline obrigatório:** mediana de MAPE **por `Modelo`** no treino do fold.
5. Amostra pequena (~223 animais). Target encoding de país **não** é usado.
6. Interações texto `Modelo|Status` e `Modelo|Sexo` no 03; one-hot no 04.

## Pipeline

```
01 EDA
  → 02 Nichos (país, status, sexo) e ICC entre estudos
    → 03 Features (cluster k-means no CSV; sem Estudo como preditor)
      → 04 Baseline (escala original) + dois estágios no log (OLS / splines / Ridge / Lasso / ElasticNet)
        → 05 Decisão: seletor vs sempre o campeão do fold
          → 06 Figuras da validação
```

Como rodar (pasta `projeto-rgna`):

```
Rscript scripts/01_eda_diagnostico.R
Rscript scripts/02_analise_bivariada.R
Rscript scripts/03_engenharia_features.R
python3 scripts/04_modelagem_validacao.py
python3 scripts/05_avaliacao_recomendacao.py
Rscript scripts/06_figuras_modelagem.R
Rscript scripts/render_relatorio.R
```

### 01 — EDA (`01_eda_diagnostico.R`)

Tipos, nulos, contagens; distribuição do MAPE; MAPE por modelo → **campeão global** (`01_campeao_global.txt`); `Sistema_Producao`. Figuras: histograma e boxplot.

### 02 — Nichos (`02_analise_bivariada.R`)

Vencedor por mediana em país, status e sexo; correlação com peso, consumo e frações; ANOVA/ICC por `Estudo`; tabelas dos 5 países do case. Figuras: boxplots e heatmap país × modelo.

### 03 — Features (`03_engenharia_features.R`)

Saída `data/processed/03_modelagem.csv`:

- **Fora do CSV de modelagem:** `Sistema_Producao`, `Fracao_Perda_B` (idêntica a A no treino). `Estudo` permanece só como grupo.
- **Inclui:** `Modelo_Status`, `Modelo_Sexo`, `MAPE_log`, peso, consumo, fração A, país, status, sexo, modelo.
- **Cluster:** k-means (k = 3, variáveis padronizadas: peso, consumo, fração A; `set.seed(123)`). A coluna `cluster` vai para o CSV, mas o script 04 **não** a usa (`# "cluster"` em `CAT`).
- **`categoria_paises`:** calculada no 03 (faixas por contagem de animais no país) e **não** entra no `select` final — o comentário no script marca vazamento se usada no treino completo.
- Encoding de país **não** é feito aqui: one-hot no 04.

### 04 — Validação (`04_modelagem_validacao.py`)

**Preditores efetivos (listas `NUM` / `CAT` no script):**

- Numéricos: `Peso_Corporal_kg`, `Fracao_Perda_A`. **`Consumo_MS_kg` está comentado** (não entra no modelo).
- Categóricos: `Pais_Estudo`, `Status_Metabolico`, `Sexo_Animal`, `Modelo`, `Modelo_Status`, `Modelo_Sexo`.

**Baseline:** mediana de MAPE por `Modelo` no treino do fold (escala original). Não passa pelos dois estágios.

**Regressores (OLS, splines, Ridge, Lasso, ElasticNet) — dois estágios em `fit_predict`:**

1. Uma linha por animal no treino. Alvo: mediana de `MAPE_log` do animal. Estimador: Ridge (α = 10), sem a coluna `Modelo` (as interações `Modelo_*` ainda estão nas categorias do animal).
2. Nas cinco linhas: alvo = `MAPE_log` menos a mediana de log do animal. O pipeline do fold (OLS / spline / penalizado) ajusta esse **resíduo**.
3. No teste: `MAPÊ = exp(patamar̂ + resíduô)`.

**Splines:** `SplineTransformer` (4 nós, grau 3) só em `Peso_Corporal_kg`; `Fracao_Perda_A` linear.

**Penalização:** no GroupKFold, α (e `l1_ratio` no ElasticNet) em CV interna `GroupKFold` (2 splits) sobre resíduos de log; o MAE de sintonização usa `exp(resíduô + mediana global de log no fold)`. No LOSO: Ridge α = 10, Lasso α = 0,01, ElasticNet α = 0,01 e `l1_ratio` = 0,5 (sem grid).

**Protocolos:** `GroupKFold` (5 folds) por `Estudo`; LOSO nos estudos dos 5 países do case.

**Diagnóstico:** VIF nos numéricos usados (peso e fração A). **Incerteza:** bootstrap por Estudo (2.000 réplicas) no OOF do GroupKFold.

**Métricas:** MAE, RMSE, \(R^2\), Top-1.

### 05 — Avaliação (`05_avaliacao_recomendacao.py`)

Grava `05_decisao.csv` / `.json`. O seletor só substitui “sempre o campeão do fold” se o Top-1 do **modelo de menor MAE** for **estritamente maior** que `top1_sempre_campeao` naquele protocolo.

As apostas do case vêm da **mediana no treino daquele país** (fase 02), não do ML.

### 06 — Figuras (`06_figuras_modelagem.R`)

MAE/RMSE com IC bootstrap, MAE por protocolo e por fold, Top-1, VIF, previsto vs observado, resíduos OOF.

## Achados 01–02 (treino; teste lacrado)

- 1.115 linhas, 223 animais, 93 estudos, 16 países; 5 linhas por ID.
- `Sistema_Producao` é só **Confinamento**.
- MAPE assimétrico (assimetria ≈ 1,61); sempre positivo.
- **Campeão global (menor mediana):** `Modelo_4` (mediana 0,156).
- O campeão **muda** por país e por status.
- `Fracao_Perda_A` = `Fracao_Perda_B` no treino.
- ICC entre estudos ≈ **0,54**.

## Apostas do case (após 01–02; não olhar o teste)

Critério: menor **mediana** de MAPE nos estudos ainda presentes no treino daquele país.

| País | Modelo apostado | Por quê |
|---|---|---|
| Campo Rico | Modelo_1 | Único dos cinco em que o vencedor por mediana no treino não é 3/4. |
| Ilha Verdejante | Modelo_4 | Vencedor local e campeão global. |
| Terra Nortenha | Modelo_4 | Vencedor local e campeão global. |
| Monção Dourada | Modelo_3 | Vencedor por mediana no treino desse país. |
| Costa Austral | Modelo_4 | Vencedor local e campeão global. |

Campeão global no treino: **Modelo_4**.

## Achados 03–05 (números atuais em `04_metricas_resumo.csv` / `05_decisao.json`)

No **GroupKFold**, o baseline continua com o menor MAE. Os regressores em dois estágios ficam um pouco atrás. Média dos folds:

| Protocolo | Melhor MAE | baseline | Ridge | Lasso | ElasticNet | OLS | Splines |
|---|---|---|---|---|---|---|---|
| GroupKFold (5) | baseline 0,140 | 0,140 | 0,145 | 0,146 | 0,146 | 0,147 | 0,149 |
| LOSO (38 estudos dos 5 países) | splines 0,115 | 0,117 | 0,116 | 0,120 | 0,118 | 0,115 | 0,115 |

IC 95% do MAE (bootstrap por Estudo, GroupKFold): baseline [0,122; 0,161]; Ridge [0,125; 0,171]; Lasso [0,125; 0,171]; ElasticNet [0,126; 0,170]; OLS [0,127; 0,170]; splines [0,128; 0,173]. Os intervalos se sobrepõem.

**VIF** (só peso e fração A, consumo fora do modelo): ambos ≈ 1,26 (ok).

No LOSO, splines/OLS/Ridge têm MAE médio ligeiramente **menor** que o baseline, mas a regra do 05 olha o Top-1 do modelo de menor MAE: splines 0,432 vs sempre-campeão 0,439 → **`seletor_supera_campeao`: falso**. Ridge no LOSO tem Top-1 0,445 (acima do campeão do fold), porém não é o de menor MAE, então não dispara a regra.

**Decisão vigente:** para erro no MAPE no protocolo principal (GroupKFold), fica o **baseline por modelo**. Para o case, as apostas da tabela acima. O seletor ML **não** substitui o campeão do fold.

## O que ainda não foi feito

- Abrir `dados_teste.xlsx` e confrontar apostas e baseline.
- Entregar predições na competição (se houver arquivo de submissão).
- Target encoding OOF e métrica de ordem além do Top-1.
- Usar `cluster` / `categoria_paises` / `Consumo_MS_kg` no 04 (existem no CSV ou no 03, mas estão desligados no estimador atual).
