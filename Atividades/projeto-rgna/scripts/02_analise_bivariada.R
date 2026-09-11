# Fase 2 — Nichos e variação entre estudos (treino apenas)
# Relatório consolidado: relatorio/relatorio-eda.Rmd

# Carrega o tema e as funções de leitura a partir do caminho do próprio script.
source(file.path(dirname(this.path::this.path()), "tema_rgna.R"), encoding = "UTF-8")

# --- Setup ---
root <- find_root()
df <- ler_treino(root)
theme_set(theme_rgna())
out_dir <- file.path(root, "data", "processed")

# --- Funções de Análise ---
resumo_grupo <- function(data, grupo) {
  data |>
    group_by({{ grupo }}, Modelo) |>
    summarise(n = n(), media = mean(MAPE), mediana = median(MAPE), .groups = "drop")
}

vencedor <- function(tab, grupo) {
  tab |>
    group_by({{ grupo }}) |>
    slice_min(mediana, n = 1, with_ties = TRUE) |>
    ungroup()
}

# --- Análises ---
tab_pais <- resumo_grupo(df, Pais_Estudo)
tab_status <- resumo_grupo(df, Status_Metabolico)
tab_sexo <- resumo_grupo(df, Sexo_Animal)

win_pais <- vencedor(tab_pais, Pais_Estudo)
win_status <- vencedor(tab_status, Status_Metabolico)
win_sexo <- vencedor(tab_sexo, Sexo_Animal)

win_id <- df |>
  group_by(ID_Observacao) |>
  slice_min(MAPE, n = 1, with_ties = TRUE) |>
  ungroup()

vitorias_modelo <- win_id |>
  count(Modelo, name = "n_vitorias") |>
  mutate(fracao = n_vitorias / sum(n_vitorias)) |>
  arrange(desc(n_vitorias))

vitorias_pais_modelo <- win_id |>
  distinct(ID_Observacao, Pais_Estudo, Modelo) |>
  count(Pais_Estudo, Modelo, name = "n_vitorias") |>
  group_by(Pais_Estudo) |>
  mutate(fracao = n_vitorias / sum(n_vitorias)) |>
  ungroup()

tab_pais_oculto <- tab_pais |>
  filter(Pais_Estudo %in% paises_ocultos) |>
  arrange(Pais_Estudo, mediana)

win_pais_oculto <- win_pais |>
  filter(Pais_Estudo %in% paises_ocultos)

cors <- as.data.frame(cor(select(df, where(is.numeric)), use = "complete.obs"))
cors$variavel <- rownames(cors)

# Variância entre vs dentro de estudo (ANOVA e ICC)
aov_est <- aov(MAPE ~ Estudo, data = df)
ss <- summary(aov_est)[[1]]

var_estudo <- tibble(
  fonte = c("Estudo", "Residual (dentro)"),
  gl = ss[["Df"]],
  soma_quadrados = ss[["Sum Sq"]],
  fracao_ss = ss[["Sum Sq"]] / sum(ss[["Sum Sq"]])
)

# Note. Qual abordagem metodológica utilizar para calcular o ICC? Também posso usar a lib library(lme4)
# # ICC aproximado: componentes de variância (método dos momentos)
# ms <- ss[["Mean Sq"]]
# n_por_estudo <- df |> count(Estudo)
# n0 <- (sum(n_por_estudo$n) - sum(n_por_estudo$n^2) / sum(n_por_estudo$n)) / (nrow(n_por_estudo) - 1)
# sigma2_w <- ms[2]
# sigma2_b <- max(0, (ms[1] - ms[2]) / n0)
# icc <- sigma2_b / (sigma2_b + sigma2_w)
# icc_tab <- tibble(icc_estudo = icc, sigma2_entre = sigma2_b, sigma2_dentro = sigma2_w, n0 = n0)

# Cálculo manual do ICC (1,1) via Quadrados Médios (MS) da ANOVA
ms_entre <- ss[["Mean Sq"]][1]
ms_res <- ss[["Mean Sq"]][2]

# k_0 é o tamanho médio Harmônico dos grupos para dados desbalanceados
n_obs <- length(df$MAPE)
n_grupos <- length(unique(df$Estudo))
k_0 <- (n_obs - sum(table(df$Estudo)^2) / n_obs) / (n_grupos - 1)

# Variâncias
var_entre <- max(0, (ms_entre - ms_res) / k_0)
var_dentro <- ms_res
icc_val <- var_entre / (var_entre + var_dentro)

icc_tab <- tibble(icc_estudo = icc_val)

# --- Salvando Saídas ---
outputs_csv <- list(
  "02_mape_pais_modelo" = tab_pais,
  "02_mape_status_modelo" = tab_status,
  "02_mape_sexo_modelo" = tab_sexo,
  "02_vencedor_pais" = win_pais,
  "02_vencedor_status" = win_status,
  "02_vencedor_sexo" = win_sexo,
  "02_vitorias_por_modelo" = vitorias_modelo,
  "02_vitorias_pais_modelo" = vitorias_pais_modelo,
  "02_mape_paises_estudo_oculto" = tab_pais_oculto,
  "02_vencedor_paises_estudo_oculto" = win_pais_oculto,
  "02_correlacao_numericas" = cors,
  "02_anova_estudo" = var_estudo,
  "02_icc_estudo" = icc_tab
)

iwalk(outputs_csv, ~ write.csv(.x, file.path(out_dir, paste0(.y, ".csv")), row.names = FALSE))

# --- Gráficos ---
p_status <- ggplot(df, aes(x = Status_Metabolico, y = MAPE)) +
  geom_boxplot(fill = fill_soft, color = fill_main, outlier.alpha = 0.35, width = 0.55, linewidth = 0.4) +
  facet_wrap(~ Modelo, nrow = 1) +
  labs(title = "MAPE por status metabólico", x = NULL, y = "MAPE") +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 8))

p_sexo <- ggplot(df, aes(x = Sexo_Animal, y = MAPE)) +
  geom_boxplot(fill = fill_soft, color = fill_main, outlier.alpha = 0.35, width = 0.55, linewidth = 0.4) +
  facet_wrap(~ Modelo, nrow = 1) +
  labs(title = "MAPE por sexo", x = NULL, y = "MAPE") +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 8))

ord_pais <- tab_pais |>
  group_by(Pais_Estudo) |>
  summarise(m = min(mediana), .groups = "drop") |>
  arrange(m) |>
  pull(Pais_Estudo)
heat <- tab_pais |>
  mutate(Pais_Estudo = factor(Pais_Estudo, levels = ord_pais))
p_heat <- ggplot(heat, aes(x = Modelo, y = Pais_Estudo, fill = mediana)) +
  geom_tile(color = paper, linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", mediana), color = mediana > 0.22), size = 2.7) +
  scale_fill_gradient(low = "#eef1ee", high = fill_main, name = "Mediana") +
  scale_color_manual(values = c("TRUE" = "white", "FALSE" = ink), guide = "none") +
  labs(title = "Mediana do MAPE: país × modelo", x = NULL, y = NULL) +
  theme(panel.grid = element_blank(), axis.line = element_blank(), axis.ticks = element_blank(), legend.position = "right")

# --- Mensagens Finais ---
message("ICC (fração da variância do MAPE entre estudos): ", round(icc_tab$icc_estudo, 3))
message("Vencedores (mediana) nos 5 países do case:")
print(win_pais_oculto)
message("Saídas em ", out_dir)