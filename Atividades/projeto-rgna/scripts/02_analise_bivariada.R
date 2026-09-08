# Fase 2 — Nichos e variação entre estudos (treino apenas)
# Relatório consolidado: relatorio/relatorio-eda.Rmd

# Carrega o tema e as funções de leitura a partir de um caminho relativo simples.
# A função find_root() dentro de tema_rgna.R cuidará de localizar a raiz do projeto.
source(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), "tema_rgna.R"), encoding = "UTF-8")

root <- find_root()
df <- ler_treino(root)
theme_set(theme_rgna())
out_dir <- file.path(root, "data", "processed")
fig_dir <- file.path(out_dir, "figuras")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

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

tab_pais <- resumo_grupo(df, Pais_Estudo)
tab_status <- resumo_grupo(df, Status_Metabolico)
tab_sexo <- resumo_grupo(df, Sexo_Animal)

win_pais <- vencedor(tab_pais, Pais_Estudo)
win_status <- vencedor(tab_status, Status_Metabolico)
win_sexo <- vencedor(tab_sexo, Sexo_Animal)

# Ranking dentro do animal: quem tem o menor MAPE observado
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

num <- df |>
  select(MAPE, Peso_Corporal_kg, Consumo_MS_kg, Fracao_Perda_A, Fracao_Perda_B)
cors <- as.data.frame(cor(num, use = "complete.obs"))
cors$variavel <- rownames(cors)

# Variância entre vs dentro de estudo (ANOVA one-way)
aov_est <- aov(MAPE ~ Estudo, data = df)
ss <- summary(aov_est)[[1]]
var_estudo <- tibble(
  fonte = c("Estudo", "Residual (dentro)"),
  gl = ss[["Df"]],
  soma_quadrados = ss[["Sum Sq"]],
  fracao_ss = ss[["Sum Sq"]] / sum(ss[["Sum Sq"]])
)

# ICC aproximado: componentes de variância (método dos momentos)
ms <- ss[["Mean Sq"]]
n_por_estudo <- df |> count(Estudo)
n0 <- (sum(n_por_estudo$n) - sum(n_por_estudo$n^2) / sum(n_por_estudo$n)) / (nrow(n_por_estudo) - 1)
sigma2_w <- ms[2]
sigma2_b <- max(0, (ms[1] - ms[2]) / n0)
icc <- sigma2_b / (sigma2_b + sigma2_w)
icc_tab <- tibble(icc_estudo = icc, sigma2_entre = sigma2_b, sigma2_dentro = sigma2_w, n0 = n0)

write.csv(tab_pais, file.path(out_dir, "02_mape_pais_modelo.csv"), row.names = FALSE)
write.csv(tab_status, file.path(out_dir, "02_mape_status_modelo.csv"), row.names = FALSE)
write.csv(tab_sexo, file.path(out_dir, "02_mape_sexo_modelo.csv"), row.names = FALSE)
write.csv(win_pais, file.path(out_dir, "02_vencedor_pais.csv"), row.names = FALSE)
write.csv(win_status, file.path(out_dir, "02_vencedor_status.csv"), row.names = FALSE)
write.csv(win_sexo, file.path(out_dir, "02_vencedor_sexo.csv"), row.names = FALSE)
write.csv(vitorias_modelo, file.path(out_dir, "02_vitorias_por_modelo.csv"), row.names = FALSE)
write.csv(vitorias_pais_modelo, file.path(out_dir, "02_vitorias_pais_modelo.csv"), row.names = FALSE)
write.csv(tab_pais_oculto, file.path(out_dir, "02_mape_paises_estudo_oculto.csv"), row.names = FALSE)
write.csv(win_pais_oculto, file.path(out_dir, "02_vencedor_paises_estudo_oculto.csv"), row.names = FALSE)
write.csv(cors, file.path(out_dir, "02_correlacao_numericas.csv"), row.names = FALSE)
write.csv(var_estudo, file.path(out_dir, "02_anova_estudo.csv"), row.names = FALSE)
write.csv(icc_tab, file.path(out_dir, "02_icc_estudo.csv"), row.names = FALSE)

p_status <- ggplot(df, aes(x = Status_Metabolico, y = MAPE)) +
  geom_boxplot(fill = fill_soft, color = fill_main, outlier.alpha = 0.35, width = 0.55, linewidth = 0.4) +
  facet_wrap(~ Modelo, nrow = 1) +
  labs(title = "MAPE por status metabólico", x = NULL, y = "MAPE") +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 8))
ggsave(file.path(fig_dir, "02_mediana_status_modelo.png"), p_status, width = 9, height = 5, dpi = 140)

p_sexo <- ggplot(df, aes(x = Sexo_Animal, y = MAPE)) +
  geom_boxplot(fill = fill_soft, color = fill_main, outlier.alpha = 0.35, width = 0.55, linewidth = 0.4) +
  facet_wrap(~ Modelo, nrow = 1) +
  labs(title = "MAPE por sexo", x = NULL, y = "MAPE") +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 8))
ggsave(file.path(fig_dir, "02_mediana_sexo_modelo.png"), p_sexo, width = 9, height = 5, dpi = 140)

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
ggsave(file.path(fig_dir, "02_heatmap_pais_modelo.png"), p_heat, width = 8, height = 7.2, dpi = 140)

message("ICC (fração da variância do MAPE entre estudos): ", round(icc, 3))
message("Vencedores (mediana) nos 5 países do case:")
print(win_pais_oculto)
message("Saídas em ", out_dir)