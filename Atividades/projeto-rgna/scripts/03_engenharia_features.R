# Fase 3 — Features para modelagem (treino)
# Relatório consolidado: relatorio/relatorio-eda.Rmd
# Estudo permanece só como grupo de validação, não como preditor.

# Carrega o tema e as funções de leitura a partir do caminho do próprio script.
source(file.path(dirname(this.path::this.path()), "tema_rgna.R"), encoding = "UTF-8")

# --- Setup ---
root <- find_root()
df <- ler_treino(root)
out_dir <- file.path(root, "data", "processed")
library(MASS) # Para boxcox

# --- Análises ---

# Verificações iniciais: Frações A e B são idênticas, Sistema_Producao é constante.
stopifnot(isTRUE(all.equal(df$Fracao_Perda_A, df$Fracao_Perda_B)))
if (n_distinct(df$Sistema_Producao) != 1L) {
  warning("Sistema_Producao não é constante: ", paste(unique(df$Sistema_Producao), collapse = ", "))
}

# Transformação Box-Cox para a variável resposta (MAPE)
box_result <- boxcox(MAPE ~ 1, data = df, lambda = seq(-2, 2, 0.1), plotit = FALSE)
best_lambda <- box_result$x[which.max(box_result$y)]

# Criação da base de modelagem
modelagem <- df |>
  mutate(
    MAPE_log = log(MAPE),
    MAPE_boxcox = if (best_lambda == 0) log(MAPE) else (MAPE^best_lambda - 1) / best_lambda
  ) |>
  dplyr::select(
    ID_Observacao,
    Estudo,
    Pais_Estudo,
    Status_Metabolico,
    Sexo_Animal,
    Peso_Corporal_kg,
    Consumo_MS_kg,
    Fracao_Perda_A,
    Modelo,
    MAPE,
    MAPE_log,
    MAPE_boxcox
  )

# Modelo campeão (menor mediana de MAPE global)
campeao <- df |>
  group_by(Modelo) |>
  summarise(mediana = median(MAPE), .groups = "drop") |>
  slice_min(mediana, n = 1, with_ties = FALSE) |>
  pull(Modelo) |>
  as.character()

# Metadados do processo para rastreabilidade
meta <- tibble(
  n_linhas = nrow(modelagem),
  n_id = n_distinct(modelagem$ID_Observacao),
  n_estudo = n_distinct(modelagem$Estudo),
  campeao_global = campeao,
  lambda_boxcox = best_lambda,
  colunas_excluidas = "Sistema_Producao; Fracao_Perda_B",
  info = "Estudo usado apenas como grupo de CV. País será one-hot encoded."
)

# --- Salvando Saídas ---
outputs_csv <- list(
  "03_modelagem" = modelagem,
  "03_meta" = meta
)
iwalk(outputs_csv, ~ write.csv(.x, file.path(out_dir, paste0(.y, ".csv")), row.names = FALSE, fileEncoding = "UTF-8"))
writeLines(campeao, file.path(out_dir, "01_campeao_global.txt"))

# --- Gráficos ---
p_dens_boxcox <- ggplot(modelagem, aes(x = MAPE_boxcox, color = Modelo)) +
  geom_density(adjust = 1.1, linewidth = 0.55) +
  scale_color_manual(values = pal_modelo) +
  labs(
    title = "Densidade do MAPE por modelo (transformado por Box-Cox)",
    x = paste0("MAPE (Box-Cox, λ = ", round(best_lambda, 2), ")"),
    y = "Densidade",
    caption = "Fonte: dados de treino. Kernel gaussiano."
  )

# --- Mensagens Finais ---
message("Campeão global (mediana): ", campeao)
message("Lambda de Box-Cox para MAPE: ", round(best_lambda, 3))
message("Saídas salvas em ", out_dir)
