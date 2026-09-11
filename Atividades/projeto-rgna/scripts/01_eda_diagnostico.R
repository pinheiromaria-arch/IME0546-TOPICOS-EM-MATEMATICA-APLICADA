# Fase 1 — EDA e inspeção do alvo (treino apenas)
# Relatório consolidado: relatorio/relatorio-eda.Rmd

# Carrega o tema e as funções de leitura a partir do caminho do próprio script.
source(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), "tema_rgna.R"), encoding = "UTF-8")

# --- Setup ---
root <- find_root()
df <- ler_treino(root)
theme_set(theme_rgna())
out_dir <- file.path(root, "data", "processed")

# --- Análise ---
nulos <- tibble(
  coluna = names(df),
  n_na = colSums(is.na(df)),
  classe = vapply(df, function(x) paste(class(x), collapse = ","), character(1))
)

contagens <- tibble(
  n_linhas = nrow(df),
  n_id = n_distinct(df$ID_Observacao),
  n_estudo = n_distinct(df$Estudo),
  n_pais = n_distinct(df$Pais_Estudo),
  n_modelo = n_distinct(df$Modelo),
  linhas_por_id_min = min(count(df, ID_Observacao)$n),
  linhas_por_id_max = max(count(df, ID_Observacao)$n)
)

sistema <- as.data.frame(table(df$Sistema_Producao, useNA = "ifany"), stringsAsFactors = FALSE)
names(sistema) <- c("Sistema_Producao", "n")
sistema$constante <- nrow(sistema) == 1L

mape_global <- df |>
  summarise(
    n = n(),
    media = mean(MAPE),
    mediana = median(MAPE),
    desvio = sd(MAPE),
    minimo = min(MAPE),
    maximo = max(MAPE),
    q1 = quantile(MAPE, 0.25),
    q3 = quantile(MAPE, 0.75),
    assimetria = mean((MAPE - mean(MAPE))^3) / (sd(MAPE)^3),
    n_nao_positivo = sum(MAPE <= 0)
  )

mape_modelo <- df |>
  group_by(Modelo) |>
  summarise(
    n = n(),
    media = mean(MAPE),
    mediana = median(MAPE),
    desvio = sd(MAPE),
    .groups = "drop"
  ) |>
  arrange(mediana)

campeao <- as.character(mape_modelo$Modelo[which.min(mape_modelo$mediana)])

estudos_pais <- df |>
  group_by(Pais_Estudo) |>
  summarise(
    n_estudo = n_distinct(Estudo),
    n_id = n_distinct(ID_Observacao),
    n_linhas = n(),
    .groups = "drop"
  ) |>
  arrange(n_estudo, Pais_Estudo)

# --- Salvando Saídas ---

# Agrupa todos os dataframes a serem salvos em uma lista nomeada
outputs_csv <- list(
  "01_nulos_tipos" = nulos,
  "01_contagens" = contagens,
  "01_sistema_producao" = sistema,
  "01_mape_global" = mape_global,
  "01_mape_por_modelo" = mape_modelo,
  "01_estudos_por_pais" = estudos_pais
)

# Itera sobre a lista e salva cada dataframe como um CSV
iwalk(outputs_csv, ~ write.csv(.x, file.path(out_dir, paste0(.y, ".csv")), row.names = FALSE))

# Salva o campeão global em um arquivo de texto
writeLines(campeao, file.path(out_dir, "01_campeao_global.txt"))

# --- Gráficos ---
p_hist <- ggplot(df, aes(x = MAPE)) +
  geom_histogram(bins = 36, fill = fill_main, color = paper, linewidth = 0.2) +
  geom_vline(xintercept = median(df$MAPE), linetype = "22", color = ink, linewidth = 0.4) +
  labs(title = "Distribuição do MAPE", x = "MAPE", y = "Frequência")

p_box <- ggplot(df, aes(x = Modelo, y = MAPE)) +
  geom_boxplot(
    fill = fill_soft, color = fill_main, outlier.colour = muted,
    outlier.alpha = 0.45, outlier.size = 1.1, width = 0.55, linewidth = 0.45
  ) +
  labs(title = "MAPE por modelo empírico", x = NULL, y = "MAPE")

# --- Mensagens Finais ---
message("Campeão global (menor mediana de MAPE): ", campeao)
message("Sistema_Producao constante? ", sistema$constante[1], " -> ", paste(sistema$Sistema_Producao, collapse = ", "))
message("Saídas em ", out_dir)