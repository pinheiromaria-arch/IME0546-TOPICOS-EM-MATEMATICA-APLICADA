# Fase 1 — EDA e inspeção do alvo (treino apenas)
# Relatório consolidado: relatorio/relatorio-eda.Rmd

locate_tema <- function() {
  cands <- character()
  push <- function(p) if (length(p) && nzchar(p[[1]])) cands <<- c(cands, p)
  for (i in seq_len(max(1L, sys.nframe()))) {
    ofile <- tryCatch(sys.frame(i)$ofile, error = function(e) NULL)
    if (!is.null(ofile) && nzchar(ofile)) {
      push(file.path(dirname(normalizePath(ofile, mustWork = FALSE)), "tema_rgna.R"))
    }
  }
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grepl("^--file=", args)])
  if (length(f)) push(file.path(dirname(normalizePath(f[[1]], mustWork = FALSE)), "tema_rgna.R"))
  wd <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  d <- wd
  for (i in seq_len(12)) {
    push(file.path(d, "tema_rgna.R"))
    push(file.path(d, "scripts", "tema_rgna.R"))
    push(file.path(d, "projeto-rgna", "scripts", "tema_rgna.R"))
    push(file.path(d, "Atividades", "projeto-rgna", "scripts", "tema_rgna.R"))
    parent <- dirname(d)
    if (identical(parent, d)) break
    d <- parent
  }
  cands <- unique(cands[file.exists(cands)])
  cands <- cands[!grepl("projeto-rgna3", cands)]
  if (!length(cands)) {
    stop("Não achei scripts/tema_rgna.R. Working directory atual: ", wd)
  }
  normalizePath(cands[[1]], winslash = "/", mustWork = TRUE)
}
source(locate_tema(), encoding = "UTF-8")

root <- find_root()
df <- ler_treino(root)
theme_set(theme_rgna())
out_dir <- file.path(root, "data", "processed")
fig_dir <- file.path(out_dir, "figuras")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Estima o lambda ótimo para o MAPE por verossimilhança
box_result <- boxcox(MAPE ~ 1, data = df, lambda = seq(-2, 2, 0.1))

# Extrai o lambda que maximiza a log-verossimilhança
best_lambda <- box_result$x[which.max(box_result$y)]
# best_lambda <- 0.5

# Transforma a variável aplicando a fórmula matematicamente
if (best_lambda == 0) {
  df$MAPE_boxcox <- log(df$MAPE)
} else {
  df$MAPE_boxcox <- (df$MAPE^best_lambda - 1) / best_lambda
}

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
    media = mean(MAPE_boxcox),
    mediana = median(MAPE_boxcox),
    desvio = sd(MAPE_boxcox),
    minimo = min(MAPE_boxcox),
    maximo = max(MAPE_boxcox),
    q1 = quantile(MAPE_boxcox, 0.25),
    q3 = quantile(MAPE_boxcox, 0.75),
    assimetria = mean((MAPE_boxcox - mean(MAPE_boxcox))^3) / (sd(MAPE_boxcox)^3),
    n_nao_positivo = sum(MAPE_boxcox <= 0)
  )

mape_modelo <- df |>
  group_by(Modelo) |>
  summarise(
    n = n(),
    media = mean(MAPE_boxcox),
    mediana = median(MAPE_boxcox),
    desvio = sd(MAPE_boxcox),
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

write.csv(nulos, file.path(out_dir, "01_nulos_tipos.csv"), row.names = FALSE)
write.csv(contagens, file.path(out_dir, "01_contagens.csv"), row.names = FALSE)
write.csv(sistema, file.path(out_dir, "01_sistema_producao.csv"), row.names = FALSE)
write.csv(mape_global, file.path(out_dir, "01_mape_global.csv"), row.names = FALSE)
write.csv(mape_modelo, file.path(out_dir, "01_mape_por_modelo.csv"), row.names = FALSE)
write.csv(estudos_pais, file.path(out_dir, "01_estudos_por_pais.csv"), row.names = FALSE)
writeLines(campeao, file.path(out_dir, "01_campeao_global.txt"))

p_hist <- ggplot(df, aes(x = MAPE_boxcox)) +
  geom_histogram(bins = 36, fill = fill_main, color = paper, linewidth = 0.2) +
  geom_vline(xintercept = median(df$MAPE_boxcox), linetype = "22", color = ink, linewidth = 0.4) +
  labs(title = "Distribuição do MAPE_boxcox", x = "MAPE_boxcox", y = "Frequência")
ggsave(file.path(fig_dir, "01_hist_mape.png"), p_hist, width = 8, height = 4.5, dpi = 140)

p_box <- ggplot(df, aes(x = Modelo, y = MAPE_boxcox)) +
  geom_boxplot(
    fill = fill_soft, color = fill_main, outlier.colour = muted,
    outlier.alpha = 0.45, outlier.size = 1.1, width = 0.55, linewidth = 0.45
  ) +
  labs(title = "MAPE_boxcox por modelo empírico", x = NULL, y = "MAPE")
ggsave(file.path(fig_dir, "01_box_mape_modelo.png"), p_box, width = 8, height = 4.5, dpi = 140)

message("Campeão global (menor mediana de MAPE_boxcox): ", campeao)
message("Sistema_Producao constante? ", sistema$constante[1], " -> ", paste(sistema$Sistema_Producao, collapse = ", "))
message("Saídas em ", out_dir)
