# Fase 3 — Features para modelagem (treino)
# Estudo permanece só como grupo de validação, não como preditor.
library(MASS)

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
out_dir <- file.path(root, "data", "processed")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Frações A e B são idênticas no treino; Sistema_Producao é constante.
stopifnot(isTRUE(all.equal(df$Fracao_Perda_A, df$Fracao_Perda_B)))
sistema_niveis <- unique(df$Sistema_Producao)
if (length(sistema_niveis) != 1L) {
  warning("Sistema_Producao não é constante: ", paste(sistema_niveis, collapse = ", "))
}

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

modelagem <- df |>
  mutate(
    MAPE_log = log(MAPE)
  ) |>
  select(
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

campeao <- df |>
  group_by(Modelo) |>
  summarise(mediana = median(MAPE), .groups = "drop") |>
  arrange(mediana) |>
  slice(1) |>
  pull(Modelo) |>
  as.character()

meta <- tibble(
  n_linhas = nrow(modelagem),
  n_id = n_distinct(modelagem$ID_Observacao),
  n_estudo = n_distinct(modelagem$Estudo),
  campeao_global = campeao,
  colunas_excluidas = "Estudo (só grupo); Sistema_Producao; Fracao_Perda_B",
  encoding_pais = "one-hot no script 04, handle_unknown=ignore (não target-encoding)"
)

write.csv(modelagem, file.path(out_dir, "03_modelagem.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(meta, file.path(out_dir, "03_meta.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(campeao, file.path(out_dir, "01_campeao_global.txt"))


message("03_modelagem.csv: ", nrow(modelagem), " linhas, campeão global = ", campeao)
message("Preditores: país, categoria_paises, status, sexo, modelo (+ interações texto), peso, consumo, fração A")
message("Grupo de CV: Estudo. Alvo: MAPE (MAPE_log disponível).")

# Visualização da densidade do MAPE por modelo
ggplot(modelagem, aes(x = MAPE_boxcox, color = Modelo)) +
  geom_density(adjust = 1.1, linewidth = 0.55) +
  scale_color_manual(values = pal_modelo) +
  labs(
    title = "Densidade do MAPE por modelo",
    x = "MAPE",
    y = "Densidade",
    caption = "Fonte: dados_treino.xlsx. Kernel gaussiano."
  )