#!/usr/bin/env Rscript
# Gera o .RDA de entrega da prova pratica (RGNA_Marcelo_Roner_Maria_Pinheiro.RDA).
#
# O professor pediu: "O arquivo .RDA deve permitir a avaliacao do erro de
# predicao nos dados de teste que estao comigo." Este script produz um
# arquivo autossuficiente: apos load(), o objeto predict_rgna() nao depende
# de nenhum script, CSV intermediario ou caminho deste repositorio.
#
# Modelo escolhido: OLS com features Sexo_Animal + Modelo (one-hot, n-1
# dummy por categorica - correcao de posto deficiente ja validada), alvo
# MAPE_boxcox, lambda = 03_meta.csv (fonte unica de verdade, 0.3 atualmente).
#
# Justificativa (ver detalhes em info_modelo$criterio_selecao): validado por
# GroupKFold (5-fold, sklearn.GroupKFold, agregacao weighted_mean) + LOSO por
# paises ocultos, ambos so com dados de treino. Essa candidata vence a
# baseline (mediana historica por Modelo) em MAPE, MAE e Top-1 nos dois
# protocolos.
#
# ATENCAO - fold splits R vs Python NAO sao equivalentes: o make_group_kfold()
# do script R monolitico e o GroupKFold do sklearn implementam o mesmo
# algoritmo guloso, mas com desempate diferente, gerando conjuntos de
# Estudos por fold DIFERENTES (mesmo tamanho, membros diferentes). Por isso
# a selecao do modelo final usa os resultados do pipeline Python
# (scripts/04_modelagem_validacao.py), nao do script R monolitico - o
# sklearn.GroupKFold e a implementacao de referencia, o make_group_kfold()
# em R e uma tentativa de replica-lo que diverge e nao foi corrigida.
#
# Uso:
#   Rscript scripts/06_gerar_entrega_rda.R [estudo_para_ocultar] [caminho_saida]
#
# Sem argumentos: ajusta com TODOS os dados de treino (entrega final).
# Com 1o argumento (nome de um Estudo): esse Estudo e excluido do ajuste -
# usado apenas no teste de pseudo-holdout.

args <- commandArgs(trailingOnly = TRUE)
estudo_oculto <- if (length(args) >= 1) args[[1]] else ""
out_path_override <- if (length(args) >= 2) args[[2]] else NA_character_

script_path <- tryCatch(this.path::this.path(), error = function(e) NA_character_)
if (!is.character(script_path) || is.na(script_path) || !nzchar(script_path)) {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  script_path <- if (length(file_arg) > 0L) sub("^--file=", "", file_arg[[1]]) else stop("Nao foi possivel identificar o caminho do script.")
}
script_path <- normalizePath(script_path, winslash = "/", mustWork = TRUE)
source(file.path(dirname(script_path), "tema_rgna.R"), encoding = "UTF-8")

root <- find_root()
csv_path <- file.path(root, "data", "processed", "03_modelagem.csv")
meta_path <- file.path(root, "data", "processed", "03_meta.csv")
if (!file.exists(csv_path) || !file.exists(meta_path)) {
  stop("Rode antes scripts/03_engenharia_features.R (03_modelagem.csv/03_meta.csv nao encontrados).")
}

df <- read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)
df$Estudo <- as.character(df$Estudo)
lambda_boxcox <- as.numeric(read.csv(meta_path, stringsAsFactors = FALSE)$lambda_boxcox[[1]])

if (nzchar(estudo_oculto)) {
  message("Ocultando Estudo '", estudo_oculto, "' do ajuste (pseudo-teste, arquivo temporario).")
  treino <- df[df$Estudo != estudo_oculto, , drop = FALSE]
  out_path <- if (!is.na(out_path_override)) out_path_override else file.path(tempdir(), "temp_pseudo_teste.RDA")
} else {
  message("Ajustando com TODOS os dados de treino disponiveis (entrega final).")
  treino <- df
  out_dir <- file.path(root, "entrega")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- if (!is.na(out_path_override)) out_path_override else file.path(out_dir, "RGNA_Marcelo_Roner_Maria_Pinheiro.RDA")
}

message("N linhas usadas no ajuste: ", nrow(treino), " (Estudos: ", length(unique(treino$Estudo)), ")")
message("Lambda Box-Cox (de 03_meta.csv): ", lambda_boxcox)

# --- preprocessing (n-1 dummy por categorica, aprendido no treino) ---------
CAT_COLS <- c("Sexo_Animal", "Modelo")

safe_mode <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(x)]
  names(sort(table(x), decreasing = TRUE))[[1]]
}

niveis_por_coluna <- list()
niveis_completos_por_coluna <- list()
moda_por_coluna <- list()
for (col in CAT_COLS) {
  valores <- as.character(treino[[col]])
  valores[is.na(valores) | !nzchar(valores)] <- safe_mode(valores)
  todos_os_niveis <- unique(valores)
  niveis_completos_por_coluna[[col]] <- todos_os_niveis
  # Dropa o primeiro nivel (referencia) para evitar colinearidade exata com
  # o intercepto quando ha mais de uma categorica one-hot no mesmo modelo.
  # O nivel de referencia continua sendo um valor VALIDO na predicao (vira
  # uma linha de dummies toda zero) - so nao ganha coluna propria.
  niveis_por_coluna[[col]] <- if (length(todos_os_niveis) > 1L) todos_os_niveis[-1] else todos_os_niveis
  moda_por_coluna[[col]] <- safe_mode(valores)
}

construir_dummies <- function(dados, cat_cols, niveis, modas) {
  mats <- list()
  nomes <- character()
  for (col in cat_cols) {
    valores <- as.character(dados[[col]])
    valores[is.na(valores) | !nzchar(valores)] <- modas[[col]]
    levs <- niveis[[col]]
    mat <- matrix(0, nrow = nrow(dados), ncol = length(levs))
    colnames(mat) <- paste0(col, "_", levs)
    idx <- match(valores, levs, nomatch = 0L)
    validos <- which(idx > 0L)
    if (length(validos) > 0L) mat[cbind(validos, idx[validos])] <- 1
    mats[[length(mats) + 1L]] <- mat
    nomes <- c(nomes, colnames(mat))
  }
  x <- do.call(cbind, mats)
  colnames(x) <- nomes
  x
}

x_treino <- construir_dummies(treino, CAT_COLS, niveis_por_coluna, moda_por_coluna)
y_treino <- treino$MAPE_boxcox

fit <- .lm.fit(x = cbind("(Intercept)" = 1, x_treino), y = y_treino)
coef_all <- fit$coefficients
coef_all[is.na(coef_all)] <- 0
intercepto <- coef_all[[1]]
beta <- coef_all[-1]
names(beta) <- colnames(x_treino)

# --- modelo_final ------------------------------------------------------
modelo_final <- list(
  tipo = "ols_sexo_modelo",
  formula_texto = "MAPE_boxcox ~ Sexo_Animal + Modelo (one-hot n-1, sem numericos)",
  intercepto = intercepto,
  coeficientes = beta,
  niveis_por_coluna = niveis_por_coluna,
  niveis_completos_por_coluna = niveis_completos_por_coluna,
  moda_por_coluna = moda_por_coluna,
  cat_cols = CAT_COLS,
  lambda_boxcox = lambda_boxcox
)

# --- funcao de predicao externa (autossuficiente, so usa base R) ----------
predict_rgna <- function(newdata) {
  if (!is.data.frame(newdata)) {
    stop("predict_rgna: 'newdata' precisa ser um data.frame.")
  }
  obrigatorias <- modelo_final$cat_cols
  faltando <- setdiff(obrigatorias, names(newdata))
  if (length(faltando) > 0L) {
    stop("predict_rgna: colunas obrigatorias ausentes em newdata: ", paste(faltando, collapse = ", "))
  }

  n <- nrow(newdata)
  mats <- list()
  nomes <- character()
  for (col in modelo_final$cat_cols) {
    valores <- as.character(newdata[[col]])
    # Trata NA/vazio/nivel nao visto no treino como a moda do treino (fallback seguro).
    niveis_validos <- modelo_final$niveis_por_coluna[[col]]
    valores_tratados <- valores
    desconhecido <- is.na(valores) | !nzchar(valores) |
      !(valores %in% modelo_final$niveis_completos_por_coluna[[col]])
    if (any(desconhecido)) {
      valores_tratados[desconhecido] <- modelo_final$moda_por_coluna[[col]]
    }
    levs <- niveis_validos
    mat <- matrix(0, nrow = n, ncol = length(levs))
    colnames(mat) <- paste0(col, "_", levs)
    idx <- match(valores_tratados, levs, nomatch = 0L)
    validos <- which(idx > 0L)
    if (length(validos) > 0L) mat[cbind(validos, idx[validos])] <- 1
    mats[[length(mats) + 1L]] <- mat
    nomes <- c(nomes, colnames(mat))

    if (any(desconhecido)) {
      niveis_desconhecidos <- unique(valores[desconhecido])
      warning(
        "predict_rgna: ", length(niveis_desconhecidos),
        " valor(es) de '", col, "' nao visto(s)/invalido(s) no treino (usando a moda do treino '",
        modelo_final$moda_por_coluna[[col]], "' como fallback): ",
        paste(niveis_desconhecidos, collapse = ", ")
      )
    }
  }
  x <- do.call(cbind, mats)
  colnames(x) <- nomes
  x <- x[, names(modelo_final$coeficientes), drop = FALSE]

  pred_boxcox <- as.numeric(modelo_final$intercepto + x %*% modelo_final$coeficientes)

  lambda <- modelo_final$lambda_boxcox
  if (lambda == 0) {
    pred <- exp(pred_boxcox)
  } else {
    inner <- pmax(lambda * pred_boxcox + 1, 1e-12)
    pred <- inner^(1 / lambda)
  }

  out <- data.frame(MAPE_pred = as.numeric(pred))
  if ("Modelo" %in% names(newdata)) {
    out <- cbind(Modelo = newdata$Modelo, out)
  }
  if ("ID_Observacao" %in% names(newdata)) {
    out <- cbind(ID_Observacao = newdata$ID_Observacao, out)
  }
  stopifnot(nrow(out) == n)
  out
}

info_modelo <- list(
  grupo = "Marcelo Roner, Maria Pinheiro",
  data_geracao = as.character(Sys.time()),
  modelo_escolhido = "ols_sexo_modelo",
  criterio_selecao = paste(
    "Selecionado por validacao interna do pipeline Python (scripts/04_modelagem_validacao.py,",
    "sklearn.GroupKFold real, agregacao weighted_mean) + LOSO por paises ocultos, ambos so com",
    "dados de treino. ols_(sexo_modelo) supera a baseline (mediana historica por Modelo) em",
    "MAPE, MAE e Top-1 nos dois protocolos. IMPORTANTE: o make_group_kfold() do script R",
    "monolitico NAO reproduz os mesmos folds do sklearn.GroupKFold (tamanhos iguais, Estudos",
    "diferentes por fold) - por isso a escolha usa os resultados do pipeline Python, a",
    "implementacao de referencia, e nao do script R (nao corrigido para essa divergencia)."
  ),
  metricas_validacao_groupkfold_python_weighted_mean = list(
    ols_sexo_modelo = list(mape = 2.8100, mae = 0.1394, rmse = 0.1953, r2 = -0.0946, top1 = 0.3184),
    baseline = list(mape = 2.9604, mae = 0.1399, rmse = 0.1939, r2 = -0.0787, top1 = 0.2646)
  ),
  metricas_validacao_loso_paises_ocultos_python_weighted_mean = list(
    ols_sexo_modelo = list(mape = 2.7410, mae = 0.1052, rmse = 0.1214, r2 = -1.8293, top1 = 0.4133),
    baseline = list(mape = 3.1888, mae = 0.1081, rmse = 0.1239, r2 = -1.7465, top1 = 0.4133)
  ),
  colunas_obrigatorias = c("Sexo_Animal", "Modelo"),
  colunas_opcionais_para_output = c("ID_Observacao", "Modelo"),
  colunas_ignoradas_se_presentes = c("MAPE", "MAPE_boxcox"),
  niveis_sexo_animal_treino = niveis_completos_por_coluna[["Sexo_Animal"]],
  niveis_modelo_treino = niveis_completos_por_coluna[["Modelo"]],
  tratamento_nivel_desconhecido = "Usa a moda (categoria mais frequente) do treino como fallback, com warning.",
  estudo_como_preditor = FALSE,
  box_cox = paste0("lambda = ", lambda_boxcox, " (de 03_meta.csv, mesma fonte usada por config.get_best_lambda() no Python)."),
  preprocessing = "One-hot n-1 (dummy, categoria de referencia = 1o nivel por ordem de aparicao no treino) para Sexo_Animal e Modelo. Sem numericos, sem spline, sem padronizacao.",
  dependencias_r = "Nenhuma alem de base R.",
  uso_exemplo = 'load("RGNA_Marcelo_Roner_Maria_Pinheiro.RDA"); pred <- predict_rgna(dados_teste)'
)

save(modelo_final, predict_rgna, info_modelo, file = out_path)
message("Salvo em: ", out_path)
