#!/usr/bin/env Rscript
# Gera o .RData de entrega da prova pratica (RGNA_Marcelo_Roner_Maria_Pinheiro.RData).
#
# O professor pediu: "O arquivo .RDA deve permitir a avaliacao do erro de
# predicao nos dados de teste que estao comigo." Este script produz um
# arquivo autossuficiente: apos load(), o objeto predict_rgna() nao depende
# de nenhum script, CSV intermediario ou caminho deste repositorio.
#
# Modelo escolhido (ver justificativa completa em info_modelo$criterio_selecao):
# baseline de mediana historica de MAPE por Modelo. Selecionado por validacao
# interna (GroupKFold 5-fold + LOSO por paises ocultos, ambos usando somente
# dados de treino) porque superou todas as variantes OLS/spline testadas em
# MAE, RMSE, R2 e Top-1 nos dois protocolos - so perdeu em MAPE, metrica
# instavel quando o MAPE real e proximo de zero.
#
# Uso:
#   Rscript scripts/06_gerar_entrega_rda.R [estudo_para_ocultar]
#
# Sem argumento: ajusta com TODOS os dados de treino (entrega final).
# Com argumento (nome de um Estudo): esse Estudo e excluido do ajuste -
# usado apenas no teste de pseudo-holdout (nao gera o arquivo de entrega).

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
if (!file.exists(csv_path)) {
  stop("Rode antes scripts/03_engenharia_features.R (03_modelagem.csv nao encontrado).")
}

df <- read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)
df$Estudo <- as.character(df$Estudo)

if (nzchar(estudo_oculto)) {
  message("Ocultando Estudo '", estudo_oculto, "' do ajuste (pseudo-teste, arquivo temporario).")
  treino <- df[df$Estudo != estudo_oculto, , drop = FALSE]
  out_path <- if (!is.na(out_path_override)) out_path_override else file.path(tempdir(), "temp_pseudo_teste.RData")
} else {
  message("Ajustando com TODOS os dados de treino disponiveis (entrega final).")
  treino <- df
  out_dir <- file.path(root, "entrega")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(out_dir, "RGNA_Marcelo_Roner_Maria_Pinheiro.RData")
}

message("N linhas usadas no ajuste: ", nrow(treino), " (Estudos: ", length(unique(treino$Estudo)), ")")

# --- modelo_final: baseline de mediana historica de MAPE por Modelo -------
medianas_modelo <- stats::aggregate(MAPE ~ Modelo, data = treino, FUN = stats::median)
names(medianas_modelo) <- c("Modelo", "mediana_mape")
fallback_global <- stats::median(treino$MAPE)

modelo_final <- list(
  tipo = "baseline_mediana_historica_por_modelo",
  medianas_por_modelo = medianas_modelo,
  fallback_global = fallback_global,
  niveis_modelo_treino = as.character(medianas_modelo$Modelo)
)

# --- funcao de predicao externa (autossuficiente, so usa base R) ----------
predict_rgna <- function(newdata) {
  if (!is.data.frame(newdata)) {
    stop("predict_rgna: 'newdata' precisa ser um data.frame.")
  }
  if (!("Modelo" %in% names(newdata))) {
    stop("predict_rgna: a coluna obrigatoria 'Modelo' nao esta presente em newdata. ",
         "Valores esperados: ", paste(modelo_final$niveis_modelo_treino, collapse = ", "))
  }

  n <- nrow(newdata)
  modelo_chr <- as.character(newdata$Modelo)

  lookup <- modelo_final$medianas_por_modelo
  idx <- match(modelo_chr, as.character(lookup$Modelo))
  pred <- ifelse(is.na(idx), modelo_final$fallback_global, lookup$mediana_mape[idx])

  if (any(is.na(idx))) {
    niveis_desconhecidos <- unique(modelo_chr[is.na(idx)])
    warning(
      "predict_rgna: ", length(niveis_desconhecidos),
      " valor(es) de 'Modelo' nao vistos no treino (usando mediana global como fallback): ",
      paste(niveis_desconhecidos, collapse = ", ")
    )
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
  modelo_escolhido = "baseline_mediana_modelo",
  criterio_selecao = paste(
    "Selecionado por validacao interna (GroupKFold 5-fold + LOSO por paises ocultos,",
    "somente dados de treino). O baseline (mediana historica de MAPE por Modelo, sem",
    "regressao) superou todos os candidatos OLS/spline testados em MAE, RMSE, R2 e",
    "Top-1 nos dois protocolos; so perdeu em MAPE (metrica instavel perto de zero)."
  ),
  metricas_validacao_groupkfold = list(
    baseline = list(mape = 2.9640, mae = 0.1395, rmse = 0.1908, r2 = -0.0629, top1 = 0.3330),
    runner_up_ols_peso_modelo = list(mape = 2.8599, mae = 0.1400, rmse = 0.1936, r2 = -0.0941, top1 = 0.2511)
  ),
  metricas_validacao_loso_paises_ocultos = list(
    baseline = list(mape = 2.8968, mae = 0.1166, rmse = 0.1325, r2 = -1.9934, top1 = 0.4386),
    runner_up_ols_consumo_pais_status_sexo_modelo = list(mape = 2.0667, mae = 0.1193, rmse = 0.1378, r2 = -2.5483, top1 = 0.4386)
  ),
  colunas_obrigatorias = "Modelo",
  colunas_opcionais_para_output = c("ID_Observacao", "Modelo"),
  colunas_ignoradas_se_presentes = c("MAPE", "MAPE_boxcox"),
  niveis_modelo_treino = modelo_final$niveis_modelo_treino,
  tratamento_nivel_desconhecido = "Usa a mediana global de MAPE do treino como fallback (com warning).",
  estudo_como_preditor = FALSE,
  box_cox = "Nao aplicavel (baseline opera diretamente na escala original de MAPE).",
  dependencias_r = "Nenhuma alem de base R.",
  uso_exemplo = 'load("RGNA_Marcelo_Roner_Maria_Pinheiro.RData"); pred <- predict_rgna(dados_teste)'
)

save(modelo_final, predict_rgna, info_modelo, file = out_path)
message("Salvo em: ", out_path)
