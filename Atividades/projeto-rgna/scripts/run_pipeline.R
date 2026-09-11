#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || is.na(x) || !nzchar(x)) y else x
}

suppressPackageStartupMessages({
  library(processx)
})

find_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0L) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
  }

  fallback <- tryCatch(this.path::this.path(), error = function(e) NA_character_)
  if (is.character(fallback) && length(fallback) == 1L && !is.na(fallback) && nzchar(fallback)) {
    return(normalizePath(fallback, winslash = "/", mustWork = TRUE))
  }

  stop("Não foi possível identificar o caminho do script run_pipeline.R.")
}

script_path <- find_script_path()
scripts_dir <- dirname(script_path)
source(file.path(scripts_dir, "tema_rgna.R"), encoding = "UTF-8")
root <- find_root()
Sys.setenv(RGNA_ROOT = root)

run_step <- function(label, command, args) {
  message("")
  message("==> ", label)
  result <- processx::run(
    command = command,
    args = args,
    wd = root,
    echo = TRUE,
    error_on_status = FALSE
  )

  if (!identical(result$status, 0L)) {
    stop(
      paste0(
        "Etapa falhou: ", label,
        "\nComando: ", paste(c(command, args), collapse = " "),
        "\nSaída:\n", result$stdout,
        if (nzchar(result$stderr)) paste0("\n", result$stderr) else ""
      ),
      call. = FALSE
    )
  }
}

rscript_bin <- Sys.getenv("R_SCRIPT_BIN", unset = Sys.getenv("Rscript", unset = "Rscript"))
python_bin <- Sys.getenv("PYTHON_BIN", unset = "python3")

steps <- list(
  list(label = "EDA e diagnóstico", command = rscript_bin, args = c(file.path("scripts", "01_eda_diagnostico.R"))),
  list(label = "Análise bivariada", command = rscript_bin, args = c(file.path("scripts", "02_analise_bivariada.R"))),
  list(label = "Engenharia de features", command = rscript_bin, args = c(file.path("scripts", "03_engenharia_features.R"))),
  list(label = "Modelagem e validação", command = python_bin, args = c(file.path("scripts", "04_modelagem_validacao.py"))),
  list(label = "Avaliação e recomendação", command = python_bin, args = c(file.path("scripts", "05_avaliacao_recomendacao.py")))
)

message("Raiz do projeto: ", root)
for (step in steps) {
  run_step(step$label, step$command, step$args)
}

message("")
message("Pipeline concluída. Artefatos em: ", file.path(root, "data", "processed"))
