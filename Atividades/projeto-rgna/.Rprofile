# Roda quando o R inicia com working directory nesta pasta
# (RStudio ao abrir o .Rproj; terminal com `cd` até aqui).
# Não carrega pacotes: só opções e a raiz.

options(
  encoding = "UTF-8",
  stringsAsFactors = FALSE,
  dplyr.summarise.inform = FALSE
)

invisible(Sys.setenv(RGNA_ROOT = normalizePath(getwd(), winslash = "/", mustWork = FALSE)))
