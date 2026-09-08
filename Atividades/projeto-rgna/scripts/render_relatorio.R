# Gera relatorio/relatorio-eda.html e .pdf (PDF = impressão do HTML no Chrome)

find_root <- function() {
  env <- Sys.getenv("RGNA_ROOT", unset = "")
  if (nzchar(env) && file.exists(file.path(env, "projeto-rgna.Rproj"))) {
    return(normalizePath(env, winslash = "/", mustWork = TRUE))
  }
  d <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  for (i in seq_len(12L)) {
    if (file.exists(file.path(d, "projeto-rgna.Rproj"))) {
      return(normalizePath(d, winslash = "/", mustWork = TRUE))
    }
    parent <- dirname(d)
    if (identical(parent, d)) break
    d <- parent
  }
  stop("Não achei projeto-rgna.Rproj. Rode a partir da pasta do projeto.")
}

need <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Instale o pacote ", pkg, ": install.packages(\"", pkg, "\")")
  }
}

root <- find_root()
rmd <- file.path(root, "relatorio", "relatorio-eda.Rmd")
html <- file.path(root, "relatorio", "relatorio-eda.html")
pdf <- file.path(root, "relatorio", "relatorio-eda.pdf")

need("rmarkdown")
message("Knit HTML → ", html)
rmarkdown::render(
  input = rmd,
  output_format = "html_document",
  output_file = basename(html),
  output_dir = dirname(html),
  encoding = "UTF-8",
  quiet = FALSE
)

need("pagedown")
chrome <- tryCatch(pagedown::find_chrome(), error = function(e) NULL)
if (is.null(chrome) || !nzchar(chrome)) {
  stop(
    "Chrome/Chromium não encontrado. Instale o Google Chrome ",
    "ou abra o HTML e use Imprimir → Salvar como PDF."
  )
}

message("PDF via Chrome → ", pdf)
pagedown::chrome_print(
  input = html,
  output = pdf,
  wait = 2,
  timeout = 120
)

message("Pronto:\n  ", html, "\n  ", pdf)
