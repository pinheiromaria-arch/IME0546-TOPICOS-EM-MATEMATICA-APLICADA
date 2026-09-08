# Tema e leitura do treino — compartilhado entre scripts e o relatório

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(rprojroot)
})

# Função unificada para encontrar a raiz do projeto.
# A raiz é definida pela presença de 'projeto-rgna.Rproj' ou 'data/raw/dados_treino.xlsx'.
find_root <- function() {
  # Primeiro, tenta a variável de ambiente, se definida.
  env <- Sys.getenv("RGNA_ROOT", unset = "")
  if (nzchar(env)) {
    return(normalizePath(env, winslash = "/", mustWork = TRUE))
  }
  # Senão, usa rprojroot para buscar a partir do diretório atual ou do script.
  rprojroot::find_root(
    criterion = rprojroot::has_file("projeto-rgna.Rproj") |
      rprojroot::has_file("data/raw/dados_treino.xlsx")
  )
}

ink <- "#1c1917"
muted <- "#6b6560"
paper <- "#f7f4ee"
fill_main <- "#2f4a3c"
fill_soft <- "#d5ddd6"

pal_modelo <- c(
  Modelo_1 = "#8a8174",
  Modelo_2 = "#6d7a82",
  Modelo_3 = "#b56a4a",
  Modelo_4 = "#2f4a3c",
  Modelo_5 = "#5e7f96"
)

theme_rgna <- function(base_size = 12) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "plain", size = base_size + 1, color = ink, hjust = 0),
      plot.subtitle = element_text(size = base_size - 1.5, color = muted, hjust = 0, margin = margin(b = 10)),
      plot.caption = element_text(size = base_size - 2.5, color = muted, hjust = 0, margin = margin(t = 8)),
      axis.title = element_text(size = base_size - 1, color = muted),
      axis.text = element_text(color = ink),
      axis.line = element_line(color = "#d4cfc6", linewidth = 0.4),
      axis.ticks = element_line(color = "#d4cfc6", linewidth = 0.4),
      panel.grid.major.y = element_line(color = "#efeae2", linewidth = 0.4),
      panel.grid.major.x = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.position = "bottom",
      legend.title = element_text(size = base_size - 1.5, color = muted),
      legend.text = element_text(size = base_size - 1.5),
      legend.background = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "plain", color = ink, hjust = 0)
    )
}

ler_treino <- function(root = find_root()) {
  read_excel(
    file.path(root, "data", "raw", "dados_treino.xlsx"),
    .name_repair = "minimal"
  ) |>
    select(
      ID_Observacao, Estudo, Pais_Estudo, Sistema_Producao, Status_Metabolico,
      Sexo_Animal, Peso_Corporal_kg, Consumo_MS_kg, Fracao_Perda_A,
      Fracao_Perda_B, Modelo, MAPE
    ) |>
    mutate(
      Modelo = factor(Modelo, levels = paste0("Modelo_", 1:5)),
      Pais_Estudo = as.character(Pais_Estudo)
    )
}

paises_ocultos <- c(
  "Campo Rico", "Ilha Verdejante", "Terra Nortenha",
  "Monção Dourada", "Costa Austral"
)