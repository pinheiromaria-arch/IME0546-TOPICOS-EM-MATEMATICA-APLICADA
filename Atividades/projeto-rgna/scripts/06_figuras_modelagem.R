# Fase 4 — Figuras da validação (Versão Refatorada: Suporte Completo a MAPE)

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
library(dplyr)
library(ggplot2)
library(tidyr)
library(stringr)

root <- find_root()
theme_set(theme_rgna())
out_dir <- file.path(root, "data", "processed")
fig_dir <- file.path(out_dir, "figuras")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ── 1. CARREGAR DADOS ──────────────────────────────────────────────────────────
resumo <- read.csv(file.path(out_dir, "04_metricas_resumo.csv"), check.names = FALSE)
folds  <- read.csv(file.path(out_dir, "04_metricas_folds.csv"), check.names = FALSE)
boot   <- read.csv(file.path(out_dir, "04_bootstrap_ic.csv"), check.names = FALSE)
vif    <- read.csv(file.path(out_dir, "04_vif.csv"), check.names = FALSE)
oof    <- read.csv(file.path(out_dir, "04_predicoes_oof.csv"), check.names = FALSE)

# ── 1.1 PADRONIZAÇÃO DAS COLUNAS DE BOOTSTRAP ─────────────────────────────────
colnames(boot) <- tolower(colnames(boot))

obter_coluna_ic <- function(df, prefixo, sufixo) {
  padroes <- c(
    paste0(prefixo, "_ic95_", sufixo),
    paste0(prefixo, "_ic_", sufixo),
    paste0("ic95_", sufixo),
    paste0("ic_", sufixo)
  )
  col_encontrada <- intersect(padroes, colnames(df))
  if (length(col_encontrada) > 0) return(df[[col_encontrada[1]]])
  return(NA_real_)
}

boot <- boot %>%
  mutate(
    mae_ic_inf  = obter_coluna_ic(boot, "mae", "inf"),
    mae_ic_sup  = obter_coluna_ic(boot, "mae", "sup"),
    rmse_ic_inf = obter_coluna_ic(boot, "rmse", "inf"),
    rmse_ic_sup = obter_coluna_ic(boot, "rmse", "sup"),
    mape_ic_inf = obter_coluna_ic(boot, "mape", "inf"),
    mape_ic_sup = obter_coluna_ic(boot, "mape", "sup")
  )

# ── 2. FILTRAGEM DOS TOP MODELOS ───────────────────────────────────────────────
TOP_N <- 8

# Ordenado pelo alvo principal (MAPE se disponível, fallback para MAE)
col_ordenacao <- if ("mape" %in% colnames(resumo)) "mape" else "mae"

top_modelos <- resumo %>%
  filter(protocolo == "GroupKFold") %>%
  arrange(.data[[col_ordenacao]]) %>%
  slice_head(n = TOP_N) %>%
  pull(modelo)

modelos_filtrados <- unique(c("baseline_mediana_modelo", top_modelos))

limpar_nome <- function(x) {
  x %>%
    str_replace("baseline_mediana_modelo", "Baseline") %>%
    str_replace("__num\\(", " [") %>%
    str_replace("\\)__cat\\(", " | ") %>%
    str_replace("\\)", "]") %>%
    str_replace_all("_", " ") %>%
    str_wrap(width = 28)
}

resumo_f <- resumo %>% filter(modelo %in% modelos_filtrados) %>% mutate(rotulo = limpar_nome(modelo))
folds_f  <- folds  %>% filter(modelo %in% modelos_filtrados)  %>% mutate(rotulo = limpar_nome(modelo))
boot_f   <- boot   %>% filter(modelo %in% modelos_filtrados)   %>% mutate(rotulo = limpar_nome(modelo))
oof_f    <- oof    %>% filter(modelo_ml %in% modelos_filtrados) %>% mutate(rotulo = limpar_nome(modelo_ml))

ordem_rotulos <- resumo_f %>%
  filter(protocolo == "GroupKFold") %>%
  arrange(.data[[col_ordenacao]]) %>%
  pull(rotulo)

resumo_f$rotulo <- factor(resumo_f$rotulo, levels = ordem_rotulos)
folds_f$rotulo  <- factor(folds_f$rotulo, levels = ordem_rotulos)
boot_f$rotulo   <- factor(boot_f$rotulo, levels = ordem_rotulos)
oof_f$rotulo    <- factor(oof_f$rotulo, levels = ordem_rotulos)

folds_gkf <- folds_f %>% filter(protocolo == "GroupKFold")
oof_gkf   <- oof_f   %>% filter(protocolo == "GroupKFold")

# ── 3. GRÁFICOS DEDICADOS AO MAPE ──────────────────────────────────────────────

# A. MAPE + IC 95% Bootstrap (Barras com indicação numérica do erro relativo)
p_mape_boot <- ggplot(boot_f %>% filter(protocolo == "GroupKFold"), aes(x = reorder(rotulo, mape), y = mape)) +
  geom_col(width = 0.55, fill = fill_main, alpha = 0.85) +
  {
    if (!all(is.na(boot_f$mape_ic_inf))) {
      geom_errorbar(aes(ymin = mape_ic_inf, ymax = mape_ic_sup), width = 0.15, color = ink, linewidth = 0.35)
    }
  } +
  geom_text(aes(label = paste0(round(mape * 100, 1), "%")), hjust = -0.2, size = 3.1, color = ink) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100, 1), "%"), expand = expansion(mult = c(0, 0.18))) +
  coord_flip() +
  labs(
    title = paste("Top", TOP_N, "Modelos: MAPE no GroupKFold (IC 95% Bootstrap)"),
    subtitle = "Menor percentual indica melhor ajuste relativo ao valor observado.",
    x = NULL, y = "MAPE (%)"
  ) +
  theme(panel.grid.major.y = element_blank())

ggsave(file.path(fig_dir, "04_mape_bootstrap.png"), p_mape_boot, width = 9.5, height = 6.0, dpi = 140)

# B. MAPE por Pasta (Heatmap de estabilidade das folds)
p_mape_heatmap <- ggplot(folds_gkf, aes(x = factor(fold), y = rotulo, fill = mape)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = paste0(round(mape * 100, 1), "%")), color = ink, size = 3.0) +
  scale_fill_gradient(low = "#e0ede0", high = "#c48874", name = "MAPE", labels = function(x) paste0(round(x * 100), "%")) +
  labs(
    title = "MAPE em cada pasta do GroupKFold (Heatmap)",
    subtitle = "Identificação visual de agrupamentos/folds onde os modelos perdem precisão.",
    x = "Fold", y = NULL
  ) +
  theme(panel.grid = element_blank())

ggsave(file.path(fig_dir, "04_mape_por_pasta_heatmap.png"), p_mape_heatmap, width = 9.5, height = 5.5, dpi = 140)

# C. MAPE por Fold (Linhas de variabilidade)
p_mape_fold <- ggplot(folds_gkf, aes(x = factor(fold), y = mape, group = rotulo, color = rotulo)) +
  geom_line(alpha = 0.5, linewidth = 0.6) +
  geom_point(size = 1.8) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100, 1), "%")) +
  scale_color_viridis_d(option = "mako", name = "Modelo") +
  labs(
    title = "Variabilidade do MAPE entre Folds",
    subtitle = "Oscilação do erro percentual absoluto ao longo do GroupKFold.",
    x = "Número do Fold", y = "MAPE (%)"
  ) +
  theme(legend.position = "right", legend.text = element_text(size = 8))

ggsave(file.path(fig_dir, "04_mape_por_fold.png"), p_mape_fold, width = 10.0, height = 5.5, dpi = 140)

# D. Resíduos Relativos OOF: Erro Relativo Percentual vs Alvo Observado
# Identifica a coluna alvo no OOF (procura 'mape', 'target', 'y' ou 'y_true')
col_alvo_oof <- intersect(c("mape", "target", "y", "y_true", "observado"), colnames(oof_gkf))[1]

if (!is.na(col_alvo_oof)) {
  oof_gkf <- oof_gkf %>%
    mutate(
      alvo_val = .data[[col_alvo_oof]],
      residuo_rel = (pred - alvo_val) / alvo_val
    )

  set.seed(42)
  oof_sub <- if (nrow(oof_gkf) > 5000) oof_gkf[sample(seq_len(nrow(oof_gkf)), 5000), ] else oof_gkf

  p_residuos_mape <- ggplot(oof_sub, aes(x = alvo_val, y = residuo_rel)) +
    geom_hline(yintercept = 0, color = "#b56a4a", linetype = "dashed", linewidth = 0.6) +
    geom_point(alpha = 0.3, size = 1.1, color = fill_main) +
    geom_smooth(method = "loess", color = ink, se = FALSE, linewidth = 0.5) +
    scale_y_continuous(labels = function(x) paste0(round(x * 100), "%"), limits = c(-1.5, 1.5)) +
    facet_wrap(~ rotulo, ncol = 3) +
    labs(
      title = "Erro Percentual OOF vs Valor Observado",
      subtitle = "Erro Relativo ((Previsto - Observado) / Observado). Tendência próxima de 0% indica bom ajuste.",
      x = "Valor Observado", y = "Erro Relativo (%)"
    )

  ggsave(file.path(fig_dir, "04_residuos_relativos_mape.png"), p_residuos_mape, width = 11.0, height = 7.5, dpi = 140)

  # E. Previsto vs Observado
  p_oof_mape <- ggplot(oof_sub, aes(x = alvo_val, y = pred)) +
    geom_abline(slope = 1, intercept = 0, color = muted, linetype = "22", linewidth = 0.4) +
    geom_point(alpha = 0.28, size = 1.05, color = fill_main) +
    facet_wrap(~ rotulo, ncol = 3) +
    labs(
      title = "Previsão OOF vs Valor Observado (Top Modelos)",
      x = "Observado", y = "Previsto"
    )

  ggsave(file.path(fig_dir, "04_mape_previsto_vs_observado.png"), p_oof_mape, width = 10.0, height = 7.5, dpi = 140)
}

# ── 4. GRÁFICOS COMPLEMENTARES (MAE E RMSE) ───────────────────────────────────

# MAE Bootstrap
p_mae_boot <- ggplot(boot_f %>% filter(protocolo == "GroupKFold"), aes(x = reorder(rotulo, mae), y = mae)) +
  geom_col(width = 0.55, fill = fill_main, alpha = 0.85) +
  geom_errorbar(aes(ymin = mae_ic_inf, ymax = mae_ic_sup), width = 0.15, color = ink, linewidth = 0.35) +
  geom_text(aes(label = sprintf("%.3f", mae)), hjust = -0.2, size = 3.1, color = ink) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = paste("Top", TOP_N, "Modelos: MAE no GroupKFold (IC 95%)"), x = NULL, y = "MAE") +
  theme(panel.grid.major.y = element_blank())
ggsave(file.path(fig_dir, "04_mae_bootstrap.png"), p_mae_boot, width = 9.5, height = 6.0, dpi = 140)

# RMSE Bootstrap
p_rmse_boot <- ggplot(boot_f %>% filter(protocolo == "GroupKFold"), aes(x = reorder(rotulo, rmse), y = rmse)) +
  geom_col(width = 0.55, fill = fill_main, alpha = 0.85) +
  geom_errorbar(aes(ymin = rmse_ic_inf, ymax = rmse_ic_sup), width = 0.15, color = ink, linewidth = 0.35) +
  geom_text(aes(label = sprintf("%.3f", rmse)), hjust = -0.2, size = 3.1, color = ink) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = paste("Top", TOP_N, "Modelos: RMSE no GroupKFold (IC 95%)"), x = NULL, y = "RMSE") +
  theme(panel.grid.major.y = element_blank())
ggsave(file.path(fig_dir, "04_rmse_bootstrap.png"), p_rmse_boot, width = 9.5, height = 6.0, dpi = 140)

# VIF
vif_plot <- vif %>%
  mutate(
    VIF_plot = ifelse(!is.finite(VIF) | VIF > 20, 20, VIF),
    variavel = factor(variavel, levels = rev(variavel))
  )
p_vif <- ggplot(vif_plot, aes(x = VIF_plot, y = variavel, fill = alerta)) +
  geom_col(width = 0.62) +
  geom_vline(xintercept = 5, linetype = "22", color = muted, linewidth = 0.4) +
  geom_vline(xintercept = 10, linetype = "22", color = "#b56a4a", linewidth = 0.4) +
  scale_fill_manual(values = c("ok" = fill_soft, "VIF>=5" = "#c4a574", "VIF>=10" = "#b56a4a"), guide = "none") +
  labs(title = "VIF dos Preditores Numéricos", x = "VIF", y = NULL)
ggsave(file.path(fig_dir, "04_vif.png"), p_vif, width = 8, height = 4.2, dpi = 140)

message("Novos gráficos do MAPE gerados com sucesso em: ", fig_dir)