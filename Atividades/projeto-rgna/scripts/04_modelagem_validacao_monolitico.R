#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(tidyr)
  library(tibble)
  library(jsonlite)
  library(splines)
})

script_path <- tryCatch(this.path::this.path(), error = function(e) NA_character_)
if (!is.character(script_path) || is.na(script_path) || !nzchar(script_path)) {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0L) {
    script_path <- sub("^--file=", "", file_arg[[1]])
  } else {
    stop("Nao foi possivel identificar o caminho do script.")
  }
}
script_path <- normalizePath(script_path, winslash = "/", mustWork = TRUE)
source(file.path(dirname(script_path), "tema_rgna.R"), encoding = "UTF-8")

PAISES_OCULTOS <- c(
  "Campo Rico",
  "Ilha Verdejante",
  "Terra Nortenha",
  "Monção Dourada",
  "Costa Austral"
)
TARGET <- "MAPE_boxcox"
NUM_CANDIDATES <- c("Peso_Corporal_kg", "Consumo_MS_kg", "Fracao_Perda_A")
CAT_CANDIDATES <- c("Pais_Estudo", "Status_Metabolico", "Sexo_Animal", "Modelo")
MAP_COLUMNS <- c(
  Peso_Corporal_kg = "peso",
  Consumo_MS_kg = "consumo",
  Fracao_Perda_A = "fracao",
  Pais_Estudo = "pais",
  Status_Metabolico = "status",
  Sexo_Animal = "sexo",
  Modelo = "modelo"
)
N_BOOT <- 1000L
BOOT_SEED <- 42L
AGGREGATION_STRATEGY <- "mean"
TOP_N_MODELOS_POR_PROTOCOLO <- 5L
WEIGHT_FEATURE <- "Peso_Corporal_kg"

log_msg <- function(msg) {
  message(sprintf("[%s] %s", format(Sys.time(), "%d/%m/%Y %H:%M:%S"), msg))
}

inv_boxcox <- function(y_trans, lambda_param) {
  if (lambda_param == 0) {
    return(exp(y_trans))
  }
  inner <- pmax(lambda_param * y_trans + 1, 1e-12)
  inner^(1 / lambda_param)
}

load_frame <- function(root) {
  csv_path <- file.path(root, "data", "processed", "03_modelagem.csv")
  if (!file.exists(csv_path)) {
    stop("Rode antes scripts/03_engenharia_features.R")
  }
  df <- read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)
  df$Estudo <- as.character(df$Estudo)
  df
}

rmse_vec <- function(y_true, y_pred) {
  sqrt(mean((y_true - y_pred)^2))
}

mape_metric <- function(y_true, y_pred) {
  denom <- abs(y_true)
  rel <- ifelse(denom > 0, (y_true - y_pred) / denom, 0)
  mean(abs(rel))
}

regression_scores <- function(y_true, y_pred) {
  sse <- sum((y_true - y_pred)^2)
  sst <- sum((y_true - mean(y_true))^2)
  r2 <- 1 - sse / sst
  list(
    mae = mean(abs(y_true - y_pred)),
    rmse = rmse_vec(y_true, y_pred),
    r2 = r2,
    mape = mape_metric(y_true, y_pred)
  )
}

top1_accuracy <- function(frame, pred_col) {
  real <- frame |>
    arrange(ID_Observacao, MAPE, Modelo) |>
    distinct(ID_Observacao, .keep_all = TRUE) |>
    select(ID_Observacao, Modelo_real = Modelo)
  hat <- frame |>
    arrange(ID_Observacao, .data[[pred_col]], Modelo) |>
    distinct(ID_Observacao, .keep_all = TRUE) |>
    select(ID_Observacao, Modelo_hat = Modelo)
  merged <- inner_join(real, hat, by = "ID_Observacao")
  mean(merged$Modelo_real == merged$Modelo_hat)
}

baseline_predict <- function(train, test) {
  med <- train |>
    group_by(Modelo) |>
    summarise(med = median(MAPE), .groups = "drop")
  fallback <- median(train$MAPE)
  test |>
    left_join(med, by = "Modelo") |>
    mutate(pred = if_else(is.na(med), fallback, med)) |>
    pull(pred)
}

campeao_do_fold <- function(train) {
  med <- train |>
    group_by(Modelo) |>
    summarise(mediana = median(MAPE), .groups = "drop")
  as.character(med$Modelo[[which.min(med$mediana)]])
}

generate_feature_sets <- function() {
  feature_sets <- list()
  num_combinations <- unlist(
    lapply(seq_along(NUM_CANDIDATES), function(r) combn(NUM_CANDIDATES, r, simplify = FALSE)),
    recursive = FALSE
  )
  cat_combinations <- unlist(
    lapply(seq_along(CAT_CANDIDATES), function(r) combn(CAT_CANDIDATES, r, simplify = FALSE)),
    recursive = FALSE
  )

  for (num_cols in num_combinations) {
    feature_sets[[length(feature_sets) + 1L]] <- list(
      num = num_cols,
      cat = character(),
      x_cols = num_cols,
      name = sprintf("(%s)", paste(MAP_COLUMNS[num_cols], collapse = "_"))
    )
  }
  for (cat_cols in cat_combinations) {
    feature_sets[[length(feature_sets) + 1L]] <- list(
      num = character(),
      cat = cat_cols,
      x_cols = cat_cols,
      name = sprintf("(%s)", paste(MAP_COLUMNS[cat_cols], collapse = "_"))
    )
  }
  for (num_cols in num_combinations) {
    for (cat_cols in cat_combinations) {
      feature_sets[[length(feature_sets) + 1L]] <- list(
        num = num_cols,
        cat = cat_cols,
        x_cols = c(num_cols, cat_cols),
        name = sprintf("(%s)_(%s)", paste(MAP_COLUMNS[num_cols], collapse = "_"), paste(MAP_COLUMNS[cat_cols], collapse = "_"))
      )
    }
  }
  feature_sets
}

make_group_kfold <- function(df, n_splits = 5L) {
  group_sizes <- df |>
    count(Estudo, name = "n") |>
    arrange(desc(n), Estudo)
  n_splits <- min(as.integer(n_splits), nrow(group_sizes))
  if (n_splits < 2L) {
    stop("GroupKFold requer pelo menos 2 estudos distintos.")
  }
  fold_groups <- vector("list", n_splits)
  fold_sizes <- rep(0L, n_splits)
  for (i in seq_len(nrow(group_sizes))) {
    idx <- which.min(fold_sizes)
    fold_groups[[idx]] <- c(fold_groups[[idx]], group_sizes$Estudo[[i]])
    fold_sizes[[idx]] <- fold_sizes[[idx]] + group_sizes$n[[i]]
  }
  lapply(seq_len(n_splits), function(i) {
    te_idx <- which(df$Estudo %in% fold_groups[[i]])
    tr_idx <- setdiff(seq_len(nrow(df)), te_idx)
    list(train = tr_idx, test = te_idx, name = paste0("gkf", i))
  })
}

compute_spline_spec <- function(x) {
  quantiles <- as.numeric(stats::quantile(x, probs = seq(0, 1, length.out = 4), na.rm = TRUE, names = FALSE))
  list(
    degree = 3L,
    knots = quantiles,
    lower = quantiles[[1]],
    upper = quantiles[[4]]
  )
}

safe_mode <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(x)]
  names(sort(table(x), decreasing = TRUE))[[1]]
}

prepare_cat_levels <- function(train, cat_cols) {
  levels_map <- list()
  mode_map <- list()
  for (col in cat_cols) {
    values <- as.character(train[[col]])
    values[is.na(values) | !nzchar(values)] <- safe_mode(values)
    levels_map[[col]] <- unique(values)
    mode_map[[col]] <- safe_mode(values)
  }
  list(levels = levels_map, modes = mode_map)
}

build_one_hot <- function(df, cat_cols, cat_info) {
  mats <- list()
  names_out <- character()
  if (length(cat_cols) == 0L) {
    return(list(matrix = matrix(numeric(), nrow = nrow(df), ncol = 0), names = character()))
  }
  for (col in cat_cols) {
    values <- as.character(df[[col]])
    values[is.na(values) | !nzchar(values)] <- cat_info$modes[[col]]
    levs <- cat_info$levels[[col]]
    mat <- matrix(0, nrow = nrow(df), ncol = length(levs))
    colnames(mat) <- paste0(col, "_", levs)
    matched <- match(values, levs, nomatch = 0L)
    valid <- which(matched > 0L)
    if (length(valid) > 0L) {
      mat[cbind(valid, matched[valid])] <- 1
    }
    mats[[length(mats) + 1L]] <- mat
    names_out <- c(names_out, colnames(mat))
  }
  list(matrix = do.call(cbind, mats), names = names_out)
}

prepare_numeric <- function(train, test, num_cols, use_spline) {
  if (length(num_cols) == 0L) {
    return(list(train = matrix(numeric(), nrow = nrow(train), ncol = 0), test = matrix(numeric(), nrow = nrow(test), ncol = 0), names = character()))
  }

  train_imp <- train
  test_imp <- test
  for (col in num_cols) {
    med <- median(train[[col]], na.rm = TRUE)
    train_imp[[col]][is.na(train_imp[[col]])] <- med
    test_imp[[col]][is.na(test_imp[[col]])] <- med
  }

  mats_train <- list()
  mats_test <- list()
  names_out <- character()

  if (use_spline && WEIGHT_FEATURE %in% num_cols) {
    spec <- compute_spline_spec(train_imp[[WEIGHT_FEATURE]])
    bs_train <- splines::bs(
      train_imp[[WEIGHT_FEATURE]],
      df = 5,
      degree = spec$degree,
      knots = spec$knots[2:3],
      Boundary.knots = c(spec$lower, spec$upper),
      intercept = FALSE
    )
    bs_test <- splines::bs(
      test_imp[[WEIGHT_FEATURE]],
      df = 5,
      degree = spec$degree,
      knots = spec$knots[2:3],
      Boundary.knots = c(spec$lower, spec$upper),
      intercept = FALSE
    )
    colnames(bs_train) <- paste0("Peso_Corporal_kg_bs_", seq_len(ncol(bs_train)))
    colnames(bs_test) <- colnames(bs_train)
    mats_train[[length(mats_train) + 1L]] <- bs_train
    mats_test[[length(mats_test) + 1L]] <- bs_test
    names_out <- c(names_out, colnames(bs_train))
    num_cols <- setdiff(num_cols, WEIGHT_FEATURE)
  }

  if (length(num_cols) > 0L) {
    train_num <- as.matrix(train_imp[num_cols])
    test_num <- as.matrix(test_imp[num_cols])
    colnames(train_num) <- num_cols
    colnames(test_num) <- num_cols
    mats_train[[length(mats_train) + 1L]] <- train_num
    mats_test[[length(mats_test) + 1L]] <- test_num
    names_out <- c(names_out, num_cols)
  }

  x_train <- do.call(cbind, mats_train)
  x_test <- do.call(cbind, mats_test)
  means <- colMeans(x_train)
  sds <- apply(x_train, 2, stats::sd)
  sds[is.na(sds) | sds == 0] <- 1
  x_train <- scale(x_train, center = means, scale = sds)
  x_test <- scale(x_test, center = means, scale = sds)
  colnames(x_train) <- names_out
  colnames(x_test) <- names_out
  list(train = x_train, test = x_test, names = names_out)
}

prepare_design_matrices <- function(train, test, num_cols, cat_cols, use_spline = FALSE) {
  cat_info <- prepare_cat_levels(train, cat_cols)
  num_part <- prepare_numeric(train, test, num_cols, use_spline)
  cat_train <- build_one_hot(train, cat_cols, cat_info)
  cat_test <- build_one_hot(test, cat_cols, cat_info)

  x_train <- cbind(num_part$train, cat_train$matrix)
  x_test <- cbind(num_part$test, cat_test$matrix)
  if (is.null(dim(x_train))) {
    x_train <- matrix(x_train, ncol = 1)
  }
  if (is.null(dim(x_test))) {
    x_test <- matrix(x_test, ncol = 1)
  }
  if (ncol(x_train) == 0L) {
    x_train <- matrix(1, nrow = nrow(train), ncol = 1)
    x_test <- matrix(1, nrow = nrow(test), ncol = 1)
    colnames(x_train) <- "(constant)"
    colnames(x_test) <- "(constant)"
  } else {
    colnames(x_train) <- c(num_part$names, cat_train$names)
    colnames(x_test) <- c(num_part$names, cat_test$names)
  }
  list(train = x_train, test = x_test)
}

extract_equation <- function(fit, feature_names, model_name) {
  beta <- as.numeric(fit$coefficients)
  intercept <- fit$intercept
  terms <- c(sprintf("%.4f", intercept))
  if (length(beta) > 0L) {
    terms <- c(
      terms,
      map2_chr(beta, feature_names, function(coef, name) {
        sign <- if (coef >= 0) "+" else "-"
        sprintf("%s %.4f * (%s)", sign, abs(coef), name)
      })
    )
  }
  sprintf("%s: y_hat = %s", model_name, paste(terms, collapse = " "))
}

fit_linear_regression <- function(x_train, y_train, x_test) {
  x_train <- as.matrix(x_train)
  x_test <- as.matrix(x_test)
  fit <- .lm.fit(x = cbind("(Intercept)" = 1, x_train), y = y_train)
  coef_all <- fit$coefficients
  coef_all[is.na(coef_all)] <- 0
  intercept <- coef_all[[1]]
  beta <- coef_all[-1]
  pred <- drop(cbind(1, x_test) %*% c(intercept, beta))
  list(
    intercept = intercept,
    coefficients = beta,
    pred = pred
  )
}

fit_predict_simple <- function(train, test, num_cols, cat_cols, model_name, lambda_param, use_spline = FALSE) {
  mats <- prepare_design_matrices(train, test, num_cols, cat_cols, use_spline = use_spline)
  fit <- fit_linear_regression(mats$train, train[[TARGET]], mats$test)
  pred_final <- inv_boxcox(fit$pred, lambda_param)
  list(
    pred = as.numeric(pred_final),
    equations = extract_equation(fit, colnames(mats$train), model_name)
  )
}

collect_fold_results <- function(name, protocol, fold, test, pred, campeao) {
  tmp <- test
  tmp$pred <- pred
  real_win <- tmp |>
    arrange(ID_Observacao, MAPE, Modelo) |>
    distinct(ID_Observacao, .keep_all = TRUE) |>
    pull(Modelo)
  scores <- regression_scores(tmp$MAPE, pred)
  c(
    list(
      modelo = name,
      protocolo = protocol,
      fold = fold,
      n = nrow(tmp),
      top1 = top1_accuracy(tmp, "pred"),
      top1_sempre_campeao = mean(real_win == campeao)
    ),
    scores
  )
}

run_all_models_for_fold <- function(train, test, feature_sets, lambda_param) {
  predictions <- list(
    baseline_mediana_modelo = list(pred = baseline_predict(train, test), equations = character())
  )
  for (fset in feature_sets) {
    has_weight <- WEIGHT_FEATURE %in% fset$x_cols
    model_key <- paste0("ols_", fset$name)
    predictions[[model_key]] <- fit_predict_simple(train, test, fset$num, fset$cat, model_key, lambda_param, use_spline = FALSE)
    if (has_weight) {
      model_key <- paste0("spl_", fset$name)
      predictions[[model_key]] <- fit_predict_simple(train, test, fset$num, fset$cat, model_key, lambda_param, use_spline = TRUE)
    }
  }
  predictions
}

run_group_kfold <- function(df, feature_sets, lambda_param, n_splits = 5L) {
  folds <- make_group_kfold(df, n_splits)
  rows <- list()
  oof_parts <- list()
  all_equations <- character()
  test_rows <- sum(vapply(folds, function(f) length(f$test), integer(1)))
  if (test_rows == 0L) {
    stop("GroupKFold nao produziu folds de teste validos.")
  }
  for (i in seq_along(folds)) {
    fold <- folds[[i]]
    train <- df[fold$train, , drop = FALSE]
    test <- df[fold$test, , drop = FALSE]
    campeao <- campeao_do_fold(train)
    log_msg(sprintf("Executando GroupKFold Fold %d...", i))
    preds_and_eqs <- run_all_models_for_fold(train, test, feature_sets, lambda_param)
    for (name in names(preds_and_eqs)) {
      pred <- preds_and_eqs[[name]]$pred
      eqs <- preds_and_eqs[[name]]$equations
      rows[[length(rows) + 1L]] <- collect_fold_results(name, "GroupKFold", fold$name, test, pred, campeao)
      all_equations <- c(all_equations, eqs)
      oof_parts[[length(oof_parts) + 1L]] <- test |>
        select(ID_Observacao, Estudo, Pais_Estudo, Modelo, MAPE) |>
        mutate(pred = pred, modelo_ml = name, protocolo = "GroupKFold", fold = fold$name)
    }
  }
  list(rows = rows, oof = bind_rows(oof_parts), equations = all_equations)
}

run_loso_paises <- function(df, feature_sets, lambda_param) {
  estudos <- df |>
    filter(Pais_Estudo %in% PAISES_OCULTOS) |>
    pull(Estudo) |>
    unique()
  rows <- list()
  oof_parts <- list()
  all_equations <- character()
  for (estudo in estudos) {
    test <- df[df$Estudo == estudo, , drop = FALSE]
    train <- df[df$Estudo != estudo, , drop = FALSE]
    if (nrow(test) == 0L || nrow(train) == 0L) {
      next
    }
    campeao <- campeao_do_fold(train)
    log_msg(sprintf("Executando LOSO para Estudo: %s...", estudo))
    preds_and_eqs <- run_all_models_for_fold(train, test, feature_sets, lambda_param)
    for (name in names(preds_and_eqs)) {
      pred <- preds_and_eqs[[name]]$pred
      eqs <- preds_and_eqs[[name]]$equations
      rows[[length(rows) + 1L]] <- collect_fold_results(name, "LOSO_paises_case", as.character(estudo), test, pred, campeao)
      all_equations <- c(all_equations, eqs)
      oof_parts[[length(oof_parts) + 1L]] <- test |>
        select(ID_Observacao, Estudo, Pais_Estudo, Modelo, MAPE) |>
        mutate(pred = pred, modelo_ml = name, protocolo = "LOSO_paises_case", fold = as.character(estudo))
    }
  }
  list(
    rows = rows,
    oof = if (length(oof_parts) > 0L) bind_rows(oof_parts) else tibble(),
    equations = all_equations
  )
}

summarize_results <- function(rows) {
  log_msg(sprintf("Agregando resultados dos folds com estratégia '%s'...", AGGREGATION_STRATEGY))
  raw <- bind_rows(lapply(rows, as_tibble))
  if (nrow(raw) == 0L) {
    cols <- c("protocolo", "modelo", "n_folds", "mape", "mae", "rmse", "r2", "top1", "top1_sempre_campeao")
    return(list(raw = raw, agg = tibble::as_tibble(setNames(replicate(length(cols), logical(0), simplify = FALSE), cols))))
  }
  agg_metric <- switch(
    tolower(AGGREGATION_STRATEGY),
    weighted_mean = function(series, weights) weighted.mean(series, w = weights),
    weightedmean = function(series, weights) weighted.mean(series, w = weights),
    media_pesada = function(series, weights) weighted.mean(series, w = weights),
    mean = function(series, weights) mean(series),
    median = function(series, weights) median(series),
    stop("AGGREGATION_STRATEGY invalida. Use 'weighted_mean', 'mean' ou 'median'.")
  )
  agg <- raw |>
    group_by(protocolo, modelo) |>
    summarise(
      n_folds = n_distinct(fold),
      n = sum(n),
      mape = agg_metric(mape, n),
      mae = agg_metric(mae, n),
      rmse = agg_metric(rmse, n),
      r2 = agg_metric(r2, n),
      top1 = agg_metric(top1, n),
      top1_sempre_campeao = agg_metric(top1_sempre_campeao, n),
      .groups = "drop"
    ) |>
    arrange(protocolo, mape)
  list(raw = raw, agg = agg)
}

vif_from_matrix <- function(X) {
  X <- as.data.frame(X)
  keep <- names(X)[vapply(X, function(col) stats::sd(col) > 1e-12, logical(1))]
  X <- X[keep]
  bind_rows(lapply(names(X), function(col) {
    y <- X[[col]]
    Z <- X[setdiff(names(X), col)]
    if (ncol(Z) == 0L) {
      r2 <- 0
      vif <- 1
    } else {
      fit <- stats::lm(y ~ ., data = Z)
      r2 <- summary(fit)$r.squared
      vif <- if (r2 < 1 - 1e-9) 1 / (1 - r2) else Inf
    }
    tibble(bloco = NA_character_, variavel = col, r2_auxiliar = r2, VIF = vif)
  }))
}

compute_vif <- function(df) {
  missing <- setdiff(NUM_CANDIDATES, names(df))
  if (length(missing) > 0L) {
    stop("Colunas numericas ausentes para VIF: ", paste(missing, collapse = ", "))
  }
  vif_num <- vif_from_matrix(df[NUM_CANDIDATES])
  vif_num$bloco <- "numericos_brutos"
  vif_num$alerta <- ifelse(vif_num$VIF >= 10, "VIF>=10", ifelse(vif_num$VIF >= 5, "VIF>=5", "ok"))
  vif_num |>
    arrange(desc(VIF))
}

cluster_bootstrap_error <- function(oof) {
  log_msg("Calculando erro via bootstrap clusterizado por Estudo...")
  gkf <- oof |>
    filter(protocolo == "GroupKFold")
  if (nrow(gkf) == 0L) {
    return(tibble())
  }
  set.seed(BOOT_SEED)
  rows <- list()
  split_models <- split(gkf, gkf$modelo_ml, drop = TRUE)
  for (name in names(split_models)) {
    sub <- split_models[[name]] |>
      arrange(Estudo, ID_Observacao)
    grouped <- split(sub, sub$Estudo, drop = TRUE)
    studies <- names(grouped)
    n_studies <- length(studies)
    y_by_study <- lapply(grouped, function(df) df$MAPE)
    yhat_by_study <- lapply(grouped, function(df) df$pred)
    boot_idx <- replicate(N_BOOT, sample.int(n_studies, size = n_studies, replace = TRUE), simplify = FALSE)
    mae_b <- numeric(N_BOOT)
    rmse_b <- numeric(N_BOOT)
    mape_b <- numeric(N_BOOT)
    for (b in seq_len(N_BOOT)) {
      idx <- boot_idx[[b]]
      y_b <- unlist(y_by_study[idx], use.names = FALSE)
      yhat_b <- unlist(yhat_by_study[idx], use.names = FALSE)
      err <- y_b - yhat_b
      mae_b[[b]] <- mean(abs(err))
      rmse_b[[b]] <- sqrt(mean(err^2))
      mape_b[[b]] <- mape_metric(y_b, yhat_b)
    }
    y_obs <- sub$MAPE
    yhat_obs <- sub$pred
    err_obs <- y_obs - yhat_obs
    rows[[length(rows) + 1L]] <- tibble(
      protocolo = "GroupKFold",
      modelo = name,
      n_boot = N_BOOT,
      agrupamento = "Estudo",
      mape = mape_metric(y_obs, yhat_obs),
      mape_ic95_inf = as.numeric(stats::quantile(mape_b, 0.025)),
      mape_ic95_sup = as.numeric(stats::quantile(mape_b, 0.975)),
      mae = mean(abs(err_obs)),
      mae_ic95_inf = as.numeric(stats::quantile(mae_b, 0.025)),
      mae_ic95_sup = as.numeric(stats::quantile(mae_b, 0.975)),
      rmse = sqrt(mean(err_obs^2)),
      rmse_ic95_inf = as.numeric(stats::quantile(rmse_b, 0.025)),
      rmse_ic95_sup = as.numeric(stats::quantile(rmse_b, 0.975))
    )
  }
  bind_rows(rows) |>
    arrange(mae)
}

filter_models <- function(agg) {
  log_msg("Filtrando modelos com desempenho inferior à baseline...")
  baseline_metrics <- agg |>
    filter(modelo == "baseline_mediana_modelo") |>
    select(protocolo, mape, mae, rmse)
  models_to_keep <- unique(agg$modelo)
  for (i in seq_len(nrow(agg))) {
    row <- agg[i, , drop = FALSE]
    protocol <- row$protocolo[[1]]
    if (row$modelo[[1]] == "baseline_mediana_modelo") {
      next
    }
    base <- baseline_metrics |>
      filter(protocolo == protocol)
    if (nrow(base) == 1L && row$mape[[1]] > base$mape[[1]]) {
      models_to_keep <- setdiff(models_to_keep, row$modelo[[1]])
    }
  }
  models_to_keep
}

top_models_union_after_median_filter <- function(agg) {
  n_top <- as.integer(TOP_N_MODELOS_POR_PROTOCOLO)
  if (n_top <= 0L || nrow(agg) == 0L) {
    return(character())
  }
  baseline_name <- "baseline_mediana_modelo"
  selected <- character()
  if (baseline_name %in% unique(agg$modelo)) {
    selected <- c(selected, baseline_name)
  }
  for (protocolo_atual in c("LOSO_paises_case", "GroupKFold")) {
    pd_proto <- agg |>
      filter(protocolo == protocolo_atual)
    if (nrow(pd_proto) == 0L) {
      next
    }
    if (baseline_name %in% unique(pd_proto$modelo)) {
      selected <- c(selected, baseline_name)
    }
    top <- pd_proto |>
      arrange(mape, mae, rmse, modelo) |>
      slice_head(n = n_top)
    selected <- c(selected, top$modelo)
  }
  unique(selected)
}

main <- function() {
  root <- find_root()
  out_dir <- file.path(root, "data", "processed")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  df_raw <- ler_treino(root)
  stopifnot(isTRUE(all.equal(df_raw$Fracao_Perda_A, df_raw$Fracao_Perda_B)))
  if (n_distinct(df_raw$Sistema_Producao) != 1L) {
    warning("Sistema_Producao nao e constante: ", paste(unique(df_raw$Sistema_Producao), collapse = ", "))
  }

  box_result <- MASS::boxcox(MAPE ~ 1, data = df_raw, lambda = seq(-2, 2, 0.001), plotit = FALSE)
  best_lambda <- box_result$x[which.max(box_result$y)]

  modelagem <- df_raw |>
    mutate(
      MAPE_log = log(MAPE),
      MAPE_boxcox = if (best_lambda == 0) log(MAPE) else (MAPE^best_lambda - 1) / best_lambda
    ) |>
    dplyr::select(
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

  campeao <- df_raw |>
    group_by(Modelo) |>
    summarise(mediana = median(MAPE), .groups = "drop") |>
    slice_min(mediana, n = 1, with_ties = FALSE) |>
    pull(Modelo) |>
    as.character()

  meta <- tibble(
    n_linhas = nrow(modelagem),
    n_id = n_distinct(modelagem$ID_Observacao),
    n_estudo = n_distinct(modelagem$Estudo),
    campeao_global = campeao,
    lambda_boxcox = round(best_lambda, 4),
    colunas_excluidas = "Sistema_Producao; Fracao_Perda_B",
    info = "Estudo usado apenas como grupo de CV. País será one-hot encoded."
  )

  write.csv(modelagem, file.path(out_dir, "03_modelagem_r_monolitico.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(meta, file.path(out_dir, "03_meta_r_monolitico.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  writeLines(campeao, file.path(out_dir, "01_campeao_global_r_monolitico.txt"))

  df <- modelagem
  df$Estudo <- as.character(df$Estudo)
  log_msg(sprintf("Dados carregados: n=%d, estudos=%d", nrow(df), n_distinct(df$Estudo)))

  vif <- compute_vif(df)
  write.csv(vif, file.path(out_dir, "04_vif_r_monolitico.csv"), row.names = FALSE)
  log_msg("VIF calculado.")

  feature_sets <- generate_feature_sets()
  gkf <- run_group_kfold(df, feature_sets, best_lambda)
  loso <- run_loso_paises(df, feature_sets, best_lambda)

  summary <- summarize_results(c(gkf$rows, loso$rows))
  raw <- summary$raw
  agg <- summary$agg
  oof_parts <- Filter(function(part) nrow(part) > 0L, list(gkf$oof, loso$oof))
  oof <- if (length(oof_parts) > 0L) bind_rows(oof_parts) else tibble()
  all_equations <- c(gkf$equations, loso$equations)

  boot <- if (nrow(oof) > 0L) cluster_bootstrap_error(oof) else tibble()

  keep_mask <- rep(FALSE, nrow(agg))
  for (protocol in unique(agg$protocolo)) {
    proto <- agg |>
      filter(protocolo == protocol)
    models_to_keep <- filter_models(proto)
    keep_mask <- keep_mask | (agg$protocolo == protocol & agg$modelo %in% models_to_keep)
  }
  agg_f <- agg[keep_mask, , drop = FALSE]

  top_union_models <- top_models_union_after_median_filter(agg_f)
  if (length(top_union_models) > 0L) {
    agg_f <- agg_f |>
      filter(modelo %in% top_union_models)
  }

  raw_f <- raw |>
    inner_join(agg_f |>
      distinct(protocolo, modelo), by = c("protocolo", "modelo"))
  keep_pairs <- agg_f |>
    distinct(protocolo, modelo)
  oof_f <- oof |>
    inner_join(keep_pairs, by = c("protocolo", "modelo_ml" = "modelo"))
  boot_f <- boot |>
    inner_join(keep_pairs, by = c("protocolo", "modelo"))

  write.csv(agg_f, file.path(out_dir, "04_metricas_resumo_r_monolitico.csv"), row.names = FALSE)
  write.csv(raw_f, file.path(out_dir, "04_metricas_folds_r_monolitico.csv"), row.names = FALSE)
  write.csv(oof_f, file.path(out_dir, "04_predicoes_oof_r_monolitico.csv"), row.names = FALSE)
  write.csv(boot_f, file.path(out_dir, "04_bootstrap_ic_r_monolitico.csv"), row.names = FALSE)
  writeLines(sort(unique(all_equations)), file.path(out_dir, "04_equacoes_modelos_r_monolitico.txt"), useBytes = TRUE)

  summary_json <- list(
    metricas = jsonlite::fromJSON(jsonlite::toJSON(agg_f, dataframe = "rows", auto_unbox = TRUE)),
    vif_numericos = jsonlite::fromJSON(jsonlite::toJSON(vif, dataframe = "rows", auto_unbox = TRUE))
  )
  writeLines(
    jsonlite::toJSON(summary_json, ensure_ascii = FALSE, pretty = TRUE, auto_unbox = TRUE),
    file.path(out_dir, "04_resumo_r_monolitico.json"),
    useBytes = TRUE
  )

  log_msg("\n--- Resultados ---")
  log_msg("VIF (Numéricos):")
  log_msg(paste(capture.output(print(vif)), collapse = "\n"))
  log_msg("\nMelhores Modelos (Métricas Agregadas):")
  log_msg(paste(capture.output(print(head(agg_f, 10))), collapse = "\n"))
  log_msg("\nBootstrap (IC 95% do Erro - GroupKFold):")
  log_msg(paste(capture.output(print(head(boot_f, 10))), collapse = "\n"))
  log_msg(sprintf("\nResultados salvos em: %s", out_dir))
}

main()
