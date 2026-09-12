#' EBLUPs based on a Spatial Fay-Herriot Model.
#'
#' @description This function gives the Spatial Empirical Best Linear Unbiased Prediction (EBLUP) or Empirical Best (EB) predictor under normality based on a Fay-Herriot model.
#'
#' @references
#' \enumerate{
#'  \item Rao, J. N., & Molina, I. (2015). Small area estimation. John Wiley & Sons.
#' }
#'
#' @param formula an object of class formula that contains a description of the model to be fitted. The variables included in the formula must be contained in the data.
#' @param data a data frame or a data frame extension (e.g. a tibble).
#' @param vardir vector or column names from data that contain variance sampling from the direct estimator for each area.
#' @param method Fitting method can be chosen between 'ML' and 'REML'.
#' @param W A square matrix with dimension equal to the TOTAL number of domains in `data`
#' (including any unsampled domains where the response is `NA`). It should contain the
#' row-standardized spatial weights (proximities) between ALL domains, with values ranging
#' from 0 to 1. Rows and columns must be ordered consistently with the domain identifiers in
#' `data`. Do NOT pre-subset `W` to sampled domains only -- unsampled domains' spatial MSE/
#' prediction relies on their relation (in `W`) to sampled neighbors.
#' @param mse_method a character string determining the estimation method of the MSE.
#'   Methods that can be chosen: "analytical", "pbmse", or "npbmse".
#' When there are unsampled domains, "pbmse"/"npbmse" bootstrap MSE is only defined for the
#' sampled domains; unsampled domains automatically get a full-spatial synthetic (kriging)
#' prediction and analytical MSE merged into the result regardless of `mse_method`.
#' @param B Number of bootstrap replications when mse_methpd = "pbmse" or "npbmse".
#' @param n_threads Number of threads used in parallel computation.
#'   Values less than or equal to 0 use the default OpenMP configuration (all cores).
#' @param seed Integer seed for bootstrap resampling.
#'   A value of -1 leaves the current R RNG state unchanged.
#' @param maxiter maximum number of iterations allowed in the Fisher-scoring algorithm. Default is 100 iterations.
#' @param precision convergence tolerance limit for the Fisher-scoring algorithm. Default value is 0.0001.
#' @param print_result print coefficient or not, default value is TRUE.
#'
#' @returns The function returns a list with the following objects (\code{df_res} and \code{fit}):
#' \code{estcoef} a data frame with the estimated model coefficients in the first column (beta),
#'    their asymptotic standard errors in the second column (std.error),
#'    the t-statistics in the third column (tvalue) and the p-values of the significance of each coefficient
#'    in last column (pvalue) \cr
#' \code{formula} model formula applied \cr
#' \code{random_effect_var} estimated random effect variance \cr
#' \code{goodness} vector containing several goodness-of-fit measures: loglikehood, AIC, and BIC \cr
#' \code{df_eblup} a data frame that contains the following columns: \cr
#'    * \code{y} variable response \cr
#'    * \code{eblup} estimated results for each area \cr
#'    * \code{random_effect} random effect for each area \cr
#'    * \code{vardir} variance sampling from the direct estimator for each area \cr
#'    * \code{mse} Mean Square Error \cr
#'    * \code{rse} Relative Standart Error (%) \cr
#'    * \code{mse_pb} or \code{mse_npb} Parametric / Non Parametric bootstrap MSE\cr
#'    * \code{mse_pbbc} or \code{mse_npbbc} Bias Corrected Parametric / Non Parametric bootstrap MSE \cr
#'
#' @details
#' The model has a form that is response ~ auxiliary variables.
#' where numeric type response variables can contain NA.
#' When the response variable contains NA, that domain is treated as unsampled: it is
#' predicted with a full-spatial synthetic (kriging) estimator that borrows strength from
#' sampled neighbors via `W`, with an analytical MSE approximation.
#'
#' @export
#' @examples
#' library(fastsae)
#'
#' # Spatial Fay-Herriot model
#' m1 <- seblup_area(
#'   y ~ x1 + x2 + x3,
#'   data = mys,
#'   vardir = ~vardir,
#'   W = mys_proxmat
#' )
#'
#' # Spatial Fay-Herriot model with Parametric Bootstrap MSE
#' m2 <- seblup_area(
#'   y ~ x1 + x2 + x3,
#'   data = mys,
#'   vardir = ~vardir,
#'   mse_method = 'pbmse',
#'   B = 50,
#'   W = mys_proxmat
#' )
#'
#' @md
seblup_area <- function(
    formula,
    vardir,
    data,
    method = c("REML", "ML"),
    mse_method = c("analytical", "pbmse", "npbmse"),
    W = NULL,
    B = 100,
    n_threads = 0,
    seed = -1,
    maxiter = 100,
    precision = 1e-4,
    print_result = TRUE
) {
  method <- match.arg(toupper(method), choices = c("REML", "ML"))
  mse_method <- match.arg(tolower(mse_method), choices = c("analytical", "pbmse", "npbmse"))

  # model frame & validasi
  mf <- stats::model.frame(formula, data, na.action = stats::na.pass)
  vardir <- .get_variable(data, vardir)


  if (nrow(mf) != length(vardir)) {
    stop("Length of 'vardir' must equal number of observations in data")
  }

  y <- stats::model.response(mf, "numeric")
  X <- stats::model.matrix(attr(mf, "terms"), mf)

  # cek Auxiliary variabels mengandung NA atau tidak
  if (anyNA(X)) {
    cli::cli_abort("Auxiliary variabels contains NA values.")
  }

  # Validasi vardir untuk area tersampel saja (y tidak NA)
  y_valid <- !is.na(y)
  if (any(vardir[y_valid] <= 0, na.rm = TRUE)) {
    stop("vardir must be strictly positive for sampled areas")
  }


  if (is.null(W)) {
    cli::cli_abort("Argument `W` (spatial weight matrix) must be provided when `spatial = TRUE`.")
  }

  n_total <- nrow(X)
  if (!is.matrix(W) || nrow(W) != n_total || ncol(W) != n_total) {
    cli::cli_abort(c(
      "`W` must be a square matrix with dimension equal to the total number of domains in `data` ({n_total}), including any unsampled ones.",
      "i" = "Got a {nrow(W)}x{ncol(W)} matrix. Do not pre-subset `W` to sampled domains only."
    ))
  }




  idx_s <- !is.na(y)
  has_unsampled <- any(!idx_s)

  if (mse_method %in% c("pbmse", "npbmse")) {
    if (has_unsampled) {
      cli::cli_alert_info(
        "{sum(!idx_s)} unsampled domain(s) detected. Bootstrap MSE ({mse_method}) is computed for the {sum(idx_s)} sampled domain(s); unsampled domain(s) get a full-spatial synthetic (kriging) prediction and analytical MSE instead."
      )
    }

    # Fungsi bootstrap butuh input complete-case (sampled-only, tanpa NA).
    Xs      <- X[idx_s, , drop = FALSE]
    ys      <- y[idx_s]
    vardirs <- vardir[idx_s]
    Ws      <- W[idx_s, idx_s, drop = FALSE]

    if (mse_method == "pbmse") {
      res <- .seblup_pbmse(
        X = Xs, y = ys, vardir = vardirs, method = method, W = Ws,
        maxiter = maxiter, precision = precision, B = B,
        n_threads = n_threads, seed = seed
      )
    } else {
      res <- .seblup_npbmse(
        X = Xs, y = ys, vardir = vardirs, method = method, W = Ws,
        maxiter = maxiter, precision = precision, B = B,
        n_threads = n_threads, seed = seed
      )
    }



    if (has_unsampled) {
      # Lengkapi df_eblup dengan area tak-tersampel via kriging spasial
      # penuh (mse_method = "analytical" style, dihitung sekali lagi
      # dengan Wall utuh), digabung ke hasil bootstrap.
      res_full <- .seblup_core(
        Xall = X, yall = y, vardirall = vardir, method = method, Wall = W,
        maxiter = maxiter, precision = precision
      )

      df_full <- res_full$df_eblup
      # timpa baris area TERSAMPEL dengan hasil dari fit bootstrap (sama
      # titik estimasinya, tapi mse/rse dari bootstrap dipakai)
      df_full$eblup[idx_s] <- res$df_eblup$eblup
      df_full$mse[idx_s]   <- res$df_eblup$mse
      df_full$rse[idx_s]   <- res$df_eblup$rse

      # tempelkan kolom bootstrap-spesifik (mse_pb/mse_pbbc atau
      # mse_npb/mse_npbbc) -- NA untuk area tak-tersampel, karena metode
      # bootstrap memang tidak didefinisikan untuk area itu (MSE-nya
      # berasal dari estimator kriging spasial di atas).
      boot_cols <- setdiff(names(res$df_eblup), c("y", "vardir", "eblup", "mse", "rse"))
      for (col in boot_cols) {
        full_col <- rep(NA_real_, n_total)
        full_col[idx_s] <- res$df_eblup[[col]]
        df_full[[col]] <- full_col
      }

      res$df_eblup <- df_full
    }
  } else {
    res <- .seblup_core(
      Xall = X,
      yall = y,
      vardirall = vardir,
      method = method,
      Wall = W,
      maxiter = maxiter,
      precision = precision
    )
  }

  # attach beberapa info tambahan
  row.names(res$estcoef) <- colnames(X)
  res$call <- match.call()
  class(res) <- "fastsae"

  if (!res$convergence) {
    cli::cli_alert_danger("After {res$n_iter} iterations, there is no convergence.")
    return(res)
  }

  if (print_result) {
    cli::cli_alert_success("Convergence after {.orange {res$n_iter}} iterations")
    cli::cli_alert("Method : {method}")
    cli::cli_h1("Coefficient")
    stats::printCoefmat(res$estcoef, signif.stars = TRUE)
  }
  return(res)
}



# Fungsi Penolong ---------------------------------------------------------

# extract variable from data frame
.get_variable <- function(data, variable) {
  if (length(variable) == nrow(data)) {
    return(variable)
  } else if (methods::is(variable, "character")) {
    if (variable %in% colnames(data)) {
      variable <- data[[variable]]
    } else {
      cli::cli_abort('variable "{variable}" is not found in the data')
    }
  } else if (methods::is(variable, "formula")) {
    # extract column name (class character) from formula
    variable <- data[[all.vars(variable)]]
  } else {
    cli::cli_abort('variable "{variable}" is not found in the data')
  }
  return(variable)
}
