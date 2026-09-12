#' EBLUPs based on a Fay-Herriot Model.
#'
#' @description This function gives the Empirical Best Linear Unbiased Prediction (EBLUP) or Empirical Best (EB) predictor under normality based on a Fay-Herriot model.
#'
#' @references
#' \enumerate{
#'  \item Rao, J. N., & Molina, I. (2015). Small area estimation. John Wiley & Sons.
#' }
#'
#' @param formula an object of class formula that contains a description of the model to be fitted.
#' @param data a data frame or a data frame extension (e.g. a tibble).
#' @param vardir vector or column names from data that contain variance sampling from the direct estimator.
#' @param method Fitting method can be chosen between 'ML' and 'REML'.
#' @param maxiter maximum number of iterations allowed in the Fisher-scoring algorithm.
#' @param precision convergence tolerance limit for the Fisher-scoring algorithm.
#' @param print_result print coefficient or not, default value is TRUE.
#'
#' @returns The function returns a list with the following objects:
#' \code{estcoef} a data frame with the estimated model coefficients,
#' \code{random_effect_var} estimated random effect variance,
#' \code{goodness} vector containing several goodness-of-fit measures,
#' \code{df_eblup} a data frame that contains y, eblup, vardir, mse, and rse.
#'
#' @details
#' The model has a form that is response ~ auxiliary variables.
#' where numeric type response variables can contain NA.
#' When the response variable contains NA it will be estimated with synthetic estimator.
#'
#' @export
#' @examples
#' library(fastsae)
#'
#' # Standard Fay-Herriot model
#' m1 <- eblup_fh(
#'   y ~ x1 + x2 + x3,
#'   data = mys,
#'   vardir = "vardir"
#' )
#'
#' @md
eblup_fh <- function(
  formula,
  vardir,
  data,
  method = c("REML", "ML"),
  maxiter = 100,
  precision = 1e-4,
  print_result = TRUE
) {
  method <- match.arg(method, choices = c("REML", "ML"))

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

  res <- .eblup_core(
    Xall = X,
    yall = y,
    vardirall = vardir,
    method = method,
    maxiter = maxiter,
    precision = precision
  )

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
