#' EBLUPs based on a Fay-Herriot Model.
#'
#' @description This function gives the Empirical Best Linear Unbiased Prediction (EBLUP) or Empirical Best (EB) predictor under normality based on a Fay-Herriot model.
#'
#' @references
#' \enumerate{
#'  \item Rao, J. N., & Molina, I. (2015). Small area estimation. John Wiley & Sons.
#'}
#'
#' @param formula an object of class formula that contains a description of the model to be fitted. The variables included in the formula must be contained in the data.
#' @param data a data frame or a data frame extension (e.g. a tibble).
#' @param vardir vector or column names from data that contain variance sampling from the direct estimator for each area.
#' @param method Fitting method can be chosen between 'ML' and 'REML'.
#' @param maxiter maximum number of iterations allowed in the Fisher-scoring algorithm. Default is 100 iterations.
#' @param precision convergence tolerance limit for the Fisher-scoring algorithm. Default value is 0.0001.
#' @param scale scaling auxiliary variable or not, default value is FALSE.
#' @param print_result print coefficient or not, default value is TRUE.
#'
#' @returns The function returns a list with the following objects (\code{df_res} and \code{fit}):
#' \code{df_res} a data frame that contains the following columns: \cr
#'    * \code{y} variable response \cr
#'    * \code{eblup} estimated results for each area \cr
#'    * \code{random_effect} random effect for each area \cr
#'    * \code{vardir} variance sampling from the direct estimator for each area \cr
#'    * \code{mse} Mean Square Error \cr
#'    * \code{rse} Relative Standart Error (%) \cr
#'
#' \code{fit} a list containing the following objects: \cr
#'    * \code{estcoef} a data frame with the estimated model coefficients in the first column (beta),
#'    their asymptotic standard errors in the second column (std.error),
#'    the t-statistics in the third column (tvalue) and the p-values of the significance of each coefficient
#'    in last column (pvalue) \cr
#'    * \code{model_formula} model formula applied \cr
#'    * \code{method} type of fitting method applied (`ML` or `REML`) \cr
#'    * \code{random_effect_var} estimated random effect variance \cr
#'    * \code{convergence} logical value that indicates the Fisher-scoring algorithm has converged or not \cr
#'    * \code{n_iter} number of iterations performed by the Fisher-scoring algorithm. \cr
#'    * \code{goodness} vector containing several goodness-of-fit measures: loglikehood, AIC, and BIC \cr
#'
#'
#' @details
#' The model has a form that is response ~ auxiliary variables.
#' where numeric type response variables can contain NA.
#' When the response variable contains NA it will be estimated with cluster information.
#'
#' @export
#' @examples
#' library(fastsae)
#'
#' m1 <- eblup_area(y ~ x1 + x2 + x3, data = na.omit(mys), vardir = "var")
#' m1 <- eblup_area(y ~ x1 + x2 + x3, data = na.omit(mys), vardir = ~var)
#'
#' @md
eblup_area <- function(
    formula,
    vardir,
    data,
    method = c("REML", "ML"),
    maxiter = 100,
    precision = 1e-4,
    return_fullP = FALSE
  ) {
  method <- match.arg(method, choices = c("REML", "ML"))

  # model frame & validasi
  mf <- stats::model.frame(formula, data, na.action = stats::na.omit)
  if (anyNA(vardir)) stop("'vardir' contains NA values")
  if (nrow(mf) != length(vardir)) {
    stop("Length of 'vardir' must equal number of observations in data")
  }

  y <- stats::model.response(mf, "numeric")
  X <- stats::model.matrix(attr(mf, "terms"), mf)

  # panggil core C++
  res <- eblup_core_optimized(
    X = X,
    y = y,
    vardir = vardir,
    method = method,
    maxiter = maxiter,
    precision = precision,
    return_fullP = return_fullP
  )

  # attach beberapa info tambahan
  res$formula <- formula
  res$method <- 'eblup'
  res$level <- 'area'
  class(res) <- "eblupfh"

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
  res
}
