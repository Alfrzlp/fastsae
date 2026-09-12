#' EBLUPs based on a Spatio-Temporal Fay-Herriot Model.
#'
#' @description This function gives the Spatio-Temporal Empirical Best Linear
#' Unbiased Prediction (EBLUP) under normality based on a spatio-temporal
#' Fay-Herriot model. It reimplements the same Fisher-scoring algorithm as
#' \code{eblupSTFH()} (package \pkg{sae}, Marhuenda, Molina & Morales 2013),
#' but the estimation loop runs in compiled C++/Armadillo (\code{.steblup_core}),
#' making it substantially faster and more memory-efficient than the original
#' R implementation -- especially for a large number of domains/time periods.
#'
#' @references
#' \enumerate{
#'  \item Marhuenda, Y., Molina, I., & Morales, D. (2013). Small area estimation
#'    with spatio-temporal Fay-Herriot models. Computational Statistics & Data
#'    Analysis, 58, 308-325.
#'  \item Rao, J. N., & Molina, I. (2015). Small area estimation. John Wiley & Sons.
#' }
#'
#' @param formula an object of class formula describing the model to fit
#'   (response ~ auxiliary variables). Variables must be present in \code{data}.
#' @param data a data frame (or extension) with \code{D * Time} rows, sorted so
#'   that all \code{Time} periods of domain 1 come first, then all periods of
#'   domain 2, and so on (i.e. domain-major order) -- exactly as required by
#'   \code{eblupSTFH()}.
#' @param vardir vector, or column name / one-sided formula referencing a column
#'   in \code{data}, with the sampling variances of the direct estimator.
#' @param D number of domains (areas).
#' @param Time number of time periods per domain. Equivalent to the \code{T}
#'   argument of \code{eblupSTFH()}
#' @param W a square proximity/spatial weights matrix of dimension
#'   \code{D x D} (row-standardized, values typically in \eqn{[0,1]}).
#' @param model character, either \code{"ST"} (spatio-temporal, default) or
#'   \code{"S"} (spatial only, no AR(1) temporal component).
#' @param maxiter maximum number of Fisher-scoring iterations. Default 100.
#' @param precision convergence tolerance for the Fisher-scoring algorithm.
#'   Default \code{1e-4}.
#' @param sigma21_start,rho1_start,sigma22_start,rho2_start starting values for
#'   the variance/autocorrelation components. Defaults mirror \code{eblupSTFH()}:
#'   \code{0.5 * median(vardir)} for the variances and \code{0.5} for the
#'   autocorrelations. \code{rho2_start} is ignored when \code{model = "S"}.
#' @param print_result print the estimated coefficients or not. Default \code{TRUE}.
#'
#' @returns A list with the same shape as \code{eblupSTFH()}'s output:
#' \describe{
#'   \item{\code{eblup}}{vector of length \code{D*Time} with the EBLUP estimates.}
#'   \item{\code{fit}}{a list with:
#'     \code{model}, \code{convergence}, \code{iterations},
#'     \code{estcoef} (data frame: beta, std.error, tvalue, pvalue),
#'     \code{estvarcomp} (data frame: estimate, std.error, for sigma21, rho1,
#'     sigma22, rho2), and \code{goodness} (loglike, AIC, BIC).}
#' }
#'
#' @details
#' This function does not (yet) compute an analytical or bootstrap MSE, matching
#' \code{eblupSTFH()}'s scope. Unlike \code{eblupSTFH()}, unsampled domain/time
#' cells (\code{NA} in the response) are not supported -- \code{data} must be a
#' complete panel, again matching \code{eblupSTFH()}'s requirements.
#'
#' @export
#' @examples
#' \dontrun{
#' library(fastsae)
#'
#' m1 <- steblup_area(
#'   y ~ x1 + x2,
#'   data = panel_data,
#'   vardir = ~vardir,
#'   D = 50,
#'   Time = 6,
#'   W = W,
#'   model = "ST"
#' )
#' }
#'
#' @md
steblup_area <- function(
    formula,
    vardir,
    data,
    D,
    Time,
    W,
    model = c("ST", "S"),
    maxiter = 100,
    precision = 1e-4,
    sigma21_start = NULL,
    rho1_start = 0.5,
    sigma22_start = NULL,
    rho2_start = 0.5,
    print_result = TRUE
) {
  model <- match.arg(model, choices = c("ST", "S"))

  # ---- model frame & matriks desain ----
  mf <- stats::model.frame(formula, data, na.action = stats::na.omit)
  if (nrow(mf) != nrow(data)) {
    stop("Argument formula=", deparse(formula), " contains NA values (unsampled domains/periods are not supported).")
  }
  X <- stats::model.matrix(attr(mf, "terms"), mf)
  y <- stats::model.response(mf, "numeric")

  vardir <- .get_variable(data, vardir)
  if (anyNA(vardir)) {
    stop("Argument vardir contains NA values.")
  }

  # ---- validasi dimensi ----
  M <- D * Time
  if (nrow(X) != M || length(y) != M || length(vardir) != M) {
    stop(
      "formula=", deparse(formula), " [rows=", nrow(X), "] and vardir [rows=",
      length(vardir), "] must have D*Time = ", D, "*", Time, " = ", M, " rows."
    )
  }

  if (!is.matrix(W)) W <- as.matrix(W)
  if (anyNA(W)) stop("Argument W contains NA values.")
  if (nrow(W) != D || ncol(W) != D) {
    stop("Argument W must be a square matrix of size D=", D, ".")
  }

  if (!is.null(sigma21_start) && sigma21_start < 0) {
    stop("Argument sigma21_start must be >= 0.")
  }
  if (!is.null(sigma22_start) && sigma22_start < 0) {
    stop("Argument sigma22_start must be >= 0.")
  }
  if (rho1_start <= -1 || rho1_start >= 1) {
    stop("Argument rho1_start must be in the interval (-1,1).")
  }
  if (model == "ST" && (rho2_start <= -1 || rho2_start >= 1)) {
    stop("Argument rho2_start must be in the interval (-1,1).")
  }

  # ---- panggil core C++ ----
  res <- .steblup_core(
    X = X,
    y = y,
    vardir = vardir,
    proxmat = W,
    D = as.integer(D),
    Tt = as.integer(Time),
    model = model,
    maxiter = as.integer(maxiter),
    precision = precision,
    sigma21_start = if (is.null(sigma21_start)) -1 else sigma21_start,
    rho1_start = rho1_start,
    sigma22_start = if (is.null(sigma22_start)) -1 else sigma22_start,
    rho2_start = rho2_start
  )

  if (!is.null(res$fit$estcoef)) {
    row.names(res$fit$estcoef) <- colnames(X)
  }
  res$call <- match.call()
  class(res) <- "fastsae_st"

  if (!isTRUE(res$fit$convergence)) {
    cli::cli_alert_danger(
      "After {res$fit$iterations} iteration(s), there is no convergence."
    )
    return(res)
  }

  if (print_result) {
    cli::cli_alert_success("Convergence after {.orange {res$fit$iterations}} iterations")
    cli::cli_alert("Model : {model}")
    cli::cli_h1("Coefficient")
    stats::printCoefmat(res$fit$estcoef[, c("beta", "std.error", "tvalue", "pvalue")],
                        signif.stars = TRUE)
    cli::cli_h1("Variance / autocorrelation components")
    print(res$fit$estvarcomp)
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
