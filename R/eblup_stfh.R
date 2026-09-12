#' EBLUPs based on a Spatio-Temporal Fay-Herriot Model.
#'
#' @description This function gives the Spatio-Temporal Empirical Best Linear
#' Unbiased Prediction (EBLUP) under normality based on a spatio-temporal
#' Fay-Herriot model. It reimplements the same Fisher-scoring algorithm as
#' \code{eblupSTFH()} (package \pkg{sae}, Marhuenda, Molina & Morales 2013),
#' but the estimation loop runs in compiled C++/Armadillo (\code{.eblup_stfh_core}),
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
#' @param data a data frame (or extension) with \code{domain * time} rows, sorted so
#'   that all \code{time} periods of domain 1 come first, then all periods of
#'   domain 2, and so on (i.e. domain-major order) -- exactly as required by
#'   \code{eblupSTFH()}.
#' @param vardir vector, column name or one-sided formula referencing a column
#'   in \code{data}, with the sampling variances of the direct estimator.
#' @param domain vector, column name or one-sided formula referencing a domain names column
#'   in \code{data}.
#' @param time vector, column name, or one-sided formula referencing a time names column
#'   in \code{data}.
#' @param W a square proximity/spatial weights matrix of dimension
#'   \code{domain x domain} (row-standardized, values typically in \eqn{[0,1]}).
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
#' @returns A list with the same structure as \code{seblup_area()}, with additional
#'   \code{estvarcomp} for spatio-temporal variance/autocorrelation components:
#'   \describe{
#'     \item{\code{estcoef}}{data frame with beta, std.error, tvalue, pvalue.}
#'     \item{\code{estvarcomp}}{data frame with estimate, std.error for sigma21, rho1, sigma22, rho2.}
#'     \item{\code{goodness}}{vector with loglike, AIC, BIC.}
#'     \item{\code{df_eblup}}{data frame with eblup, random_effect_u1, random_effect_u2.}
#'     \item{\code{model}}{model type ("ST" or "S").}
#'     \item{\code{convergence}}{logical, whether the algorithm converged.}
#'     \item{\code{n_iter}}{number of iterations.}
#'   }
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
#' m1 <- eblup_stfh(
#'   y ~ x1 + x2,
#'   data = mys_panel,
#'   vardir = ~vardir,
#'   domain = ~area,
#'   time = ~year,
#'   W = W,
#'   model = "ST"
#' )
#' }
#'
#' @md
eblup_stfh <- function(
    formula,
    vardir,
    data,
    domain,
    time,
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

  # ---- model frame & matrix design ----
  mf <- stats::model.frame(formula, data, na.action = stats::na.omit)
  if (nrow(mf) != nrow(data)) {
    stop("Argument formula=", deparse(formula), " contains NA values (unsampled domains/periods are not supported).")
  }
  X <- stats::model.matrix(attr(mf, "terms"), mf)
  y <- stats::model.response(mf, "numeric")

  # vardir
  vardir <- .get_variable(data, vardir)
  if (nrow(mf) != length(vardir)) {
    stop("Length of 'vardir' must equal number of observations in data")
  }

  y_valid <- !is.na(y)
  if (any(vardir[y_valid] <= 0, na.rm = TRUE)) {
    stop("vardir must be strictly positive for sampled areas")
  }

  # domain & time
  domain <- .get_variable(data, domain)
  time <- .get_variable(data, time)
  n_domain <- length(unique(domain))
  n_time <- length(unique(time))

  # check dimension
  M <- n_domain * n_time
  if (nrow(X) != M || length(y) != M || length(vardir) != M) {
    stop(
      "formula=", deparse(formula), " [rows=", nrow(X), "] and vardir [rows=",
      length(vardir), "] must have domain * time = ", n_domain, "*", n_time, " = ", M, " rows."
    )
  }

  # check W
  if (!is.matrix(W)) W <- as.matrix(W)
  if (anyNA(W)) stop("Argument W contains NA values.")
  if (nrow(W) != n_domain || ncol(W) != n_domain) {
    stop("Argument W must be a square matrix of size Domain =", n_domain, ".")
  }

  # check sigma and rho
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

  # ---- call C++ core ----
  res <- .eblup_stfh_core(
    X = X,
    y = y,
    vardir = vardir,
    proxmat = W,
    D = as.integer(n_domain),
    Tt = as.integer(n_time),
    model = model,
    maxiter = as.integer(maxiter),
    precision = precision,
    sigma21_start = if (is.null(sigma21_start)) -1 else sigma21_start,
    rho1_start = rho1_start,
    sigma22_start = if (is.null(sigma22_start)) -1 else sigma22_start,
    rho2_start = rho2_start
  )

  # ---- attach additional info (matching seblup_area structure) ----
  # Row names for estcoef
  if (!is.null(res$estcoef)) {
    row.names(res$estcoef) <- colnames(X)
  }

  # Attach formula
  res$formula <- formula

  # Add class
  res$call <- match.call()
  class(res) <- "fastsae"

  # Convergence check
  if (!isTRUE(res$convergence)) {
    cli::cli_alert_danger(
      "After {res$n_iter} iteration(s), there is no convergence."
    )
    return(res)
  }

  # Print results
  if (print_result) {
    cli::cli_alert_success("Convergence after {.orange {res$n_iter}} iterations")
    cli::cli_alert("Model : {model}")
    cli::cli_h1("Coefficient")
    stats::printCoefmat(res$estcoef, signif.stars = TRUE)
    cli::cli_h1("Variance / autocorrelation components")
    print(res$estvarcomp)
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
