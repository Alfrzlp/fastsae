#' A Fast Implementation of Hierarchical Bayes Algorithms for Small Area Estimation Area Level
#'
#' @param formula an object of class formula that contains a description of the model to be fitted. The variables included in the formula must be contained in the data.
#' @param data a data frame or a data frame extension (e.g. a tibble).
#' @param likelihood Sampling likelihood to be used. The choices are "beta" (default),
#' @param iter_sampling A positive integer specifying how many iterations for each chain (including warmup). The default is 2000.
#' @param warmup A positive integer specifying the number of warmup (aka burnin) iterations. If step-size adaptation is on (which it is by default), this also controls the number of iterations for which adaptation is run (and hence the samples should not be
#' @param chains A positive integer specifying number of chains; defaults to 4.
#' @param thin A positive integer specifying the period for saving samples; defaults to 1.
#' @param parallel_chains he maximum number of MCMC chains to run in parallel. If parallel_chains is not specified then the default is to look for the option "mc.cores", which can be set for an entire R session by options(mc.cores=value). If the "mc.cores" option has not been set then the default is 1.
#' @param max_treedepth The maximum allowed tree depth for the NUTS engine. See the Tree Depth section of the CmdStan User's Guide for more details.
#' @param adapt_delta The adaptation target acceptance statistic value is a real number in the interval (0, 1).
#' @param seed The seed for random number generation. The default is generated from 1 to the maximum integer supported by R on the machine. Even if multiple chains are used, only one seed is needed, with other chains having seeds derived from that of the first chain.
#' @param ... Arguments passed to $sample() (see https://mc-stan.org/cmdstanr/reference/model-method-sample.html).
#'
#' @returns The function returns a list with the following objects (\code{df_res} and \code{fit}):
#' \code{df_res} a data frame that contains the following columns: \cr
#'    * \code{y} variable response \cr
#'    * \code{eblup} estimated results for each area \cr
#'    * \code{mse} Mean Square Error \cr
#'    * \code{rse} Relative Standart Error (%) \cr
#'
#' @export
#' @examples
#' library(fastsae)
#'
#' m1 <- hb_area(y ~ x1 + x2 + x3, data = mys)
#'
#' @details
#' The model has a form that is response ~ auxiliary variables. Where numeric type response variables can contain NA.
#'
#' @examples
hb_area <- function(
    formula,
    data,
    likelihood = 'beta',
    iter_sampling = 1000,
    warmup = floor(iter_sampling/2),
    chains = 4,
    thin = 1,
    parallel_chains = parallel::detectCores() - 1,
    max_treedepth = 15,
    adapt_delta = 0.95,
    seed = 1,
    ...
){
  result <- list(
    method = 'hb',
    level = 'area'
  )

  formuladata <- stats::model.frame(formula, data, na.action = NULL)
  if (any(is.na(formuladata[, -1]))) {
    stop("Auxiliary Variables contains NA values.")
  }

  y_all <- formuladata[[1]]
  idx_na <- is.na(y_all)

  y_sampled <- y_all[!idx_na]
  x_sampled <- formuladata[!idx_na, -1]
  x_nonsampled <- formuladata[idx_na, -1]
  nvar <- ncol(x_sampled)

  data_list <- list(
    # number of sampled areas
    n1 = nrow(x_sampled),
    # number of nonsampled areas
    n2 = nrow(x_nonsampled),
    # number of covariates
    nvar = nvar,
    # vector of proportions for sampled areas
    y_sampled = y_sampled,
    x_sampled = x_sampled,
    # matrix of covariates for sampled
    x_nonsampled = x_nonsampled,
    # prior means for regression coefficients
    mu_b = rep(0, nvar),
    # prior precisions
    tau_b = rep(1, nvar),
    phi_aa = 2,
    phi_ab = 1,
    phi_ba = 2,
    phi_bb = 1,
    tau_ua = 2,
    tau_ub = 1
  )

  mod <- cmdstanr::cmdstan_model("include/hierarchical_beta_sae.stan")

  # Fit model
  fit <- mod$sample(
    data = data_list,
    seed = seed,
    chains = chains,
    parallel_chains = parallel_chains,
    iter_sampling = iter_sampling,
    max_treedepth = max_treedepth,
    adapt_delta = adapt_delta,
    iter_warmup = iter_warmup,
    thin = thin,
    ...
  )

  result$fit <- fit

  class(result) <- 'fastsae_result'
  return(result)
}

