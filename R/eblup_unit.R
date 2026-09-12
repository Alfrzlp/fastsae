#' Empirical Best Linear Unbiased Prediction (EBLUP) for the Battese-Harter-Fuller Model
#'
#' This function estimates small area means or totals using the unit-level model
#' proposed by Battese, Harter, and Fuller (1988), which combines survey data
#' (sample units) and auxiliary population information.
#'
#' @param formula An object of class `formula` describing the model.
#' @param unit_data A `data.frame` containing the unit-level survey data.
#' @param Xpop A `data.frame` containing auxiliary variables and domain info.
#' @param domain_var A character string giving the column name for domain identifier.
#' @param popsize_var A character string for population size variable.
#' @param method Fitting method: "REML" (default) or "ML".
#' @param popnmean_xpop Population mean of auxiliary variables per domain.
#' @param B Number of bootstrap replicates for MSE (if mse = TRUE).
#' @param mse If TRUE, compute bootstrap MSE.
#' @param n_threads Number of threads for parallel computation.
#' @param seed Random seed for reproducibility.
#' @param print_result Print results (default TRUE).
#'
#' @returns List containing EBLUP estimates, fit, and optionally MSE.
#'
#' @references
#' Battese, G. E., Harter, R. M., and Fuller, W. A. (1988). An error-components
#' model for prediction of county crop areas using survey and satellite data.
#' *Journal of the American Statistical Association*, 83(401), 28-36.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df_meanpop <- cornsoybeanmeans |>
#'   rename(CornPix = MeanCornPixPerSeg, SoyBeansPix = MeanSoyBeansPixPerSeg)
#' df_cornsoybean <- cornsoybean |>
#'   rename(CountyIndex = County)
#'
#' res <- eblup_unit(
#'   formula = CornHec ~ CornPix + SoyBeansPix,
#'   Xpop = df_meanpop,
#'   unit_data = df_cornsoybean,
#'   domain_var = "CountyIndex",
#'   popsize_var = "PopnSegments"
#' )
#' }
#' @export
eblup_unit <- function(
  formula,
  unit_data,
  Xpop,
  domain_var,
  popsize_var,
  method = c("REML", "ML"),
  popnmean_xpop = NULL,
  B = 100,
  mse = FALSE,
  n_threads = 1,
  seed = -1,
  print_result = TRUE
) {
  method <- match.arg(method, choices = c("REML", "ML"))

  # --- data preparation ---
  formuladata_all <- stats::model.frame(formula, na.action = stats::na.pass, unit_data)
  formuladata <- stats::model.frame(formula, na.action = stats::na.omit, unit_data)
  dom <- .get_variable(unit_data, domain_var)
  selectdom <- unique(dom)

  Xs <- stats::model.matrix(formula, unit_data)
  ys <- formuladata[[1]]

  # --- fit linear mixed model ---
  formula_lmer <- ys ~ -1 + Xs + (1 | dom)
  fit <- lme4::lmer(formula_lmer, data = unit_data)
  betaest <- matrix(lme4::fixef(fit), ncol = 1)
  upred <- lme4::ranef(fit)$dom

  # --- prepare population data ---
  Xpop <- dplyr::arrange(Xpop, match(domain_var, selectdom))
  popnsize <- .get_variable(Xpop, popsize_var)

  if (is.null(popnmean_xpop)) {
    formula_noy <- stats::reformulate(attr(stats::terms(formula), "term.labels"))
    meanxpop <- stats::model.matrix(formula_noy, Xpop)
  } else {
    meanxpop <- as.matrix(popnmean_xpop)
  }

  # --- extract variance components ---
  sigma2_u <- lme4::VarCorr(fit)$dom[1, 1]
  sigma2_e <- attr(lme4::VarCorr(fit), "sc")^2

  # --- compute EBLUP ---
  result <- .eblup_unit_cpp(
    selectdom = selectdom,
    dom = dom,
    Xs = Xs,
    meanxpop = meanxpop,
    ys = ys,
    popnsize = popnsize,
    betaest = betaest,
    upred = upred
  )

  eblup_df <- data.frame(
    domain = selectdom,
    eblup = result$eblup,
    samp_size = result$samp_size
  )

  if (length(result$warn_domains) > 0) {
    cli::cli_warn(
      "{length(result$warn_domains)} domain(s) have no sample units: {paste(result$warn_domains, collapse = ', ')}"
    )
  }

  # --- bootstrap MSE ---
  if (mse) {
    if (print_result) {
      cli::cli_alert_info("Computing bootstrap MSE with B = {B} replicates...")
    }

    mse_result <- pbmse_unit(
      formula = formula,
      unit_data = unit_data,
      Xpop = Xpop,
      domain_var = domain_var,
      popsize_var = popsize_var,
      method = method,
      B = B,
      n_threads = n_threads,
      seed = seed
    )

    mse_df <- data.frame(
      domain = selectdom,
      mse = mse_result$mse
    )

    eblup_df <- merge(eblup_df, mse_df, by = "domain", sort = FALSE)
  }

  # --- assemble output ---
  out <- list(
    eblup = eblup_df,
    fit = list(
      method = method,
      sigma2_u = sigma2_u,
      sigma2_e = sigma2_e,
      beta = betaest,
      random_effect = upred
    ),
    call = match.call()
  )

  class(out) <- "fastsae_unit"

  if (print_result) {
    cli::cli_alert_success("EBLUP estimation completed")
    cli::cli_h1("Summary")
    print(head(out$eblup, 10))
    if (nrow(out$eblup) > 10) {
      cli::cli_text("... and {nrow(out$eblup) - 10} more rows")
    }
  }

  return(out)
}

# ============================================================================
# Parametric Bootstrap MSE for Unit-level Model
# ============================================================================
#' @rdname eblup_unit
#' @export
pbmse_unit <- function(
  formula,
  unit_data,
  Xpop,
  domain_var,
  popsize_var,
  method = c("REML", "ML"),
  B = 100,
  n_threads = 1,
  seed = -1
) {
  method <- match.arg(method, choices = c("REML", "ML"))

  # --- data preparation ---
  formuladata <- stats::model.frame(formula, na.action = stats::na.omit, unit_data)
  dom <- .get_variable(unit_data, domain_var)
  selectdom <- unique(dom)

  Xs <- stats::model.matrix(formula, unit_data)
  ys <- formuladata[[1]]

  # --- prepare population data ---
  Xpop <- dplyr::arrange(Xpop, match(domain_var, selectdom))
  popnsize <- .get_variable(Xpop, popsize_var)

  formula_noy <- stats::reformulate(attr(stats::terms(formula), "term.labels"))
  meanxpop <- stats::model.matrix(formula_noy, Xpop)

  # --- initial fit ---
  formula_lmer <- ys ~ -1 + Xs + (1 | dom)
  fit <- lme4::lmer(formula_lmer, data = unit_data)
  betaest <- matrix(lme4::fixef(fit), ncol = 1)
  upred <- lme4::ranef(fit)$dom

  sigma2_u <- lme4::VarCorr(fit)$dom[1, 1]
  sigma2_e <- attr(lme4::VarCorr(fit), "sc")^2

  # --- initial EBLUP ---
  init_result <- .eblup_unit_cpp(
    selectdom = selectdom,
    dom = dom,
    Xs = Xs,
    meanxpop = meanxpop,
    ys = ys,
    popnsize = popnsize,
    betaest = betaest,
    upred = upred
  )

  eblup_init <- init_result$eblup
  names(eblup_init) <- as.character(selectdom)

  # --- bootstrap loop ---
  if (seed >= 0) set.seed(seed)

  mse <- numeric(length(selectdom))
  n_domain <- length(selectdom)

  # Group indices by domain
  dom_factor <- factor(dom, levels = as.character(selectdom))
  dom_idx <- split(seq_along(dom), dom_factor)

  # Unique domains in sample and their indices
  sample_domains <- names(dom_idx)

  for (b in seq_len(B)) {
    # Generate bootstrap sample
    u_boot <- rnorm(length(selectdom), sd = sqrt(sigma2_u))
    names(u_boot) <- as.character(selectdom)

    e_boot <- rnorm(length(ys), sd = sqrt(sigma2_e))

    # y_boot = X * beta + u_boot[dom] + e_boot
    y_boot <- ys
    for (i in seq_along(ys)) {
      d <- as.character(dom[i])
      y_boot[i] <- as.numeric(Xs[i, ] %*% betaest) + u_boot[d] + e_boot[i]
    }

    # Fit bootstrap model
    fit_boot <- lme4::lmer(formula_lmer, data = data.frame(y_boot, Xs, dom))
    beta_boot <- matrix(lme4::fixef(fit_boot), ncol = 1)
    upred_boot <- lme4::ranef(fit_boot)$dom

    # Compute bootstrap EBLUP
    eblup_boot <- .eblup_unit_cpp(
      selectdom = selectdom,
      dom = dom,
      Xs = Xs,
      meanxpop = meanxpop,
      ys = y_boot,
      popnsize = popnsize,
      betaest = beta_boot,
      upred = upred_boot
    )$eblup

    # Accumulate MSE
    mse <- mse + (eblup_boot - eblup_init)^2
  }

  mse <- mse / B
  names(mse) <- as.character(selectdom)

  return(list(
    eblup = eblup_init,
    mse = mse,
    method = method,
    B = B
  ))
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
    variable <- data[[all.vars(variable)]]
  } else {
    cli::cli_abort('variable "{variable}" is not found in the data')
  }
  return(variable)
}
