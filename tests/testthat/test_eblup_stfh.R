library(testthat)
library(fastsae)

skip_if_not_installed("sae")

# ------------------------------------------------------------------
# Shared objects
# ------------------------------------------------------------------

library(dplyr)
mys_panel_nona <- mys_panel |>
  filter(!is.na(y))
mys_proxmat_nona <- mys_proxmat[-c(21, 25), -c(21, 25)]


fit_fast <- eblup_stfh(
  y ~ x1 + x2 + x3,
  data = mys_panel_nona,
  vardir = ~vardir,
  domain = ~area,
  time = ~year,
  W = mys_proxmat_nona,
  model = "ST",
  print_result = FALSE
)

fit_sae <- sae::eblupSTFH(
  mys_panel_nona$y ~ mys_panel_nona$x1 + mys_panel_nona$x2 + mys_panel_nona$x3,
  vardir = mys_panel_nona$vardir,
  D = length(unique(mys_panel_nona$area)),
  T = length(unique(mys_panel_nona$year)),
  proxmat = mys_proxmat_nona
)


tol <- 1e-6


# -----------------------------------------------------------------
# Structure
# ------------------------------------------------------------------

test_that("eblup_stfh returns valid structure matching seblup_area", {
  expect_true(is.list(fit_fast))

  # Top-level fields (matching seblup_area structure)
  expect_true("estcoef" %in% names(fit_fast))
  expect_true("goodness" %in% names(fit_fast))
  expect_true("df_eblup" %in% names(fit_fast))
  expect_true("convergence" %in% names(fit_fast))
  expect_true("n_iter" %in% names(fit_fast))

  # Additional for spatio-temporal
  expect_true("estvarcomp" %in% names(fit_fast))
  expect_true("model" %in% names(fit_fast))
  expect_true("formula" %in% names(fit_fast))

  # df_eblup has eblup column
  expect_true("eblup" %in% names(fit_fast$df_eblup))
  expect_true("random_effect_u1" %in% names(fit_fast$df_eblup))
  expect_true("random_effect_u2" %in% names(fit_fast$df_eblup))

  expect_length(fit_fast$df_eblup$eblup, nrow(mys_panel_nona))
})

# ------------------------------------------------------------------
# EBLUP
# ------------------------------------------------------------------

test_that("EBLUP agrees with sae::eblupSTFH", {
  expect_equal(
    fit_fast$df_eblup$eblup,
    as.numeric(fit_sae$eblup),
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Variance component (estvarcomp - spatio-temporal specific)
# ------------------------------------------------------------------

test_that("estvarcomp agrees with sae::eblupSTFH", {
  expect_equal(
    fit_fast$estvarcomp$estimate,
    as.numeric(fit_sae$fit$estvarcomp$estimate),
    tolerance = tol
  )
  expect_equal(
    fit_fast$estvarcomp$std.error,
    as.numeric(fit_sae$fit$estvarcomp$std.error),
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Goodness of fit
# ------------------------------------------------------------------

test_that("goodness statistics agree with sae::eblupSTFH", {
  expect_equal(
    as.numeric(fit_fast$goodness),
    as.numeric(fit_sae$fit$goodness),
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Regression coefficients
# ------------------------------------------------------------------

test_that("beta estimates agree with sae::eblupSTFH", {
  expect_identical(
    names(fit_fast$estcoef$beta),
    names(fit_sae$fit$estcoef$beta)
  )

  expect_equal(
    fit_fast$estcoef$beta,
    fit_sae$fit$estcoef$beta,
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Beta Standard errors
# ------------------------------------------------------------------

test_that("beta standard errors agree with sae::eblupSTFH", {
  expect_equal(
    fit_fast$estcoef$std.error,
    fit_sae$fit$estcoef$std.error,
    tolerance = tol
  )
})


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

test_that("non-square proximity matrix throws error", {
  W_bad <- mys_proxmat_nona[-1, ]

  expect_error(
    eblup_stfh(
      y ~ x1 + x2 + x3,
      data = mys_panel_nona,
      vardir = ~vardir,
      domain = ~area,
      time = ~year,
      W = W_bad,
      model = "ST"
    )
  )
})

test_that("negative sampling variance throws error", {
  dat_bad <- mys_panel_nona
  dat_bad$vardir[1] <- -1

  expect_error(
    eblup_stfh(
      y ~ x1 + x2 + x3,
      data = dat_bad,
      vardir = ~vardir,
      domain = ~area,
      time = ~year,
      W = mys_proxmat_nona,
      model = "ST"
    )
  )
})
