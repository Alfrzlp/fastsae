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


fit_fast <- steblup_area(
  y ~ x1 + x2 + x3,
  data = mys_panel_nona,
  vardir = ~vardir,
  D = length(unique(mys_panel_nona$area)),
  Time = length(unique(mys_panel_nona$year)),
  W = mys_proxmat_nona,
  model = "ST"
)

fit_sae <- sae::eblupSTFH(
  mys_panel_nona$y ~ mys_panel_nona$x1 + mys_panel_nona$x2 + mys_panel_nona$x3,
  vardir = mys_panel_nona$vardir,
  D = length(unique(mys_panel_nona$area)),
  T= length(unique(mys_panel_nona$year)),
  proxmat = mys_proxmat_nona
)


tol <- 1e-6


# -----------------------------------------------------------------
# Structure
# ------------------------------------------------------------------
#
# test_that("seblup_area returns valid structure", {
#   expect_true(is.list(fit_fast))
#
#   expect_true("df_eblup" %in% names(fit_fast))
#   expect_true("sigma2_u" %in% names(fit_fast))
#   expect_true("estcoef" %in% names(fit_fast))
#
#   expect_length(
#     fit_fast$df_eblup$eblup,
#     nrow(mysnona)
#   )
#
#   expect_length(
#     fit_fast$df_eblup$mse,
#     nrow(mysnona)
#   )
# })

# ------------------------------------------------------------------
# EBLUP
# ------------------------------------------------------------------

test_that("EBLUP agrees with sae::eblupSTFH", {
  expect_equal(
    fit_fast$eblup,
    as.numeric(fit_sae$eblup),
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# MSE
# ------------------------------------------------------------------

# test_that("MSE agrees with sae::mseSFH", {
#   expect_equal(
#     fit_fast$fit$estcoef,
#     fit_sae$mse,
#     tolerance = tol
#   )
# })

# ------------------------------------------------------------------
# Variance component
# ------------------------------------------------------------------

# test_that("random effect variance agrees with sae::mseSFH", {
#   expect_equal(
#     fit_fast$sigma2_u,
#     fit_sae$est$fit$refvar,
#     tolerance = tol
#   )
# })

# ------------------------------------------------------------------
# Goodness of fit
# ------------------------------------------------------------------

test_that("goodness statistics agree with sae::eblupSTFH", {
  expect_equal(
    as.numeric(fit_fast$fit$goodness),
    as.numeric(fit_sae$fit$goodness),
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Regression coefficients
# ------------------------------------------------------------------

test_that("beta estimates agree with sae::eblupSTFH", {
  expect_identical(
    names(fit_fast$fit$estcoef$beta),
    names(fit_sae$fit$estcoef$beta)
  )

  expect_equal(
    fit_fast$fit$estcoef$beta,
    fit_sae$fit$estcoef$beta,
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Standard errors
# ------------------------------------------------------------------

test_that("beta standard errors agree with sae::eblupSTFH", {
  expect_equal(
    fit_fast$fit$estcoef$std.error,
    fit_sae$fit$estcoef$std.error,
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

