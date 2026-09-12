library(testthat)
library(fastsae)

skip_if_not_installed("sae")

# ------------------------------------------------------------------
# Shared objects
# ------------------------------------------------------------------

mysnona <- mys[!is.na(mys$y), ]

fit_fast <- eblup_area(
  y ~ x1 + x2 + x3,
  vardir = "vardir",
  method = "REML",
  data = mysnona,
  print_result = FALSE
)

fit_sae <- sae::mseFH(
  mysnona$y ~ mysnona$x1 + mysnona$x2 + mysnona$x3,
  vardir = mysnona$vardir,
  method = "REML"
)

tol <- 1e-5

# ------------------------------------------------------------------
# Structure
# ------------------------------------------------------------------

test_that("eblup_area returns valid structure", {
  expect_true(is.list(fit_fast))

  expect_true("df_eblup" %in% names(fit_fast))
  expect_true("random_effect_var" %in% names(fit_fast))
  expect_true("estcoef" %in% names(fit_fast))

  expect_length(
    fit_fast$df_eblup$eblup,
    nrow(mysnona)
  )

  expect_length(
    fit_fast$df_eblup$mse,
    nrow(mysnona)
  )
})

# ------------------------------------------------------------------
# EBLUP
# ------------------------------------------------------------------

test_that("EBLUP agrees with sae::mseFH", {
  expect_equal(
    fit_fast$df_eblup$eblup,
    as.numeric(fit_sae$est$eblup),
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# MSE
# ------------------------------------------------------------------

test_that("MSE agrees with sae::mseFH", {
  expect_equal(
    fit_fast$df_eblup$mse,
    fit_sae$mse,
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Random effect variance
# ------------------------------------------------------------------

test_that("random effect variance agrees with sae::mseFH", {
  expect_equal(
    fit_fast$random_effect_var,
    fit_sae$est$fit$refvar,
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Goodness of fit
# ------------------------------------------------------------------

test_that("goodness statistics agree with sae::mseFH", {
  expect_equal(
    as.numeric(fit_fast$goodness),
    as.numeric(fit_sae$est$fit$goodness[-4]),
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Regression coefficients and Standard errors
# ------------------------------------------------------------------

test_that("beta estimates agree with sae::mseFH", {
  expect_equal(
    fit_fast$estcoef$beta,
    fit_sae$est$fit$estcoef$beta,
    tolerance = tol
  )

  expect_equal(
    fit_fast$estcoef$stderr_beta,
    fit_sae$est$fit$estcoef$std.error,
    tolerance = tol
  )
})


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------

test_that("negative or zero vardir throws error", {
  dat_bad <- mysnona
  dat_bad$vardir[1] <- -1
  dat_bad$vardir[2] <- 0

  expect_error(
    eblup_area(
      y ~ x1 + x2 + x3,
      vardir = "vardir",
      data = dat_bad,
      print_result = FALSE
    )
  )
})


# ------------------------------------------------------------------
# Method argument
# ------------------------------------------------------------------

test_that("invalid method throws error", {
  expect_error(
    eblup_area(
      y ~ x1 + x2 + x3,
      vardir = "vardir",
      method = "INVALID",
      data = mysnona,
      print_result = FALSE
    )
  )
})
