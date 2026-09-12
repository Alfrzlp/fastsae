library(testthat)
library(fastsae)

skip_if_not_installed("sae")

# ------------------------------------------------------------------
# Shared objects
# ------------------------------------------------------------------

idx <- !is.na(mys$y)

mysnona <- mys[idx, ]
mys_proxmat_nona <- mys_proxmat[idx, idx]

fit_fast <- seblup_area(
  y ~ x1 + x2 + x3,
  vardir = "vardir",
  method = "REML",
  data = mysnona,
  W = mys_proxmat_nona,
  print_result = FALSE
)

fit_sae <- sae::mseSFH(
  mysnona$y ~ mysnona$x1 + mysnona$x2 + mysnona$x3,
  vardir = mysnona$vardir,
  proxmat = mys_proxmat_nona,
  method = "REML"
)

tol <- 1e-6


# -----------------------------------------------------------------
# Structure
# ------------------------------------------------------------------

test_that("seblup_area returns valid structure", {
  expect_true(is.list(fit_fast))

  expect_true("df_eblup" %in% names(fit_fast))
  expect_true("sigma2_u" %in% names(fit_fast))
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

test_that("EBLUP agrees with sae::mseSFH", {
  expect_equal(
    fit_fast$df_eblup$eblup,
    as.numeric(fit_sae$est$eblup),
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# MSE
# ------------------------------------------------------------------

test_that("MSE agrees with sae::mseSFH", {
  expect_equal(
    fit_fast$df_eblup$mse,
    fit_sae$mse,
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Variance component
# ------------------------------------------------------------------

test_that("random effect variance agrees with sae::mseSFH", {
  expect_equal(
    fit_fast$sigma2_u,
    fit_sae$est$fit$refvar,
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Goodness of fit
# ------------------------------------------------------------------

test_that("goodness statistics agree with sae::mseSFH", {
  expect_equal(
    as.numeric(fit_fast$goodness),
    as.numeric(fit_sae$est$fit$goodness[-4]),
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Regression coefficients
# ------------------------------------------------------------------

test_that("beta estimates agree with sae::mseSFH", {
  expect_identical(
    names(fit_fast$estcoef$beta),
    names(fit_sae$est$fit$estcoef$beta)
  )

  expect_equal(
    fit_fast$estcoef$beta,
    fit_sae$est$fit$estcoef$beta,
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Standard errors
# ------------------------------------------------------------------

test_that("beta standard errors agree with sae::mseSFH", {
  expect_equal(
    fit_fast$estcoef$stderr_beta,
    fit_sae$est$fit$estcoef$std.error,
    tolerance = tol
  )
})

# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

test_that("non-square proximity matrix throws error", {
  W_bad <- mys_proxmat_nona[-1, ]

  expect_error(
    seblup_area(
      y ~ x1 + x2 + x3,
      vardir = "vardir",
      data = mysnona,
      W = W_bad,
      print_result = FALSE
    )
  )
})

test_that("negative sampling variance throws error", {
  dat_bad <- mysnona
  dat_bad$vardir[1] <- -1

  expect_error(
    seblup_area(
      y ~ x1 + x2 + x3,
      vardir = "vardir",
      data = dat_bad,
      W = mys_proxmat_nona,
      print_result = FALSE
    )
  )
})
