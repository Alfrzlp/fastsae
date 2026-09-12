library(testthat)
library(fastsae)
library(dplyr)
library(sae)

skip_if_not_installed("sae")

# ============================================================================
# Helper
# ============================================================================
quiet <- function(expr) {
  result <- NULL
  invisible(capture.output(result <- suppressWarnings(suppressMessages(expr))))
  result
}

# ============================================================================
# Load data
# ============================================================================
data(cornsoybean, package = "sae")
data(cornsoybeanmeans, package = "sae")

# Attach formula variables to environment
CornHec <- cornsoybean$CornHec
CornPix <- cornsoybean$CornPix
SoyBeansPix <- cornsoybean$SoyBeansPix

# Prepare data for fastsae
df_cornsoybean <- cornsoybean
df_cornsoybean$CountyIndex <- df_cornsoybean$County
df_cornsoybean$County <- NULL

# Prepare Xmean from cornsoybeanmeans
Xmean <- cornsoybeanmeans[, c("CountyIndex", "MeanCornPixPerSeg", "MeanSoyBeansPixPerSeg")]
names(Xmean) <- c("CountyIndex", "CornPix", "SoyBeansPix")

# Prepare population sizes
Popn <- cornsoybeanmeans[, c("CountyIndex", "PopnSegments")]

# Merge Xmean with Popn
Xpop <- merge(Xmean, Popn, by = "CountyIndex")

# ============================================================================
# Fit using fastsae
# ============================================================================
fit_fast <- eblup_bhf(
  formula = CornHec ~ CornPix + SoyBeansPix,
  unit_data = df_cornsoybean,
  Xpop = Xpop,
  domain_var = "CountyIndex",
  popsize_var = "PopnSegments",
  method = "REML",
  print_result = FALSE
)

# ============================================================================
# Fit using sae package
# ============================================================================
# Prepare meanxpop and popnsize for sae
Xmean_sae <- data.frame(
  CountyIndex = cornsoybeanmeans$CountyIndex,
  MeanCornPixPerSeg = cornsoybeanmeans$MeanCornPixPerSeg,
  MeanSoyBeansPixPerSeg = cornsoybeanmeans$MeanSoyBeansPixPerSeg
)
Popn_sae <- data.frame(
  CountyIndex = cornsoybeanmeans$CountyIndex,
  PopnSegments = cornsoybeanmeans$PopnSegments
)

fit_sae <- quiet(sae::eblupBHF(
  CornHec ~ CornPix + SoyBeansPix,
  dom = cornsoybean$County,
  selectdom = unique(cornsoybean$County),
  meanxpop = Xmean_sae,
  popnsize = Popn_sae,
  method = "REML"
))

tol <- 1e-4

# ============================================================================
# Tests: EBLUP
# ============================================================================
test_that("eblup_bhf returns valid structure", {
  expect_s3_class(fit_fast, "fastsae_unit")
  expect_true("eblup" %in% names(fit_fast))
  expect_true("fit" %in% names(fit_fast))
  expect_true("sigma2_u" %in% names(fit_fast$fit))
  expect_true("sigma2_e" %in% names(fit_fast$fit))
})

test_that("EBLUP agrees with sae::eblupBHF", {
  # Order by domain
  fast_eblup <- fit_fast$eblup[order(fit_fast$eblup$domain), ]
  sae_eblup <- fit_sae$eblup[order(fit_sae$eblup$domain), ]

  expect_equal(
    fast_eblup$eblup,
    as.numeric(sae_eblup$eblup),
    tolerance = tol
  )
})

test_that("Variance components agree with sae::eblupBHF", {
  expect_equal(
    fit_fast$fit$sigma2_u,
    fit_sae$fit$refvar,
    tolerance = tol
  )
})

test_that("Fixed effects agree with sae::eblupBHF", {
  fast_beta <- as.numeric(fit_fast$fit$beta)
  sae_beta <- as.numeric(fit_sae$fit$fixed)

  expect_equal(
    fast_beta,
    sae_beta,
    tolerance = tol
  )
})

# ============================================================================
# Tests: Bootstrap MSE
# ============================================================================
test_that("pbmse_unit returns valid structure", {
  skip_on_cran()

  B_test <- 50
  mse_fast <- pbmse_unit(
    formula = CornHec ~ CornPix + SoyBeansPix,
    unit_data = df_cornsoybean,
    Xpop = Xpop,
    domain_var = "CountyIndex",
    popsize_var = "PopnSegments",
    method = "REML",
    B = B_test,
    seed = 123
  )

  expect_type(mse_fast$mse, "double")
  expect_length(mse_fast$mse, length(unique(df_cornsoybean$CountyIndex)))
  expect_true(all(mse_fast$mse >= 0))
})

test_that("pbmse_unit produces valid MSE estimates", {
  skip_on_cran()

  B_test <- 50
  set.seed(123)

  mse_result <- pbmse_unit(
    formula = CornHec ~ CornPix + SoyBeansPix,
    unit_data = df_cornsoybean,
    Xpop = Xpop,
    domain_var = "CountyIndex",
    popsize_var = "PopnSegments",
    method = "REML",
    B = B_test,
    seed = 123
  )

  expect_type(mse_result$mse, "double")
  expect_length(mse_result$mse, 12)
  expect_true(all(mse_result$mse > 0))

  # MSE should be in reasonable range (based on response scale)
  expect_true(all(mse_result$mse < 1000))
})

# ============================================================================
# Tests: Error handling
# ============================================================================
test_that("Invalid method throws error", {
  expect_error(
    eblup_bhf(
      formula = CornHec ~ CornPix + SoyBeansPix,
      unit_data = df_cornsoybean,
      Xpop = Xpop,
      domain_var = "CountyIndex",
      popsize_var = "PopnSegments",
      method = "INVALID"
    )
  )
})
