library(emdi)
library(MASS) # untuk mvrnorm jika diperlukan


# helper ------------------------------------------------------------------
quiet <- function(expr) {
  result <- NULL
  # Tangkap output ke memori (bukan file/nullfile()) -- beberapa lingkungan
  # sandbox memblokir penulisan ke /dev/null (nullfile() di Unix), jadi ini
  # lebih portabel sekaligus tetap cross-platform.
  invisible(capture.output(result <- suppressWarnings(suppressMessages(expr))))
  result
}


# ------------------------- 1. Desain simulasi -------------------------------
# Populasi "aman" (lihat helper-simulate_safe_sfh.R): W dari kontiguitas
# grid row-standardized & rasio sinyal/noise dijaga supaya rho tidak mentok
# batas +-1 di hampir semua replikasi (beda dari desain lama berbasis
# mys/mys_proxmat, yang mentok batas di ~19 dari 20 replikasi).
pop <- simulate_safe_sfh()
m           <- pop$m
W           <- pop$W
X           <- pop$X
beta_true   <- pop$beta_true
sigma2_true <- pop$sigma2_true
rho_true    <- pop$rho_true
vardir      <- pop$vardir
u_true      <- pop$u_true
theta_true  <- pop$theta_true

# ------------------------- 2. Pengaturan simulasi ---------------------------
nsim <- 20
B <- 50

# Struktur penyimpanan hasil: array [replikasi, area]
eblup_fast <- eblup_emdi <- matrix(NA_real_, nsim, m)
mse_fast <- mse_emdi <- matrix(NA_real_, nsim, m)




# ------------------------- 3. Loop simulasi Monte Carlo ---------------------
for (s in seq_len(nsim)) {
  # message(sprintf("Replikasi %d / %d", s, nsim))
  e_s <- rnorm(m, mean = 0, sd = sqrt(vardir))
  y_s <- theta_true + e_s

  dat_s <- data.frame(
    y      = y_s,
    x1     = X[, "x1"],
    x2     = X[, "x2"],
    x3     = X[, "x3"],
    vardir = vardir
  )


  seed_s <- 1000 + s

  set.seed(seed_s)
  fit_fast_s <- tryCatch(
    eblup_sfh(
      y ~ x1 + x2 + x3,
      vardir = "vardir",
      method = "REML",
      data = dat_s,
      W = W,
      mse_method = "pbmse",
      print_result = FALSE,
      B = B
    ),
    error = function(e) NULL
  )

  set.seed(seed_s)
  fit_emdi_s <- tryCatch(
    quiet(
      emdi::fh(
        y ~ x1 + x2 + x3,
        vardir = "vardir",
        method = "reml",
        combined_data = dat_s,
        correlation = "spatial",
        corMatrix = W,
        MSE = TRUE,
        mse_type = "spatialparbootbc",
        B = B
      )
    ),
    error = function(e) NULL
  )

  # simpan hasil
  if (!is.null(fit_fast_s)) {
    eblup_fast[s, ] <- fit_fast_s$df_eblup$eblup
    mse_fast[s, ] <- fit_fast_s$df_eblup$mse_pbbc
  }

  if (!is.null(fit_emdi_s)) {
    eblup_emdi[s, ] <- fit_emdi_s$ind$FH
    mse_emdi[s, ] <- fit_emdi_s$MSE$FH
  }
}




# ------------------------- 4. True MSE empiris per area ---------------------
true_mse <- function(eblup_mat, theta_true) {
  colMeans((sweep(eblup_mat, 2, theta_true, "-"))^2, na.rm = TRUE)
}

true_mse_fast <- true_mse(eblup_fast, theta_true)
true_mse_emdi <- true_mse(eblup_emdi, theta_true)


test_that("Parametric Bootstrap MSE agrees with emdi::fh", {
  expect_equal(
    true_mse_fast,
    true_mse_emdi,
    tolerance = 1e-6
  )
})


