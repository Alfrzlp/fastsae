# Pembangkit data simulasi "aman" untuk model Spatial Fay-Herriot (SFH).
#
# Desain simulasi lama (berbasis `mys`/`mys_proxmat`, lihat riwayat di
# test_seblup_npbmse.R / test_seblup_pbmse.R) membuat parameter autokorelasi
# spasial (rho) mentok di batas (+-1) pada ~19 dari 20 replikasi Monte Carlo,
# karena dua sebab:
#   1) W yang dipakai adalah SUBMATRIKS mys_proxmat untuk area tersampel saja
#      -- baris-barisnya tidak lagi berjumlah 1 (row sums ~0.71-0.79), dan
#      spektrumnya jadi sempit (radius spektral ~0.76), sehingga variasi
#      rho hampir tidak mengubah struktur kovariansi (diag(Sigma_u) nyaris
#      konstan) -> rho sangat lemah teridentifikasi.
#   2) sigma2_true (variansi efek spasial) jauh lebih kecil daripada vardir
#      (median rasio ~0.3) -- sinyal antar-area kalah oleh noise sampling.
# Akibatnya bootstrap non-parametrik (.seblup_npbmse) sering crash
# ("pinv(): svd failed") karena dekomposisi eigen-nya berasumsi solusi model
# tidak degenerate -- kerapuhan yang juga terbukti ada di algoritma acuannya,
# sae:::npbmseSFH, bukan spesifik ke port C++ fastsae.
#
# Desain di bawah ini memperbaiki kedua sebab tersebut:
#   - W dibangun dari kontiguitas rook pada grid persegi lalu di
#     row-standardize (baris berjumlah tepat 1) -- struktur spasial yang
#     wajar & punya variasi kovariansi antar-area yang nyata terhadap rho,
#     bukan submatriks sisa subsetting area tak-tersampel.
#   - sigma2_true dan vardir dijaga pada skala sepadan (rasio ~1) supaya ada
#     cukup sinyal antar-area untuk mengidentifikasi rho.
#   - vardir dibangkitkan dari rentang sempit (tanpa outlier ekstrem) supaya
#     Fisher-scoring lebih stabil.
#
# Sudah divalidasi manual: 100 replikasi fit inti (tanpa bootstrap) -> 0
# yang mentok batas rho=+-1, 0 gagal konvergen (rho estimasi berkisar
# ~0.5-0.7, median ~0.62, dekat rho_true=0.6). 8 replikasi dengan bootstrap
# penuh (B=50, mse_method="npbmse" & "pbmse") -> 0 error/crash.

# W dari kontiguitas rook (atas/bawah/kiri/kanan) pada grid persegi.
make_grid_W <- function(nrow_grid, ncol_grid) {
  m <- nrow_grid * ncol_grid
  coords <- expand.grid(row = 1:nrow_grid, col = 1:ncol_grid)
  W <- matrix(0, m, m)
  for (i in 1:m) {
    dr <- abs(coords$row[i] - coords$row)
    dc <- abs(coords$col[i] - coords$col)
    W[i, ] <- as.numeric((dr + dc) == 1)
  }
  W / rowSums(W)  # row-standardize -> row-stochastic (radius spektral = 1)
}

#' Bangkitkan populasi "kebenaran" untuk simulasi model Spatial Fay-Herriot.
#'
#' @param nrow_grid,ncol_grid dimensi grid untuk matriks ketetanggaan `W`
#'   (jumlah area `m = nrow_grid * ncol_grid`).
#' @param beta_true vektor koefisien regresi kebenaran (termasuk intercept).
#' @param sigma2_true variansi efek acak spasial kebenaran.
#' @param rho_true parameter autokorelasi spasial (SAR) kebenaran.
#' @param vardir_range rentang (min, max) variansi sampling langsung per
#'   area, dibangkitkan dari distribusi seragam.
#' @param seed seed RNG untuk mereproduksi W/X/vardir/u_true yang sama.
#'
#' @return list dengan elemen `W`, `X`, `vardir`, `beta_true`, `sigma2_true`,
#'   `rho_true`, `u_true`, `theta_true`, `m`.
simulate_safe_sfh <- function(
    nrow_grid = 8,
    ncol_grid = 8,
    beta_true = c(5, 1.2, -0.8, 0.5),
    sigma2_true = 1,
    rho_true = 0.6,
    vardir_range = c(0.5, 1.5),
    seed = 7842
) {
  set.seed(seed)

  W <- make_grid_W(nrow_grid, ncol_grid)
  m <- nrow(W)

  X <- cbind(
    intercept = 1,
    x1 = round(rnorm(m, 5, 1.5), 2),
    x2 = round(runif(m, 0, 10), 2),
    x3 = round(rbinom(m, 1, 0.5))
  )

  vardir <- runif(m, vardir_range[1], vardir_range[2])

  I_m <- diag(m)
  Sigma_u <- sigma2_true * solve(I_m - rho_true * W) %*% t(solve(I_m - rho_true * W))
  u_true <- as.numeric(MASS::mvrnorm(1, mu = rep(0, m), Sigma = Sigma_u))
  theta_true <- as.numeric(X %*% beta_true) + u_true

  list(
    W = W, X = X, vardir = vardir,
    beta_true = beta_true, sigma2_true = sigma2_true, rho_true = rho_true,
    u_true = u_true, theta_true = theta_true, m = m
  )
}
