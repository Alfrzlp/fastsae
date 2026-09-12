devtools::load_all(".", quiet = TRUE)
library(sae)

cmp <- function(a,b,tol=1e-6) isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance=tol, check.attributes=FALSE))

## ---- 1. cross-check vs sae on package's own real dataset (mys) ----
data(mys); data(mys_proxmat)
mys_s <- subset(mys, !is.na(y))
Wm <- mys_proxmat[!is.na(mys$y), !is.na(mys$y)]

fit_fast <- seblup_area(y ~ x1 + x2 + x3, "vardir", mys_s, W = Wm, print_result = FALSE)
fit_sae  <- sae::eblupSFH(mys_s$y ~ mys_s$x1 + mys_s$x2 + mys_s$x3, mys_s$vardir, Wm, method = "REML")

cat("=== mys real data ===\n")
cat("beta fastsae vs sae: ", cmp(fit_fast$estcoef$beta, fit_sae$fit$estcoef$beta), "\n")
cat("rho fastsae vs sae:  ", cmp(fit_fast$rho, fit_sae$fit$spatialcorr), "\n")
cat("sigma2 fastsae vs sae:", cmp(fit_fast$sigma2_u, fit_sae$fit$refvar), "\n")
cat("eblup fastsae vs sae (sampled): ", cmp(fit_fast$df_eblup$eblup, fit_sae$eblup), "\n")

## ---- 2. synthetic mid-size data ----
generate_spatial_data <- function(n, beta = c(10,2,-1.5,0.8), A=4, sigma2=100, seed=1){
  set.seed(seed)
  x1 <- rnorm(n, 50, 10); x2 <- rnorm(n, 20, 5); x3 <- runif(n, 0, 100)
  X <- cbind(1, x1, x2, x3)
  u <- rnorm(n, 0, sqrt(A))
  theta <- drop(X %*% beta + u)
  ni <- sample(20:500, n, replace = TRUE)
  vardir <- sigma2 / ni
  y <- theta + rnorm(n, 0, sqrt(vardir))
  dat <- data.frame(area=seq_len(n), y=y, x1=x1, x2=x2, x3=x3, vardir=vardir)
  W <- matrix(runif(n*n, 0, 1), n, n)
  W_std <- W / rowSums(W)
  list(df=dat, W=W_std)
}
ds <- generate_spatial_data(200, seed = 55)
ff <- seblup_area(y ~ x1+x2+x3, "vardir", ds$df, W = ds$W, print_result = FALSE)
fs <- sae::eblupSFH(ds$df$y ~ ds$df$x1+ds$df$x2+ds$df$x3, ds$df$vardir, ds$W, method="REML")
cat("\n=== synthetic n=200 ===\n")
cat("beta fastsae vs sae: ", cmp(ff$estcoef$beta, fs$fit$estcoef$beta), "\n")
cat("rho fastsae vs sae:  ", cmp(ff$rho, fs$fit$spatialcorr), "\n")
cat("sigma2 fastsae vs sae:", cmp(ff$sigma2_u, fs$fit$refvar), "\n")
cat("eblup fastsae vs sae:", cmp(ff$df_eblup$eblup, fs$eblup), "\n")
cat("mse(posterior) fastsae vs sae:", cmp(ff$df_eblup$mse, fs$mse), "\n")

## ---- 3. eblup_unit old-logic baseline (cornsoybean) ----
df_meanpop <- cornsoybeanmeans |> dplyr::rename(CornPix = MeanCornPixPerSeg, SoyBeansPix = MeanSoyBeansPixPerSeg)
df_cornsoybean <- cornsoybean |> dplyr::rename(CountyIndex = County)
res_unit <- eblup_unit(
  formula = CornHec ~ CornPix + SoyBeansPix, Xpop = df_meanpop, unit_data = df_cornsoybean,
  domain_var = "CountyIndex", popsize_var = "PopnSegments"
)

## ---- 4. PB/NPB baseline (pre warm-start) on synthetic n=80 ----
ds2 <- generate_spatial_data(80, seed = 909)
fit2 <- seblup_area(y ~ x1+x2+x3, "vardir", ds2$df, W = ds2$W, print_result = FALSE)
t_pb  <- system.time(pb  <- .seblup_pbmse(fit2, B = 300, seed = 77, n_threads = 1))
t_npb <- system.time(npb <- .seblup_npbmse(fit2, B = 300, seed = 77, n_threads = 1))

saveRDS(list(
  eblup_unit = res_unit,
  pb_mse = pb$mse_pb, pb_failed = pb$B_failed, pb_time = t_pb[["elapsed"]],
  npb_mse = npb$mse_npb, npb_bc = npb$mse_npb_bc, npb_failed = npb$B_failed, npb_time = t_npb[["elapsed"]],
  fit2_beta = fit2$estcoef$beta, fit2_rho = fit2$rho, fit2_sigma2 = fit2$sigma2_u
), "./.Rtmp/baseline2.rds")
cat("\nAll baseline captured OK.\n")
