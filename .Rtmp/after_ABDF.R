devtools::load_all(".", quiet = TRUE)
base <- readRDS("./.Rtmp/baseline2.rds")
cmp <- function(a,b,tol=1e-8) isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance=tol, check.attributes=FALSE))

## ---- B: eblup_unit hash-map vs old nested-loop baseline ----
df_meanpop <- cornsoybeanmeans |> dplyr::rename(CornPix = MeanCornPixPerSeg, SoyBeansPix = MeanSoyBeansPixPerSeg)
df_cornsoybean <- cornsoybean |> dplyr::rename(CountyIndex = County)
res_unit <- eblup_unit(
  formula = CornHec ~ CornPix + SoyBeansPix, Xpop = df_meanpop, unit_data = df_cornsoybean,
  domain_var = "CountyIndex", popsize_var = "PopnSegments"
)
cat("=== B: eblup_unit hash-map vs nested-loop baseline ===\n")
cat("eblup identical:  ", cmp(res_unit$eblup, base$eblup_unit$eblup), "\n")
cat("samp_size identical:", cmp(res_unit$samp_size, base$eblup_unit$samp_size), "\n")
cat("warn_domains identical:", identical(res_unit$warn_domains, base$eblup_unit$warn_domains), "\n")

## ---- A: dead code removed, confirm .seblup_pbmse still resolves to R wrapper ----
cat("\n=== A: dead-code check ===\n")
cat(".eblup_core exists (should be FALSE):", exists(".eblup_core", mode="function"), "\n")
cat(".seblup_core exists (should be FALSE):", exists(".seblup_core", mode="function"), "\n")
cat(".seblup_pbmse2 exists (should be FALSE):", exists(".seblup_pbmse2", mode="function"), "\n")
cat(".seblup_pbmse resolves to R wrapper (formals 'object'):", "object" %in% names(formals(.seblup_pbmse)), "\n")

## ---- D: warm-start bootstrap - regenerate on SAME synthetic n=80 dataset/seed ----
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
ds2 <- generate_spatial_data(80, seed = 909)
fit2 <- seblup_area(y ~ x1+x2+x3, "vardir", ds2$df, W = ds2$W, print_result = FALSE)
cat("\n=== D: warm-start bootstrap ===\n")
cat("base fit beta/rho/sigma2 unchanged: ", cmp(fit2$estcoef$beta, base$fit2_beta), cmp(fit2$rho, base$fit2_rho), cmp(fit2$sigma2_u, base$fit2_sigma2), "\n")

t_pb  <- system.time(pb  <- .seblup_pbmse(fit2, B = 300, seed = 77, n_threads = 1))
t_npb <- system.time(npb <- .seblup_npbmse(fit2, B = 300, seed = 77, n_threads = 1))

cat("PB : successes/failed before=", 300-base$pb_failed, "/", base$pb_failed,
    " after=", 300-pb$B_failed, "/", pb$B_failed, "\n")
cat("NPB: successes/failed before=", 300-base$npb_failed, "/", base$npb_failed,
    " after=", 300-npb$B_failed, "/", npb$B_failed, "\n")
cat("PB time before:", base$pb_time, " after (warm-start):", t_pb[["elapsed"]], "\n")
cat("NPB time before:", base$npb_time, " after (warm-start):", t_npb[["elapsed"]], "\n")

cat("PB mean(mse) before:", mean(base$pb_mse), " after:", mean(pb$mse_pb), "\n")
cat("NPB mean(mse) before:", mean(base$npb_mse), " after:", mean(npb$mse_npb), "\n")
cat("NPB mean(mse_bc) before:", mean(base$npb_bc), " after:", mean(npb$mse_npb_bc), "\n")
cat("PB mse correlation before/after:", cor(base$pb_mse, pb$mse_pb), "\n")
cat("NPB mse correlation before/after:", cor(base$npb_mse, npb$mse_npb), "\n")
cat("NPB mse_bc correlation before/after:", cor(base$npb_bc, npb$mse_npb_bc), "\n")
