devtools::load_all(".", quiet = TRUE)
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
t_pb  <- system.time(pb  <- .seblup_pbmse(fit2, B = 300, seed = 77, n_threads = 1))
t_npb <- system.time(npb <- .seblup_npbmse(fit2, B = 300, seed = 77, n_threads = 1))
saveRDS(list(
  pb_mse = pb$mse_pb, pb_failed = pb$B_failed, pb_time = t_pb[["elapsed"]],
  npb_mse = npb$mse_npb, npb_bc = npb$mse_npb_bc, npb_failed = npb$B_failed, npb_time = t_npb[["elapsed"]],
  fit2_beta = fit2$estcoef$beta, fit2_rho = fit2$rho, fit2_sigma2 = fit2$sigma2_u
), "./.Rtmp/coldstart.rds")
cat("coldstart captured. pb_time=", t_pb[["elapsed"]], " npb_time=", t_npb[["elapsed"]],
    " pb_failed=", pb$B_failed, " npb_failed=", npb$B_failed, "\n")
