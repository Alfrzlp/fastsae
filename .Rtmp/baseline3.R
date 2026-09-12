devtools::load_all(".", quiet = TRUE)
library(sae)
cmp <- function(a,b,tol=1e-5) isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance=tol, check.attributes=FALSE))

gen <- function(n, rho_true=0.4, beta=c(10,2,-1.5,0.8), sigma2u=4, sigma2e=100, seed=1){
  set.seed(seed)
  x1 <- rnorm(n, 50, 10); x2 <- rnorm(n, 20, 5); x3 <- runif(n, 0, 100)
  X <- cbind(1, x1, x2, x3)
  W <- matrix(runif(n*n, 0, 1), n, n); diag(W) <- 0
  W <- W / rowSums(W)
  Bmat <- diag(n) - rho_true * W
  u <- solve(Bmat, rnorm(n, 0, sqrt(sigma2u)))
  theta <- drop(X %*% beta + u)
  ni <- sample(20:500, n, replace = TRUE)
  vardir <- sigma2e / ni
  y <- theta + rnorm(n, 0, sqrt(vardir))
  list(df = data.frame(area=seq_len(n), y=y, x1=x1, x2=x2, x3=x3, vardir=vardir), W = W)
}

ok <- FALSE
for (seed in c(11,22,33,44,55,66)) {
  ds <- gen(120, seed = seed)
  ff <- try(seblup_area(y ~ x1+x2+x3, "vardir", ds$df, W = ds$W, print_result = FALSE), silent=TRUE)
  fs <- try(sae::eblupSFH(ds$df$y ~ ds$df$x1+ds$df$x2+ds$df$x3, ds$df$vardir, ds$W, method="REML"), silent=TRUE)
  if (!inherits(ff,"try-error") && !inherits(fs,"try-error") && isTRUE(ff$convergence) && isTRUE(fs$fit$convergence)) {
    cat("seed", seed, "OK\n")
    ok <- TRUE
    break
  } else cat("seed", seed, "skipped (non-convergence or error)\n")
}
if (!ok) stop("no seed converged for both")

cat("\nbeta fastsae:", ff$estcoef$beta, "\n")
cat("beta sae:    ", fs$fit$estcoef$beta, "\n")
cat("beta match:", cmp(ff$estcoef$beta, fs$fit$estcoef$beta), "\n")
cat("rho fastsae:", ff$rho, " rho sae:", fs$fit$spatialcorr, " match:", cmp(ff$rho, fs$fit$spatialcorr), "\n")
cat("sigma2 fastsae:", ff$sigma2_u, " sigma2 sae:", fs$fit$refvar, " match:", cmp(ff$sigma2_u, fs$fit$refvar), "\n")
cat("eblup match:", cmp(ff$df_eblup$eblup, fs$eblup), "\n")
cat("mse(posterior) match:", cmp(ff$df_eblup$mse, fs$mse), "\n")

saveRDS(list(seed_used = seed, ds = ds, ff = ff, fs = fs), "./.Rtmp/crosscheck_seed.rds")
