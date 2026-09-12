devtools::load_all(".", quiet = TRUE)
library(sae)
data(mys); data(mys_proxmat)
mys_s <- subset(mys, !is.na(y))
Wm <- mys_proxmat[!is.na(mys$y), !is.na(mys$y)]
cat("Wm dim:", dim(Wm), " rowSums range:", range(rowSums(Wm)), "\n")
cat("Wm spectral norm (2-norm):", norm(as.matrix(Wm), "2"), "\n")

fit_fast <- seblup_area(y ~ x1 + x2 + x3, "vardir", mys_s, W = Wm, print_result = FALSE)
fit_sae  <- sae::eblupSFH(mys_s$y ~ mys_s$x1 + mys_s$x2 + mys_s$x3, mys_s$vardir, Wm, method = "REML")

cat("\nbeta fastsae:", fit_fast$estcoef$beta, "\n")
cat("beta sae:    ", fit_sae$fit$estcoef$beta, "\n")
cat("rho fastsae:", fit_fast$rho, " rho_bound:", fit_fast$rho_bound, "\n")
cat("rho sae:    ", fit_sae$fit$spatialcorr, "\n")
cat("sigma2 fastsae:", fit_fast$sigma2_u, "  sigma2 sae:", fit_sae$fit$refvar, "\n")
cat("convergence fastsae:", fit_fast$convergence, " n_iter:", fit_fast$n_iter, "\n")
cat("convergence sae:", fit_sae$fit$convergence, "\n")
cat("loglik fastsae:", fit_fast$goodness[["loglikelihood"]], " loglik sae:", fit_sae$fit$goodness[["loglikelihood"]], "\n")
