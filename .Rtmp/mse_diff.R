r <- readRDS("./.Rtmp/crosscheck_seed.rds")
ff <- r$ff; fs <- r$fs
d_beta <- abs(ff$estcoef$beta - fs$fit$estcoef$beta) / abs(fs$fit$estcoef$beta)
cat("relative beta diff:", d_beta, "\n")
cat("relative rho diff:", abs(ff$rho - fs$fit$spatialcorr)/abs(fs$fit$spatialcorr), "\n")
cat("n_iter fastsae:", ff$n_iter, "\n")
d_mse <- (ff$df_eblup$mse - fs$mse) / fs$mse
cat("mse relative diff summary:\n"); print(summary(d_mse))
cat("max abs mse diff:", max(abs(ff$df_eblup$mse - fs$mse)), "\n")
cat("mean(mse) fastsae:", mean(ff$df_eblup$mse), " sae:", mean(fs$mse), "\n")
