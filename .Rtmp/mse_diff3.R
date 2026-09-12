library(sae)
r <- readRDS("./.Rtmp/crosscheck_seed.rds")
ff <- r$ff; ds <- r$ds
fs_mse <- sae::mseSFH(ds$df$y ~ ds$df$x1+ds$df$x2+ds$df$x3, ds$df$vardir, ds$W, method="REML")
mse_sae <- fs_mse$mse
d <- (ff$df_eblup$mse - mse_sae) / mse_sae
cat("relative mse diff summary:\n"); print(summary(d))
cat("max abs mse diff:", max(abs(ff$df_eblup$mse - mse_sae)), "\n")
cat("mean mse fastsae:", mean(ff$df_eblup$mse), " sae:", mean(mse_sae), "\n")
