r <- readRDS("./.Rtmp/crosscheck_seed.rds")
ff <- r$ff; fs <- r$fs
str(fs$mse)
str(ff$df_eblup$mse)
mse_sae <- fs$mse[[1]]
d <- (ff$df_eblup$mse - mse_sae) / mse_sae
cat("relative mse diff summary:\n"); print(summary(d))
cat("max abs mse diff:", max(abs(ff$df_eblup$mse - mse_sae)), "\n")
