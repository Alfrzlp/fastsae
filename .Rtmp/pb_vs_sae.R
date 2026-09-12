devtools::load_all(".", quiet = TRUE)
library(sae)
r <- readRDS("./.Rtmp/crosscheck_seed.rds")
ff <- r$ff; ds <- r$ds
fs_mse <- sae::mseSFH(ds$df$y ~ ds$df$x1+ds$df$x2+ds$df$x3, ds$df$vardir, ds$W, method="REML")$mse

pb  <- .seblup_pbmse(ff, B = 500, seed = 5, n_threads = 1)
npb <- .seblup_npbmse(ff, B = 500, seed = 5, n_threads = 1)

cat("mean mse: fastsae-posterior=", mean(ff$df_eblup$mse),
    " fastsae-PB=", mean(pb$mse_pb),
    " fastsae-NPB(bc)=", mean(npb$mse_npb_bc),
    " sae-analytic=", mean(fs_mse), "\n")
cat("correlation posterior vs sae:", cor(ff$df_eblup$mse, fs_mse), "\n")
cat("correlation PB vs sae:", cor(pb$mse_pb, fs_mse), "\n")
cat("correlation NPB-bc vs sae:", cor(npb$mse_npb_bc, fs_mse), "\n")
cat("mean relative diff PB vs sae:", mean((pb$mse_pb - fs_mse)/fs_mse), "\n")
cat("mean relative diff NPB-bc vs sae:", mean((npb$mse_npb_bc - fs_mse)/fs_mse), "\n")
