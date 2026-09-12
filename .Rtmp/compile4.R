Rcpp::compileAttributes(".")
pkgbuild::compile_dll(".", force = TRUE)
cat("compiled ok\n")
