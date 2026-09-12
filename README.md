
# fastsae

<!-- badges: start -->

[![R-CMD-check](https://github.com/ridsonap/fastsae/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/ridsonap/fastsae/actions/workflows/R-CMD-check.yaml)

<!-- badges: end -->

## Overview

**fastsae** provides fast implementations of small area estimation (SAE)
methods using C++ (RcppArmadillo) for computational efficiency. It is
designed for large-scale applications where standard R implementations
are too slow.

### Supported Models

| Model | Description | MSE Estimation |
|----|----|----|
| Fay-Herriot (Area-level) | `eblup_fh()` | Analytical |
| Spatial Fay-Herriot | `seblup_fh()` | Analytical, Parametric Bootstrap MSE, Non Parametric Bootstrap MSE (Bias Corrected & Non Bias Corrected) |
| Battese-Harter-Fuller (Unit-level) | `eblup_unit()` | Parametric Bootstrap MSE |

## Installation

``` r
# Install from GitHub
install.packages("remotes")
remotes::install_github("ridsonap/fastsae")

# Or from CRAN
install.packages("fastsae")
```

### Dependencies

**Imports:** cli, dplyr, lme4, Rcpp, RcppArmadillo, RcppParallel,
methods

## Quick Start

### Fay-Herriot Model

``` r
library(fastsae)

# Using sampled areas only
mys_sampled <- mys[!is.na(mys$y), ]
m1 <- eblup_fh(
  y ~ x1 + x2 + x3,
  data = mys_sampled,
  vardir = "vardir",
  method = "REML"
)
```

### Spatial Fay-Herriot Model

``` r
# Using sampled areas only
mys_sampled <- mys[!is.na(mys$y), ]
W_sampled <- mys_proxmat[!is.na(mys$y), !is.na(mys$y)]

m2 <- seblup_fh(
  y ~ x1 + x2 + x3,
  data = mys_sampled,
  vardir = "vardir",
  W = W_sampled,
  method = "REML"
)
```

#### Bootstrap MSE

``` r
# Parametric Bootstrap MSE
m_pb <- seblup_fh(
  y ~ x1 + x2 + x3,
  data = mys_sampled,
  vardir = "vardir",
  W = W_sampled,
  mse_method = "pbmse",
  B = 100,
  seed = 123
)

# Nonparametric Bootstrap MSE
m_npb <- seblup_fh(
  y ~ x1 + x2 + x3,
  data = mys_sampled,
  vardir = "vardir",
  W = W_sampled,
  mse_method = "npbmse",
  B = 100,
  seed = 123
)
```

#### Parallel Computation

fastsae supports multi-threaded bootstrap MSE estimation via OpenMP:

``` r
# Use multiple threads for bootstrap
m_parallel <- seblup_fh(
  y ~ x1 + x2 + x3,
  data = mys_sampled,
  vardir = "vardir",
  W = W_sampled,
  mse_method = "pbmse",
  B = 200,
  n_threads = 4  # Use 4 threads
)
```

### Battese-Harter-Fuller Model (Unit-Level)

``` r
library(dplyr)

# Prepare data
df_meanpop <- cornsoybeanmeans |>
  rename(CornPix = MeanCornPixPerSeg, SoyBeansPix = MeanSoyBeansPixPerSeg)
df_cornsoybean <- cornsoybean |>
  rename(CountyIndex = County)

m3 <- eblup_unit(
  formula = CornHec ~ CornPix + SoyBeansPix,
  unit_data = df_cornsoybean,
  Xpop = df_meanpop,
  domain_var = "CountyIndex",
  popsize_var = "PopnSegments"
)
```

## Output Structure

All fitting functions return a list with:

| Component           | Description                                       |
|---------------------|---------------------------------------------------|
| `df_eblup`          | Data frame with eblup, mse, rse estimates         |
| `estcoef`           | Regression coefficients with SE, z-value, p-value |
| `random_effect_var` | Estimated variance of random effects              |
| `goodness`          | Log-likelihood, AIC, BIC                          |
| `n_iter`            | Number of iterations                              |
| `convergence`       | Logical: did algorithm converge?                  |

## Benchmark

fastsae is benchmarked against **sae** and **emdi** packages across
sample sizes ranging from n=30 to n=1000 with 5 covariates.

### Performance Summary

| Metric                   | fastsae  | sae     | emdi   |
|--------------------------|----------|---------|--------|
| **Mean Time (EBLUP)**    | 0.0015s  | 0.291s  | 9.64s  |
| **Mean Time (SEBLUP)**   | 0.165s   | 12.6s   | 8.69s  |
| **Mean Memory (EBLUP)**  | 0.055 MB | 16.3 MB | 824 MB |
| **Mean Memory (SEBLUP)** | 5.15 MB  | 408 MB  | 824 MB |

------------------------------------------------------------------------

### Execution Time Comparison

![](README_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

The execution time chart shows how long each package takes to complete
estimation across different sample sizes. The Y-axis shows median
execution time in seconds on a logarithmic scale where each tick
represents a 10x increase.

### Memory Usage Comparison

![](README_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

The memory usage chart displays peak RAM consumption during estimation.
Memory is measured in megabytes on a logarithmic Y-axis, allowing
comparison between packages that use vastly different amounts. fastsae
maintains minimal memory usage of approximately 0.02 MB for EBLUP and 23
MB for SEBLUP even at n=1000.

------------------------------------------------------------------------

### Speedup Factor

![](README_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

The speedup chart illustrates the performance advantage of fastsae over
competitors by showing how many times faster it executes. The dashed
horizontal line at y=1 represents equal performance; points above this
line indicate fastsae is faster. For EBLUP with n=1000, fastsae achieves
a 12,343x speedup over emdi and a 34x speedup over sae.

------------------------------------------------------------------------

### Key Findings

fastsae demonstrates substantial performance advantages over both sae
and emdi packages across all tested configurations. The speedup factor
ranges from 19x at small sample sizes (n=30) to 12,343x at large sample
sizes (n=1000) when compared to emdi. Memory consumption follows a
similar pattern, with fastsae using 100 to 20,000 times less RAM
depending on sample size and algorithm type. The EBLUP algorithm shows
the most dramatic improvements, while SEBLUP maintains consistent
advantages of 11-81x over emdi.

------------------------------------------------------------------------

### Detailed Benchmark Results

**EBLUP (p = 5)**

| Method  |  Min (s) | Median (s) | Iterations/sec | Memory (MB) |    n |
|:--------|---------:|-----------:|---------------:|------------:|-----:|
| fastsae |  0.00050 |    0.00059 |     1622.53774 |     0.00507 |   30 |
| emdi    |  0.01060 |    0.01088 |       91.19345 |     4.18307 |   30 |
| sae     |  0.00152 |    0.00155 |      631.90597 |     0.15338 |   30 |
| fastsae |  0.00050 |    0.00051 |     1912.06792 |     0.00884 |   50 |
| emdi    |  0.01678 |    0.01806 |       51.94885 |    10.46759 |   50 |
| sae     |  0.00188 |    0.00195 |      494.42923 |     0.37314 |   50 |
| fastsae |  0.00053 |    0.00055 |     1761.40355 |     0.01628 |  100 |
| emdi    |  0.04959 |    0.05051 |       19.42356 |    37.33941 |  100 |
| sae     |  0.00336 |    0.00344 |      288.10834 |     0.91754 |  100 |
| fastsae |  0.00087 |    0.00090 |     1080.76612 |     0.03860 |  250 |
| emdi    |  0.83982 |    0.85735 |        1.16289 |   240.89332 |  250 |
| sae     |  0.03729 |    0.03758 |       26.01528 |     6.34988 |  250 |
| fastsae |  0.00138 |    0.00140 |      696.95570 |     0.07579 |  500 |
| emdi    |  5.82064 |    5.86366 |        0.16988 |   880.31369 |  500 |
| sae     |  0.19652 |    0.19673 |        5.02802 |    18.23907 |  500 |
| fastsae |  0.00396 |    0.00413 |      175.75489 |     0.18570 | 1000 |
| emdi    | 50.64281 |   51.02236 |        0.01958 |  3773.59894 | 1000 |
| sae     |  1.49633 |    1.50363 |        0.66408 |    71.97652 | 1000 |

**SEBLUP (p = 5)**

| Method  |  Min (s) | Median (s) | Iterations/sec | Memory (MB) |    n |
|:--------|---------:|-----------:|---------------:|------------:|-----:|
| fastsae |  0.00072 |    0.00073 |     1312.17465 |     0.02981 |   30 |
| emdi    |  0.01052 |    0.01071 |       93.46346 |     4.18307 |   30 |
| sae     |  0.00389 |    0.00410 |      246.21949 |     1.37314 |   30 |
| fastsae |  0.00110 |    0.00115 |      841.21692 |     0.07378 |   50 |
| emdi    |  0.01726 |    0.01800 |       43.83976 |    10.46759 |   50 |
| sae     |  0.00939 |    0.00968 |      102.57065 |     3.31219 |   50 |
| fastsae |  0.00462 |    0.00473 |      208.13301 |     0.25861 |  100 |
| emdi    |  0.04976 |    0.05299 |       17.47209 |    37.33941 |  100 |
| sae     |  0.06985 |    0.07054 |       12.79223 |    17.97552 |  100 |
| fastsae |  0.02359 |    0.02383 |       37.01162 |     1.49972 |  250 |
| emdi    |  0.83931 |    0.84378 |        1.17486 |   240.89332 |  250 |
| sae     |  1.05539 |    1.07476 |        0.92557 |   110.38794 |  250 |
| fastsae |  0.13310 |    0.14205 |        6.95660 |     5.85706 |  500 |
| emdi    |  3.08699 |    3.13662 |        0.31768 |   949.86449 |  500 |
| sae     |  7.10022 |    7.11793 |        0.13965 |   776.40673 |  500 |
| fastsae |  0.82337 |    0.83216 |        1.19408 |    23.15480 | 1000 |
| emdi    | 50.82626 |   51.05931 |        0.01955 |  3772.56870 | 1000 |
| sae     | 66.98470 |   67.15175 |        0.01488 |  1920.00576 | 1000 |

## References

- Fay, R. E., & Herriot, R. A. (1979). Estimates of income for small
  places: An application of James-Stein procedures to Census data.
  *JASA*, 74(366), 269-277.
- Battese, G. E., Harter, R. M., & Fuller, W. A. (1988). An
  error-components model for prediction of county crop areas using
  survey and satellite data. *JASA*, 83(401), 28-36.
- Rao, J. N. K., & Molina, I. (2015). *Small Area Estimation*, 2nd
  Edition. Wiley.
