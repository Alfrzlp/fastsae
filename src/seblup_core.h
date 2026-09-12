// src/seblup_core.h
// Forward declarations for seblup functions

#ifndef SEBLUP_CORE_H
#define SEBLUP_CORE_H

#include <RcppArmadillo.h>

struct SeblupFitArma {
  arma::vec theta;
  arma::vec g1d;
  arma::vec g2d;
  double sigma2;
  double rho_fix;
  int n_iter;
  bool converged;
};

void set_r_seed(int seed);

SeblupFitArma seblup_fit_core_arma(
    const arma::mat& X,
    const arma::vec& y,
    const arma::vec& vardir,
    const arma::mat& W,
    const std::string& method,
    int maxiter,
    double precision
);

// Forward declare seblup_core (Rcpp export)
Rcpp::List seblup_core(
    const arma::mat& Xall,
    const arma::vec& yall,
    const arma::vec& vardirall,
    const arma::mat& Wall,
    std::string method = "REML",
    int maxiter = 100,
    double precision = 1e-4,
    bool only_core = false
);

#endif // SEBLUP_CORE_H
