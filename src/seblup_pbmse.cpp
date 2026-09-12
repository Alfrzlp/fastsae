// src/seblup_pbmse.cpp
// Parametric Bootstrap MSE for Spatial Fay-Herriot Model
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "seblup_core.h"
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export(.seblup_pbmse)]]
List seblup_pbmse(
    const arma::mat& X,
    const arma::vec& y,
    const arma::vec& vardir,
    const arma::mat& W,
    std::string method = "REML",
    int maxiter = 100,
    double precision = 1e-4,
    int B = 100,
    int n_threads = 0,
    int seed = -1
) {
  if (seed >= 0) set_r_seed(seed);

  // ================================================================
  // Main fit
  // ================================================================
  List result_awal = seblup_core(
    X, y, vardir, W,
    method, maxiter, precision,
    false
  );

  const int m = X.n_rows;
  double sigma2_boot = result_awal["sigma2_u"];
  double rho_boot = result_awal["rho"];
  mat beta_boot = result_awal["beta"];
  mat Xbeta = result_awal["Xbeta"];
  mat GVi = result_awal["GVi"];
  vec g1 = result_awal["g1d"];
  vec g2 = result_awal["g2d"];

  mat Wt = W.t();
  mat I = eye<mat>(m, m);

  // ================================================================
  // 1) Generate all random numbers SEQUENTIALLY
  // ================================================================
  mat U_boot(m, B);
  mat E_boot(m, B);
  for (int b = 0; b < B; ++b) {
    U_boot.col(b) = as<arma::vec>(rnorm(m, 0.0, std::sqrt(sigma2_boot)));
    for (int i = 0; i < m; ++i) {
      E_boot(i, b) = R::rnorm(0.0, std::sqrt(vardir(i)));
    }
  }

  // ================================================================
  // 2) Storage matrices (1 column = 1 replicate)
  // ================================================================
  mat Theta_mat(m, B, fill::zeros);
  mat G1_mat(m, B, fill::zeros);
  mat G2_mat(m, B, fill::zeros);
  mat SqDiff_mat(m, B, fill::zeros);
  mat SqDiffPb_mat(m, B, fill::zeros);

#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
#endif

  // ================================================================
  // 3) Parallel loop: call pure Armadillo function
  // ================================================================
#pragma omp parallel for schedule(dynamic)
  for (int b = 0; b < B; ++b) {
    vec v_boot = (I - rho_boot * Wt) * U_boot.col(b);
    vec theta_boot = X * beta_boot + v_boot;
    vec direct_boot = theta_boot + E_boot.col(b);

    SeblupFitArma res = seblup_fit_core_arma(
      X, direct_boot, vardir, W, method, maxiter, precision
    );

    vec resid = theta_boot - Xbeta;
    vec theta_pb = Xbeta + GVi * resid;

    Theta_mat.col(b) = res.theta;
    G1_mat.col(b) = res.g1d;
    G2_mat.col(b) = res.g2d;
    SqDiff_mat.col(b) = arma::square(res.theta - theta_boot);
    SqDiffPb_mat.col(b) = arma::square(theta_boot - theta_pb);
  }

  // ================================================================
  // 4) Sequential aggregation
  // ================================================================
  vec mse_pb = arma::mean(SqDiff_mat, 1);
  vec g1_pb = arma::mean(G1_mat, 1);
  vec g2_pb = arma::mean(G2_mat, 1);
  vec g3_pb = arma::mean(SqDiffPb_mat, 1);

  // bias corrected
  vec mse_pb2 = 2.0 * (g1 + g2) - g1_pb - g2_pb + g3_pb;

  DataFrame df_eblup = Rcpp::as<Rcpp::DataFrame>(result_awal["df_eblup"]);
  df_eblup.push_back(mse_pb, "mse_pb");
  df_eblup.push_back(mse_pb2, "mse_pbbc");
  result_awal["df_eblup"] = df_eblup;

  return result_awal;
}
