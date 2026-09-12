// src/seblup_npbmse.cpp
// Non-Parametric Bootstrap MSE for Spatial Fay-Herriot Model
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

// [[Rcpp::export(.seblup_npbmse)]]
List seblup_npbmse(
    const arma::mat& X,
    const arma::vec& y,
    const arma::vec& vardir,
    const arma::mat& W,
    std::string method = "REML",
    int maxiter = 100,
    double precision = 1e-4,
    int B = 100,
    int n_threads = 0,
    int max_attempts_factor = 5,
    int seed = -1
) {
  if (seed >= 0) set_r_seed(seed);

  // Nonparametric bootstrap only defined for REML
  if (method != "REML") {
    stop("method must be 'REML' for nonparametric bootstrap MSE (seblup_npbmse).");
  }

  // ================================================================
  // Main fit
  // ================================================================
  List result_awal = seblup_core(
    X, y, vardir, W,
    method, maxiter, precision,
    false
  );

  if (!(bool) result_awal["convergence"]) {
    warning("Initial fit did not converge; bootstrap MSE results may be unreliable.");
  }

  const int m = X.n_rows;
  const int p = X.n_cols;

  double sigma2_boot = result_awal["sigma2_u"];
  double rho_boot = result_awal["rho"];
  mat beta_boot = result_awal["beta"];
  mat Xbeta = result_awal["Xbeta"];
  mat GVi = result_awal["GVi"];
  mat Q_mat = result_awal["Q"];
  mat XtVi_mat = result_awal["XtVi"];
  mat Vi_mat = result_awal["Vi"];
  mat G_mat = result_awal["G"];
  vec g1 = result_awal["g1d"];
  vec g2 = result_awal["g2d"];

  mat I = eye<mat>(m, m);
  mat Irhoproxmat = I - rho_boot * W;

  // ================================================================
  // 1) Compute standardized residuals
  // ================================================================
  vec resid_fit = y - Xbeta;
  vec vstim = GVi * resid_fit;

  mat VG = diagmat(vardir);

  mat QXtVi = Q_mat * XtVi_mat;
  mat ViX = XtVi_mat.t();
  mat P = Vi_mat - ViX * QXtVi;

  mat Ve = VG * P * VG;
  mat Vu = Irhoproxmat * G_mat * P * G_mat * Irhoproxmat.t();

  // symmetrize for numerical stability
  Ve = 0.5 * (Ve + Ve.t());
  Vu = 0.5 * (Vu + Vu.t());

  // eigenvalue decomposition
  vec eigval_e; mat eigvec_e;
  eig_sym(eigval_e, eigvec_e, Ve);
  mat VecVe = eigvec_e.cols(p, m - 1);
  vec ValVe = eigval_e.subvec(p, m - 1);
  mat Vei05 = VecVe * diagmat(1.0 / arma::sqrt(ValVe)) * VecVe.t();

  vec eigval_u; mat eigvec_u;
  eig_sym(eigval_u, eigvec_u, Vu);
  mat VecVu = eigvec_u.cols(p, m - 1);
  vec ValVu = eigval_u.subvec(p, m - 1);
  mat Vui05 = VecVu * diagmat(1.0 / arma::sqrt(ValVu)) * VecVu.t();

  vec ustim = Vui05 * (Irhoproxmat * vstim);
  vec estim = Vei05 * (resid_fit - vstim);

  double sdu = std::sqrt(sigma2_boot);

  double mean_u = arma::mean(ustim);
  double sdemp_u = std::sqrt(arma::mean(arma::square(ustim - mean_u)));
  vec u_std = sdu * (ustim - mean_u) / sdemp_u;

  double mean_e = arma::mean(estim);
  double sdemp_e = std::sqrt(arma::mean(arma::square(estim - mean_e)));
  vec e_std = (estim - mean_e) / sdemp_e;

  // ================================================================
  // 2) Batch scheme: repeat until exactly B valid replicates
  // ================================================================
  vec sum_mse(m, fill::zeros);
  vec sum_g3(m, fill::zeros);
  vec sum_g1(m, fill::zeros);
  vec sum_g2(m, fill::zeros);

  int n_valid_total = 0;
  int n_attempted_total = 0;
  int round_no = 0;
  const int max_total_attempts = B * max_attempts_factor;

#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
#endif

  while (n_valid_total < B) {
    ++round_no;
    int need = B - n_valid_total;

    if (n_attempted_total + need > max_total_attempts) {
      int remaining_budget = max_total_attempts - n_attempted_total;
      if (remaining_budget <= 0) {
        stop("Failed to collect %d valid replicates after %d total attempts "
             "(too many replicates failed to converge).", B, n_attempted_total);
      }
      need = remaining_budget;
    }

    // --- 2a) Generate resampling indices SEQUENTIALLY ---
    umat U_idx(m, need), E_idx(m, need);
    for (int b = 0; b < need; ++b) {
      for (int i = 0; i < m; ++i) {
        U_idx(i, b) = (arma::uword) std::floor(R::unif_rand() * m);
        E_idx(i, b) = (arma::uword) std::floor(R::unif_rand() * m);
      }
    }

    // --- 2b) Temporary storage for this round ---
    mat SqDiff_r(m, need, fill::zeros);
    mat SqDiffPb_r(m, need, fill::zeros);
    mat G1_r(m, need, fill::zeros);
    mat G2_r(m, need, fill::zeros);
    vec valid_flag_r(need, fill::zeros);

    // --- 2c) Parallel loop: call pure Armadillo function ---
#pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < need; ++b) {
      vec u_boot(m), e_samp(m);
      for (int i = 0; i < m; ++i) {
        u_boot(i) = u_std(U_idx(i, b));
        e_samp(i) = e_std(E_idx(i, b));
      }
      vec e_boot = arma::sqrt(vardir) % e_samp;
      vec v_boot = arma::solve(Irhoproxmat, u_boot);
      vec theta_boot = X * beta_boot + v_boot;
      vec direct_boot = theta_boot + e_boot;

      SeblupFitArma res = seblup_fit_core_arma(
        X, direct_boot, vardir, W, method, maxiter, precision
      );

      bool valid = res.converged &&
        res.sigma2 >= 0.0 &&
        res.rho_fix >= -1.0 && res.rho_fix <= 1.0;

      if (!valid) continue;

      // sblup estimate using initial fit weights
      vec Bstim_sblup = QXtVi * direct_boot;
      vec theta_sblup = X * Bstim_sblup + GVi * (direct_boot - X * Bstim_sblup);

      SqDiff_r.col(b) = arma::square(res.theta - theta_boot);
      SqDiffPb_r.col(b) = arma::square(res.theta - theta_sblup);
      G1_r.col(b) = res.g1d;
      G2_r.col(b) = res.g2d;
      valid_flag_r(b) = 1.0;
    }

    // --- 2d) Round aggregation ---
    int n_valid_round = (int) arma::accu(valid_flag_r);

    sum_mse += arma::sum(SqDiff_r, 1);
    sum_g3 += arma::sum(SqDiffPb_r, 1);
    sum_g1 += arma::sum(G1_r, 1);
    sum_g2 += arma::sum(G2_r, 1);

    n_valid_total += n_valid_round;
    n_attempted_total += need;

    if (n_valid_round == 0) {
      Rcpp::warning(
        "Round %d: 0 out of %d replicates valid (all failed to converge).",
        round_no, need
      );
    }
  }

  // ================================================================
  // 3) Final means
  // ================================================================
  vec mse_npb = sum_mse / (double) B;
  vec g3_npb = sum_g3 / (double) B;
  vec g1_npb = sum_g1 / (double) B;
  vec g2_npb = sum_g2 / (double) B;

  // bias corrected
  vec mse_npb2 = 2.0 * (g1 + g2) - g1_npb - g2_npb + g3_npb;

  DataFrame df_eblup = Rcpp::as<Rcpp::DataFrame>(result_awal["df_eblup"]);
  df_eblup.push_back(mse_npb, "mse_npb");
  df_eblup.push_back(mse_npb2, "mse_npbbc");
  result_awal["df_eblup"] = df_eblup;

  result_awal.push_back(round_no, "n_rounds");
  result_awal.push_back(n_attempted_total, "n_attempted_total");
  return result_awal;
}
