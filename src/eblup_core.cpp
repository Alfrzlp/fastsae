// src/eblup_core.cpp
#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export(.eblup_core)]]
List eblup_core(
    const arma::mat& Xall,
    const arma::vec& yall,
    const arma::vec& vardirall,
    std::string method = "REML",
    int maxiter = 100,
    double precision = 1e-4) {

  // ================================================================
  // hasil
  // ================================================================
  arma::vec eblup_all(Xall.n_rows);
  arma::vec mse_all(Xall.n_rows);

  // ================================================================
  // Handle unsampled areas (y=NA only)
  // Note: For sampled areas, vardir must be > 0
  // ================================================================
  arma::mat Xs, Xns;
  arma::vec y, vardir;
  arma::uvec idx_ns, idx_s;
  
  // Find unsampled areas: y is NA (any vardir)
  idx_ns = arma::find_nonfinite(yall);
  idx_s = arma::find_finite(yall);

  bool adaNA = idx_ns.n_elem > 0;

  if (adaNA) {
    Xns = Xall.rows(idx_ns);
    Xs = Xall.rows(idx_s);
    y = yall.elem(idx_s);
    vardir = vardirall.elem(idx_s);
  } else {
    Xs = Xall;
    y = yall;
    vardir = vardirall;
  }

  const int m = Xs.n_rows;
  const int p = Xs.n_cols;

  // safety checks
  if (m == 0) stop("No sampled areas found.");
  if (!(method == "ML" || method == "REML")) stop("method must be 'ML' or 'REML'.");

  // Check vardir > 0 for sampled areas
  if (arma::any(vardir <= 0)) {
    stop("vardir must be strictly positive for sampled areas.");
  }

  // ================================================================
  // initial value for sigma2
  // ================================================================
  double sigma2 = as_scalar(median(vardir));
  if (sigma2 < 0.0) sigma2 = 0.0;

  double diff = precision + 1.0;
  int k = 0;

  // ================================================================
  // pre-allocate temporaries
  // ================================================================
  mat Xt = Xs.t();
  mat eye_p = eye<mat>(p, p);
  vec Vi(m);
  vec Vi_sq(m);
  mat XtVi(p, m);
  mat XtViX(p, p);
  mat Q(p, p);
  mat QXtVi(p, m);
  vec Py(m);

  const bool is_ml = (method == "ML");

  // ================================================================
  // Fisher-Scoring Iteration
  // ================================================================
  while ((diff > precision) && (k < maxiter)) {
    Vi = 1.0 / (vardir + sigma2);
    Vi_sq = square(Vi);

    XtVi = trans(Xs.each_col() % Vi);
    XtViX = XtVi * Xs;

    bool ok = inv_sympd(Q, XtViX);
    if (!ok) Q = solve(XtViX, eye_p);

    QXtVi = Q * XtVi;
    Py = Vi % y - trans(XtVi) * (QXtVi * y);

    mat XtViXtVi = XtVi * XtVi.t();
    double trP = accu(Vi) - trace(Q * XtViXtVi);
    double s = (-0.5) * trP + 0.5 * as_scalar(trans(Py) * Py);

    double Isigma2;
    if (is_ml) {
      Isigma2 = 0.5 * accu(Vi_sq);
    } else {
      mat M = trans(XtVi) * QXtVi;
      double tracePP = accu(Vi_sq) - 2.0 * accu(Vi % M.diag()) + trace(M * M);
      Isigma2 = 0.5 * tracePP;
    }

    if (Isigma2 <= 0) Isigma2 = std::numeric_limits<double>::min();

    double sigma2_new = sigma2 + s / Isigma2;
    if (!std::isfinite(sigma2_new) || sigma2_new < 0.0) sigma2_new = 0.0;

    diff = std::fabs((sigma2_new - sigma2) / std::max(sigma2, 1e-12));
    sigma2 = sigma2_new;
    ++k;
  }

  if (sigma2 < 0) sigma2 = 0.0;

  // ================================================================
  // Final parameter estimates
  // ================================================================
  Vi = 1.0 / (vardir + sigma2);
  mat XtVi_final = trans(Xs.each_col() % Vi);
  XtViX = XtVi_final * Xs;
  mat XtViy = Xt * (Vi % y);

  bool ok_final = inv_sympd(Q, XtViX);
  if (!ok_final) Q = solve(XtViX, eye_p);

  mat beta = Q * XtViy;
  vec stderr_beta = sqrt(Q.diag());

  // ================================================================
  // p-value
  // ================================================================
  vec zvalue = beta / stderr_beta;
  vec abs_z = arma::abs(zvalue);
  vec pvalue(zvalue.n_elem);
  for (uword i = 0; i < pvalue.n_elem; ++i) {
    pvalue(i) = 2.0 * R::pnorm5(-abs_z(i), 0.0, 1.0, 1, 0);
  }

  // ================================================================
  // EBLUP computation
  // ================================================================
  vec Xbeta_s = Xs * beta;
  vec resid = y - Xbeta_s;
  vec sigma_vardir = sigma2 + vardir;
  vec Bd = vardir / sigma_vardir;

  vec u = (sigma2 / sigma_vardir) % resid;
  vec eblup_s = Xbeta_s + u;

  // ================================================================
  // Goodness of fit
  // ================================================================
  vec resid_sq = resid % resid;
  vec log_term = log(2.0 * M_PI * sigma_vardir) + resid_sq / sigma_vardir;
  const double loglike = -0.5 * accu(log_term);
  const double AIC = -2.0 * loglike + 2.0 * (p + 1);
  const double BIC = -2.0 * loglike + (p + 1) * std::log((double) m);

  Rcpp::NumericVector goodness = Rcpp::NumericVector::create(
    Rcpp::Named("loglikelihood") = loglike,
    Rcpp::Named("AIC") = AIC,
    Rcpp::Named("BIC") = BIC
  );

  // ================================================================
  // MSE computation
  // ================================================================
  double SumAD2 = accu(square(Vi));
  double VarA = 2.0 / SumAD2;

  vec Bd2 = square(Bd);
  vec h = sum(Xs * Q % Xs, 1);

  vec g1 = vardir % (1.0 - Bd);
  vec g2 = Bd2 % h;
  vec g3 = Bd2 % (VarA / sigma_vardir);

  vec mse_s(m);
  if (is_ml) {
    double b = -trace(Q * XtViX) / std::max(SumAD2, 1e-30);
    mse_s = g1 + g2 + 2.0 * g3 - b * Bd2;
  } else {
    mse_s = g1 + g2 + 2.0 * g3;
  }

  // ================================================================
  // Synthetic estimators for unsampled areas
  // ================================================================
  if (adaNA) {
    vec eblup_ns = Xns * beta;
    vec mse_ns = sigma2 + sum(Xns * Q % Xns, 1);

    eblup_all.elem(idx_s) = eblup_s;
    eblup_all.elem(idx_ns) = eblup_ns;
    mse_all.elem(idx_s) = mse_s;
    mse_all.elem(idx_ns) = mse_ns;
  } else {
    eblup_all = eblup_s;
    mse_all = mse_s;
  }

  // ================================================================
  // Relative Standard Error (%)
  // ================================================================
  vec rse = sqrt(mse_all);
  vec relerr = rse / arma::abs(eblup_all);
  relerr.elem(arma::find(eblup_all == 0)).fill(datum::nan);
  rse = 100.0 * relerr;

  // ================================================================
  // Output
  // ================================================================
  DataFrame df_coef = DataFrame::create(
    _["beta"] = beta,
    _["stderr_beta"] = stderr_beta,
    _["zvalue"] = zvalue,
    _["pvalue"] = pvalue
  );

  DataFrame df_eblup = DataFrame::create(
    _["y"] = yall,
    _["eblup"] = eblup_all,
    _["vardir"] = vardirall,
    _["mse"] = mse_all,
    _["rse"] = rse
  );

  List out = List::create(
    _["random_effect_var"] = sigma2,
    _["estcoef"] = df_coef,
    _["df_eblup"] = df_eblup,
    _["goodness"] = goodness,
    _["n_iter"] = k,
    _["convergence"] = (k < maxiter),
    _["method"] = "eblup",
    _["level"] = "area"
  );

  return out;
}
