#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
List eblup_core(const arma::mat& X,
                const arma::vec& y,
                const arma::vec& vardir,
                std::string method = "REML",
                int maxiter = 100,
                double precision = 1e-4) {

  int m = X.n_rows;
  int p = X.n_cols;
  mat Xt = X.t();
  vec sigma2_u(maxiter + 2, fill::zeros);

  // initial value
  sigma2_u(0) = median(vardir);
  double diff = 1 + precision;
  int k = 0;

  mat Z = eye(m, m);
  mat R = diagmat(vardir);
  mat V, Vi, XtVi, Q, BETA, P, PZtZ, Py;
  double s, Isigma2_u;

  // Iterasi Fisher scoring
  if (method == "ML") {
    while ((diff > precision) && (k < maxiter)) {
      mat G = eye(m, m) * sigma2_u(k);
      V = Z * G * Z.t() + R;
      Vi = inv_sympd(V);
      XtVi = Xt * Vi;
      // Q = inv_sympd(XtVi * X);
      Q = solve(XtVi * X, eye(p, p));
      BETA = Q * XtVi * y;

      vec yXBeta = y - X * BETA;
      mat ViZtZ = Vi * Z * Z.t();

      s = -0.5 * trace(ViZtZ) - 0.5 * as_scalar(yXBeta.t() * (-ViZtZ * Vi) * yXBeta);
      Isigma2_u = trace(ViZtZ * ViZtZ) / 2.0;

      k++;
      sigma2_u(k) = sigma2_u(k - 1) + s / Isigma2_u;
      diff = fabs((sigma2_u(k) - sigma2_u(k - 1)) / sigma2_u(k - 1));
    }
  } else { // REML
    while ((diff > precision) && (k < maxiter)) {
      mat G = eye(m, m) * sigma2_u(k);
      V = Z * G * Z.t() + R;
      Vi = inv_sympd(V);
      XtVi = Xt * Vi;
      // Q = inv_sympd(XtVi * X);
      Q = solve(XtVi * X, eye(p, p));
      P = Vi - Vi * X * Q * XtVi;
      PZtZ = P * Z * Z.t();
      Py = P * y;

      s = -0.5 * trace(PZtZ) + 0.5 * as_scalar(y.t() * PZtZ * Py);
      Isigma2_u = trace(PZtZ * PZtZ) / 2.0;

      k++;
      sigma2_u(k) = sigma2_u(k - 1) + s / Isigma2_u;
      diff = fabs((sigma2_u(k) - sigma2_u(k - 1)) / sigma2_u(k - 1));
    }
  }

  double sigma2_final = std::max(sigma2_u(k), 0.0);

  // Estimasi beta
  mat G = eye(m, m) * sigma2_final;
  V = Z * G * Z.t() + R;
  Vi = inv_sympd(V);
  XtVi = Xt * Vi;
  // Q = inv_sympd(XtVi * X);
  Q = solve(XtVi * X, eye(p, p));
  BETA = Q * XtVi * y;
  vec zvalue = BETA / sqrt(Q.diag());

  // EBLUP
  vec Xbeta = X * BETA;
  vec resid = y - Xbeta;
  vec u = (sigma2_final / (sigma2_final + vardir)) % resid;
  vec eblup_est = Xbeta + u;

  // Goodness of fit
  double loglike = -0.5 * accu(log(2 * M_PI * (sigma2_final + vardir)) + (resid % resid) / (sigma2_final + vardir));
  double AIC = -2 * loglike + 2 * (p + 1);
  double BIC = -2 * loglike + (p + 1) * log(m);

  return List::create(
    _["sigma2_u"] = sigma2_final,
    _["beta"] = BETA,
    _["eblup"] = eblup_est,
    _["u"] = u,
    _["loglike"] = loglike,
    _["AIC"] = AIC,
    _["BIC"] = BIC,
    _["n_iter"] = k,
    _["convergence"] = (k < maxiter)
  );
}
