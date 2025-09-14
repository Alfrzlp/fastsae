// src/eblup_core.cpp
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
List eblup_core_optimized(const arma::mat& X,
                          const arma::vec& y,
                          const arma::vec& vardir,
                          std::string method = "REML",
                          int maxiter = 100,
                          double precision = 1e-4,
                          bool return_fullP = false) {

  const int m = X.n_rows;
  const int p = X.n_cols;

  // safety checks
  if ((int) vardir.n_elem != m) {
    stop("Length of 'vardir' must equal nrow(X).");
  }
  if ((int) y.n_elem != m) {
    stop("Length of 'y' must equal nrow(X).");
  }
  if (!(method == "ML" || method == "REML")) {
    stop("method must be 'ML' or 'REML'.");
  }

  // Precompute transpose
  mat Xt = X.t();

  // initializations
  double sigma2 = as_scalar(median(vardir)); // initial
  double diff = precision + 1.0;
  int k = 0;

  vec w(m);         // weights = 1 / (vardir + sigma2)
  mat A;            // A = diag(w) * X  (implemented as X.each_col() % w)
  mat XtWiX(p, p);
  vec XtWy(p);
  vec beta(p);
  vec resid(m);
  vec u(m);
  vec eblup(m);

  // Pre-alloc temporary matrices used in REML when needed
  mat Q(p, p);
  mat P;
  vec Py(m);

  // Iteration loop (ML or REML)
  if (method == "ML") {
    while ((diff > precision) && (k < maxiter)) {
      // compute weights
      w = 1.0 / (vardir + sigma2);        // diag(V) inverse entries
      // A = diag(w) * X  implemented as element-wise multiply of rows
      A = X.each_col() % w;

      // Xt * W * X  and Xt * W * y
      XtWiX = Xt * A;                     // p x p
      XtWy = Xt * (w % y);                // p vector

      // solve for beta: XtWiX * beta = XtWy
      beta = solve(XtWiX, XtWy);

      resid = y - X * beta;
      // yXBeta = resid
      // s = -0.5 * sum(w) + 0.5 * sum(resid^2 * w^2)
      vec r2 = resid % resid;
      double s = -0.5 * accu(w) + 0.5 * accu(r2 % (w % w));
      double Isigma2 = 0.5 * accu(w % w);

      // update sigma2
      double sigma2_new = sigma2 + s / Isigma2;
      if (sigma2_new < 0) sigma2_new = 0.0;

      diff = std::fabs(sigma2_new - sigma2);
      sigma2 = sigma2_new;
      ++k;
    }
  } else { // REML
    while ((diff > precision) && (k < maxiter)) {
      w = 1.0 / (vardir + sigma2);
      A = X.each_col() % w;               // m x p

      XtWiX = Xt * A;                     // p x p
      XtWy = Xt * (w % y);                // p vector

      // Q = inv(X' V^-1 X)
      Q = inv_sympd(XtWiX);               // p x p (p typically small)

      // P = V^-1 - V^-1 X Q X' V^-1
      // but V^-1 diag = diag(w)
      // Vi*y = w % y
      vec t1 = Q * XtWy;                  // p vector
      Py = (w % y) - A * t1;              // P %*% y  (m vector)

      // trace(P) = sum(w) - trace(A * Q * A.t()) = sum(w) - p  (since Q * XtWiX = I)
      double trP = accu(w) - (double) p;

      // s = -0.5 * tr(P) + 0.5 * t(Py) %*% Py
      double s = -0.5 * trP + 0.5 * as_scalar(Py.t() * Py);

      // Isigma2 = 0.5 * trace(P * P) -> we must build P to compute this
      // P = diagmat(w) - A * Q * A.t()
      if (return_fullP) {
        // If user wants full P returned we construct it anyway
        mat W = diagmat(w);
        P = W - A * Q * A.t();
        double trPP = accu(square(P));
        double Isigma2 = 0.5 * trPP;

        double sigma2_new = sigma2 + s / Isigma2;
        if (sigma2_new < 0) sigma2_new = 0.0;
        diff = std::fabs(sigma2_new - sigma2);
        sigma2 = sigma2_new;
      } else {
        // construct P only to compute trace(P*P). For large m this is the heavy step.
        // But we can compute P explicitly here (still cheaper than general V^-1)
        mat W = diagmat(w);
        P = W - A * Q * A.t();
        double trPP = accu(square(P));
        double Isigma2 = 0.5 * trPP;

        double sigma2_new = sigma2 + s / Isigma2;
        if (sigma2_new < 0) sigma2_new = 0.0;
        diff = std::fabs(sigma2_new - sigma2);
        sigma2 = sigma2_new;
      }

      ++k;
    }
  }

  // Final estimation with sigma2
  // compute Beta, eblup, u, goodness
  w = 1.0 / (vardir + sigma2);
  A = X.each_col() % w;
  XtWiX = Xt * A;
  XtWy = Xt * (w % y);
  beta = solve(XtWiX, XtWy);
  vec stderr_beta = sqrt(Q.diag());
  vec zvalue = beta / stderr_beta;
  vec pvalue(zvalue.n_elem);
  for (uword i = 0; i < zvalue.n_elem; i++) {
    pvalue(i) = 2.0 * R::pnorm5(-std::abs(zvalue(i)), 0.0, 1.0, 1, 0);
  }

  vec Xbeta = X * beta;
  resid = y - Xbeta;
  u = (sigma2 / (sigma2 + vardir)) % resid;
  eblup = Xbeta + u;

  const double loglike = -0.5 * accu(log(2.0 * M_PI * (sigma2 + vardir)) + (resid % resid) / (sigma2 + vardir));
  const double AIC = -2.0 * loglike + 2.0 * (p + 1);
  const double BIC = -2.0 * loglike + (p + 1) * std::log((double) m);

  // Hitung MSE components dengan cara vektorized
  vec Bd = sigma2 / (sigma2 + vardir);
  Q = inv_sympd(XtWiX);

  // precompute h = diag(X * Q * X^T)
  vec h = sum((X * Q) % X, 1);

  double SumAD2 = accu(square(w));
  double VarA = 2.0 / SumAD2;

  vec g1 = vardir % (1.0 - Bd);
  vec g2 = square(Bd) % h;
  vec g3 = square(Bd) % (VarA / (sigma2 + vardir));
  vec mse;

  if (method == "ML") {
    double b = - trace(Q * XtWiX) / SumAD2;
    mse = g1 + g2 + 2.0 * g3 - b * square(Bd);
  } else { // REML
    mse = g1 + g2 + 2.0 * g3;
  }
  vec rse = sqrt(mse) * 100 / eblup;

  // return
  List out = List::create(
    _["sigma2_u"] = sigma2,
    _["beta"]     = beta,
    _["stderr_beta"] = beta,
    _["zvalue"]   = zvalue,
    _["pvalue"]   = pvalue,

    _["eblup"]    = eblup,
    _["u"]        = u,
    _["mse"]      = mse,
    _["rse"]      = rse,
    _["loglike"]  = loglike,
    _["AIC"]      = AIC,
    _["BIC"]      = BIC,
    _["n_iter"]   = k,
    _["convergence"] = (k < maxiter)
  );

  return out;
}
