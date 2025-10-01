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

  // hasil
  arma::vec eblup_all(Xall.n_rows);
  arma::vec mse_all(Xall.n_rows);

  // deklarasi global di fungsi
  arma::mat Xns, X;
  arma::vec y, vardir;
  arma::uvec idx_ns, idx_s;
  bool adaNA = yall.has_nan();

  if(adaNA){
    idx_ns = arma::find_nonfinite(yall);   // sama dengan is.na(y)
    idx_s  = arma::find_finite(yall);      // !is.na(y)

    // X untuk area non-sampled
    Xns = Xall.rows(idx_ns);
    // X dan y untuk sampled
    X  = Xall.rows(idx_s);
    y  = yall.elem(idx_s);
    vardir  = vardirall.elem(idx_s);
  }else{
    X  = Xall;
    y  = yall;
    vardir = vardirall;
  }

  const int m = X.n_rows;
  const int p = X.n_cols;

  // safety checks
  // if ((int) vardir.n_elem != m) stop("Length of 'vardir' must equal nrow(X).");
  // if ((int) y.n_elem != m) stop("Length of 'y' must equal nrow(X).");
  if (!(method == "ML" || method == "REML")) stop("method must be 'ML' or 'REML'.");


  // initial value for sigma2
  double sigma2 = as_scalar(median(vardir));
  if (sigma2 < 0.0) sigma2 = 0.0;

  double diff = precision + 1.0;
  int k = 0;

  // Pre-allocate all temporaries reused in loop
  mat Xt = X.t();
  mat eye_p = eye<mat>(p, p);
  vec Vi(m);
  mat XtVi(p, m);
  mat XtViX(p, p);
  mat Q(p, p);
  mat temp_pp(p, p);
  mat temp_pm(p, m);
  vec Py(m);
  mat P(m, m);

  // ML
  if (method == "ML") {
    while ((diff > precision) && (k < maxiter)) {
      Vi = 1.0 / (vardir + sigma2);

      // XtVi <- t(Vi * X) p x m
      XtVi = trans(X.each_col() % Vi);
      XtViX = XtVi * X;

      // Q = inv_sympd(XtViX) with fallback to solve
      bool ok = inv_sympd(Q, XtViX); // fast Cholesky-based inverse if SPD
      if (!ok) {
        Q = solve(XtViX, eye_p);
      }

      // P = diag(Vi) - t(XtVi) * Q * XtVi
      // t(XtVi) is m x p, XtVi is p x m
      // compute temp_pp = Q * XtVi  (p x m) but reuse variables:
      temp_pm = Q * XtVi;               // p x m
      // compute t(XtVi) * temp_pm  => (m x p) * (p x m) = m x m
      P = diagmat(Vi);                 // m x m
      P -= trans(XtVi) * temp_pm;      // m x m

      // Py = P * y
      Py = P * y;

      // s = (-0.5) * trace(P) + 0.5 * (t(Py) * Py)
      double trP = trace(P); // sum of diagonal
      double s = (-0.5) * trP + 0.5 * as_scalar(trans(Py) * Py);

      // Fisher info approx for ML: 0.5 * sum(Vi^2)
      double Isigma2 = 0.5 * accu(square(Vi));
      if (Isigma2 <= 0) Isigma2 = std::numeric_limits<double>::min();

      double sigma2_new = sigma2 + s / Isigma2;
      if (!std::isfinite(sigma2_new) || sigma2_new < 0.0) sigma2_new = 0.0; // guard

      diff = std::fabs((sigma2_new - sigma2) / std::max(sigma2, 1e-12));
      sigma2 = sigma2_new;
      ++k;
    }
  } else {
    // REML
    while ((diff > precision) && (k < maxiter)) {
      Vi = 1.0 / (vardir + sigma2);
      XtVi = trans(X.each_col() % Vi); // p x m
      XtViX = XtVi * X;

      bool ok = inv_sympd(Q, XtViX);
      if (!ok) Q = solve(XtViX, eye_p);

      temp_pm = Q * XtVi;             // p x m
      P = diagmat(Vi);
      P -= trans(XtVi) * temp_pm;

      Py = P * y;

      double trP = trace(P);
      double s = (-0.5) * trP + 0.5 * as_scalar(trans(Py) * Py);

      // Fisher info approx for REML: 0.5 * sum(diag(P %*% P))
      // compute trace(P*P) as sum of elementwise product on diag? use trace(P*P)
      double tracePP = trace(P * P);
      double Isigma2 = 0.5 * tracePP;
      if (Isigma2 <= 0) Isigma2 = std::numeric_limits<double>::min();

      double sigma2_new = sigma2 + s / Isigma2;
      if (!std::isfinite(sigma2_new) || sigma2_new < 0.0) sigma2_new = 0.0;

      diff = std::fabs((sigma2_new - sigma2) / std::max(sigma2, 1e-12));
      sigma2 = sigma2_new;
      ++k;
    }
  }

  // final clamp
  if (sigma2 < 0) sigma2 = 0.0;

  // Final parameter estimates
  Vi = 1.0 / (vardir + sigma2);

  // A = X.each_col() % Vi  (n x p) but we'll compute XtViX and XtViy
  // XtViX = Xt * (X.each_col() % Vi)  => p x p
  mat A_mat = X.each_col() % Vi;   // n x p
  XtViX = Xt * A_mat;              // p x p
  mat XtViy = Xt * (Vi % y);

  // Solve for beta: XtViX * beta = XtViy
  bool ok_final = inv_sympd(Q, XtViX); // reuse Q as inverse of XtViX
  if (!ok_final){
    Q = solve(XtViX, eye_p);
  }

  mat beta = Q * XtViy;
  vec stderr_beta = sqrt(Q.diag());
  vec zvalue = beta / stderr_beta;
  vec pvalue(zvalue.n_elem);

  for (uword i = 0; i < zvalue.n_elem; ++i) {
    pvalue(i) = 2.0 * R::pnorm5(-std::abs(zvalue(i)), 0.0, 1.0, 1, 0);
  }

  vec Xbeta = X * beta;
  vec resid = y - Xbeta;
  vec sigma_vardir = sigma2 + vardir;
  vec u = (sigma2 / sigma_vardir) % resid;
  vec eblup = Xbeta + u;

  const double loglike = -0.5 * accu(log(2.0 * M_PI * sigma_vardir) + (resid % resid) / sigma_vardir);
  const double AIC = -2.0 * loglike + 2.0 * (p + 1);
  const double BIC = -2.0 * loglike + (p + 1) * std::log((double) m);

  Rcpp::NumericVector goodness = Rcpp::NumericVector::create(
    Rcpp::Named("loglikelihood") = loglike,
    Rcpp::Named("AIC") = AIC,
    Rcpp::Named("BIC") = BIC
  );

  // MSE computations
  double SumAD2 = accu(square(Vi));
  double VarA = 2.0 / SumAD2;
  vec Bd = vardir / sigma_vardir;
  vec Bd2 = square(Bd);

  // g2
  vec h = sum((X * Q) % X, 1); // n x 1
  vec g2 = Bd2 % h;

  vec g1 = vardir % (1.0 - Bd);
  vec g3 = Bd2 % (VarA / sigma_vardir);
  vec mse(m);

  if (method == "ML") {
    double b = - trace(Q * XtViX) / std::max(SumAD2, 1e-30);
    mse = g1 + g2 + 2.0 * g3 - b * Bd2;
  } else {
    mse = g1 + g2 + 2.0 * g3;
  }


  // synthetic estimators
  if(adaNA){
    vec eblup_ns = Xns * beta;
    vec mse_ns = sum((Xns * Q) % Xns, 1); // n x 1
    mse_ns += sigma2;

    eblup_all.elem(idx_s)  = eblup;
    eblup_all.elem(idx_ns) = eblup_ns;
    mse_all.elem(idx_ns) = mse_ns;
    mse_all.elem(idx_s)  = mse;
  }else{
    eblup_all = eblup;
    mse_all = mse;
  }

  vec rse = sqrt(mse_all);
  // be careful dividing by eblup (avoid div by zero)
  for (uword i = 0; i < Xall.n_rows; ++i) {
    if (std::abs(eblup_all(i)) > 1e-16) rse(i) = rse(i) * 100.0 / eblup_all(i);
    else rse(i) = datum::nan; // not defined
  }

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

  // collect output
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


// [[Rcpp::export(.seblup_core)]]
List seblup_core(
    const arma::mat& Xall,
    const arma::vec& yall,
    const arma::vec& vardirall,
    const arma::mat& W,   // spatial weight matrix n x n
    std::string method = "REML",
    int maxiter = 100,
    double precision = 1e-4) {

  // hasil
  arma::vec eblup_all(Xall.n_rows);
  arma::vec mse_all(Xall.n_rows);

  // deklarasi global di fungsi
  arma::mat Xns, X;
  arma::vec y, vardir;
  arma::uvec idx_ns, idx_s;
  bool adaNA = yall.has_nan();

  if(adaNA){
    idx_ns = arma::find_nonfinite(yall);   // sama dengan is.na(y)
    idx_s  = arma::find_finite(yall);      // !is.na(y)

    // X untuk area non-sampled
    Xns = Xall.rows(idx_ns);
    // X dan y untuk sampled
    X  = Xall.rows(idx_s);
    y  = yall.elem(idx_s);
    vardir  = vardirall.elem(idx_s);
  }else{
    X  = Xall;
    y  = yall;
    vardir = vardirall;
  }

  const int m = X.n_rows;
  const int p = X.n_cols;

  // safety checks
  if ((int) vardir.n_elem != m) stop("Length of 'vardir' must equal nrow(X).");
  if ((int) y.n_elem != m) stop("Length of 'y' must equal nrow(X).");
  if (!(method == "ML" || method == "REML")) stop("method must be 'ML' or 'REML'.");

  // --- precomputations & preallocations ---
  mat Xt = X.t();
  mat yt = y.t();
  mat Wt = W.t();
  mat eye_p = eye<mat>(p, p);
  mat I = eye<mat>(m, m);

  // initial value for sigma2
  vec sigma2_u(maxiter + 1, fill::zeros);
  vec rho(maxiter + 1, fill::zeros);

  // initial values
  sigma2_u(0) = median(vardir);
  rho(0) = 0.5;

  vec s(2, fill::zeros);
  mat Idev(2, 2, fill::zeros);
  vec par_stim(2, fill::zeros);
  vec stime_fin(2, fill::zeros);

  double diff = precision + 1.0;
  int k = 0;

  // Pre-allocate all temporaries reused in loop
  mat XtVi(p, m);
  mat XtViX(p, p);
  mat Q(p, p);
  mat temp_pm(p, m);
  vec Py(m);
  mat P(m, m);
  mat Vi(m, m);
  mat derSigma(m, m);
  mat WtW = Wt * W;
  mat WpWt = W + Wt;


  if (method == "ML") {
    // ML
    while ((diff > precision) && (k < maxiter)) {
      ++k;
      // seblup --------
      mat A = (I - rho(k-1) * Wt) * (I - rho(k-1) * W);
      bool ok_inverse = inv_sympd(derSigma, A);
      if (!ok_inverse) derSigma = pinv(A);

      mat derRho = 2 * rho(k-1) * WtW - WpWt;
      mat derVRho = -sigma2_u(k-1) * (derSigma * derRho * derSigma);

      mat V = sigma2_u(k-1) * derSigma + diagmat(vardir);
      //  --------------


      bool okV = inv_sympd(Vi, V);
      if (!okV) Vi = pinv(V);

      mat XtVi = X.t() * Vi;     // p x n
      mat XtViX = XtVi * X;      // p x p

      // Q = inv_sympd(XtViX) with fallback to solve
      bool ok = inv_sympd(Q, XtViX); // fast Cholesky-based inverse if SPD
      if (!ok) {
        Q = solve(XtViX, eye_p);
      }

      // P = diag(Vi) - t(XtVi) * Q * XtVi
      // t(XtVi) is m x p, XtVi is p x m
      // compute temp_pp = Q * XtVi  (p x m) but reuse variables:
      temp_pm = Q * XtVi;               // p x m
      // compute t(XtVi) * temp_pm  => (m x p) * (p x m) = m x m
      P = Vi;                 // m x m
      P -= trans(XtVi) * temp_pm;      // m x m
      Py = P * y;

      // seblup ---------
      mat PD = P * derSigma;
      mat PR = P * derVRho;

      mat VID = Vi * derSigma; // bedanya ML dan REML
      mat VIR = Vi * derVRho;

      s(0) = -0.5 * trace(VID) + 0.5 * as_scalar(y.t() * PD * Py);
      s(1) = -0.5 * trace(VIR) + 0.5 * as_scalar(y.t() * PR * Py);

      Idev(0,0) = 0.5 * trace(VID * VID);
      Idev(0,1) = 0.5 * trace(VID * VIR);
      Idev(1,0) = Idev(0,1);
      Idev(1,1) = 0.5 * trace(VIR * VIR);

      par_stim(0) = sigma2_u(k-1);
      par_stim(1) = rho(k-1);

      stime_fin = par_stim + solve(Idev, s);

      // bound rho between (-1,1)
      if (stime_fin(1) <= -1) stime_fin(1) = -0.999;
      if (stime_fin(1) >= 1)  stime_fin(1) = 0.999;

      sigma2_u(k) = stime_fin(0);
      rho(k) = stime_fin(1);

      diff = max(abs(stime_fin - par_stim) / abs(par_stim + 1e-12));
      //  ---------------
    }
  } else {
    // REML
    while ((diff > precision) && (k < maxiter)) {
      ++k;
      // seblup --------
      mat A = (I - rho(k-1) * Wt) * (I - rho(k-1) * W);
      bool ok_inverse = inv_sympd(derSigma, A);
      if (!ok_inverse) derSigma = pinv(A);

      mat derRho = 2 * rho(k-1) * WtW - WpWt;
      mat derVRho = -sigma2_u(k-1) * (derSigma * derRho * derSigma);

      mat V = sigma2_u(k-1) * derSigma + diagmat(vardir);
      //  --------------

      bool okV = inv_sympd(Vi, V);
      if (!okV) Vi = pinv(V);

      mat XtVi = X.t() * Vi;     // p x n
      mat XtViX = XtVi * X;      // p x p

      bool ok = inv_sympd(Q, XtViX);
      if (!ok) Q = solve(XtViX, eye_p);

      temp_pm = Q * XtVi;             // p x m
      P = Vi;
      P -= trans(XtVi) * temp_pm;
      Py = P * y;

      // seblup ---------
      mat PD = P * derSigma;
      mat PR = P * derVRho;

      s(0) = -0.5 * trace(PD) + 0.5 * as_scalar(y.t() * PD * Py);
      s(1) = -0.5 * trace(PR) + 0.5 * as_scalar(y.t() * PR * Py);

      Idev(0,0) = 0.5 * trace(PD * PD);
      Idev(0,1) = 0.5 * trace(PD * PR);
      Idev(1,0) = Idev(0,1);
      Idev(1,1) = 0.5 * trace(PR * PR);

      par_stim(0) = sigma2_u(k-1);
      par_stim(1) = rho(k-1);

      stime_fin = par_stim + solve(Idev, s);

      // bound rho between (-1,1)
      if (stime_fin(1) <= -1) stime_fin(1) = -0.999;
      if (stime_fin(1) >= 1)  stime_fin(1) = 0.999;

      sigma2_u(k) = stime_fin(0);
      rho(k) = stime_fin(1);

      diff = max(abs(stime_fin - par_stim) / abs(par_stim + 1e-12));
      //  ---------------
    }
  }

  // final rho and sigma2u ----
  double rho_fix = rho(k);
  if(rho_fix == -0.999){
    rho_fix = -1;
  }else if(rho_fix == 0.999){
    rho_fix = 1;
  }

  double sigma2 = sigma2_u(k);
  if (sigma2 < 0.0) sigma2 = 0.0;

  // --- Final parameter estimates ---
  mat A = (I - rho_fix * Wt) * (I - rho_fix * W);
  bool ok_inverse = inv_sympd(derSigma, A);
  if (!ok_inverse) derSigma = pinv(A);

  mat G = sigma2 * derSigma;
  mat V = G + diagmat(vardir);

  bool okV = inv_sympd(Vi, V);
  if (!okV) Vi = pinv(V);

  // A = X.each_col() % Vi  (n x p) but we'll compute XtViX and XtViy
  // XtViX = Xt * (X.each_col() % Vi)  => p x p
  XtVi = X.t() * Vi;     // p x n
  XtViX = XtVi * X;      // p x p
  mat XtViy = XtVi * y;


  // Solve for beta: XtViX * beta = XtViy
  bool ok_final = inv_sympd(Q, XtViX); // reuse Q as inverse of XtViX
  if (!ok_final){
    Q = solve(XtViX, eye_p);
  }

  mat beta = Q * XtViy;
  vec stderr_beta = sqrt(Q.diag());
  vec zvalue = beta / stderr_beta;
  vec pvalue(zvalue.n_elem);

  for (uword i = 0; i < zvalue.n_elem; ++i) {
    pvalue(i) = 2.0 * R::pnorm5(-std::abs(zvalue(i)), 0.0, 1.0, 1, 0);
  }

  vec Xbeta = X * beta;
  vec resid = y - Xbeta;
  mat GVi = G * Vi;
  vec eblup = Xbeta + GVi * resid;


  // loglike
  double sign;
  double logdetV;
  log_det(logdetV, sign, V);
  double quadform = dot(resid, Vi * resid);
  const double loglike = -0.5 * ( m * std::log(2.0 * M_PI) + logdetV + quadform );
  // seblup p + 2
  const double AIC = -2.0 * loglike + 2.0 * (p + 2);
  const double BIC = -2.0 * loglike + (p + 2) * std::log((double) m);

  Rcpp::NumericVector goodness = Rcpp::NumericVector::create(
    Rcpp::Named("loglikelihood") = loglike,
    Rcpp::Named("AIC") = AIC,
    Rcpp::Named("BIC") = BIC
  );


  // --- MSE computations ---
  mat Ga = G - GVi * G;
  mat Gb = GVi * X;

  // komponen mse
  vec g1d = Ga.diag();
  vec g2d(m, fill::zeros);
  vec g3d(m, fill::zeros);
  vec g4d(m, fill::zeros);

  // loop kecil per i
  mat R = X - Gb;                 // m × p
  g2d = sum((R * Q) % R, 1);  // row-wise quadratic form

  // turunan rho
  mat derRho = 2 * rho_fix * WtW - WpWt;
  mat rhosigma = derRho * derSigma;
  mat der3 = derSigma * rhosigma;
  mat Amat = - sigma2 * der3;

  mat QXtVi = Q * XtVi;
  P = Vi - XtVi.t() * QXtVi;
  mat PD = P * derSigma;
  mat PR = P * Amat;

  // informasi deviance
  Idev(0,0) = 0.5 * trace(PD * PD);
  Idev(0,1) = 0.5 * trace(PD * PR);
  Idev(1,0) = Idev(0,1);
  Idev(1,1) = 0.5 * trace(PR * PR);

  mat Idevi = inv_sympd(Idev);

  mat ViD = Vi * derSigma;
  mat ViR = Vi * Amat;

  mat l1 = ViD - sigma2 * ViD * ViD;
  mat l2 = ViR - sigma2 * ViR * ViD;
  mat l1t = l1.t();
  mat l2t = l2.t();

  // hitung g3d
  for (int i = 0; i < m; i++) {
    mat L(2,m,fill::zeros);
    L.row(0) = l1t.row(i);
    L.row(1) = l2t.row(i);
    g3d(i) = trace(L * V * L.t() * Idevi);
  }

  vec mse2d_aux = g1d + g2d + 2 * g3d;

  // g4d
  mat psi = diagmat(vardir);
  mat D12aux = -der3;
  mat D22aux = 2 * sigma2 * der3 * rhosigma
  - 2 * sigma2 * derSigma * WtW * derSigma;

  mat psiVi = psi * Vi;
  mat Vipsi = Vi * psi;
  mat D = (psiVi * D12aux * Vipsi) * (Idevi(0,1) + Idevi(1,0))
    + (psiVi * D22aux * Vipsi) * Idevi(1,1);

  for (int i = 0; i < m; i++) {
    g4d(i) = 0.5 * D(i,i);
  }

  vec mse2d = mse2d_aux - g4d;

  // Koreksi ML
  if (method == "ML") {
    mat ViX   = Vi * X;
    double h1 = -trace(QXtVi * derSigma * ViX);
    double h2 = -trace(QXtVi * Amat * ViX);

    vec h = {h1, h2};
    vec bML = Idevi * h / 2.0;
    rowvec tbML = bML.t();

    mat GViCi   = GVi * derSigma;
    mat GViAmat = GVi * Amat;

    mat dg1_dA = derSigma - 2 * GViCi + sigma2 * GViCi * ViD;
    mat dg1_dp = Amat - 2 * GViAmat + sigma2 * GViAmat * ViD;

    vec bMLgradg1(m, fill::zeros);
    for (int i = 0; i < m; i++) {
      vec gradg1d = { dg1_dA(i,i), dg1_dp(i,i) };
      bMLgradg1(i) = as_scalar(tbML * gradg1d);
    }

    mse2d -= bMLgradg1;
  }

  // synthetic estimators
  if(adaNA){
    vec eblup_ns = Xns * beta;
    eblup_all.elem(idx_s)  = eblup;
    eblup_all.elem(idx_ns) = eblup_ns;

    mse_all.elem(idx_s)  = mse2d;
  }else{
    eblup_all = eblup;
    mse_all = mse2d;
  }

  vec rse = sqrt(mse_all);
  // be careful dividing by eblup (avoid div by zero)
  for (uword i = 0; i < Xall.n_rows; ++i) {
    if (std::abs(eblup_all(i)) > 1e-16) rse(i) = rse(i) * 100.0 / eblup_all(i);
    else rse(i) = datum::nan; // not defined
  }


  DataFrame df_coef = DataFrame::create(
    _["beta"] = beta,
    _["stderr_beta"] = stderr_beta,
    _["zvalue"] = zvalue,
    _["pvalue"] = pvalue
  );

  DataFrame df_eblup = DataFrame::create(
    _["y"] = yall,
    _["vardir"] = vardirall,
    _["eblup"] = eblup_all,
    _["mse"] = mse_all,
    _["rse"] = rse
  );

  // collect output
  List out = List::create(
    _["sigma2_u"] = sigma2,
    _["rho"] = rho_fix,
    _["estcoef"] = df_coef,
    _["df_eblup"] = df_eblup,
    _["goodness"] = goodness,
    _["n_iter"] = k,
    _["convergence"] = (k < maxiter)
  );

  return out;
}


