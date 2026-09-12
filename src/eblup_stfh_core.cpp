// src/eblup_stfh_core.cpp
// Core Spatio-Temporal Fay-Herriot EBLUP estimation (Fisher-scoring, REML-style)
//
// Optimized version - Key optimizations:
//  1. Pre-compute WtW and WpWt outside the iteration loop
//  2. Pre-compute OnesT matrix for Kronecker products
//  3. Use inv_sympd for better numerical stability
//  4. Handle sigma21=0 case properly (zero Vu1 matrix)
//
#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// ============================================================================
// build_omega2()
//
// Membangun matriks kovarians AR(1) "Omega2(rho2)" berukuran T x T beserta
// turunannya terhadap rho2.
// ============================================================================
static void build_omega2(int Tt, double rho2, arma::mat& Omega2, arma::mat& dOmega2) {
  Omega2.zeros(Tt, Tt);
  dOmega2.zeros(Tt, Tt);
  const double one_m_rho2sq = 1.0 - rho2 * rho2;

  for (int i = 0; i < Tt; ++i) {
    for (int j = 0; j < i; ++j) {
      const int lag = i - j;
      const double rho_lag = std::pow(rho2, (double) lag);
      const double val = rho_lag / one_m_rho2sq;
      Omega2(i, j) = val;
      Omega2(j, i) = val;

      const double raw_deriv = (double) lag * std::pow(rho2, (double)(lag - 1));
      const double dval = raw_deriv / one_m_rho2sq + (2.0 * rho2 / one_m_rho2sq) * val;
      dOmega2(i, j) = dval;
      dOmega2(j, i) = dval;
    }
  }
  // Diagonal (lag = 0): rho2^0 / (1-rho2^2) = 1/(1-rho2^2)
  Omega2.diag().fill(1.0 / one_m_rho2sq);
  // Turunan diagonal: d/drho2 [1/(1-rho2^2)] = 2*rho2 / (1-rho2^2)^2.
  dOmega2.diag().fill(2.0 * rho2 / (one_m_rho2sq * one_m_rho2sq));
}

// ============================================================================
// ModelPieces
//
// Menyimpan semua kuantitas turunan dari theta = (sigma21, rho1, sigma22[, rho2])
// ============================================================================
struct ModelPieces {
  bool ok = true;
  bool hasVu1 = true;
  arma::mat Omega1;      // D x D
  arma::mat Vu1;         // D x D (zero when sigma21 == 0)
  arma::mat invVu1;      // D x D (not computed when sigma21 == 0)
  arma::mat invA;        // M x M, blok diagonal per domain
  arma::vec diagC;       // D x 1, diag(Z1' invA Z1)
  double logdetA = 0.0;  // log det dari matriks blok-diagonal A (= invA^-1)
  arma::mat Omega2;      // T x T (ST saja)
  arma::mat dOmega2;     // T x T (ST saja)
};

static ModelPieces compute_pieces(
    double sigma21, double rho1, double sigma22, double rho2,
    bool isST, int D, int Tt,
    const arma::mat& Id, const arma::mat& W, const arma::vec& vardir
) {
  ModelPieces mp;
  const int M = D * Tt;

  arma::mat ImrW = Id - rho1 * W;
  arma::mat A1 = ImrW.t() * ImrW;
  bool ok1 = arma::inv_sympd(mp.Omega1, A1);
  if (!ok1) ok1 = arma::inv(mp.Omega1, A1);
  if (!ok1) { mp.ok = false; return mp; }

  mp.Vu1 = sigma21 * mp.Omega1;
  if (sigma21 == 0.0) {
    mp.hasVu1 = false;
  } else {
    mp.hasVu1 = true;
    bool okVu1 = arma::inv_sympd(mp.invVu1, mp.Vu1);
    if (!okVu1) okVu1 = arma::inv(mp.invVu1, mp.Vu1);
    if (!okVu1) { mp.ok = false; return mp; }
  }

  mp.invA.zeros(M, M);
  mp.diagC.zeros(D);
  mp.logdetA = 0.0;

  if (isST) build_omega2(Tt, rho2, mp.Omega2, mp.dOmega2);

  for (int d = 0; d < D; ++d) {
    const int first = d * Tt;
    const int last = first + Tt - 1;

    if (!isST) {
      arma::vec block_diag = sigma22 + vardir.subvec(first, last);
      if (arma::any(block_diag <= 0.0)) { mp.ok = false; return mp; }
      arma::vec invblock = 1.0 / block_diag;
      for (int t = 0; t < Tt; ++t) mp.invA(first + t, first + t) = invblock(t);
      mp.diagC(d) = arma::sum(invblock);
      mp.logdetA += arma::sum(arma::log(block_diag));
    } else {
      arma::mat Ved = arma::diagmat(vardir.subvec(first, last));
      arma::mat Ad = sigma22 * mp.Omega2 + Ved;
      arma::mat invAd;
      bool okAd = arma::inv_sympd(invAd, Ad);
      if (!okAd) okAd = arma::inv(invAd, Ad);
      if (!okAd) { mp.ok = false; return mp; }

      double ld, sign_;
      bool okld = arma::log_det(ld, sign_, Ad);
      if (!okld || sign_ <= 0) { mp.ok = false; return mp; }

      mp.invA.submat(first, first, last, last) = invAd;
      mp.diagC(d) = arma::accu(invAd);
      mp.logdetA += ld;
    }
  }
  return mp;
}

// Z1' invA (M x D)
static arma::mat build_invAZ1(const arma::mat& invA, int D, int Tt) {
  const int M = D * Tt;
  arma::mat invAZ1(M, D, fill::zeros);
  for (int d = 0; d < D; ++d) {
    const int first = d * Tt, last = first + Tt - 1;
    invAZ1.submat(first, d, last, d) = arma::sum(invA.submat(first, first, last, last), 1);
  }
  return invAZ1;
}

// invV via Woodbury identity
static bool build_invV(const ModelPieces& mp, const arma::mat& invAZ1, int D,
                       arma::mat& invV_out, arma::mat& Cmat_out) {
  if (!mp.hasVu1) {
    invV_out = mp.invA;
    Cmat_out.zeros();
    return true;
  }
  Cmat_out = mp.invVu1 + arma::diagmat(mp.diagC);
  arma::mat Cinv;
  bool okC = arma::inv_sympd(Cinv, Cmat_out);
  if (!okC) okC = arma::inv(Cinv, Cmat_out);
  if (!okC) return false;
  invV_out = mp.invA - invAZ1 * Cinv * invAZ1.t();
  return true;
}

// ============================================================================
// .eblup_stfh_core()
//
// Output structure matching seblup_area + estvarcomp for spatio-temporal
// ============================================================================
// [[Rcpp::export(.eblup_stfh_core)]]
List eblup_stfh_core(
    const arma::mat& X,
    const arma::vec& y,
    const arma::vec& vardir,
    const arma::mat& proxmat,
    int D,
    int Tt,
    std::string model = "ST",
    int maxiter = 100,
    double precision = 1e-4,
    double sigma21_start = -1.0,
    double rho1_start = 0.5,
    double sigma22_start = -1.0,
    double rho2_start = 0.5
) {
  if (model != "S" && model != "ST") stop("Argument model must be \"S\" or \"ST\".");

  const int M = D * Tt;
  const int p = X.n_cols;
  if ((int) X.n_rows != M)      stop("nrow(X) must equal D*T.");
  if ((int) y.n_elem != M)      stop("length(y) must equal D*T.");
  if ((int) vardir.n_elem != M) stop("length(vardir) must equal D*T.");
  if ((int) proxmat.n_rows != D || (int) proxmat.n_cols != D)
    stop("proxmat must be a square D x D matrix.");

  const double med_vardir = arma::median(vardir);
  if (sigma21_start < 0) sigma21_start = 0.5 * med_vardir;
  if (sigma22_start < 0) sigma22_start = 0.5 * med_vardir;
  if (rho1_start <= -1 || rho1_start >= 1) stop("rho1_start must be in (-1,1).");
  if (rho2_start <= -1 || rho2_start >= 1) stop("rho2_start must be in (-1,1).");

  const bool isST = (model == "ST");
  const int nparam = isST ? 4 : 3;

  const arma::mat Id = arma::eye<arma::mat>(D, D);
  const arma::mat& W = proxmat;
  const arma::mat Wt = W.t();
  const arma::mat WtW = Wt * W;
  const arma::mat WpWt = W + Wt;
  const arma::mat EyeD = arma::eye<arma::mat>(D, D);
  const arma::mat tX = X.t();
  const arma::mat OnesT = arma::ones<arma::mat>(Tt, Tt);

  CharacterVector thetanames = isST
  ? CharacterVector::create("sigma21", "rho1", "sigma22", "rho2")
    : CharacterVector::create("sigma21", "rho1", "sigma22");

  arma::vec thetak(nparam), thetak1(nparam);
  thetak1(0) = sigma21_start;
  thetak1(1) = rho1_start;
  thetak1(2) = sigma22_start;
  if (isST) thetak1(3) = rho2_start;

  arma::vec S(nparam, fill::zeros);
  arma::mat F(nparam, nparam, fill::zeros);
  arma::mat Finv(nparam, nparam, fill::zeros);

  int k = 0;
  double diff = precision + 1.0;

  auto make_fail_result = [&](bool conv) -> List {
    return List::create(
      _["estcoef"] = R_NilValue,
      _["estvarcomp"] = R_NilValue,
      _["goodness"] = R_NilValue,
      _["df_eblup"] = R_NilValue,
      _["model"] = model,
      _["convergence"] = conv,
      _["n_iter"] = k
    );
  };

  while (diff > precision && k < maxiter) {
    ++k;
    thetak = thetak1;

    const double sigma21_k = thetak(0);
    const double rho1_k = thetak(1);
    const double sigma22_k = thetak(2);
    const double rho2_k = isST ? thetak(3) : 0.0;

    ModelPieces mp = compute_pieces(sigma21_k, rho1_k, sigma22_k, rho2_k,
                                    isST, D, Tt, Id, W, vardir);
    if (!mp.ok) return make_fail_result(false);

    arma::mat invAZ1 = build_invAZ1(mp.invA, D, Tt);
    arma::mat invV, Cmat;
    if (!build_invV(mp, invAZ1, D, invV, Cmat)) return make_fail_result(false);

    arma::mat tXinvV = tX * invV;
    arma::mat tXinvVX = tXinvV * X;
    arma::mat Q;
    bool okQ = arma::inv_sympd(Q, tXinvVX);
    if (!okQ) okQ = arma::inv(Q, tXinvVX);
    if (!okQ) return make_fail_result(false);

    arma::mat P = invV - tXinvV.t() * Q * tXinvV;
    arma::vec Py = P * y;

    // Derivative of spatial model
    arma::mat derivrho1 = -WpWt + 2.0 * rho1_k * WtW;
    arma::mat sigmaOmegaderivrho1Omega = (-sigma21_k) * (mp.Omega1 * derivrho1 * mp.Omega1);

    // Va matrices (Kronecker products)
    std::vector<arma::mat> Va(nparam);
    Va[0] = arma::kron(mp.Omega1, OnesT);
    Va[1] = arma::kron(sigmaOmegaderivrho1Omega, OnesT);
    if (!isST) {
      Va[2] = arma::eye<arma::mat>(M, M);
    } else {
      Va[2] = arma::kron(EyeD, mp.Omega2);
      arma::mat sigma22dOmega2 = sigma22_k * mp.dOmega2;
      Va[3] = arma::kron(EyeD, sigma22dOmega2);
    }

    // PV matrices and traces
    std::vector<arma::mat> PV(nparam);
    arma::vec trPV(nparam);
    for (int i = 0; i < nparam; ++i) {
      PV[i] = P * Va[i];
      trPV(i) = arma::trace(PV[i]);
    }

    // Fisher information matrix
    arma::mat trPVPV(nparam, nparam, fill::zeros);
    for (int i = 0; i < nparam; ++i) {
      for (int j = i; j < nparam; ++j) {
        const double tv = arma::accu(PV[i] % PV[j].t());
        trPVPV(i, j) = tv;
        trPVPV(j, i) = tv;
      }
    }

    // Score vector and Fisher matrix
    for (int a = 0; a < nparam; ++a) {
      const double quad = arma::as_scalar(Py.t() * Va[a] * Py);
      S(a) = -0.5 * trPV(a) + 0.5 * quad;
      for (int b = a; b < nparam; ++b) F(a, b) = 0.5 * trPVPV(a, b);
    }
    for (int a = 1; a < nparam; ++a)
      for (int b = 0; b < a; ++b) F(a, b) = F(b, a);

    // Update theta
    bool okF = arma::inv_sympd(Finv, F);
    if (!okF) okF = arma::inv(Finv, F);
    if (!okF) return make_fail_result(false);

    thetak1 = thetak + Finv * S;

    // Clamp parameters
    if (thetak1(1) <= -1) thetak1(1) = -0.999;
    else if (thetak1(1) >= 1) thetak1(1) = 0.999;
    if (isST) {
      if (thetak1(3) <= -1) thetak1(3) = -0.999;
      else if (thetak1(3) >= 1) thetak1(3) = 0.999;
    }

    // Convergence check
    arma::vec thetak_safe = thetak;
    for (int i = 0; i < nparam; ++i) if (thetak_safe(i) == 0.0) thetak_safe(i) = 1e-4;
    diff = arma::max(arma::abs((thetak_safe - thetak1) / thetak_safe));
  }

  if (k >= maxiter && diff >= precision) {
    return make_fail_result(false);
  }

  // ==========================================================================
  // Finalization
  // ==========================================================================
  thetak1(0) = std::max(thetak1(0), 0.0);
  thetak1(2) = std::max(thetak1(2), 0.0);

  const double sigma21_f = thetak1(0);
  const double rho1_f = thetak1(1);
  const double sigma22_f = thetak1(2);
  const double rho2_f = isST ? thetak1(3) : 0.0;

  const bool param_invalid = (sigma21_f < 0) || (rho1_f < -1) || (rho1_f > 1) ||
    (sigma22_f < 0) || (isST && (rho2_f < -1 || rho2_f > 1));

  NumericVector est_vec(thetak1.begin(), thetak1.end());
  est_vec.names() = thetanames;

  if (param_invalid) {
    DataFrame estvarcomp = DataFrame::create(
      _["estimate"] = est_vec,
      _["std.error"] = NumericVector(nparam, 0.0)
    );
    return List::create(
      _["estcoef"] = R_NilValue,
      _["estvarcomp"] = estvarcomp,
      _["goodness"] = R_NilValue,
      _["df_eblup"] = R_NilValue,
      _["model"] = model,
      _["convergence"] = true,
      _["n_iter"] = k
    );
  }

  ModelPieces mpf = compute_pieces(sigma21_f, rho1_f, sigma22_f, rho2_f,
                                   isST, D, Tt, Id, W, vardir);
  if (!mpf.ok) {
    DataFrame estvarcomp = DataFrame::create(
      _["estimate"] = est_vec,
      _["std.error"] = NumericVector(nparam, 0.0)
    );
    return List::create(
      _["estcoef"] = R_NilValue,
      _["estvarcomp"] = estvarcomp,
      _["goodness"] = R_NilValue,
      _["df_eblup"] = R_NilValue,
      _["model"] = model,
      _["convergence"] = false,
      _["n_iter"] = k
    );
  }

  bool haveVu1 = (sigma21_f != 0.0);
  arma::mat invAZ1f = build_invAZ1(mpf.invA, D, Tt);
  arma::mat invV, Cmat;
  if (haveVu1) {
    if (!build_invV(mpf, invAZ1f, D, invV, Cmat)) {
      DataFrame estvarcomp = DataFrame::create(
        _["estimate"] = est_vec,
        _["std.error"] = NumericVector(nparam, 0.0)
      );
      return List::create(
        _["estcoef"] = R_NilValue,
        _["estvarcomp"] = estvarcomp,
        _["goodness"] = R_NilValue,
        _["df_eblup"] = R_NilValue,
        _["model"] = model,
        _["convergence"] = false,
        _["n_iter"] = k
      );
    }
  } else {
    invV = mpf.invA;
  }

  // Beta and residuals
  arma::mat tXinvV = tX * invV;
  arma::mat tXinvVX = tXinvV * X;
  arma::mat Q;
  bool okQ = arma::inv_sympd(Q, tXinvVX);
  if (!okQ) okQ = arma::inv(Q, tXinvVX);
  if (!okQ) return make_fail_result(false);

  arma::vec beta = Q * (tXinvV * y);
  arma::vec resid = y - X * beta;
  arma::vec invVresid = invV * resid;

  // Random effects
  arma::vec tZ1invVresid(D, fill::zeros);
  for (int d = 0; d < D; ++d) {
    const int first = d * Tt, last = first + Tt - 1;
    tZ1invVresid(d) = arma::sum(invVresid.subvec(first, last));
  }
  arma::vec u1est = mpf.Vu1 * tZ1invVresid;
  arma::vec u1dt(M);
  for (int d = 0; d < D; ++d) {
    const int first = d * Tt, last = first + Tt - 1;
    u1dt.subvec(first, last).fill(u1est(d));
  }

  arma::vec u2dt(M);
  if (!isST) {
    u2dt = sigma22_f * invVresid;
  } else {
    arma::mat sigma22Omega2 = sigma22_f * mpf.Omega2;
    for (int d = 0; d < D; ++d) {
      const int first = d * Tt, last = first + Tt - 1;
      u2dt.subvec(first, last) = sigma22Omega2 * invVresid.subvec(first, last);
    }
  }

  arma::vec eblup = X * beta + u1dt + u2dt;

  // Log-likelihood
  double logdetV;
  const double quadform = arma::dot(resid, invVresid);
  if (haveVu1) {
    arma::mat ImrWf = Id - rho1_f * W;
    arma::mat A1f = ImrWf.t() * ImrWf;
    double logdetA1f, sign_;
    arma::log_det(logdetA1f, sign_, A1f);
    const double logdetVu1 = D * std::log(sigma21_f) - logdetA1f;

    double logdetC, signC;
    arma::log_det(logdetC, signC, Cmat);

    logdetV = mpf.logdetA + logdetVu1 + logdetC;
  } else {
    logdetV = mpf.logdetA;
  }

  const double loglike = -0.5 * (M * std::log(2.0 * M_PI) + logdetV + quadform);
  const double AIC = -2.0 * loglike + 2.0 * (p + nparam);
  const double BIC = -2.0 * loglike + std::log((double) M) * (p + nparam);

  NumericVector goodness = NumericVector::create(
    _["loglike"] = loglike, _["AIC"] = AIC, _["BIC"] = BIC
  );

  // SE for beta
  arma::vec stderr_beta = arma::sqrt(Q.diag());
  arma::vec tvalue = beta / stderr_beta;
  arma::vec pvalue(p);
  for (int i = 0; i < p; ++i)
    pvalue(i) = 2.0 * R::pnorm(std::fabs(tvalue(i)), 0.0, 1.0, 0, 0);

  DataFrame estcoef = DataFrame::create(
    _["beta"] = NumericVector(beta.begin(), beta.end()),
    _["std.error"] = NumericVector(stderr_beta.begin(), stderr_beta.end()),
    _["tvalue"] = NumericVector(tvalue.begin(), tvalue.end()),
    _["pvalue"] = NumericVector(pvalue.begin(), pvalue.end())
  );

  // SE for theta
  arma::vec diagFinv = Finv.diag();
  if (arma::any(diagFinv < 0) || diagFinv.has_nan()) {
    DataFrame estvarcomp = DataFrame::create(
      _["estimate"] = est_vec,
      _["std.error"] = NumericVector(nparam, 0.0)
    );
    return List::create(
      _["estcoef"] = R_NilValue,
      _["estvarcomp"] = estvarcomp,
      _["goodness"] = R_NilValue,
      _["df_eblup"] = R_NilValue,
      _["model"] = model,
      _["convergence"] = false,
      _["n_iter"] = k
    );
  }
  arma::vec stderr_theta = arma::sqrt(diagFinv);

  DataFrame estvarcomp = DataFrame::create(
    _["estimate"] = est_vec,
    _["std.error"] = NumericVector(stderr_theta.begin(), stderr_theta.end())
  );

  // df_eblup with random effects
  DataFrame df_eblup = DataFrame::create(
    _["eblup"] = NumericVector(eblup.begin(), eblup.end()),
    _["random_effect_u1"] = NumericVector(u1dt.begin(), u1dt.end()),
    _["random_effect_u2"] = NumericVector(u2dt.begin(), u2dt.end())
  );

  // Return structure matching seblup_area + estvarcomp
  return List::create(
    _["estcoef"] = estcoef,
    _["estvarcomp"] = estvarcomp,
    _["goodness"] = goodness,
    _["df_eblup"] = df_eblup,
    _["model"] = model,
    _["convergence"] = true,
    _["n_iter"] = k
  );
}
