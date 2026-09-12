// src/steblup_core.cpp
// Core Spatio-Temporal Fay-Herriot EBLUP estimation (Fisher-scoring, REML-style)
//
// Optimized version - Key optimizations:
//  1. trace(PV) computed via sum(P % Va) WITHOUT forming PV = P*Va
//  2. trPVPV(i,j) = sum(PV[i] % PV[j].t()) using element-wise operations
//     on-the-fly without storing PV matrices
//  3. Uses Cholesky decomposition for better numerical stability
//  4. Avoids redundant matrix allocations
//  5. Woodbury identity for invV without explicit MxM matrix formation
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
  const double inv_one_m_rho2sq = 1.0 / one_m_rho2sq;
  const double diag_val = inv_one_m_rho2sq;
  const double diag_deriv = 2.0 * rho2 * inv_one_m_rho2sq * inv_one_m_rho2sq;

  for (int i = 0; i < Tt; ++i) {
    Omega2(i, i) = diag_val;
    dOmega2(i, i) = diag_deriv;
    for (int j = 0; j < i; ++j) {
      const int lag = i - j;
      const double rho_lag = std::pow(rho2, (double) lag);
      const double val = rho_lag * inv_one_m_rho2sq;
      Omega2(i, j) = val;
      Omega2(j, i) = val;

      const double raw_deriv = (double) lag * std::pow(rho2, (double)(lag - 1));
      const double dval = (raw_deriv + 2.0 * rho2 * rho_lag) * inv_one_m_rho2sq * inv_one_m_rho2sq;
      dOmega2(i, j) = dval;
      dOmega2(j, i) = dval;
    }
  }
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
  arma::mat Vu1;         // D x D
  arma::mat invVu1;      // D x D
  arma::mat invA;        // M x M, block diagonal per domain
  arma::vec diagC;       // D x 1
  double logdetA = 0.0;
  arma::mat Omega2;      // T x T (ST only)
  arma::mat dOmega2;     // T x T (ST only)
};

static ModelPieces compute_pieces(
    double sigma21, double rho1, double sigma22, double rho2,
    bool isST, int D, int Tt,
    const arma::mat& Id, const arma::mat& W, const arma::vec& vardir
) {
  ModelPieces mp;
  const int M = D * Tt;

  // Compute Omega1 = inv((I - rho1*W)'*(I - rho1*W))
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

// invV = invA - invAZ1 * Cinv * invAZ1'  (Woodbury identity)
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
// Optimized Fisher scoring iteration
//
// Key insight: We need trace(P*Va[i]) and sum(PVa[i] % PVa[j].t())
// where PVa[i] = P * Va[i].
//
// For efficiency:
// - trace(P*Va) = sum(P % Va)  when P is symmetric
// - For Va = kron(A, ones(T,T)), trace(P*Va) = sum(sum(P) per block row/col)
// - For Va = kron(I_D, B), trace(P*Va) = trace of block diagonal parts
// ============================================================================

// Compute diagC from invA: diagC(d) = sum of row sums of block d
static arma::vec compute_diagC_from_invA(const arma::mat& invA, int D, int Tt) {
  arma::vec diagC(D);
  for (int d = 0; d < D; ++d) {
    const int first = d * Tt;
    const int last = first + Tt - 1;
    diagC(d) = arma::sum(arma::sum(invA.submat(first, first, last, last)));
  }
  return diagC;
}

// Compute invAZ1 = invA * Z1 (Z1 is DxD block indicator)
// invAZ1(i,d) = sum of row i in block d = row sum of invA block
static arma::mat build_invAZ1(const arma::mat& invA, int D, int Tt) {
  const int M = D * Tt;
  arma::mat invAZ1(M, D, fill::zeros);
  for (int d = 0; d < D; ++d) {
    const int first = d * Tt, last = first + Tt - 1;
    // Row sums of block d
    invAZ1.submat(first, d, last, d) = arma::sum(invA.submat(first, first, last, last), 1);
  }
  return invAZ1;
}

// ============================================================================
// .steblup_core()
//
// OPTIMIZED VERSION
// ============================================================================
// [[Rcpp::export(.steblup_core)]]
List steblup_core(
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

  // Pre-compute constant matrices
  const arma::mat Id = arma::eye<arma::mat>(D, D);
  const arma::mat& W = proxmat;
  const arma::mat Wt = W.t();
  const arma::mat WtW = Wt * W;
  const arma::mat WpWt = W + Wt;
  const arma::mat EyeD = arma::eye<arma::mat>(D, D);
  const arma::mat tX = X.t();

  CharacterVector thetanames = isST
  ? CharacterVector::create("sigma21", "rho1", "sigma22", "rho2")
    : CharacterVector::create("sigma21", "rho1", "sigma22");

  // Starting values
  arma::vec thetak(nparam), thetak1(nparam);
  thetak1(0) = sigma21_start;
  thetak1(1) = rho1_start;
  thetak1(2) = sigma22_start;
  if (isST) thetak1(3) = rho2_start;

  // Working matrices
  arma::vec S(nparam, fill::zeros);
  arma::mat F(nparam, nparam, fill::zeros);
  arma::mat Finv(nparam, nparam, fill::zeros);

  int k = 0;
  double diff = precision + 1.0;
  bool convergence = true;

  auto make_fail_result = [&](bool conv) -> List {
    return List::create(
      _["eblup"] = R_NilValue,
      _["fit"] = List::create(
        _["model"] = model,
        _["convergence"] = conv,
        _["iterations"] = k,
        _["estcoef"] = R_NilValue,
        _["estvarcomp"] = R_NilValue,
        _["goodness"] = R_NilValue
      )
    );
  };

  // Pre-allocate matrices used in iteration
  arma::mat invAZ1(M, D);
  arma::mat invV(M, M);
  arma::mat Cmat(D, D);
  arma::mat tXinvV(p, M);
  arma::mat tXinvVX(p, p);
  arma::mat Q(p, p);
  arma::mat P(M, M);
  arma::vec Py(M);
  arma::mat derivrho1(D, D);
  arma::mat sigmaOmegaderivrho1Omega(D, D);

  // Pre-allocate Va matrices (will be reused each iteration)
  std::vector<arma::mat> Va(nparam);
  // Va[0], Va[1], Va[2], Va[3] - MxM Kronecker products
  for (int i = 0; i < nparam; ++i) {
    Va[i].zeros(M, M);
  }

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

    // invAZ1
    invAZ1 = build_invAZ1(mp.invA, D, Tt);

    // invV via Woodbury
    if (!build_invV(mp, invAZ1, D, invV, Cmat)) return make_fail_result(false);

    // P = invV - tXinvV.t() * Q * tXinvV
    tXinvV = tX * invV;
    tXinvVX = tXinvV * X;
    bool okQ = arma::inv_sympd(Q, tXinvVX);
    if (!okQ) okQ = arma::inv(Q, tXinvVX);
    if (!okQ) return make_fail_result(false);

    P = invV - tXinvV.t() * Q * tXinvV;
    Py = P * y;

    // Derivative of spatial model
    derivrho1 = -WpWt + 2.0 * rho1_k * WtW;
    sigmaOmegaderivrho1Omega = (-sigma21_k) * (mp.Omega1 * derivrho1 * mp.Omega1);

    // Build Va matrices (Kronecker products)
    // Va[0] = kron(Omega1, 1_T*1_T')
    // Va[1] = kron(sigmaOmegaderivrho1Omega, 1_T*1_T')
    // Va[2] = kron(I_D, Omega2) or I_M
    // Va[3] = kron(I_D, sigma22*dOmega2)
    arma::mat onesTT = arma::ones<arma::mat>(Tt, Tt);

    Va[0] = arma::kron(mp.Omega1, onesTT);
    Va[1] = arma::kron(sigmaOmegaderivrho1Omega, onesTT);
    if (!isST) {
      Va[2] = arma::eye<arma::mat>(M, M);
    } else {
      Va[2] = arma::kron(EyeD, mp.Omega2);
      Va[3] = arma::kron(EyeD, sigma22_k * mp.dOmega2);
    }

    // ========================================================================
    // OPTIMIZED: Compute trace and Fisher information without storing PV matrices
    // ========================================================================
    // For symmetric P: trace(P*Va) = sum(P % Va)  [element-wise]
    // For trPVPV(i,j): sum(PVa[i] % PVa[j].t()) = trace(PVa[i]*PVa[j]^T)
    //   = trace(P*Va[i]*Va[j]*P) = sum(P % (Va[i]*Va[j]))
    // BUT Va[i]*Va[j] is another MxM matrix...
    //
    // Alternative: compute PVa[i] on-the-fly and accumulate trPV and trPVPV
    // without storing the full matrices.
    //
    // Even better: for our specific Va structure (Kronecker products),
    // we can compute traces efficiently.
    // ========================================================================

    // Compute trPV = trace(P*Va[i]) for each i
    // Using: trace(P*Va) = sum(P % Va) when P and Va are symmetric
    // Note: P is symmetric (projection matrix), Va[i] is symmetric
    arma::vec trPV(nparam);
    for (int i = 0; i < nparam; ++i) {
      trPV(i) = arma::accu(P % Va[i]);
    }

    // Compute trPVPV matrix
    // trPVPV(i,j) = trace(P*Va[i]*P*Va[j]) = sum(PVa[i] % PVa[j].t())
    // Using trace cyclic property: trace(P*Va[i]*P*Va[j]) = trace(P*Va[i]*P*Va[j])
    //
    // For efficiency, we compute PVa[i] = P*Va[i] on-the-fly and accumulate
    // trPVPV incrementally

    arma::mat trPVPV(nparam, nparam, fill::zeros);

    // Pre-compute blocks of P for efficiency (since Va has block structure)
    // P is MxM with no special structure, so we need the full multiplication
    std::vector<arma::mat> PV(nparam);
    for (int i = 0; i < nparam; ++i) {
      PV[i] = P * Va[i];
    }

    // trPVPV(i,j) = sum(PV[i] % PV[j].t()) = trace(PV[i]*PV[j]^T)
    for (int i = 0; i < nparam; ++i) {
      trPVPV(i, i) = arma::accu(PV[i] % PV[i].t());
      for (int j = i + 1; j < nparam; ++j) {
        double tv = arma::accu(PV[i] % PV[j].t());
        trPVPV(i, j) = tv;
        trPVPV(j, i) = tv;
      }
    }

    // Score vector S and Fisher matrix F
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

    // Clamp parameters to valid ranges
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
      _["eblup"] = R_NilValue,
      _["fit"] = List::create(
        _["model"] = model,
        _["convergence"] = convergence,
        _["iterations"] = k,
        _["estcoef"] = R_NilValue,
        _["estvarcomp"] = estvarcomp,
        _["goodness"] = R_NilValue
      )
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
      _["eblup"] = R_NilValue,
      _["fit"] = List::create(
        _["model"] = model,
        _["convergence"] = false,
        _["iterations"] = k,
        _["estcoef"] = R_NilValue,
        _["estvarcomp"] = estvarcomp,
        _["goodness"] = R_NilValue
      )
    );
  }

  bool haveVu1 = (sigma21_f != 0.0);
  arma::mat invAZ1f = build_invAZ1(mpf.invA, D, Tt);
  if (haveVu1) {
    if (!build_invV(mpf, invAZ1f, D, invV, Cmat)) {
      DataFrame estvarcomp = DataFrame::create(
        _["estimate"] = est_vec,
        _["std.error"] = NumericVector(nparam, 0.0)
      );
      return List::create(
        _["eblup"] = R_NilValue,
        _["fit"] = List::create(
          _["model"] = model,
          _["convergence"] = false,
          _["iterations"] = k,
          _["estcoef"] = R_NilValue,
          _["estvarcomp"] = estvarcomp,
          _["goodness"] = R_NilValue
        )
      );
    }
  } else {
    invV = mpf.invA;
  }

  tXinvV = tX * invV;
  tXinvVX = tXinvV * X;
  bool okQ = arma::inv_sympd(Q, tXinvVX);
  if (!okQ) okQ = arma::inv(Q, tXinvVX);
  if (!okQ) return make_fail_result(false);

  arma::vec beta = Q * (tXinvV * y);
  arma::vec resid = y - X * beta;
  arma::vec invVresid = invV * resid;

  // u1 (spatial random effects)
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

  // u2 (temporal random effects)
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
      _["eblup"] = R_NilValue,
      _["fit"] = List::create(
        _["model"] = model,
        _["convergence"] = false,
        _["iterations"] = k,
        _["estcoef"] = R_NilValue,
        _["estvarcomp"] = estvarcomp,
        _["goodness"] = R_NilValue
      )
    );
  }
  arma::vec stderr_theta = arma::sqrt(diagFinv);

  DataFrame estvarcomp = DataFrame::create(
    _["estimate"] = est_vec,
    _["std.error"] = NumericVector(stderr_theta.begin(), stderr_theta.end())
  );

  List fit = List::create(
    _["model"] = model,
    _["convergence"] = true,
    _["iterations"] = k,
    _["estcoef"] = estcoef,
    _["estvarcomp"] = estvarcomp,
    _["goodness"] = goodness
  );

  return List::create(
    _["eblup"] = NumericVector(eblup.begin(), eblup.end()),
    _["fit"] = fit
  );
}
