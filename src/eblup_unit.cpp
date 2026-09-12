// src/eblup_unit.cpp
// Unit-level EBLUP computation for Battese-Harter-Fuller model
#include <Rcpp.h>
#include <unordered_map>
#include <vector>
using namespace Rcpp;

// [[Rcpp::export(.eblup_bhf_cpp)]]
List eblup_bhf_cpp(
    CharacterVector selectdom,
    CharacterVector dom,
    NumericMatrix Xs,
    NumericMatrix meanxpop,
    NumericVector ys,
    NumericVector popnsize,
    NumericMatrix betaest,
    DataFrame upred
) {
  const int I = selectdom.size();
  const int n = dom.size();
  const int p = Xs.ncol();

  NumericVector eblup(I, NA_REAL);
  NumericVector samp_size(I, 0.0);
  std::vector<std::string> warn_domains;

  // Build domain -> upred mapping
  CharacterVector upred_names = upred.attr("row.names");
  NumericVector upred_values = upred[0];

  std::unordered_map<std::string, double> upred_map;
  upred_map.reserve(upred_names.size());
  for (int i = 0; i < upred_names.size(); i++) {
    upred_map[Rcpp::as<std::string>(upred_names[i])] = upred_values[i];
  }

  // Group sample-unit indices by domain (O(n) instead of O(I*n))
  std::unordered_map<std::string, std::vector<int>> dom_index;
  dom_index.reserve(I * 2);
  for (int j = 0; j < n; j++) {
    dom_index[Rcpp::as<std::string>(dom[j])].push_back(j);
  }

  // Pre-allocated buffer for meanXsd (reused across domains)
  std::vector<double> meanXsd(p, 0.0);

  for (int i = 0; i < I; i++) {
    std::string d = Rcpp::as<std::string>(selectdom[i]);
    auto it_idx = dom_index.find(d);

    if (it_idx != dom_index.end() && !it_idx->second.empty()) {
      const std::vector<int>& idx = it_idx->second;
      const int m = static_cast<int>(idx.size());
      const double fd = static_cast<double>(m) / popnsize[i];

      // Column-major access for meanXsd
      for (int c = 0; c < p; c++) {
        const double* col = &Xs(0, c);
        double sum_c = 0.0;
        for (int k = 0; k < m; k++) sum_c += col[idx[k]];
        meanXsd[c] = sum_c / m;
      }

      // meanYs
      double meanYs = 0.0;
      for (int k = 0; k < m; k++) meanYs += ys[idx[k]];
      meanYs /= m;

      // (meanxpop[i,] - fd * meanXsd) %*% betaest
      double xb = 0.0;
      for (int c = 0; c < p; c++) {
        xb += (meanxpop(i, c) - fd * meanXsd[c]) * betaest(c, 0);
      }

      // Get random effect
      double upred_val = NA_REAL;
      auto it_upred = upred_map.find(d);
      if (it_upred != upred_map.end()) upred_val = it_upred->second;

      // EBLUP formula
      eblup[i] = fd * meanYs + xb + (1.0 - fd) * upred_val;
      samp_size[i] = m;

    } else {
      // Domain not found in sample: use synthetic regression estimator
      double xb = 0.0;
      for (int c = 0; c < p; c++) xb += meanxpop(i, c) * betaest(c, 0);
      eblup[i] = xb;
      warn_domains.push_back(d);
    }
  }

  return List::create(
    _["eblup"] = eblup,
    _["samp_size"] = samp_size,
    _["warn_domains"] = wrap(warn_domains)
  );
}
