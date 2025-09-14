data {
  int<lower=1> n1;
  int<lower=1> n2;
  int<lower=1> nvar;
  vector<lower=0, upper=1>[n1] y_sampled; 
  matrix[n1, nvar] x_sampled;
  matrix[n2, nvar] x_nonsampled;

  vector[nvar] mu_b;
  vector<lower=0>[nvar] tau_b;
  real<lower=0> phi_aa;
  real<lower=0> phi_ab;
  real<lower=0> phi_ba;
  real<lower=0> phi_bb;
  real<lower=0> tau_ua;
  real<lower=0> tau_ub;
}

parameters {
  vector[nvar] b;
  vector[n1] u;
  vector[n2] v;
  real<lower=0> tau_u;
  real<lower=0> phi_a;
  real<lower=0> phi_b;
  vector<lower=0>[n1] phi;
  vector<lower=0>[n2] phi_nonsampled;
}

transformed parameters {
  vector[n1] mu_sampled;
  vector[n2] mu_nonsampled;
  vector<lower=0>[n1] alpha_sampled;
  vector<lower=0>[n1] beta_sampled;
  vector<lower=0>[n2] alpha_nonsampled;
  vector<lower=0>[n2] beta_nonsampled;

  for (i in 1:n1) {
    mu_sampled[i] = inv_logit(dot_product(x_sampled[i], b) + u[i]);
    alpha_sampled[i] = mu_sampled[i] * phi[i];
    beta_sampled[i] = (1 - mu_sampled[i]) * phi[i];
  }

  for (j in 1:n2) {
    mu_nonsampled[j] = inv_logit(dot_product(x_nonsampled[j], b) + v[j]);
    alpha_nonsampled[j] = mu_nonsampled[j] * phi_nonsampled[j];
    beta_nonsampled[j] = (1 - mu_nonsampled[j]) * phi_nonsampled[j];
  }
}

model {
  b ~ normal(mu_b, sqrt(1 ./ tau_b));
  u ~ normal(0, sqrt(1 / tau_u));
  v ~ normal(0, sqrt(1 / tau_u));
  tau_u ~ gamma(tau_ua, tau_ub);
  phi_a ~ gamma(phi_aa, phi_ab);
  phi_b ~ gamma(phi_ba, phi_bb);
  phi ~ gamma(phi_a, phi_b);
  phi_nonsampled ~ gamma(phi_a, phi_b);

  for (i in 1:n1) {
    y_sampled[i] ~ beta(alpha_sampled[i], beta_sampled[i]);
  }
}

generated quantities {
  real a_var;
  a_var = 1 / tau_u;

  vector[n2] y_pred_nonsampled;
  for (j in 1:n2) {
    y_pred_nonsampled[j] = beta_rng(alpha_nonsampled[j], beta_nonsampled[j]);
  }
}
