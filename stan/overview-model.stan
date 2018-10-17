data {
  int<lower=1> N; // N samples
  vector[N] x;
}

parameters {
  real mu;
  real<lower=0> sigma; // sigma is positive
}

model {
  x ~ normal(mu, sigma); // We expect the input data to be normally distributed, norm mu and mean sigma
}
