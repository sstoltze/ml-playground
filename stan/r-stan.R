set.seed(33)

N <- 50
x <- rnorm(N, 20, 3)

model <- "model.stan"
library(rstan)
rstan_options(auto_write = TRUE)

fit <- stan(file=model, data=list(N=N, x=x), chains=1)

print(fit)

plot(fit)
