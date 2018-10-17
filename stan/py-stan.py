import pystan
import pandas as pd
import numpy as np
import scipy as sp
import arviz as az # ??? matplotlib replacement?

np.random.seed(33)

N = 50
x = np.random.randn(N)

model="model.stan"
stan_model = pystan.StanModel(file=model)

fit = stan_model.sampling(data={'N': N, 'x' : x}, chains=1)

print(fit)

az.plot_density(fit, var_names=['mu', 'sigma'])
