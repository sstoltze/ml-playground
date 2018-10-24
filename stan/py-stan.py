import pystan
import pandas as pd
import numpy as np
import scipy as sp
import arviz as az # ??? matplotlib replacement?
import pickle

np.random.seed(33)

N = 50
x = np.random.randn(N)

model="model.stan"
try:
    with open("pickled_model.pickle", "rb") as pickle_file:
        stan_model = pickle.load(pickle_file)
except FileNotFoundError:
    stan_model = pystan.StanModel(file=model)
    with open("pickled_model.pickle", "wb") as pickle_file:
        pickle.dump(stan_model, pickle_file)

fit = stan_model.sampling(data={'N': N, 'x' : x}, chains=1)

print(fit)

# Arviz, doesn't work at the moment
az.plot_density(fit, var_names=['mu', 'sigma'])

# Old fashioned, generates a warning but works
fit.plot().savefig("./images/pygraph-savefig.png")
