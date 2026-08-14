
import numpy as np


def gbm_simulate(S0, mu, sigma, T, N, I):  # this function is now added to utils.py for later use
    dt = 1/252 # time step length
    times = np.linspace(0, T, N+1) # array of times
    S = np.zeros((N+1, I)) # matrix to store asset values
    S[0] = S0

    for t in range (1, N+1):
        Z = np.random.standard_normal(I) # array of length I of normally distrubuted noise
        S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2) * dt + sigma * np.sqrt(dt) * Z) # discretised closed form of GBM
    return S, times