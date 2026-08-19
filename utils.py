
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


def hest_simulate(S0, mu, v0, kappa, theta, xi, rho, T, N, I):
    dt    = 1/252
    times = np.linspace(0, T, N + 1)
    
    S = np.zeros((N + 1, I))
    v = np.zeros((N + 1, I))
    
    S[0] = S0
    v[0] = v0

    for t in range(1, N + 1):
        Z1 = np.random.standard_normal(I)
        Z2 = np.random.standard_normal(I)
        W_S = Z1                            # brownian motion for price
        W_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # correlated brownian motion for variance

        # variance SDE
        v[t] = np.maximum(v[t-1] + kappa * (theta - v[t-1]) * dt + 
                          xi * np.sqrt(v[t-1] * dt) * W_v, 0)

        # asset price SDE
        S[t] = S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt + 
                                np.sqrt(v[t-1] * dt) * W_S)

    return S, v, times