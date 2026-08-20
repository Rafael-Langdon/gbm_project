
import numpy as np
from scipy import stats


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

def black_scholes(S0, K, r, sigma, T, option_type='call'):

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S0 * stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
    elif option_type == 'put':
        price = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)

    return price


def implied_vol(price, S0, K, r, T, option_type):
    def objective(sigma):
        return black_scholes(S0, K, r, sigma, T, option_type) - price
    try:
        return brentq(objective, 1e-6, 10.0)
    except:
        return np.nan


def black_scholes_digital(S0, K, r, sigma, T, option_type='call', Q=1):
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return Q * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == 'put':
        return Q * np.exp(-r * T) * stats.norm.cdf(-d2)




def bs_d1d2(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2