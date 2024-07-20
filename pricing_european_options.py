import numpy as np
from scipy.stats import norm

# Parameters
S0 = 100  # initial stock price
K = 100   # strike price
T = 1     # time to maturity (1 year)
r = 0.02  # risk-free ann rate
sigma = 0.30  # volatility
mu = 0.02  # expected return == ann rfr
dt = 0.001  # time increment
N = int(T / dt)  # number of steps
M = 100000  # number of simulations

def simulate_stock_prices(S0, mu, sigma, T, dt, N, M):
    Z = np.random.standard_normal((N, M))  # standard normal variables
    S = np.zeros((N + 1, M))  # array to store simulated stock prices
    S[0] = S0  # initial stock price
    for t in range(1, N + 1):
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])
    return S

def price_call_option(S, K, r, T):
    payoffs = np.maximum(S[-1] - K, 0)  # payoff at maturity
    return np.exp(-r * T) * np.mean(payoffs)  # discounted average payoff

def price_put_option(S, K, r, T):
    payoffs = np.maximum(K - S[-1], 0)  # payoff at maturity
    return np.exp(-r * T) * np.mean(payoffs)  # discounted average payoff

# Verification using Black-Scholes formula
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def black_scholes_put(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    return put_price



strike_prices = [90, 100, 110]
tenors = [0.5, 1, 2]

results = []

for K in strike_prices:
    for T in tenors:
        print('-'*30)
        print('K :', K)
        print('T :', T)

        N = int(T / dt) 
        S = simulate_stock_prices(S0, mu, sigma, T, dt, N, M)

        # Price the European call option
        call_price = price_call_option(S, K, r, T)
        print(f"The price of the European call option is: {call_price:.2f}")

        # Price the European put option
        put_price = price_put_option(S, K, r, T)
        print(f"The price of the European put option is: {put_price:.2f}")

        bs_call_price = black_scholes_call(S0, K, T, r, sigma)
        bs_put_price = black_scholes_put(S0, K, T, r, sigma)
        print(f"Verified call price using Black-Scholes formula: {bs_call_price:.2f}")
        print(f"Verified put price using Black-Scholes formula: {bs_put_price:.2f}")

        dS = 1.0  # small change in stock price
        dSigma = 0.01*sigma  # small change in volatility

        # Calculate delta for call
        S_up = simulate_stock_prices(S0 + dS, mu, sigma, T, dt, N, M)
        call_price_up = price_call_option(S_up, K, r, T)
        delta_call = (call_price_up - call_price) / dS
        print(f"Delta of the call option: {delta_call:.4f}")

        # Calculate delta for put
        put_price_up = price_put_option(S_up, K, r, T)
        delta_put = (put_price_up - put_price) / dS
        print(f"Delta of the put option: {delta_put:.4f}")

        # Calculate vega for call
        S_sigma_up = simulate_stock_prices(S0, mu, sigma + dSigma, T, dt, N, M)
        call_price_sigma_up = price_call_option(S_sigma_up, K, r, T)
        vega_call = (call_price_sigma_up - call_price) / dSigma
        print(f"Vega of the call option: {vega_call:.4f}")

        # Calculate vega for put
        put_price_sigma_up = price_put_option(S_sigma_up, K, r, T)
        vega_put = (put_price_sigma_up - put_price) / dSigma
        print(f"Vega of the put option: {vega_put:.4f}")
        print('-'*30)
