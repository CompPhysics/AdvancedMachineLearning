import math
from scipy.stats import norm

def black_scholes_call_price(S, K, T, r, sigma):
    """
    Analytical solution to the Black-Scholes formula for a European call option.
    
    Parameters:
        S     - Current stock price
        K     - Strike price
        T     - Time to maturity (in years)
        r     - Risk-free interest rate
        sigma - Volatility of the underlying asset
    
    Returns:
        Call option price
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

# Example usage
S = 100   # Stock price
K = 100   # Strike price
T = 1     # Time to maturity (1 year)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility

price = black_scholes_call_price(S, K, T, r, sigma)
print(f"European Call Option Price: {price:.4f}")



# simple numerical solution with plain explicit finite difference


import numpy as np
import matplotlib.pyplot as plt

def black_scholes_fd_explicit(S_max, K, T, r, sigma, M=100, N=100):
    """
    Finite Difference Method (Explicit Scheme) for Black-Scholes Equation
    
    Parameters:
        S_max - Maximum stock price considered
        K     - Strike price
        T     - Time to maturity
        r     - Risk-free interest rate
        sigma - Volatility
        M     - Number of time steps
        N     - Number of price steps
        
    Returns:
        S - Stock prices
        V - Option values at t=0
    """
    dt = T / M
    dS = S_max / N
    S = np.linspace(0, S_max, N + 1)
    V = np.maximum(S - K, 0)  # Terminal payoff for a call option

    for j in range(M):
        V_old = V.copy()
        for i in range(1, N):
            delta = (V_old[i + 1] - V_old[i - 1]) / (2 * dS)
            gamma = (V_old[i + 1] - 2 * V_old[i] + V_old[i - 1]) / (dS ** 2)
            V[i] = V_old[i] + dt * (0.5 * sigma ** 2 * S[i] ** 2 * gamma +
                                    r * S[i] * delta - r * V_old[i])
        V[0] = 0  # Option worthless if stock price is 0
        V[-1] = S_max - K * np.exp(-r * (T - (j + 1) * dt))  # Approximate boundary condition

    return S, V

# Parameters
S_max = 200
K = 100
T = 1
r = 0.05
sigma = 0.2

S, V = black_scholes_fd_explicit(S_max, K, T, r, sigma)
plt.plot(S, V)
plt.xlabel("Stock Price")
plt.ylabel("Option Value")
plt.title("European Call Option Price via Finite Difference Method")
plt.grid(True)
plt.show()
