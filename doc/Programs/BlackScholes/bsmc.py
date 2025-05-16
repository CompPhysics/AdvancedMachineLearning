# Monte Carlo implementation (With Antithetic Variates)
import numpy as np
def monte_carlo_black_scholes_option(S0, K, T, r, sigma, 
                                     num_simulations=100000, 
                                     option_type='call', 
                                     use_antithetic=True):
    """
    Monte Carlo simulation for European call/put option with variance reduction.
    
    Parameters:
        S0               - Initial stock price
        K                - Strike price
        T                - Time to maturity (in years)
        r                - Risk-free interest rate
        sigma            - Volatility
        num_simulations  - Number of Monte Carlo simulations
        option_type      - 'call' or 'put'
        use_antithetic   - Whether to use antithetic variates for variance reduction
    
    Returns:
        Estimated option price
    """
    n = num_simulations // 2 if use_antithetic else num_simulations
    Z = np.random.normal(size=n)
    
    # Simulate asset prices at maturity
    ST1 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    if use_antithetic:
        ST2 = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * Z)
        ST = np.concatenate([ST1, ST2])
    else:
        ST = ST1
    
    # Calculate payoff
    if option_type.lower() == 'call':
        payoff = np.maximum(ST - K, 0)
    elif option_type.lower() == 'put':
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Discount to present value
    price = np.exp(-r * T) * np.mean(payoff)
    return price

# Example usage
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

call_price = monte_carlo_black_scholes_option(S0, K, T, r, sigma, option_type='call')
put_price = monte_carlo_black_scholes_option(S0, K, T, r, sigma, option_type='put')

print(f"Monte Carlo Estimated Call Price (Antithetic): {call_price:.4f}")
print(f"Monte Carlo Estimated Put Price  (Antithetic): {put_price:.4f}")
