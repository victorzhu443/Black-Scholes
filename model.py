import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

# Black-Scholes Option Pricing Model and Greeks Calculation for S&P 500
# --------------------------------------------------------------------
# This module implements the Black-Scholes model for pricing European call and put options
# on the S&P 500 index and calculates the option Greeks: Delta, Gamma, Theta, Vega, and Rho.

# Define the Black-Scholes option pricing function
def black_scholes(S, K, r, T, sigma, option_type="c"):
    """
    Calculate the Black-Scholes price for European call or put options.

    Parameters:
        S (float): Current price of the underlying asset (S&P 500 Index level)
        K (float): Strike price of the option
        r (float): Risk-free interest rate (annualized)
        T (float): Time to maturity (in years)
        sigma (float): Volatility of the underlying asset (annualized)
        option_type (str): 'c' for call option, 'p' for put option

    Returns:
        float: Option price
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "c":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "p":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'c' for call or 'p' for put.")

    return price

# Define the Greeks

def option_delta(S, K, r, T, sigma, option_type="c"):
    """
    Calculate Delta: Sensitivity to changes in the underlying asset price.

    Parameters:
        S (float): Underlying asset price (S&P 500 Index level)
        K (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity
        sigma (float): Volatility
        option_type (str): 'c' for call, 'p' for put

    Returns:
        float: Delta
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    if option_type == "c":
        return norm.cdf(d1)
    elif option_type == "p":
        return -norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type.")

def option_gamma(S, K, r, T, sigma):
    """
    Calculate Gamma: Sensitivity of Delta to changes in the underlying asset price.

    Parameters:
        S (float): Underlying asset price (S&P 500 Index level)
        K (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity
        sigma (float): Volatility

    Returns:
        float: Gamma
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def option_theta(S, K, r, T, sigma, option_type="c"):
    """
    Calculate Theta: Sensitivity to time decay (per day).

    Parameters:
        S (float): Underlying asset price (S&P 500 Index level)
        K (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity
        sigma (float): Volatility
        option_type (str): 'c' for call, 'p' for put

    Returns:
        float: Theta (per day)
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "c":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "p":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("Invalid option type.")

    return theta / 365  # Convert to per-day

def option_vega(S, K, r, T, sigma):
    """
    Calculate Vega: Sensitivity to changes in volatility.

    Parameters:
        S (float): Underlying asset price (S&P 500 Index level)
        K (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity
        sigma (float): Volatility

    Returns:
        float: Vega (as percentage sensitivity)
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1) * 0.01

def option_rho(S, K, r, T, sigma, option_type="c"):
    """
    Calculate Rho: Sensitivity to changes in the risk-free interest rate.

    Parameters:
        S (float): Underlying asset price (S&P 500 Index level)
        K (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity
        sigma (float): Volatility
        option_type (str): 'c' for call, 'p' for put

    Returns:
        float: Rho (as percentage sensitivity)
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "c":
        return 0.01 * K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "p":
        return -0.01 * K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type.")

# Visualization and Streamlit Interface Setup
# Use Streamlit to interactively visualize the Black-Scholes outputs.
st.set_page_config(page_title="Black-Scholes Option Pricing for S&P 500")
st.sidebar.header("Option Parameters for S&P 500")

# User Inputs
r = st.sidebar.number_input("Risk-Free Rate (annualized)", 0.0, 1.0, 0.03, step=0.01)
S = st.sidebar.number_input("S&P 500 Index Level", 1.0, 5000.0, 4500.0, step=1.0)
K = st.sidebar.number_input("Strike Price", 1.0, 5000.0, 4500.0, step=1.0)
days_to_expiry = st.sidebar.number_input("Days to Expiry", 1, 3650, 30, step=1)
sigma = st.sidebar.number_input("Volatility (annualized)", 0.0, 1.0, 0.2, step=0.01)
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

# Convert days to years
T = days_to_expiry / 365.0

# Calculate values
price = black_scholes(S, K, r, T, sigma, option_type.lower()[0])
delta = option_delta(S, K, r, T, sigma, option_type.lower()[0])
gamma = option_gamma(S, K, r, T, sigma)
theta = option_theta(S, K, r, T, sigma, option_type.lower()[0])
vega = option_vega(S, K, r, T, sigma)
rho = option_rho(S, K, r, T, sigma, option_type.lower()[0])

# Display Results
st.markdown("### Black-Scholes Option Pricing Results for S&P 500")
st.metric("Option Price", f"${price:.2f}")
st.metric("Delta", f"{delta:.4f}")
st.metric("Gamma", f"{gamma:.4f}")
st.metric("Theta", f"{theta:.4f}")
st.metric("Vega", f"{vega:.4f}")
st.metric("Rho", f"{rho:.4f}")
