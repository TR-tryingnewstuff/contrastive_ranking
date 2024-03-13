#%%
# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import var 
import matplotlib.pyplot as plt
# Fetch historical stock prices for specified tickers using Yahoo Finance
df = yf.Tickers(['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'WMT']).history(start='2010-01-01')

# Define equal weights for the portfolio, assuming a portfolio of 5 stocks with equal allocation
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Calculate daily returns for the stock prices and drop the first row as it will be NaN due to pct_change method
returns = df['Close'].pct_change().dropna()

#%%
# Define a function for backtesting Value at Risk (VaR) measures
def backtest_var(method, confidence, window=252):
    # Depending on the method ('h' for historical, 'g' for garch, initialize the VaR model with specific parameters
    if method in ['h', 'g']:
        model = var.VaR(returns, weights, distribution='t', alpha=confidence, window=window)
    else:
        model = var.VaR(returns, weights, distribution='t', alpha=confidence)
    
    # Perform backtesting using the initialized model
    hist_backtest = model.backtest(method)
    
    # Apply transformations to the backtest results for analysis (not clear without knowing the exact output format)
    hist_backtest.iloc[:, -1] = np.invert(hist_backtest.iloc[:, -1])
    hist_backtest.iloc[:, 3] = -hist_backtest.iloc[:, 3]
    hist_backtest = hist_backtest.clip(-0.2, 1)

    # Plot the results of the backtest
    plt.plot(hist_backtest['Daily PnL'].clip(-1, 0), alpha=0.5)
    plt.plot(hist_backtest.iloc[:, 1], color='g')
    plt.plot(hist_backtest.iloc[:, 2], color='red')

    plt.scatter(hist_backtest.iloc[:, -2].index, hist_backtest.iloc[:, -2].replace(False, np.nan) * -0.07, c='red', s=5)
    plt.scatter(hist_backtest.iloc[:, -3].index, hist_backtest.iloc[:, -3].replace(False, np.nan) * -0.06, c='g', s=5)
    plt.legend(['Neg Returns', 'VaR', 'ES', 'exception VaR', 'exception ES'], loc='lower right', fontsize=9)

    # Print statistics about the backtest, specifically the percentage of returns that exceeded the VaR threshold
    print('\n' )
    print('********** % of returns that went passed the threshold ***********')
    print('\n')
    print(hist_backtest.iloc[:, -3:].sum() / len(hist_backtest))

    # Return the backtest results
    return hist_backtest

# Execute the backtesting function with a garch method, 95% confidence level, and a 252-day window
res = backtest_var('h', 0.10, window=252)
