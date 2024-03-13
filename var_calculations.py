#%%
import pandas as pd
import numpy as np
import yfinance as yf
import var 

df = yf.Tickers(['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'WMT']).history(start='2010-01-01')
df

#%%
weights = weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
returns = df['Close'].pct_change().dropna()


model = var.VaR(returns, weights, distribution='t')
hist_backtest = model.backtest('h')

#%%
hist_backtest['VaR(99.0)'].plot()

# %%