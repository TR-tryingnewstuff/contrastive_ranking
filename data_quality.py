#%%
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

#%%

df = pd.read_csv('train_220k_250col.csv')
df

#%%
# ? Drop symbols with fewer than n_obs 
symbol_with_at_least_n_obs = df['symbol'].value_counts() > 20
symbol_with_at_least_n_obs = symbol_with_at_least_n_obs.loc[symbol_with_at_least_n_obs == True].index

df = df.loc[df['symbol'].isin(symbol_with_at_least_n_obs)]

# ? Drop rows where open is inferior to 1 -> some have open == 0 so the target makes no sense
df = df.loc[df['open'] > 1]

# ? Drop rows where close_target == 0 and target > 3
df = df.loc[df[['close_target']].max(axis=1) != 0]
df = df.loc[df['close_target'] < 2.5]

#%%

# ! assert no High and Lows target are below (above) the Close target 
assert (df['close_target'] <= df['high_target']).sum() == len(df) 
assert (df['close_target'] >= df['low_target']).sum() == len(df) 


#%%
symbols = df['symbol'].unique()
rand_symbols = np.random.choice(symbols)

print(rand_symbols)
df.loc[df['symbol'] == rand_symbols].reset_index()['open'].plot()

#%%

df.to_csv('train_220k_250col_asserted.csv', index=False)

# %%
df
# %%
