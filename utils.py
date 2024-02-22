#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
import seaborn as sns 
import matplotlib.pyplot as plt


def combinatorial_purged_k_fold(comb, test_size, purge, k, len_data):
    
    test_indexes = []
    test_indexes_purged = []
    for _ in range(comb):
        
        for i in range(k):

            if i == 0:
                start = np.random.randint(0, len_data)
                test_indexes.append(list(np.array(range(start, start+test_size)) % len_data)) 
                test_indexes_purged.append(list(np.array(range(np.where(start - purge > 0, start + purge, start ), start+test_size - purge)) % len_data)) 
            else:
                start = np.random.randint(0, len_data)

                while (start in test_indexes_purged[-1]) & ((start + test_size) % len_data  in test_indexes_purged[-1]):
                    start += 1

                test_indexes[-1].extend(list(np.array(range(start, start+test_size)) % len_data))     
                
                test_indexes_purged[-1].extend(list(np.array(range(np.where(start - purge > 0, start + purge, start ), start + test_size - purge)) % len_data))   


    train_indexes = [np.delete(np.arange(len_data), np.array(test).flatten()) for test in test_indexes]

    return train_indexes, test_indexes_purged
    


def turn_index_to_col(df,level):
    
    d = {}
    df_reset = df.reset_index(level)
    
    for tick in df_reset['level_0']:
        d[tick] = df_reset.loc[df_reset['level_0'] == tick]
        
    return pd.concat(d, axis=1)   


def scale(df, cols, method, end,n_quantiles=100):
    
    print(df)
    if method == 'standard':
        std_scale = StandardScaler().fit(df.iloc[:end][cols])
        df[cols] = std_scale.transform(df[cols])
    
    elif method == 'power':
        power_scale = PowerTransformer('box-cox').fit(df.iloc[:end][cols])
        df[cols] = power_scale.transform(df[cols])

    elif method == 'quantile':
        quantile_scale = QuantileTransformer(n_quantiles=n_quantiles).fit(df.iloc[:end][cols])
        df[cols] = quantile_scale.transform(df[cols])
    print(df)
    
    return df


def get_predictions_features(predictions, n_tickers, corr_estimation_frac=0.99):

    mean_preds = predictions.mean(axis=1).mean(axis=0).reshape(-1, 1).squeeze()
    var_preds = predictions.var(axis=1).mean(axis=0).reshape(-1, 1).squeeze()
    
    for i in range(n_tickers):
        
        try: 
            corr_preds = pd.DataFrame(mean_preds[i:int(len(mean_preds) * corr_estimation_frac)].reshape(-1, n_tickers)).corr()
            break
            
        except:
            pass

    return mean_preds, var_preds, corr_preds

def get_predictions_tabular(predictions, n_tickers):
    
    for i in range(n_tickers):
        
        try: 
            tabular_predictions = pd.DataFrame(predictions[i:].reshape(-1, n_tickers))
            break
            
        except:
            pass
        
    return tabular_predictions


def get_markowitz_alloc(mean_preds_tab, var_preds_tab, cov_matrix):
    
    l_w_primes = []
    
    for row_mean, row_var in zip(mean_preds_tab, var_preds_tab):
        
        print(row_mean)
        selected_tickers = np.where(row_mean > 0.5, 1, 0)
        print(selected_tickers)
        matrix_selected = (selected_tickers.reshape(-1, 1) @ selected_tickers.reshape(-1, 1).T).reshape(-1, 6) 
        
        curr_cov = cov_matrix * row_var

        trunc_cov_matrix = curr_cov.values[matrix_selected == 1].reshape(-1, selected_tickers.sum())
        
            
        ones = np.ones(trunc_cov_matrix.shape[0])  
        w_primes = (np.linalg.inv(trunc_cov_matrix) @ ones) / (ones.T @ np.linalg.inv(trunc_cov_matrix) @ ones)
        
        if selected_tickers[-1] == 0:
            w_primes = np.append(w_primes, 0)
            selected_tickers[-1] = 1
        
        
        w_primes = np.insert(w_primes, np.where(selected_tickers <= 0.5)[0], 0)
            
        l_w_primes.append(w_primes)
    
    return np.array(l_w_primes)

# %%
