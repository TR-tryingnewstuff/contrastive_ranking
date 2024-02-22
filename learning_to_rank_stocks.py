#%%#%%
import pandas as pd 
from fmp_get_data import quantile_scale, rolling_quantile_scale
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as plt
from neural_models import *
import seaborn as sns 
from utils import combinatorial_purged_k_fold
from copy import deepcopy
import xgboost as xgb
from mapie.regression import MapieRegressor
from tensorflow_addons.losses import triplet_semihard_loss, triplet_hard_loss, pinball_loss, contrastive_loss, ContrastiveLoss
import optuna as opt
# ! DONE !
# TODO - >  Test changing the target to be in the range [0, 1] for each quarter of each year
# ? Could create more granular groups to take into account differences in publication of statement and control for differences in global market returns
# ? Or even do rolling categories to increase the size of the dataset 

# ! NOT DONE !
# TODO - > redo quantile scale to accomodate look-ahead bias 
# TODO - > Test ranking by returns / (returns - (high low range)) or other sharpe related measures 
# TODO - > Do a rolling strat where stocks are reentered into next batch prediction (or updated if a new observation becomes available) and a new portfolio is constructed out of them 
# TODO - > Test Conformal Prediction 
# TODO - > Test Hyperparameter tuning on purged K-fold Cross Val

#%%
# ? ----------------------------------- DATA PREPROCESSING ----------------------------------------

df = pd.read_csv('train_220k_250col_asserted.csv')
df = df.sort_values('date').set_index('date')
df.index = pd.to_datetime(df.index)

# ? Add and Normalize date information
df['period'] = df.index.quarter
df['calendarYear'] = df.index.year
df['Year-Week'] = df.index.strftime('%Y-%U')

df['Year-Week']

# ! Create groups, WATCHOUT the groups must be ordered and the strings are being interpreted as numbers by the OrdinalEncoder
qid =  df['Year-Week']#.astype(str)#.apply(lambda x: '0' + x if len(x) == 1 else x)
qid = OrdinalEncoder().fit_transform(qid.values.reshape(-1, 1))
df

#%%

# ? Rank returns for each week and year
df['rank'] = df.groupby(['Year-Week'])["close_target"].rank(method='dense', pct=True)

for id in list(qid)[0:100]:
    # ? For each groups to rank assert that the max group rank equals the length of the dataframe
    max_rank = df.loc[qid == id]['rank'].max()
    len_group = len(df.loc[qid == id]['rank'])
    
    #assert max_rank == len_group

# ? Quantile scale DataFrame
df_scaled = quantile_scale(df, 'uniform')



# ? Create Train / Test Splits
X = df_scaled.select_dtypes(np.number).drop(['calendarYear', 'cik' ,'close_target', 'low_target', 'high_target', 'rank'], axis=1)
Y = df['rank']

X_train, X_test, Y_train, Y_test, qid_train, qid_test, df_train = X.loc[qid <= 1000], X.loc[qid > 1000],Y.loc[qid <= 1000], Y.loc[qid > 1000], qid[qid <= 1000], qid[qid > 1000], df.loc[qid <= 1000]



def make_backtest(preds, true, plot=True):

    backtest_df = deepcopy(pd.DataFrame(true))
    
    backtest_df['calendarYear'] = backtest_df.index.year
    backtest_df['period'] = backtest_df.index.quarter 
    backtest_df['preds'] = preds

    backtest_df = backtest_df.sort_values(['calendarYear', 'period', 'preds']).reset_index(drop=True)

    total_preds, total_bench = [], []
    for year in backtest_df['calendarYear'].unique():
        for quarter in backtest_df['period'].unique():
            
            loc = backtest_df.loc[(backtest_df['calendarYear'] == year) & (backtest_df['period'] == quarter)]
            
            n = 100
            best = loc.tail(n)['close_target'].reset_index(drop=True) / min(n, len(loc.tail(n)))
            worst = loc.head(n)['close_target'].reset_index(drop=True) / min(n, len(loc.tail(n)))
            bench = loc['close_target'].reset_index(drop=True) / len(loc)
            
            if len(loc) > n:
                
                if plot:
                    sns.boxenplot([best, worst, bench])
                    plt.legend(['best', 'worst','bench'])
                    plt.show()
                    
                total_preds.append((best + 1).prod())
                total_bench.append((bench + 1).prod())
                
    total_preds = np.array(total_preds)
    #print(total_preds[total_preds < 10].prod())



    total_bench = np.array(total_bench)
    #print(total_bench[total_bench < 10].prod())
    
    return total_preds, total_bench


#%%
train_idxs, test_idxs = combinatorial_purged_k_fold(5, int(len(X_train) * 0.3), 3000, 2, len(X_train))
plt.violinplot(train_idxs)
plt.violinplot(test_idxs)
plt.show()


def get_hyperparameter_df(study):
    res = []
    for t in study.get_trials():

        res.append(t.params)
        res[-1]['values'] = t.values[0]
        
    res = pd.DataFrame(res)

    return res

def hyperparam_opt_ranking(trial):
        
    param_space = {
        'objective': 'rank:pairwise',
        'lambdarank_pair_method': 'topk',
        'booster': 'gbtree',
        'n_estimators':10,
        #'learning_rate': trial.suggest_float('lr',0.1, 1),
        'gamma': trial.suggest_float('gamma', 0, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_child_weight': trial.suggest_float('min_weight', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'reg_lambda': trial.suggest_float('L2', 0, 0.05), # L2
        'reg_alpha': trial.suggest_float('L1', 0, 0.05), # L1
        'max_leaves': np.random.randint(1, 256),
        'max_bin': trial.suggest_int('bins', 100, 1000)
    }

    print(param_space)

    model = xgb.XGBRanker(**param_space)
    model.fit(X_train, Y_train, qid=qid_train)
    
    return backtest_hyperparameter_opt(model)
    
def backtest_hyperparameter_opt(model, idxs):

    preds = model.predict(X_train.iloc[idxs]).flatten()

    profit_preds = df_train.iloc[idxs]['close_target'].iloc[np.argsort(preds)].head(int(np.array([qid == id]).sum() / 4))
    profit_bench = df_train.iloc[idxs]['close_target']
    
    profit_diff = profit_preds.mean() - profit_bench.mean()
    
    return profit_diff 

def construct_nn(param_space):
    
    inputs = Input(X_train.shape[1])

    drop = Dropout(param_space['dropout'])(inputs)
    encode_branch = Dense(param_space['neurons_1'], 'sigmoid', bias_regularizer=tf.keras.regularizers.L1L2(param_space['l1'], param_space['l2']))(drop)
    #batch_norm = tf.keras.layers.BatchNormalization(scale=False)(encode_branch)
    drop = Dropout(param_space['dropout'])(encode_branch)
    encode_branch = Dense(param_space['neurons_2'], 'sigmoid')(drop)

    close_out = Dense(1, 'sigmoid', name='close')(encode_branch)

    model = tf.keras.Model(inputs=[inputs], outputs=[close_out])
    
    
    losses = {
        'close': ContrastiveLoss(margin=param_space['margin']),
    }
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(param_space['lr']), loss=losses)
    #model.fit(X_train, Y_train, verbose=1, epochs=param_space['epochs'])
    
    return model

def cross_hyperparam_opt_nn(trial):
    
    param_space = {
        'dropout': trial.suggest_float('dropout', 0, 0.5),
        'neurons_1': trial.suggest_int('neurons_1', 32, 128),
        'neurons_2': trial.suggest_int('neurons_2', 8, 64),
        'l1': trial.suggest_float('l1', 0, 0.05),
        'l2': trial.suggest_float('l2', 0, 0.1),
        'epochs': trial.suggest_int('epochs', 1, 3),
        'lr': trial.suggest_float('lr', 0.00001, 0.05),
        'margin': trial.suggest_float('margin', 0.5, 2.5)       
    }
    
    model = construct_nn(param_space)
    
    profits = 0
    for train_idx, test_idx in zip(train_idxs, test_idxs):
        model.fit(X_train.iloc[train_idx], Y_train.iloc[train_idx], verbose=1, epochs=param_space['epochs'])
        profits += backtest_hyperparameter_opt(model, test_idx)
    
    return profits / len(train_idxs)


    
#%%
study = opt.create_study(
    study_name="neural-net-simple",
    sampler=opt.samplers.TPESampler(), 
    direction='maximize'
    )


study.optimize(cross_hyperparam_opt_nn, n_trials=100)
print(study.best_params)

results = get_hyperparameter_df(study)
results.sort_values('values').tail(50).hist(bins=50)
results.sort_values('values').tail(50)


#%%




#%%
model = construct_nn(study.best_params)


for train_idx, test_idx in zip(train_idxs, test_idxs):
    model.fit(X_train.iloc[train_idx], Y_train.iloc[train_idx], epochs=study.best_params['epochs'])
    preds = model.predict(X_train.iloc[test_idx]).flatten()

    #sns.boxenplot([Y_test.loc[qid_test == id][np.argsort(preds)].head(int(np.array([qid_test == id]).sum() / 3)), Y_test.loc[qid_test == id]])
    #plt.show()

    sns.kdeplot(df_train.iloc[test_idx]['close_target'][np.argsort(preds)].head(int(len(test_idx) / 4)), cumulative=False, common_grid=True, fill=True)
    sns.kdeplot(df_train.iloc[test_idx]['close_target'], cumulative=False, common_grid=True, fill=True)
    plt.vlines(0, 0, 6, linestyles='dashed', colors='black')
    plt.legend(['model', 'bench'])
    plt.show()
    
# %%

print(study.best_params)
# ? {'dropout': 0.2573749210368332, 'neurons_1': 68, 'neurons_2': 37, 'l1': 0.015917743693902718, 'l2': 0.0025165281652827047, 'epochs': 2, 'lr': 0.0009131005298740329, 'margin': 1.8131495616237179}

#%%

from crepes import WrapRegressor
from scipy.special import expit

nn = WrapRegressor(model)
nn.fit((X_train.iloc[train_idxs[-1]]), Y_train.iloc[train_idxs[-1]])
nn.calibrate(X_train.iloc[test_idxs[-1]], Y_train.iloc[test_idxs[-1]])
nn.predict(X_train.iloc[test_idxs[-1]])

# %%
plt.scatter(y_pred_lower, y_pred_upper)
# %%
X_train.iloc[train_idxs[-1]]