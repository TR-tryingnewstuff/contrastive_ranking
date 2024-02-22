#%%
import fmpsdk
import pandas as pd
import numpy as np 
from copy import deepcopy
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import time 

API_KEY = 'cb2b98b48534f437f32eae8885fb8ce6'


#%%

#%%

from multiprocessing import Pool
from joblib import Parallel, delayed

from functools import wraps 

def retry(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except:
                pass
        
    return wrapped

@retry
def get_metrics(ticker, limit=150):
    # ? ---------------------- FINANCIALS -----------------------------------

        print(ticker)

        financial_ratios = pd.DataFrame(fmpsdk.financial_ratios(API_KEY, ticker, limit=limit, period='quarter'))

        # ? ---------------------- BALANCE SHEET ------------------------------

        bs_raw = pd.DataFrame(fmpsdk.balance_sheet_statement(API_KEY, ticker, limit=limit, period='quarter'))

        # ? ---------------------- CASH FLOW -------------------------------------


        cf_raw = pd.DataFrame(fmpsdk.cash_flow_statement(API_KEY, ticker, limit=limit, period='quarter'))

        # ? ----------------------- INCOME STATEMENT -------------------------------

        incs_raw = pd.DataFrame(fmpsdk.income_statement(API_KEY, ticker, limit=limit, period='quarter'))
            
        # ? --------------------------- METRICS --------------------------------------

        key_metrics = pd.DataFrame(fmpsdk.key_metrics(API_KEY, ticker, limit=limit, period='quarter'))

        if len(bs_raw) + len(financial_ratios) + len(cf_raw) + len(incs_raw) + len(key_metrics) == 0:
            
            return np.nan
            
        # ? ----------------------- CREATE DICTIONARY -------------------------------------------
        
        data_dict = {
            "financial_ratios": financial_ratios,
            "bs_raw": bs_raw,
            "cf_raw": cf_raw,
            "incs_raw": incs_raw,
            "key_metrics": key_metrics,
        }

        df = preprocess_quarterly_df(data_dict)
        df['fillingDate'] = pd.to_datetime(df['fillingDate'], yearfirst=True)
        

        prices = pd.DataFrame(fmpsdk.historical_price_full(API_KEY, ticker, from_date='1985-01-01'))
        prices = prices.set_index('date')

        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.resample('d').first().ffill()
        
        groupby_keys = prices.index.isin(df['fillingDate']).cumsum()
        prices = prices.reset_index()
        prices = prices.groupby(groupby_keys)
        prices = prices.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'date': 'first'})
        
        prices = add_prices_features(prices)
        print(len(prices))
        df = pd.merge(prices, df, left_on='date', right_on='fillingDate', how='inner')
        
        return df
 
        

def add_prices_features(df):
    
    df['close_target'] = (df['close'].values - df['open'].values) / (df['open'].values + 0.00001)
    df['low_target'] = (df['low'].values - df['open'].values) / (df['open'].values + 0.00001)
    df['high_target'] = (df['high'].values - df['open'].values) / (df['open'].values + 0.00001)
    
    df[['close_pct', 'low_pct', 'high_pct']] = df[['close_target', 'low_target', 'high_target']].shift(1)
    df = df.drop(['close', 'high', 'low'], axis=1)
    
    return df

def preprocess_yearly_df(df):
    
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.resample('D').ffill()
    
    return df 


def preprocess_quarterly_df(data_dict):
    
    df = pd.concat(data_dict, axis=1).droplevel(0, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]    
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    df_growth = df.select_dtypes(np.number).pct_change(-1, fill_method=None)
    df_growth = df_growth.add_suffix('_growth')
    
    df = pd.concat([df, df_growth], axis=1)
    
    return df

def get_multi_ticker_data(tickers, limit=50):
    
    d = {}
    
    for ticker in tickers:
        
        d[ticker] = get_metrics(ticker, limit)

    return pd.concat(d)

def clean_financials(df):
    """ Should contain multiple stocks from the same sector 
    
    we replace negative values from variables that should not be negative by 0
    then we quantile scale uniformely 
    
    distribution : can be either 'ratio' or 'growth'
    
    """
    df = deepcopy(df)
    
    if type == 'ratio':
    
        non_negative_variables = [
            'currentRatio',
            'quickRatio',
            'cashRatio',
            'daysOfSalesOutstanding',
            'daysOfInventoryOutstanding',
            'operatingCycle',
            'daysOfPayablesOutstanding',
            'cashConversionCycle',
            'grossProfitMargin',
            'operatingProfitMargin',
            'pretaxProfitMargin',
            'netProfitMargin',
            'effectiveTaxRate',
            'returnOnAssets',
            'returnOnEquity',
            'returnOnCapitalEmployed',
            'netIncomePerEBT',
            'ebtPerEbit',
            'ebitPerRevenue',
            'receivablesTurnover',
            'payablesTurnover',
            'inventoryTurnover',
            'fixedAssetTurnover',
            'assetTurnover',
            'operatingCashFlowPerShare',
            'freeCashFlowPerShare',
            'cashPerShare',
            'operatingCashFlowSalesRatio',
            'freeCashFlowOperatingCashFlowRatio',
            'cashFlowCoverageRatios',
            'shortTermCoverageRatios',
            'capitalExpenditureCoverageRatio',
            'dividendPaidAndCapexCoverageRatio',
            'dividendPayoutRatio',
            'dividendYield',
            'enterpriseValueMultiple',
            'priceFairValue'
        ]   
        df[non_negative_variables] = df[non_negative_variables].clip(0, np.inf)
    

    financial_ratios_col = list(pd.read_csv('financial_ratios_columns.csv').values.flatten())
    financial_growth_col = list(pd.read_csv('financial_growth_columns.csv').values.flatten())
    limit_cols = np.where(type=='ratio', financial_ratios_col, financial_growth_col)
    
    col_to_scale = df[limit_cols].select_dtypes(include=np.number).columns
    
    
    if type == 'growth':
        
        df[col_to_scale] = np.clip(-1, 100)
    
    
    dist_type = str(np.where(type=='ratio', 'uniform', 'normal'))
    scaler = QuantileTransformer(output_distribution=dist_type).fit(df.select_dtypes(include=np.number))
    
    df[col_to_scale] = scaler.transform(df[col_to_scale])
    
    if type == 'growth':
            scaler = MinMaxScaler((-1, 1)).fit(df[col_to_scale])
            df[col_to_scale] = scaler.transform(df[col_to_scale])
        
    return df

def scale_to_marketCap(df):
    """ scale by marketCapitalization and then by quantile, to be used with balance sheet statement, cash flow statement and income statement
    """
    df = deepcopy(df)
        
    key_metrics_cols = ['workingCapital', 'tangibleAssetValue', 'averageReceivables', 'averagePayables', 'averageInventory', 'netCurrentAssetValue']
    statement_cols = list(pd.read_csv('statement_columns.csv').values.flatten())
    
    statement_cols.extend(key_metrics_cols)
    statement_cols.remove('date')
    
    col_to_scale = list(df[statement_cols].select_dtypes(include=np.number).columns)
    col_to_scale.append('enterpriseValue')

    # ? Scale by market cap
    df[col_to_scale[:-1]] = df[col_to_scale].apply(lambda x:  x[:-1] / x[-1], axis=1)
    

    return df 
   
def quantile_scale(df, dist='uniform'):
    """ Use with balance sheet growth, cash flow growth, income statement growth, key_metrics """
    
    df = deepcopy(df)

    col_to_scale = df.select_dtypes(include=np.number).columns

    scaler = QuantileTransformer(output_distribution=dist).fit(df[col_to_scale])
    
    df[col_to_scale] = scaler.transform(df[col_to_scale]) 
    
    if dist == 'normal':
        
            scaler = MinMaxScaler((-1, 1)).fit(df[col_to_scale])
            df[col_to_scale] = scaler.transform(df[col_to_scale])
    
    return df

def rolling_quantile_scale(df):
    
    df['calendarYear'] = df['calendarYear'].astype(int)
    second_df = deepcopy(df)   
     
    col_to_scale = second_df.select_dtypes(include=np.number).columns
    col_to_scale = col_to_scale.drop('calendarYear')    
    
    scaler = QuantileTransformer(output_distribution='uniform')

    
    failed = 0
    
    for year in df['calendarYear'].unique()[1:-1]: # ! Watchout  
        try:
            row_to_scale = df['calendarYear'] == year

            scaler = scaler.fit(df.loc[(df['calendarYear'] < year) & (df['calendarYear'] > year - 3), col_to_scale])
            
            second_df.loc[row_to_scale, col_to_scale] = scaler.transform(second_df.loc[row_to_scale, col_to_scale])
        
        except:
            failed += 1
            print('failed : ', failed)
      
    return second_df

def add_returns(tickers, df):
    """Input df without transforming date to datetime"""
    
    d = {}
    for t in tickers:
        
        condition = 0
        loops = 0
        while (condition == 0) & (loops < 5):
            try:
                loops += 1        
                d[t] = pd.DataFrame(fmpsdk.historical_price_full(API_KEY, t, from_date='1985-01-01'))
                d[t]['symbol'] = [t] * len(d[t])
                
                condition = 1
                
            except:
                print(f'failed fetching {t}, retrying')
                time.sleep(0.3)
                
    tickers_df = pd.concat(d)[['date', 'close', 'symbol']]
    
    
    df = df.merge(tickers_df, right_on=['symbol', 'date'], left_on=['symbol', 'date'], how='left')

    for t in tickers:
        try:
            df.loc[df['symbol'] == t, df.columns == 'close'] = df.loc[df['symbol'] == t, df.columns == 'close'].pct_change(periods=-1).shift(1)
        except:
            pass
    
    return df


#%%
# ? Columns to be shared and ranked by year across all sectors : MarketCap, EnterpriseValue, Beta, Mean returns, Volatility, Liquidity 

#%%
all_industries = ['Entertainment', 'Oil & Gas Midstream', 'Semiconductors', 'Specialty Industrial Machinery', 'Banks Diversified', 'Consumer Electronics', 'Software Infrastructure', 'Broadcasting', 'Computer Hardware', 'Building Materials', 'Resorts & Casinos', 'Auto Manufacturers', 'Internet Content & Information', 'Insurance Diversified', 'Telecom Services', 'Metals & Mining', 'Capital Markets', 'Steel', 'Footwear & Accessories', 'Household & Personal Products', 'Other Industrial Metals & Mining', 'Oil & Gas E&P', 'Banks Regional', 'Drug Manufacturers General', 'Internet Retail', 'Communication Equipment', 'Semiconductor Equipment & Materials', 'Oil & Gas Services', 'Chemicals', 'Electronic Gaming & Multimedia', 'Oil & Gas Integrated', 'Credit Services', 'Online Media', 'Business Services', 'Biotechnology', 'Grocery Stores', 'Oil & Gas Equipment & Services', 'REITs', 'Copper', 'Software Application', 'Home Improvement Retail', 'Pharmaceutical Retailers', 'Communication Services', 'Oil & Gas Drilling', 'Electronic Components', 'Packaged Foods', 'Information Technology Services', 'Leisure', 'Specialty Retail', 'Oil & Gas Refining & Marketing', 'Tobacco', 'Financial Data & Stock Exchanges', 'Insurance Specialty', 'Beverages Non-Alcoholic', 'Asset Management', 'REIT Diversified', 'Residential Construction', 'Travel & Leisure', 'Gold', 'Discount Stores', 'Confectioners', 'Medical Devices', 'Banks', 'Independent Oil & Gas', 'Airlines', 'Travel Services', 'Aerospace & Defense', 'Retail Apparel & Specialty', 'Diagnostics & Research', 'Trucking', 'Insurance Property & Casualty', 'Health Care Plans', 'Consulting Services', 'Aluminum', 'Beverages Brewers', 'REIT Residential', 'Education & Training Services', 'Apparel Retail', 'Railroads', 'Apparel Manufacturing', 'Staffing & Employment Services', 'Utilities Diversified', 'Agricultural Inputs', 'Restaurants', 'Drug Manufacturers General Specialty & Generic', 'Financial Conglomerates', 'Personal Services', 'Thermal Coal', 'REIT Office', 'Advertising Agencies', 'Farm & Heavy Construction Machinery', 'Consumer Packaged Goods', 'Publishing', 'Specialty Chemicals', 'Engineering & Construction', 'Utilities Independent Power Producers', 'Utilities Regulated Electric', 'Medical Instruments & Supplies', 'Building Products & Equipment', 'Packaging & Containers', 'REIT Mortgage', 'Department Stores', 'Insurance Life', 'Luxury Goods', 'Auto Parts', 'Autos', 'REIT Specialty', 'Integrated Freight & Logistics', 'Security & Protection Services', 'Utilities Regulated Gas', 'Airports & Air Services', 'Farm Products', 'REIT Healthcare Facilities', 'REIT Industrial', 'Metal Fabrication', 'Scientific & Technical Instruments', 'Solar', 'REIT Hotel & Motel', 'Medical Distribution', 'Medical Care Facilities', 'Agriculture', 'Food Distribution', 'Health Information Services', 'Industrial Products', 'REIT Retail', 'Conglomerates', 'Health Care Providers', 'Waste Management', 'Beverages Wineries & Distilleries', 'Marine Shipping', 'Real Estate Services', 'Tools & Accessories', 'Auto & Truck Dealerships', 'Industrial Distribution', 'Uranium', 'Lodging', 'Electrical Equipment & Parts', 'Gambling', 'Specialty Business Services', 'Recreational Vehicles', 'Furnishings', 'Fixtures & Appliances', 'Forest Products', 'Silver', 'Business Equipment & Supplies', 'Medical Instruments & Equipment', 'Utilities Regulated', 'Coking Coal', 'Insurance Brokers', 'Rental & Leasing Services', 'Lumber & Wood Production', 'Medical Diagnostics & Research', 'Pollution & Treatment Controls', 'Transportation & Logistics', 'Other Precious Metals & Mining', 'Brokers & Exchanges', 'Beverages Alcoholic', 'Mortgage Finance', 'Utilities Regulated Water', 'Manufacturing Apparel & Furniture', 'Retail Defensive', 'Real Estate Development', 'Paper & Paper Products', 'Insurance Reinsurance', 'Homebuilding & Construction', 'Coal', 'Electronics & Computer Distribution', 'Health Care Equipment & Services', 'Education', 'Employment Services', 'Textile Manufacturing', 'Real Estate Diversified', 'Consulting & Outsourcing', 'Utilities Renewable', 'Tobacco Products', 'Farm & Construction Machinery', 'Shell Companies', 'N/A', 'Advertising & Marketing Services', 'Capital Goods', 'Insurance', 'Industrial Electrical Equipment', 'Utilities', 'Pharmaceuticals', 'Biotechnology & Life Sciences', 'Infrastructure Operations', 'Energy', 'NULL', 'Property Management', 'Auto Dealerships', 'Apparel Stores', 'Mortgage Investment', 'Software & Services', 'Industrial Metals & Minerals', 'Media & Entertainment', 'Diversified Financials', 'Consumer Services', 'Commercial  & Professional Services', 'Electronics Wholesale', 'Retailing', 'Automobiles & Components', 'Materials', 'Real Estate', 'Food', 'Beverage & Tobacco', 'Closed-End Fund Debt', 'Transportation', 'Food & Staples Retailing', 'Consumer Durables & Apparel', 'Technology Hardware & Equipment', 'Telecommunication Services', 'Semiconductors & Semiconductor Equipment']


from multiprocessing import Pool


#%%

#%%

#%%
#tickers = pd.DataFrame(fmpsdk.stock_screener(API_KEY, exchange=['nyse', 'nasdaq'], market_cap_more_than=50000000, limit=7000, country='US', is_actively_trading=True, is_etf=False)).symbol.unique()
#tickers = tickers[-pd.DataFrame(tickers)[0].str.contains('X')]


#%%


#%%
if __name__ == '__main__':

        
            res = Parallel(30, max_nbytes=None)(delayed(get_metrics)(t) for t in tickers)
            

            list_of_dfs = []
            for r in res:
    
                if type(r) == float:
                    print(r)
                else:
                    list_of_dfs.append(r)
        
        
            list_of_dfs = pd.concat(list_of_dfs)
            list_of_dfs



#%%

            list_of_dfs = list_of_dfs.replace([np.inf, -np.inf], np.nan)

#%%
            list_of_dfs.loc[:, list_of_dfs.isna().sum() < 10000].dropna().to_csv('train_220k_250col.csv', index=False)
