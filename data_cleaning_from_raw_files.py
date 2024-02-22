#%%
import glob 
import pandas as pd 
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

#%%
datas_path = glob.glob('/home/fast-pc-2023/Téléchargements/AAA_PYTHON/variational_inference/data_raw/*.csv')

d = {}

for path in datas_path:
    d[path] = pd.read_csv(path)
    

df = pd.concat(d).droplevel(0)
df

# ? We have many NANs for cash conversion cycle and operating cycle which can be computed based on other metrics with less NaN
# ? So we remove theme along links and reported currency 
print(df.isna().sum().sort_values().tail(20))
print(df.info())
print(df.select_dtypes('object').columns)

df.drop(['cashConversionCycle', 'operatingCycle', 'link', 'finalLink', 'reportedCurrency', 'acceptedDate'], axis=1, inplace=True)


for c in df.select_dtypes(include=np.number).columns:
    
    df[c] = df[c].clip(df[c].quantile(0.01), df[c].quantile(0.99))

df.to_csv('raw_file.csv')




#%%



df = pd.read_csv('raw_file.csv')

  

list_of_list_corr_95 = ['grossProfitMargin', 'grossProfitRatio'], ['payoutRatio', 'dividendPayoutRatio'], ['priceToSalesRatio', 'priceSalesRatio'], ['priceCashFlowRatio', 'priceToOperatingCashFlowsRatio', 'pocfratio', 'evToOperatingCashFlow'], ['daysOfSalesOutstanding', 'daysSalesOutstanding'], ['daysOfPayablesOutstanding', 'daysPayablesOutstanding'], ['daysOfInventoryOutstanding', 'daysOfInventoryOnHand'], ['operatingProfitMargin', 'ebitPerRevenue', 'operatingIncomeRatio'], ['pretaxProfitMargin', 'incomeBeforeTaxRatio'], ['netProfitMargin', 'netIncomeRatio'], ['returnOnAssets', 'returnOnTangibleAssets'], ['returnOnEquity', 'roe'], ['debtRatio', 'debtToAssets'], ['debtEquityRatio', 'debtToEquity', 'investedCapital'], ['cashFlowToDebtRatio', 'cashFlowCoverageRatios'], ['freeCashFlowOperatingCashFlowRatio', 'capexToOperatingCashFlow'], ['priceBookValueRatio', 'priceToBookRatio', 'priceFairValue', 'ptbRatio', 'pbRatio'], ['priceEarningsRatio', 'peRatio'], ['priceToFreeCashFlowsRatio', 'pfcfRatio', 'evToFreeCashFlow'], ['enterpriseValueMultiple', 'enterpriseValueOverEBITDA'], ['ebitgrowth','operatingIncomeGrowth'], ['epsgrowth', 'epsdilutedGrowth'], ['netReceivables', 'averageReceivables'], ['inventory', 'averageInventory'], ['totalAssets', 'totalLiabilities', 'totalLiabilitiesAndStockholdersEquity', 'totalLiabilitiesAndTotalEquity'], ['accountPayables', 'averagePayables'], ['longTermDebt', 'totalDebt'], ['totalStockholdersEquity', 'totalEquity'], ['totalDebt', 'netDebt'], ['netIncome', 'incomeBeforeTax'], ['netCashProvidedByOperatingActivities', 'operatingCashFlow'], ['investmentsInPropertyPlantAndEquipment', 'capitalExpenditure'], ['cashAtEndOfPeriod', 'cashAtBeginningOfPeriod'], ['revenue', 'costAndExpenses'], ['eps', 'epsdiluted', 'netIncomePerShare'], ['weightedAverageShsOut', 'weightedAverageShsOutDil'], ['bookValuePerShare', 'shareholdersEquityPerShare'], ['marketCap', 'enterpriseValue'], ['growthNetCashProvidedByOperatingActivites', 'growthOperatingCashFlow'], ['growthInvestmentsInPropertyPlantAndEquipment', 'growthCapitalExpenditure'], ['growthEPS', 'growthEPSDiluted']

list_of_list_corr_90 = [['pretaxProfitMargin', 'netProfitMargin'], ['longTermDebtToCapitalization', 'totalDebtToCapitalization'], ['priceToSalesRatio', 'evToSales'], ['netIncomeGrowth', 'epsgrowth'], ['cashAndCashEquivalents', 'cashAtEndOfPeriod'], ['goodwill', 'goodwillAndIntangibleAssets'], ['longTermInvestments', 'totalInvestments'], ['purchasesOfInvestments', 'salesMaturitiesOfInvestments'], ['ebitda', 'operatingIncome'], ['growthCashAndCashEquivalents', 'growthCashAndShortTermInvestments'], ['growthTotalAssets', 'growthTotalLiabilitiesAndStockholdersEquity'], ['growthNetIncome', 'growthNetIncome.1', 'growthNetIncomeRatio', 'growthEPS'], ['growthOperatingIncome', 'growthOperatingIncomeRatio'], ['growthIncomeBeforeTax', 'growthIncomeBeforeTaxRatio'], ['growthWeightedAverageShsOut', 'growthWeightedAverageShsOutDil']]

list_of_list_corr_85 = [['operatingProfitMargin', 'ebitdaratio'],['debtRatio', 'longTermDebtToCapitalization'], ['weightedAverageSharesGrowth', 'weightedAverageSharesDilutedGrowth'], ['propertyPlantEquipmentNet', 'investmentsInPropertyPlantAndEquipment'], ['totalNonCurrentAssets', 'totalStockholdersEquity'], ['totalAssets', 'netCurrentAssetValue'], ['longTermDebt', 'totalNonCurrentLiabilities'], ['netIncome', 'ebitda'], ['netCashProvidedByOperatingActivities', 'freeCashFlow'], ['revenue', 'costOfRevenue'], ['growthCashAndCashEquivalents', 'growthCashAtEndOfPeriod'], ['growthEBITDA', 'growthEBITDARatio']]

list_of_list_corr_80 = [['currentRatio', 'quickRatio'], ['operatingProfitMargin', 'pretaxProfitMargin'], ['operatingCashFlowPerShare', 'freeCashFlowPerShare'], ['cashAndCashEquivalents', 'cashAndShortTermInvestments'], ['totalAssets', 'totalCurrentAssets','totalNonCurrentAssets','longTermDebt'], ['revenue', 'accountPayables', 'grossProfit', 'sellingGeneralAndAdministrativeExpenses', 'operatingExpenses'], ['totalCurrentLiabilities', 'otherCurrentLiabilities'], ['netIncome', 'incomeTaxExpense'], ['bookValuePerShare', 'grahamNumber'], ['growthNetIncome', 'growthIncomeBeforeTax']]


for sub in list_of_list_corr_95:
    
    df.drop(sub[1:], axis=1, inplace=True)
    
    
    
growth_columns = df.loc[:, df.columns.str.contains('^growth|Growth', regex=True)].columns

df = df.drop(growth_columns, axis=1)    
df.drop(['cik', 'Unnamed: 0', 'otherExpenses', 'effectOfForexChangesOnCash', 'commonStockIssued', 'freeCashFlowYield'], axis=1, inplace=True)
df
# %%

from copy import deepcopy
df_test = deepcopy(df)
df_test = df_test.dropna()

# ? for those groups quantile scale and then apply PCA or dim red methods

dict_of_groups = {
    'liquidity_ratios': ['currentRatio', 'quickRatio', 'cashRatio'],
    'efficiency_ratios': ['daysOfSalesOutstanding', 'grossProfitMargin', 'receivablesTurnover', 'payablesTurnover', 'inventoryTurnover', 'fixedAssetTurnover', 'assetTurnover'],
    'sales_ratios': ['daysOfInventoryOutstanding'],
    'profitability_ratios': ['evToSales', 'daysOfPayablesOutstanding', 'earningsYield', 'operatingProfitMargin', 'pretaxProfitMargin', 'netProfitMargin', 'salesGeneralAndAdministrativeToRevenue', 'returnOnAssets', 'returnOnEquity', 'returnOnCapitalEmployed', 'researchAndDdevelopementToRevenue', 'capexToRevenue', 'capexToDepreciation', 'stockBasedCompensationToRevenue', 'roic', 'interestCoverage', 'operatingCashFlowSalesRatio', 'capitalExpenditureCoverageRatio', 'dividendPaidAndCapexCoverageRatio', 'priceToSalesRatio', 'epsgrowth', 'ebitdaratio', 'eps'],
    'tax_ratios': ['effectiveTaxRate', 'netIncomePerEBT'],
    'debt_ratios': ['netDebtToEBITDA', 'enterpriseValueMultiple', 'ebtPerEbit', 'debtRatio', 'longTermDebtToCapitalization', 'totalDebtToCapitalization'],
    'leverage_ratios': ['debtEquityRatio', 'priceBookValueRatio', 'companyEquityMultiplier'],
    'coverage_ratios': ['shortTermCoverageRatios', 'cashFlowToDebtRatio'],
    'per_share_ratios': ['revenuePerShare', 'bookValuePerShare', 'tangibleBookValuePerShare', 'interestDebtPerShare', 'grahamNumber', 'grahamNetNet', 'capexPerShare', 'operatingCashFlowPerShare', 'freeCashFlowPerShare', 'cashPerShare'],
    'dividend_ratios': ['incomeQuality', 'dividendYield', 'priceEarningsRatio', 'payoutRatio'],
    'cash_flow_ratios': ['priceToFreeCashFlowsRatio', 'freeCashFlowOperatingCashFlowRatio', 'priceCashFlowRatio'],
    'growth_ratios': ['ebitgrowth'],
    'income_statement': ['commonStockRepurchased', 'cashAtEndOfPeriod', 'freeCashFlow', 'cashAndCashEquivalents', 'retainedEarnings', 'grossProfit', 'netReceivables', 'generalAndAdministrativeExpenses', 'netIncome', 'sellingGeneralAndAdministrativeExpenses', 'operatingExpenses', 'operatingIncome', 'incomeTaxExpense'],
    'balance_sheet': ['workingCapital', 'tangibleAssetValue', 'netCurrentAssetValue', 'shortTermInvestments', 'cashAndShortTermInvestments', 'otherCurrentAssets', 'totalCurrentAssets', 'longTermInvestments', 'otherNonCurrentAssets', 'otherAssets', 'totalAssets', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'otherLiabilities', 'preferredStock', 'totalInvestments', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'interestIncome'],
    'cash_flow_statement': ['capitalLeaseObligations', 'revenue', 'costOfRevenue', 'inventory', 'accountPayables'],
    'other_assets': ['propertyPlantEquipmentNet', 'taxAssets', 'totalNonCurrentAssets', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'commonStock', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'minorityInterest', 'depreciationAndAmortization', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'netCashUsedForInvestingActivites', 'dividendsPaid', 'interestExpense', 'ebitda', 'weightedAverageShsOut'],
    'intangibles': ['intangiblesToTotalAssets', 'researchAndDevelopmentExpenses', 'sellingAndMarketingExpenses', 'intangibleAssets', 'goodwill', 'stockBasedCompensation', 'goodwillAndIntangibleAssets'],
    'other_income_expenses': ['otherNonCashItems', 'deferredIncomeTax', 'changeInWorkingCapital', 'otherWorkingCapital', 'totalOtherIncomeExpensesNet'],
    'accounts_management': ['accountsReceivables', 'accountsPayables'],
    'investing_activities': ['otherInvestingActivites', 'acquisitionsNet'],
    'financing_activities': ['otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'netChangeInCash', 'debtRepayment'],
    #'market_cap': ['marketCap']
}



scaler = QuantileTransformer()

for k in dict_of_groups.keys():
    
    scaler.fit(df_test[dict_of_groups[k]])
    print(k)
    df_test[dict_of_groups[k]] = scaler.transform(df_test[dict_of_groups[k]])
    df_test[k] = PCA(n_components=1).fit_transform(df_test[dict_of_groups[k]])
    df_test.drop(dict_of_groups[k], axis=1, inplace=True)
df_test   
#%%
# Growth Columns now 
df_test.to_csv('df_network_analysis.csv', index=False)

#%%



# %%

df_test = pd.read_csv('df_test.csv')
df_test
#%%
ProfileReport(df_test.drop(to_drop, axis=1).sample(frac=0.5)).to_file('report.html', silent=False)
#%%
import seaborn as sns 

sns.heatmap(df_test[['inventoryGrowth', 'assetGrowth']].corr())
# %%
(df_test == 0).sum(axis=0).sort_values().tail(50)#.index.to_list()

#%%

mostly_zeros = ['otherExpenses',
 'stockBasedCompensation',
 'generalAndAdministrativeExpenses',
 'dividendsPaid',
 'dividendsperShareGrowth',
 'accountsReceivables',
 'accountsPayables',
 'taxAssets',
 'debtRepayment',
 'effectOfForexChangesOnCash',
 'commonStockIssued',
 'commonStockRepurchased',
 'purchasesOfInvestments',
 'acquisitionsNet',
 'interestIncome',
 'minorityInterest',
 'researchAndDevelopmentExpenses',
 'rdexpenseGrowth',
 'capitalLeaseObligations',
 'sellingAndMarketingExpenses']


#%%

to_drop = ['minorityInterest', 'commonStockRepurchased', 'commonStockIssued', 'acquisitionsNet', 'debtRepayment', 'effectOfForexChangesOnCash', 'capitalLeaseObligations', 'weightedAverageShsOut', 'weightedAverageSharesGrowth', 'taxAssets', 'depreciationAndAmortization', 'deferredIncomeTax', 'interestDebtPerShare', 'capexPerShare', 'otherWorkingCapital', 'workingCapital']

#%%

df_test.drop(to_drop, axis=1).to_csv('df_test_65_col.csv', index=False)