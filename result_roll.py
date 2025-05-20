import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *
import warnings
warnings.filterwarnings('ignore')

industry_code_list = ['801040', '801120' ,'801200' ,'801140']
industry_name_list = ["钢铁" ,"食品饮料" ,"商贸零售" ,"轻工制造"]
industry_idx = 1
# industry_code = '801120' #'801040' #'801140' #'801200'
industry_code = industry_code_list[industry_idx]
industry_name = industry_name_list[industry_idx]

model_time = '2025-05-18_22-43-57' #'2025-05-18_02-08-13' #'2025-05-18_00-21-47' #'2025-05-18_01-31-28'
result_path = f'E:/intern/llm行业基本面量化课题数据/{industry_code}/{model_time}' #_result_factors_final
with open(result_path + f'/{industry_code}_result_factors_final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train_len = 63

factor_name_df = pd.DataFrame(data['优化因子列表'])
factor_name_df = factor_name_df.drop_duplicates(subset=['因子'], keep='first')
factor_name_df = factor_name_df[factor_name_df['累计回报(%)'] > 0]

factor_df = pd.read_csv(result_path + f'/{industry_code}_result_factors.csv')
factor_df['SecCode'] = factor_df['SecCode'].astype(str).str.zfill(6)
factor_df['FDate'] = pd.to_datetime(factor_df['FDate'])

factor_dates = sorted(factor_df['FDate'].unique())

# train_end_date = factor_dates[train_len + 1]
# factor_df = factor_df[factor_df['FDate'] >= train_end_date]

for i in range((len(factor_dates)-train_len+1)//3):
    if i != (len(factor_dates)-train_len+1)//3:
        # test_idx = list(range(train_len + i * 3 + 1, train_len + i * 3 + 4))
        test_date = factor_dates[train_len + i * 3 + 1: train_len + i * 3 + 4]
    else:
        # test_idx = list(range(train_len + i * 3 + 1, len(factor_dates)))
        test_date = factor_dates[train_len + i * 3 + 1: len(factor_dates)]

    val_df = factor_name_df.iloc[0 : train_len + i * 3 + 1]

    val_df['ICIR_val'] = val_df['信息系数ICIR_list'].apply(lambda lst: np.nanmean(lst[:train_len + i *3 + 1])/np.nanstd(lst[:train_len + i *3 + 1], ddof=1))

    val_df['return_val_1'] = val_df['累计回报(%)_list'].apply(lambda lst: lst[train_len + i *3 - 2])
    val_df['return_val_2'] = val_df['累计回报(%)_list'].apply(lambda lst: lst[train_len + i *3 - 1])
    val_df['return_val_3'] = val_df['累计回报(%)_list'].apply(lambda lst: lst[train_len + i *3])

    val_df = val_df.sort_values(by='ICIR_val', ascending=False).head(20)

    top_factors_val_1 = val_df.sort_values(by='return_val_1', ascending=False)['因子'].head(10).tolist()
    top_factors_val_2 = val_df.sort_values(by='return_val_2', ascending=False)['因子'].head(10).tolist()
    top_factors_val_3 = val_df.sort_values(by='return_val_3', ascending=False)['因子'].head(10).tolist()
    top_factors_list = [x for x in top_factors_val_3 if x in top_factors_val_1 and x in top_factors_val_2]

    corr = factor_df[factor_df['FDate'] <= factor_dates[train_len + i *3]][top_factors_list].corr().abs()
    selected_factors = []
    for factor in top_factors_list:
        if all(corr.loc[factor, sel] < 0.6 for sel in selected_factors):
            selected_factors.append(factor)
        if len(selected_factors) >= 5:
            break
    
    # test_date = factor_dates[test_idx]
    # for col in selected_factors:
    #     block = factor_df[factor_df['FDate'].isin(test_date)]
    #     block = factor_df.loc[test_idx, col]
    #     μ, std = block.mean(), block.std()
    #     factor_df.loc[test_idx,col] = (block - μ) / std
    # factor_df.loc[test_idx,'composite'] = factor_df.loc[test_idx,selected_factors].mean(axis=1)
    # # factor_df.dropna(subset='composite',inplace=True)

    mask = factor_df['FDate'].isin(test_date)
    block = factor_df.loc[mask, selected_factors]
    norm_block = block.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
    factor_df.loc[mask, selected_factors] = norm_block
    factor_df.loc[mask, 'composite'] = factor_df.loc[mask, selected_factors].mean(axis=1)
    
# factor_df.dropna(subset='composite',inplace=True)
factor_df = factor_df[factor_df['FDate']>=factor_dates[train_len+1]]

price_df = pd.read_parquet(
    "E:/intern/llm行业基本面量化课题数据/处理后的基础数据.parquet.gzip",
    engine='pyarrow',
    filters=[('I1', '==', industry_code)],
    columns=['FDate','SecCode','PRICE']
)

top_quantile = 0.1

price_df['FDate'] = pd.to_datetime(price_df['FDate'])
price_df = price_df[price_df['FDate']>=factor_dates[train_len+1]]
price_df = price_df.sort_values(['SecCode', 'FDate'])
price_df['Return'] = price_df.groupby('SecCode')['PRICE'].pct_change()


results = []

for idx, fdate in enumerate(factor_dates):
    month_factors = factor_df[factor_df['FDate'] == fdate]
    cutoff = month_factors['composite'].quantile(1 - top_quantile)
    portfolio = month_factors[month_factors['composite'] >= cutoff]['SecCode']

    start = fdate + pd.Timedelta(days=1)
    if idx + 1 < len(factor_dates):
        end = factor_dates[idx + 1]
    else:
        end = price_df['FDate'].max() + pd.Timedelta(days=1)

    period = price_df[(price_df['FDate'] >= start) & (price_df['FDate'] < end)]

    port_returns = (
        period[period['SecCode'].isin(portfolio)]
        .groupby('FDate')['Return']
        .mean()
        .rename('Portfolio_Return')
    )

    bench_returns = (
        period
        .groupby('FDate')['Return']
        .mean()
        .rename('Industry_Benchmark_Return')
    )

    df_period = pd.concat([port_returns, bench_returns], axis=1).reset_index()
    results.append(df_period)

result_df = pd.concat(results, ignore_index=True)
result_df.sort_values('FDate', inplace=True)




cum = (1 + result_df[['Portfolio_Return','Industry_Benchmark_Return']]).cumprod()
plt.figure()
plt.plot(result_df['FDate'], cum['Portfolio_Return'], label='Portfolio_Return')
plt.plot(result_df['FDate'], cum['Industry_Benchmark_Return'], label='Industry_Benchmark_Return')

plt.xlabel('Date')
plt.ylabel('Return')
plt.title(industry_code + ' Returns with Rolling Selection')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(result_path + f'/{industry_code}_rolling_cumulative_returns.png', dpi=300)
plt.close()

# test_df = result_df[result_df['FDate'] >= val_begin_date]
# test_cum = (1 + test_df[['Portfolio_Return','Industry_Benchmark_Return']]).cumprod()
# plt.figure()
# plt.plot(test_df['FDate'], test_cum['Portfolio_Return'], 
#          label='Portfolio_Return (Test)')
# plt.plot(test_df['FDate'], test_cum['Industry_Benchmark_Return'], 
#          label='Industry_Benchmark_Return (Test)')
# plt.xlabel('Date')
# plt.ylabel('Cumulative Return')
# plt.title(f'Test Period Cumulative Returns (from {val_begin_date})')
# plt.legend()
# plt.tight_layout()
# plt.savefig(result_path + '/test_cumulative_returns.png', dpi=300)