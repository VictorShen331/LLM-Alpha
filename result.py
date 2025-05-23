import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *

industry_code_list = ['801040', '801120' ,'801200' ,'801140']
industry_name_list = ["钢铁" ,"食品饮料" ,"商贸零售" ,"轻工制造"]
industry_idx = 1
# industry_code = '801120' #'801040' #'801140' #'801200'
industry_code = industry_code_list[industry_idx]
industry_name = industry_name_list[industry_idx]
model_time = '2025-05-22_18-23-07' #'2025-05-18_02-08-13' #'2025-05-18_00-21-47' #'2025-05-18_01-31-28'
result_path = f'E:/intern/llm行业基本面量化课题数据/non_numerical/{industry_code}/{model_time}' #_result_factors_final
with open(result_path + f'/{industry_code}_result_factors_final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train_len = 63
val_len = 20

factor_name_df = pd.DataFrame(data['优化因子列表'])
factor_name_df = factor_name_df.drop_duplicates(subset=['因子'], keep='first')
factor_name_df = factor_name_df[factor_name_df['累计回报(%)'] > 0]
factor_name_df['return_val'] = (
    factor_name_df['累计回报(%)_list']
    .apply(lambda lst: sum(lst[train_len + 1:train_len + val_len + 1]))
)
# factor_name_df['return_val_1'] = (
#     factor_name_df['累计回报(%)_list']
#     .apply(lambda lst: sum(lst[train_len + 1:train_len + val_len//2 + 1]))
# )

# factor_name_df['return_val_2'] = (
#     factor_name_df['累计回报(%)_list']
#     .apply(lambda lst: sum(lst[train_len + val_len//2 + 1:train_len + val_len + 1]))
# )

# top_factors_val_1 = factor_name_df.sort_values(by='return_val_1', ascending=False)['因子'].head(10).tolist()
# top_factors_val_2 = factor_name_df.sort_values(by='return_val_2', ascending=False)['因子'].head(10).tolist()

# top_factors_list = [x for x in top_factors_val_1 if x in top_factors_val_2]
top_factors_list = factor_name_df.sort_values(by='return_val', ascending=False)['因子'].head(5).tolist()

print(top_factors_list)

# factor_dates = sorted(factor_df['FDate'].unique())

# stock_pool_list = pd.DataFrame({'FDate': dates})
# for factor in top_factors_list:
#     file_path = result_path + f"/{factor_name_df[factor_name_df['因子'] == factor]['gen']}/{factor}.csv"
#     stock_df = pd.read_csv(file_path, index_col = 0)
#     stocks_df = stocks_df.reset_index().rename(columns={'index': 'FDate'})
# factor_df = pd.read_csv(result_path + f'/{industry_code}_result_factors.csv')
# selected_factors = top_factors_list
# factor_df = factor_df[['FDate','SecCode']+selected_factors]
stock_pool_df = pd.DataFrame()

for factor in top_factors_list:
    gen = factor_name_df.loc[factor_name_df['因子'] == factor, '迭代次数'].iat[0]
    file_path = f"{result_path}/{gen}/{factor}.csv"
    
    df_wide = pd.read_csv(file_path, index_col=0)
    df_wide.columns = df_wide.columns.astype(str).str.zfill(6)
    
    stock_lists = df_wide.apply(lambda row: row[row == 1].index.tolist(), axis=1)
    stock_lists.name = factor
    
    if stock_pool_df.empty:
        stock_pool_df = stock_lists.to_frame()
    else:
        stock_pool_df = stock_pool_df.join(stock_lists, how='outer')
stock_pool_df = stock_pool_df.reset_index().rename(columns={'index': 'FDate'})

# 将可能的 NaN 填成空列表
# for factor in top_factors_list:
#     stock_pool_list[factor] = stock_pool_list[factor].apply(lambda x: x if isinstance(x, list) else [])

# series_list = []

# for factor in top_factors_list:
#     gen = factor_name_df.loc[
#         factor_name_df['因子'] == factor, '迭代次数'
#     ].iat[0]
#     file_path = f"{result_path}/{gen}/{factor}.csv"
    
#     df = pd.read_csv(file_path, index_col=0)
#     df['SecCode'] = df['SecCode'].astype(str).str.zfill(6)
#     pools = df.groupby(df.index)['SecCode'] \
#               .apply(list) \
#               .rename(factor)
    
#     series_list.append(pools)

# stock_pool_df = pd.concat(series_list, axis=1)

# # 6) 把日期索引变回列
# stock_pool_df = stock_pool_df.reset_index() \
#                              .rename(columns={'index': 'FDate'})

stock_pool_df['FDate'] = pd.to_datetime(stock_pool_df['FDate'])
train_end_date = stock_pool_df['FDate'].unique()[train_len]
val_end_date   = stock_pool_df['FDate'].unique()[train_len + val_len]

# for col in selected_factors:
#     factor_df[col] = (factor_df[col] - factor_df[col].mean()) / factor_df[col].std()
# factor_df['composite'] = factor_df[selected_factors].mean(axis=1)
# factor_df.dropna(subset='composite',inplace=True)
# factor_df['SecCode'] = factor_df['SecCode'].astype(str).str.zfill(6)

price_df = pd.read_parquet(
    "E:/intern/llm行业基本面量化课题数据/处理后的基础数据.parquet.gzip",
    engine='pyarrow',
    filters=[('I1', '==', industry_code)],
    columns=['FDate','SecCode','PRICE']
)

top_quantile = 0.1

price_df['FDate'] = pd.to_datetime(price_df['FDate'])

price_df = price_df.sort_values(['SecCode', 'FDate'])
price_df['Return'] = price_df.groupby('SecCode')['PRICE'].pct_change()

factor_dates = sorted(stock_pool_df['FDate'].unique())


results = []
for idx, fdate in enumerate(factor_dates):
    port_returns_list = []
    for factor in top_factors_list:
        portfolio = stock_pool_df[stock_pool_df['FDate'] == fdate][factor].values[0]
        start = fdate + pd.Timedelta(days=1)
        if idx + 1 < len(factor_dates):
            end = factor_dates[idx + 1]
        else:
            end = price_df['FDate'].max() + pd.Timedelta(days=1)

        period = price_df[(price_df['FDate'] >= start) & (price_df['FDate'] < end)]

        port_return = (
            period[period['SecCode'].isin(portfolio)]
            .groupby('FDate')['Return']
            .mean()
            .rename(factor)
        )
        port_returns_list.append(port_return)
    port_returns = pd.concat(port_returns_list, axis=1)
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




cum = (1 + result_df[top_factors_list + ['Industry_Benchmark_Return']]).cumprod()
plt.figure()
plt.plot(result_df['FDate'], cum['Industry_Benchmark_Return'], label='Industry_Benchmark_Return')
for factor in top_factors_list:
    plt.plot(result_df['FDate'], cum[factor], label=factor)

plt.axvline(train_end_date, linestyle='--', label='train end')
plt.axvline(val_end_date,   linestyle='--', label='val end')

plt.xlabel('Date')
plt.ylabel('Return')
plt.title(industry_code + " Daily Portfolio vs Industry Benchmark Returns")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(result_path + f'/{industry_code}_all_cumulative_returns.png', dpi=300)
plt.close()

test_df = result_df[result_df['FDate'] >= val_end_date]
test_cum = (1 + test_df[top_factors_list + ['Industry_Benchmark_Return']]).cumprod()
plt.figure()
plt.plot(test_df['FDate'], test_cum['Industry_Benchmark_Return'], 
         label='Industry_Benchmark_Return (Test)')
for factor in top_factors_list:
    plt.plot(test_df['FDate'], test_cum[factor], 
            label=f'{factor} (Test)')

plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title(industry_code + " Test Period Cumulative Returns (from {val_end_date})")
plt.legend()
plt.tight_layout()
plt.savefig(result_path + f'/{industry_code}_test_cumulative_returns.png', dpi=300)