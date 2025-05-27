import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from config import *

industry_code_list = ['801040', '801120' ,'801200' ,'801140']
industry_name_list = ["钢铁" ,"食品饮料" ,"商贸零售" ,"轻工制造"]
industry_idx = 3
# industry_code = '801120' #'801040' #'801140' #'801200'
industry_code = industry_code_list[industry_idx]
industry_name = industry_name_list[industry_idx]
model_time = '2025-05-25_21-54-12' #'2025-05-18_02-08-13' #'2025-05-18_00-21-47' #'2025-05-18_01-31-28'
result_path = f'E:/intern/llm行业基本面量化课题数据/combined/{industry_code}/{model_time}' #_result_factors_final
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

stock_pool_df['FDate'] = pd.to_datetime(stock_pool_df['FDate'])
train_end_date = stock_pool_df['FDate'].unique()[train_len]
val_end_date   = stock_pool_df['FDate'].unique()[train_len + val_len]
factor_dates = sorted(stock_pool_df['FDate'].unique())

def compute_composite_pool(row, factors):
    inter = list(set(row[factors[0]]).intersection(row[factors[1]]))
    if not inter:
        return np.nan
    for factor in factors[2:5]:
        inter_next = list(set(inter).intersection(row[factor]))
        if not inter_next:
            return inter
        inter = inter_next
    return inter

stock_pool_df['composite'] = stock_pool_df.apply(
    lambda row: compute_composite_pool(row, top_factors_list),
    axis=1
)
val_full_df = factor_name_df.sort_values(by='return_val', ascending=False).head(5)
for i in range((len(factor_dates)-train_len+1)//3+1):
    if i != (len(factor_dates)-train_len+1)//3:
        test_date = factor_dates[train_len + i * 3 + 1: train_len + i * 3 + 4]
    else:
        test_date = factor_dates[train_len + i * 3 + 1: len(factor_dates)]
    val_df = val_full_df.iloc[0 : train_len + i * 3 + 1]
    val_df['return_val'] = val_df['累计回报(%)_list'].apply(lambda lst: sum(lst))
    top_factor = val_df.sort_values(by='return_val', ascending=False)['因子'].head(10).tolist()[0]
    mask = stock_pool_df['FDate'].isin(test_date)
    stock_pool_df.loc[mask, 'composite_rolling'] = stock_pool_df.loc[mask, top_factor]

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

results = []
for idx, fdate in enumerate(factor_dates):
    port_returns_list = []
    for factor in top_factors_list + ['composite','composite_rolling']:
        portfolio = stock_pool_df[stock_pool_df['FDate'] == fdate][factor].values[0]
        start = fdate + pd.Timedelta(days=1)
        if idx + 1 < len(factor_dates):
            end = factor_dates[idx + 1]
        else:
            end = price_df['FDate'].max() + pd.Timedelta(days=1)

        period = price_df[(price_df['FDate'] >= start) & (price_df['FDate'] < end)]
        if not (isinstance(portfolio, list) and len(portfolio) > 0):
            dates = period['FDate'].sort_values().unique()
            port_return = pd.Series(
                0,
                index=dates,
                name=factor
            )
            port_return.index.name = 'FDate'
        else:
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


cum = (1 + result_df[top_factors_list + ['composite','Industry_Benchmark_Return']]).cumprod()
plt.figure(figsize=(15, 9))
# plt.plot(result_df['FDate'], cum['Industry_Benchmark_Return'], label='Industry_Benchmark_Return')
# for factor in top_factors_list:
#     plt.plot(result_df['FDate'], cum[factor], label=factor)

cmap = cm.get_cmap('Blues')
plt.plot(
    result_df['FDate'],
    cum['Industry_Benchmark_Return'],
    label='Industry_Benchmark_Return',
    color='Green',
    linewidth=1
)
plt.plot(
    result_df['FDate'],
    cum['composite']+np.random.normal(scale=1e-3, size=len(cum)),
    label='composite(Test)',
    color='Purple',
    linewidth=1
)
n = len(top_factors_list)
colors = cmap(np.linspace(0.3, 0.8, n))

for col, factor in enumerate(top_factors_list):
    plt.plot(
        result_df['FDate'],
        cum[factor],
        label=factor,
        color=colors[col],
        linewidth=1
    )

plt.axvline(train_end_date, linestyle='--', label='train end')
plt.axvline(val_end_date,   linestyle='--', label='val end')

plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Portfolio vs Benchmark')
plt.legend(loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()

# plt.show()
plt.savefig(result_path + f'/{industry_code}_all_cumulative_returns.png', dpi=300)
plt.close()

test_df = result_df[result_df['FDate'] >= val_end_date]
test_cum = (1 + test_df[top_factors_list + ['composite','composite_rolling','Industry_Benchmark_Return']]).cumprod()

plt.figure(figsize=(15, 9))

# plt.plot(test_df['FDate'], test_cum['Industry_Benchmark_Return'], 
#          label='Industry_Benchmark_Return (Test)')
# for factor in top_factors_list:
#     plt.plot(test_df['FDate'], test_cum[factor], 
#             label=f'{factor} (Test)')
plt.plot(
    test_df['FDate'],
    test_cum['Industry_Benchmark_Return'],
    label='Industry_Benchmark_Return(Test)',
    color='Green',
    linewidth=1
)

plt.plot(
    test_df['FDate'],
    test_cum['composite']+np.random.normal(scale=1e-3, size=len(test_cum)),
    label='composite(Test)',
    color='Purple',
    linewidth=1
)

plt.plot(
    test_df['FDate'],
    test_cum['composite_rolling']+np.random.normal(scale=3e-3, size=len(test_cum)),
    label='composite_rolling(Test)',
    color='Red',
    linewidth=1
)

for col, factor in enumerate(top_factors_list):
    plt.plot(
        test_df['FDate'],
        test_cum[factor],
        label=factor,
        color=colors[col],
        linewidth=1
    )

plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title(industry_code + " Test Period Cumulative Returns (from {val_end_date})")
plt.legend(loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig(result_path + f'/{industry_code}_test_cumulative_returns.png', dpi=300)

# stock_pool_df.to_csv(result_path + '/stock_pool.csv')