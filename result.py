import json
import pandas as pd
import matplotlib.pyplot as plt

result_path = 'E:/intern/llm行业基本面量化课题数据/2025-05-15_20-02-48'
with open(result_path + '/801120_good_factors_final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

factor_name_df = pd.DataFrame(data['优化因子列表'])
factor_name_df.sort_values(by='累计回报(%)', ascending=False).drop_duplicates(subset=['因子'], keep='first', inplace=True)
factor_name_df.sort_values(by='累计回报(%)', ascending=False, inplace=True)
top5_factors_list = factor_name_df['因子'].head(5).tolist()

factor_df = pd.read_csv(result_path + '/801120_good_factors.csv')
factor_df = factor_df[['FDate','SecCode']+top5_factors_list]

for col in top5_factors_list:
    factor_df[col] = (factor_df[col] - factor_df[col].mean()) / factor_df[col].std()
factor_df['composite'] = factor_df[top5_factors_list].mean(axis=1)
factor_df.dropna(inplace=True)
factor_df['SecCode'] = factor_df['SecCode'].astype(str).str.zfill(6)

price_df = pd.read_parquet(
    "E:/intern/llm行业基本面量化课题数据/处理后的基础数据.parquet.gzip",
    engine='pyarrow',
    filters=[('I1', '==', '801120')],
    columns=['FDate','SecCode','PRICE']
)

top_quantile = 0.1

factor_df['FDate'] = pd.to_datetime(factor_df['FDate'])
price_df['FDate'] = pd.to_datetime(price_df['FDate'])

price_df = price_df.sort_values(['SecCode', 'FDate'])
price_df['Return'] = price_df.groupby('SecCode')['PRICE'].pct_change()

factor_dates = sorted(factor_df['FDate'].unique())
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
plt.title('Daily Portfolio vs Industry Benchmark Returns')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(result_path + '/daily_returns.png', dpi=300)