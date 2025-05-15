import pandas as pd
from config import *
import os
os.chdir(DATA_PATH)

report_industry_df = pd.read_csv(
    INDUSTRY_REPORT_FILE,
    usecols=['rpt_type_name', 'industry_code_lev1','industry_name_lev1','title','organ_name','author_name','create_date'])
report_industry_df = report_industry_df[report_industry_df['industry_code_lev1'] == INDUSTRY_L1_CODE]
industry_l1_name = report_industry_df['industry_name_lev1'].iloc[0]
report_industry_df.drop(columns=['industry_code_lev1','industry_name_lev1'], inplace  =True)

# operator_df = pd.read_excel(OPERATOR_FILE)
# alpha_df = pd.read_excel(DEFINITION_FILE, sheet_name = '因子映射表')

report_industry_df.to_csv(f'industry_report_{INDUSTRY_L1_CODE}.csv', encoding="utf-8-sig",index=False)
# alpha_df.to_csv('alpha.csv', encoding="utf-8-sig",index=False)
# operator_df.to_csv('operator.csv', encoding="utf-8-sig",index=False)

# def process_fundamental_and_alpha():
#     fundamental_df = pd.read_parquet(
#         FUNDAMENTAL_FILE,
#         engine='pyarrow',
#         filters=[('I1', '==', INDUSTRY_L1_CODE)],
#     )
#     missing_frac = fundamental_df.isna().mean()
#     cols_to_drop = missing_frac[missing_frac >= 0.2].index.tolist()
#     print(f"缺失率 >=20% 的列：{cols_to_drop}")
#     fund = fundamental_df.drop(columns=cols_to_drop).copy()

#     fund['FDate'] = pd.to_datetime(fund['FDate'])
#     fund = fund.sort_values(['SecCode', 'FDate'])

#     monthly = (
#         fund
#         .groupby('SecCode')
#         .apply(lambda g: g.set_index('FDate').resample('M').ffill())
#         .reset_index(level=0)
#         .rename_axis('FDate')
#         .reset_index()
#     )

#     monthly = monthly.dropna().reset_index(drop=True)

#     date_bounds = monthly.groupby('SecCode')['FDate'].agg(['min', 'max'])
#     common_start = date_bounds['min'].max()
#     common_end   = date_bounds['max'].min()

#     monthly_aligned = monthly[
#         (monthly['FDate'] >= common_start) &
#         (monthly['FDate'] <= common_end)
#     ].reset_index(drop=True)

#     fundamental_monthly_df = monthly_aligned
#     fundamental_monthly_df.to_csv(f'{INDUSTRY_L1_CODE}_fundamental.csv')

#     alpha_df = pd.read_excel(DEFINITION_FILE, sheet_name = '因子映射表')
#     alpha_df_filtered = alpha_df[~alpha_df['因子代码'].isin(cols_to_drop)].reset_index(drop=True)
#     alpha_df_filtered.to_csv(f'{INDUSTRY_L1_CODE}_alpha.csv', encoding="utf-8-sig",index=False)
#     return fundamental_monthly_df, alpha_df_filtered