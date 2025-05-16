import pandas as pd
from config import *
import os
os.chdir(DATA_PATH)

def process_fundamental_data():
    fundamental_df = pd.read_parquet(
        FUNDAMENTAL_FILE,
        engine='pyarrow',
        filters=[('I1', '==', INDUSTRY_L1_CODE)],
        columns=None
    )
    fundamental_df['month'] = fundamental_df['FDate'].dt.to_period('M')
    idx = fundamental_df.groupby(['SecCode', 'month'])['FDate'].idxmax()
    fundamental_df = fundamental_df.loc[idx].copy()
    fundamental_df = fundamental_df.drop(columns=['month']).reset_index(drop=True)

    missing_frac = fundamental_df.isna().mean()
    cols_to_drop = missing_frac[missing_frac > 0.2].index.tolist()

    fundamental_df = fundamental_df.drop(columns=cols_to_drop)
    num_cols = fundamental_df.columns.tolist()[10:]
    # num_cols = fundamental_df.select_dtypes(include=[np.number]).columns
    fundamental_df[num_cols] = (
        fundamental_df
        .groupby('FDate')[num_cols]
        .transform(lambda x: x.fillna(x.median()))
    )
    fundamental_df = fundamental_df.dropna().reset_index(drop=True)
    fundamental_df.sort_values(by='FDate', ascending=True, inplace=True)
    
    fundamental_df.to_csv(f'{FUNDAMENTAL_PROCESSED_PATH}/{INDUSTRY_L1_CODE}_fundamental.csv',index = False)
    return fundamental_df

if __name__ == "__main__":
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