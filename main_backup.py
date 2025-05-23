import os
import json
import pandas as pd
from new_operators import OPERATORS, FactorTest, DouFactorChoose, DouFactorChooseDrop, DouFactorChooseSig
from preprocess import process_fundamental_data
from config import *

# ALPHA_META_FILE   = 'alpha.csv'       # 因子元数据
# REPORT_FILE       = 'report.csv'      # 行业研报摘要
# FUNDAMENTAL_FILE  = 'fundamental.csv' # 包含 FDate, SecCode, PRICE 及原始因子列
OUTPUT_DIR        = 'output'
TOP_QUANTILE      = 0.2               # 示例 top 20%

# —— 1. 数据加载 ——
def load_data():
    # operator_df = pd.read_excel(OPERATOR_FILE)
    alpha_df = pd.read_excel(DEFINITION_FILE, sheet_name = '因子映射表')

    processed_fundamental_path = f'{FUNDAMENTAL_PROCESSED_PATH}/{INDUSTRY_L1_CODE}_fundamental.csv'
    if os.path.exists(processed_fundamental_path):
        fund_df = pd.read_csv(processed_fundamental_path)
    else:
        fund_df = process_fundamental_data()
    # 计算收益
    fund_df = fund_df.sort_values(['FDate','SecCode']).reset_index(drop=True)
    fund_df['return']      = fund_df.groupby('SecCode')['PRICE'].pct_change()
    fund_df['next_return'] = fund_df.groupby('SecCode')['return'].shift(-1)
    return alpha_df, fund_df # , report_df

# —— 2. 构建 Prompt ——
def build_prompt(industry_l1_name, report_df, alpha_df):
    report_sec = report_df.to_string(index=False)
    alpha_sec  = alpha_df.to_string(index=False)

    prompt = f"""
你是一名专业的量化分析师，专长在于深入分析及优化因子以提升行业内多头收益。
当前研究范围：申万一级行业“{industry_l1_name}”。
可参考的行业相关研报信息：
{report_sec}

用户将提供基本面基础数据和特定的算子。你的任务是对这些基础数据和算子进行细致的分析，理解其逻辑，并用基础数据和算子构建新的因子。
目标是在维持因子结构相对简洁的前提下，通过组合算子和数据来增强因子的有效性。

可用算子列表：
"""
    for OpCla in OPERATORS:
        prompt += f"- {OpCla.name}: {OpCla.description}\n"
    prompt += f"\n以下是基本面因子元数据：\n{alpha_sec}\n"
    return prompt

# —— 3. 实例化算子 ——
def instantiate_operators(alpha_df):
    ops = []
    # 构造原始列名
    alpha_df['raw_col'] = alpha_df['因子代码'] + '_' + alpha_df['因子后缀']
    # 对方向 -1 的列翻转符号
    for _, row in alpha_df.iterrows():
        if row.get('因子方向', 1) == -1:
            col = row['raw_col']
            # 运行时需先确保 FUNDAMENTAL_FILE 中已包含此列

    # 针对每个原始因子，都创建一个 factor_test 实例
    for col in alpha_df['raw_col']:
        ops.append(FactorTest(col, top_quantile=TOP_QUANTILE))

    # 如需添加 DouFactorChoose 等其他算子，可按实际需求加载：
    if len(alpha_df) >= 2:
        f1 = alpha_df.loc[0, 'raw_col']
        f2 = alpha_df.loc[1, 'raw_col']
        ops.append(DouFactorChoose(f1, f2))
    if len(alpha_df) >= 3:
        f1, f2, f3 = alpha_df.loc[0,'raw_col'], alpha_df.loc[1,'raw_col'], alpha_df.loc[2,'raw_col']
        ops.append(DouFactorChooseDrop(f1, f2, f3, drop_pct=TOP_QUANTILE))
        ops.append(DouFactorChooseSig(f1, f2, f3))

    return ops

# —— 4. 回测逻辑 ——
def backtest(fund_df, ops):
    dates = sorted(fund_df['FDate'].unique())[:-1]
    result_json = {"优化因子列表": []}
    pool_records = []

    for op in ops:
        monthly_rets = []
        pool_per_date = {}
        for dt in dates:
            slice_df = fund_df[fund_df['FDate'] == dt]
            mask = op.apply(slice_df)
            selected = slice_df.loc[mask, 'next_return'].dropna()
            monthly_rets.append(selected.mean() if not selected.empty else 0.0)
            pool_per_date[dt] = slice_df.loc[mask, 'SecCode'].tolist()

        cum_rets = list(pd.Series(monthly_rets).cumsum())
        result_json['优化因子列表'].append({
            '因子': op.name,
            '月度回测表现': monthly_rets,
            '累计回测表现': cum_rets
        })

        # 构建 0/1 矩阵
        codes = sorted(fund_df['SecCode'].unique())
        mat = pd.DataFrame(0, index=dates, columns=codes)
        for dt, codes_sel in pool_per_date.items():
            mat.loc[dt, codes_sel] = 1
        pool_records.append((op.name, mat))

    return result_json, pool_records

# —— 5. 保存结果 ——
def save_results(result_json, pool_records):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'factors_with_return.json'), 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)
    for name, mat in pool_records:
        mat.to_csv(os.path.join(OUTPUT_DIR, f'pool_{name}.csv'), index_label='FDate')

# —— 主流程 ——
if __name__ == '__main__':
    # 1. 加载数据
    alpha_df, report_df, fund_df = load_data()
    # 2. 构建并打印 Prompt
    prompt = build_prompt('示例行业', report_df, alpha_df)
    print(prompt)
    # 3. 实例化算子
    operators = instantiate_operators(alpha_df)
    # 4. 回测
    result_json, pool_records = backtest(fund_df, operators)
    # 5. 保存
    save_results(result_json, pool_records)
    print('回测与选股池输出完成，见 output/ 目录')
