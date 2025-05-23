import os
import re
import json
import datetime
import pandas as pd
import numpy as np
import warnings
from config import *
from openai import OpenAI
# from alpha import FactorEvaluator
# from backtest import Backtest
from preprocess import *
from new_operators import *

warnings.filterwarnings("ignore")

def build_prompt(industry_l1_name, operator_df, alpha_df, prev_factors = None, objective = '行业内TOP10%多头收益') -> str:
    operator_records = [tuple(x) for x in operator_df.itertuples(index=False, name=None)]

    # 3. 把 alpha_df 转成 list of tuples
    alpha_records = [tuple(x) for x in alpha_df.itertuples(index=False, name=None)]
    prompt = f"""
            你是一名专业的量化分析师，你的专长在于深入分析及优化因子以提升{objective}，请生成能够在同一行业对冲行业涨跌、只捕获个股超额收益的因子。
            当前研究范围：申万一级行业“{industry_l1_name}”。

            用户将提供基本面基础数据和特定的算子。 你的任务是对这些基础数据和算子进行细致的分析，理解其逻辑，并用基础数据和算子构建新的因子。
            目标是在维持因子结构相对简洁的前提下，通过组合算子和数据来增强因子{objective}。

            以下是基本面数据及其构建定义和经济学经验判断的方向与类别：
            {alpha_records}。
            对基本面数据，请注意：1.大多季度更新，具体更新时段存在个股差异；
                                2.储存为月度数据，具体为取月尾最后一天的数值。可能存在季度间数值相同的情况，请在计算因子的逻辑中考虑到这一特殊情况，避免产生误差。
            
            以下是可用算子以及其定义和适用数据类型：{operator_records}。
            请用这些算子和基础数据，生成可解释的月度多头调仓因子表达式，并说明每个因子背后的逻辑。

            因子具体格式例如：dou_factor_choose(perf7, qua8);dou_factor_choose_dropa(qua5,perf26,lia2)

            """
# ts_pct(perf26,12)；cs_rank(ts_min(qua5,24))
# 3.数据缺失处填充了0进行处理，使用算子时计算因子时，尽量避免inf和-inf，以防对分TOP10%组产生影响。
#ts_min_max_diff(perf17, 8),ts_delta(div(perf1, lia2), 4),div(ts_delta(div(perf1, lia2), 4), cs_minmax(ts_mean(occ5, 12)))
# sub(cs_rank(ts_pct(perf26,12)),cs_rank(ts_delta(lia2,24)));
#                         add(cs_rank(ts_max(perf13,12)),cs_rank(ts_min(qua5,24)))
    if prev_factors:
        prompt += "\n—— 上一代因子及其回测表现 ——\n"
        prev_text = json.dumps(prev_factors, ensure_ascii=False, indent=2)
        prompt += prev_text
        prompt += f""" 
                    请基于以上信息，改进生成新的因子, 目标是提升{objective}。
                    改进方法可以是:
                    ## 算子替换: 例如 add -> sub, ts_mean-> ts_max等
                    ## 数据替换： 例如 occ1-> qua11; 1 -> 2等
                    
                    注意这里我只是举了几个例子，具体如何替换根据你自己的理解进行。

                    请结合遗传规划GP：树形表达式结合进化算法；
                    对树结构进行交叉变异，子树变异，生成新的子代。
                    本质上是一种前向随机变异过程，进化主要是通过筛选完成。
                    锦标赛法+改进后适应度筛选相关性低的个体，增加种群多样性。

                    请确保给出新的改进因子,不要给出已有因子,在改进过程中注重实效性与因子的可解释性，避免不必要的复杂度增加。完成优化后，直接
                    """
### 算子嵌套结构/层数：谨慎使用，尽量避免不必要的复杂度增加
#： 谨慎使用，尽量避免不必要的复杂度增加
    prompt += """
                输出内容为最终优化的10个因子表达式的列表，并将其格式化为
                JSON，以便于用户直接应用及后续的分析工作, 输出格式为:
                { "优化因子列表" : 
                [
                ]
                }
                
                {"因子":"***","改进原因":"***"},
                ......
                {"因子":"***","改进原因":"***"}    
                """
    prompt += f"""
                需要参考的行业相关研报信息，请下载并阅读：{GIT_URL + f'industry_report_{INDUSTRY_L1_CODE}.csv'}。
                """
    return prompt


if __name__ == "__main__":
    os.chdir(DATA_PATH)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    trained_model_dir = INDUSTRY_PATH + f"/{now}"
    os.makedirs(trained_model_dir)
    client = OpenAI(
        # api_key = DEEPSEEK_API_KEY,
        api_key = '',
        base_url= DS_BASE_URL
    )
    operator_df = pd.read_excel(DEFINITION_FILE, sheet_name='算子适用数据')
    alpha_df = pd.read_excel(DEFINITION_FILE, sheet_name = '因子映射表')

    processed_fundamental_path = f'{FUNDAMENTAL_PROCESSED_PATH}/{INDUSTRY_L1_CODE}_fundamental.csv'
    if os.path.exists(processed_fundamental_path):
        fundamental_df = pd.read_csv(processed_fundamental_path)
    else:
        fundamental_df = process_fundamental_data()
    alpha_list = fundamental_df.columns[10:].tolist()
    fundamental_df = fundamental_df.sort_values(['FDate','SecCode']).reset_index(drop=True)
    fundamental_df['return'] = fundamental_df.groupby('SecCode')['PRICE'].pct_change()
    fundamental_df['next_return'] = fundamental_df.groupby('SecCode')['return'].shift(-1)
    # evaluator = FactorEvaluator(fundamental_df)    
    alpha_df = alpha_df[alpha_df['因子代码'].isin(fundamental_df.columns[10:])].reset_index(drop=True)
    keep_cols = ['FDate','SecCode','TotalMV','Amount','PRICE']
    result_factors_df = fundamental_df[keep_cols].copy()
    # best_factors_df = fundamental_df[keep_cols].copy()
    result_factors = []
    # best_factors = []
    all_factors = []
    train_len = 63
    # val_len = 20
    return_df = pd.DataFrame(columns=['Factor','Return'])
    saved_factors = []
    # for gen in range(1, generations + 1):
    for gen in range(1,21):
        print(f"\n=== Generation {gen} ===")
        os.makedirs(trained_model_dir + f'/{gen}')
        prompt = build_prompt(INDUSTRY_L1_NAME, operator_df, alpha_df, result_factors, '信息系数(IC)表现')
        resp   = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role":"system", "content":"你是专业的量化分析师。"},
                {"role":"user",   "content": prompt}
            ],
            temperature=0.2,
        )

        raw = resp.choices[0].message.content
        m = re.search(r'\{.*\}', raw, flags=re.S)
        json_str = m.group()
        try:
            optimized_factors = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        result_df = fundamental_df[keep_cols].copy()
        candidates = []

        for i in range(len(optimized_factors['优化因子列表']) - 1, -1, -1):
            entry = optimized_factors['优化因子列表'][i]
            raw = entry['因子']
            try:
                op = parse_operator(raw, alpha_list)
                if op == False:
                    optimized_factors['优化因子列表'].pop(i)
                    continue
            except Exception as e:
                print(f"解析因子 `{raw}` 失败：{e}")
                optimized_factors['优化因子列表'].pop(i)
                continue
            # 计算月度回报序列
            
            dates = sorted(fundamental_df['FDate'].unique())[:-1]
            codes = fundamental_df['SecCode'].dropna().astype(str).str.zfill(6).sort_values().unique()
            stocks_df = pd.DataFrame(0, index=dates, columns=codes)
            monthly = []
            for dt in dates:
                slice_df = fundamental_df[fundamental_df['FDate'] == dt]
                mask = op.apply(slice_df)
                sel = slice_df.loc[mask, 'next_return'].dropna()
                sel_codes = slice_df.loc[mask, 'SecCode'].astype(str).str.zfill(6)
                stocks_df.loc[dt, sel_codes.tolist()] = 1
                monthly.append(sel.mean() if not sel.empty else 0.0)
            stocks_df = stocks_df.reset_index().rename(columns={'index': 'FDate'})
            stocks_df.to_csv(trained_model_dir + f"/{gen}/{entry['因子']}.csv", index=False)
            cum_ret = round(np.prod([1 + r for r in monthly[:train_len+1]]) - 1, 4)
            entry['累计回报(%)'] = cum_ret
            entry['迭代次数'] = gen
            entry['累计回报(%)_list'] = monthly

            return_df.loc[i] = {'Factor': entry['因子'], 'Return': cum_ret}
            all_factors.append(entry)
            result_factors.append({'因子': raw, '累计回报(%)': cum_ret, '迭代': gen})

        return_df.sort_values(by = 'Return', ascending=False, inplace=True)
        print(return_df.head(3))
        json_path = trained_model_dir + f'/{gen}/{INDUSTRY_L1_CODE}_factors_with_perf_{gen}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_factors, f, ensure_ascii=False, indent=4)

    with open(f"{trained_model_dir}/{INDUSTRY_L1_CODE}_result_factors_final.json", 'w', encoding='utf-8') as f:
        json.dump({'优化因子列表': all_factors}, f, ensure_ascii=False, indent=4)