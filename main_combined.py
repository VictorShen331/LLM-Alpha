# main_combine.py

import os
import re
import json
import datetime
import pandas as pd
import numpy as np
import warnings

from config import *
from openai import OpenAI
from preprocess import *
from numerical_operators import FactorEvaluator
from non_numerical_operators import NAME_TO_CLASS

warnings.filterwarnings("ignore")


def build_prompt(industry_l1_name, n_operator_df, nn_operator_df, alpha_df, prev_factors=None, objective='行业内TOP10%多头收益') -> str:
    n_operator_records = [tuple(x) for x in n_operator_df.itertuples(index=False, name=None)]
    nn_operator_records = [tuple(x) for x in nn_operator_df.itertuples(index=False, name=None)]
    alpha_records = [tuple(x) for x in alpha_df.itertuples(index=False, name=None)]
    prompt = f"""
    你是一名专业的量化分析师，你的专长在于深入分析及优化因子以提升{objective}，请生成能够在同一行业对冲行业涨跌、只捕获个股超额收益的因子。
    当前研究范围：申万一级行业“{industry_l1_name}”。
    用户将提供基本面基础数据和特定的算子。 你的任务是对这些基础数据和算子进行细致的分析，理解其逻辑，并用基础数据和算子构建新的因子。
    目标是在维持因子结构相对简洁的前提下，通过组合算子和数据来增强因子{objective}。

    以下是基本面数据及其构建定义和经济学经验判断的方向与类别：
    {alpha_records}。
    对基本面数据，请注意：1. 多为季度更新，具体更新时段存在个股差异；
                          2. 储存为月度数据，取月尾最后一天的数值，可能存在季度间数值相同的情况，请在计算时考虑此特殊性。
    
    以下是可用"非数值"算子及其定义和适用数据类型：{nn_operator_records}。
    以下是可用"数值"算子及其定义和适用数据类型：{n_operator_records}。
    
    请用这些算子和基础数据，生成可解释的月度多头调仓因子表达式，说明每个因子背后的逻辑。
    对于因子，请注意，需要使用非数值算子嵌套数值算子；因子具体格式例如：dou_factor_choose(ts_pct(inp29,3), ts_pct(inp30,12)); dou_factor_choose_drop(ts_resi(perf33,12), ts_min(perf15,6), ts_max(perf15,6))
    """
    if prev_factors:
        prev_text = json.dumps(prev_factors, ensure_ascii=False, indent=2)
        prompt += "\n—— 上一代因子及其回测表现 ——\n" + prev_text
        prompt += f""" 
                    请基于以上信息，改进生成新的因子, 目标是提升{objective}。
                    改进方法可以是:
                    ## 算子替换: 例如 add -> sub, ts_dou_factor_choose-> dou_factor_choose_dropa等
                    ## 数据替换： 例如 occ1-> qua11; 1 -> 2等
                    
                    注意这里我只是举了几个例子，具体如何替换根据你自己的理解进行。

                    请结合遗传规划GP：树形表达式结合进化算法；
                    对树结构进行交叉变异，子树变异，生成新的子代。
                    本质上是一种前向随机变异过程，进化主要是通过筛选完成。
                    锦标赛法+改进后适应度筛选相关性低的个体，增加种群多样性。

                    请确保给出新的改进因子,不要给出已有因子,在改进过程中注重实效性与因子的可解释性，避免不必要的复杂度增加。完成优化后，直接
                    """
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


def split_args(argstr: str) -> list:
    """
    Split a comma-separated argument string at top level, ignoring commas inside nested parentheses.
    """
    args = []
    current = []
    depth = 0
    for ch in argstr:
        if ch == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
    if current:
        args.append(''.join(current).strip())
    return args


if __name__ == "__main__":
    os.chdir(DATA_PATH)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trained_model_dir = os.path.join(INDUSTRY_PATH, now)
    os.makedirs(trained_model_dir, exist_ok=True)

    client = OpenAI(
        api_key='',
        base_url=DS_BASE_URL
    )

    nn_operator_df = pd.read_excel(DEFINITION_FILE, sheet_name='算子适用数据')
    n_operator_df = pd.read_excel(OPERATOR_FILE)
    alpha_df = pd.read_excel(DEFINITION_FILE, sheet_name='因子映射表')

    proc_path = os.path.join(FUNDAMENTAL_PROCESSED_PATH, f"{INDUSTRY_L1_CODE}_fundamental.csv")
    if os.path.exists(proc_path):
        fundamental_df = pd.read_csv(proc_path)
    else:
        fundamental_df = process_fundamental_data()

    # prepare data
    fundamental_df = fundamental_df.sort_values(['FDate', 'SecCode']).reset_index(drop=True)
    fundamental_df['return'] = fundamental_df.groupby('SecCode')['PRICE'].pct_change()
    fundamental_df['next_return'] = fundamental_df.groupby('SecCode')['return'].shift(-1)

    evaluator = FactorEvaluator(fundamental_df)

    keep_cols = ['FDate', 'SecCode', 'TotalMV', 'Amount', 'PRICE','next_return']
    train_len = 63

    all_factors = []
    result_factors = []
    return_df = pd.DataFrame(columns=['Factor', 'Return'])
    alpha_df = alpha_df[alpha_df['因子代码'].isin(fundamental_df.columns[10:])].reset_index(drop=True)
    gen = 1
    while gen <= 20:
        print(f"\n=== Generation {gen} ===")
        gen_dir = os.path.join(trained_model_dir, str(gen))
        os.makedirs(gen_dir, exist_ok=True)

        prompt = build_prompt(INDUSTRY_L1_NAME,n_operator_df, nn_operator_df, alpha_df, result_factors) # , '信息系数(IC)表现'
        resp = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "你是专业的量化分析师。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content
        m = re.search(r'\{.*\}', raw, flags=re.S)
        if not m:
            print("未找到 JSON，跳过本代。")
            continue
        try:
            optimized = json.loads(m.group())
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败：{e}")
            continue

        # result_df = fundamental_df[keep_cols].copy()

        for i in range(len(optimized['优化因子列表']) - 1, -1, -1):
            entry = optimized['优化因子列表'][i]
            raw_expr = entry['因子'].strip()

            # parse outer operator and its args
            m_op = re.match(r'^(\w+)\((.*)\)$', raw_expr)
            if not m_op:
                print(f"表达式格式不匹配，跳过：{raw_expr}")
                optimized['优化因子列表'].pop(i)
                continue

            op_name, argstr = m_op.group(1), m_op.group(2)
            args = split_args(argstr)
            clean_args = []
            bad = False

            # compute any nested numeric expressions first
            for arg in args:
                if '(' in arg:  # numeric sub-expression
                    expr = arg
                    if expr not in fundamental_df.columns[10:]:
                        try:
                            fundamental_df[expr] = evaluator.calculate(expr)
                        except Exception as e:
                            print(f"数值因子计算失败 `{expr}`：{e}")
                            bad = True
                            break
                    clean_args.append(expr)
                elif arg in fundamental_df.columns[10:]:
                    clean_args.append(arg)
                else:
                    bad = True

            if bad:
                optimized['优化因子列表'].pop(i)
                continue

            OpClass = NAME_TO_CLASS.get(op_name)
            if OpClass is None:
                print(f"未知算子 `{op_name}`，跳过")
                optimized['优化因子列表'].pop(i)
                continue

            try:
                op_inst = OpClass(*clean_args)
            except Exception as e:
                print(f"实例化算子 `{raw_expr}` 失败：{e}")
                optimized['优化因子列表'].pop(i)
                continue

            # compute monthly returns
            dates = sorted(fundamental_df['FDate'].unique())[:-1]
            codes = fundamental_df['SecCode'].dropna().astype(str).str.zfill(6).sort_values().unique()
            stocks_df = pd.DataFrame(0, index=dates, columns=codes)
            monthly = []
            for dt in dates:
                slice_df = fundamental_df[fundamental_df['FDate'] == dt]
                try:
                    mask = op_inst.apply(slice_df)
                except Exception as e:
                    print(f"算子应用出错 `{raw_expr}`：{e}")
                    mask = pd.Series(False, index=slice_df.index)
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

            return_df.loc[i] = {'Factor': raw_expr, 'Return': cum_ret}
            all_factors.append(entry)
            result_factors.append({'因子': raw_expr, '累计回报(%)': cum_ret, '迭代': gen})

        return_df.sort_values(by='Return', ascending=False, inplace=True)
        print(return_df.head(3))

        # save this generation's JSON
        with open(os.path.join(gen_dir, f"{INDUSTRY_L1_CODE}_factors_with_perf_{gen}.json"), 'w', encoding='utf-8') as f:
            json.dump(optimized, f, ensure_ascii=False, indent=4)

        gen += 1

    # save final results
    with open(os.path.join(trained_model_dir, f"{INDUSTRY_L1_CODE}_result_factors_final.json"), 'w', encoding='utf-8') as f:
        json.dump({'优化因子列表': all_factors}, f, ensure_ascii=False, indent=4)

    print("All generations complete. 结果已保存。")
