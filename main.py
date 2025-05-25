import os
import re
import json
import pandas as pd
import numpy as np
import warnings
from config import *
from openai import OpenAI
from alpha import FactorEvaluator
from backtest import Backtest
from preprocess import *
warnings.filterwarnings("ignore")

def build_prompt(industry_l1_name, operator_df, alpha_df, prev_factors = None, objective = '行业内TOP10%多头收益') -> str:
    operator_records = [tuple(x) for x in operator_df.itertuples(index=False, name=None)]

    # 3. 把 alpha_df 转成 list of tuples
    alpha_records = [tuple(x) for x in alpha_df.itertuples(index=False, name=None)]
    prompt = f"""
            你是一名专业的量化分析师，你的专长在于深入分析及优化因子以提升{objective}。
            当前研究范围：申万一级行业“{industry_l1_name}”。

            用户将提供基本面基础数据和特定的算子。 你的任务是对这些基础数据和算子进行细致的分析，理解其逻辑，并用基础数据和算子构建新的因子。
            目标是在维持因子结构相对简洁的前提下，通过组合算子和数据来增强因子{objective}。

            以下是基本面数据及其构建定义和经济学经验判断的方向与类别：
            {alpha_records}。
            对基本面数据，请注意：1.大多季度更新，具体更新时段存在个股差异；
                                2.储存为月度数据，具体为取月尾最后一天的数值。可能存在季度间数值相同的情况，请在计算因子的逻辑中考虑到这一特殊情况，避免产生误差。
            
            以下是可用算子以及其定义和适用数据类型：{operator_records}。
            请用这些算子和基础数据，生成可解释的月度多头调仓因子表达式，并说明每个因子背后的逻辑。

            因子具体格式例如：ts_pct(perf26,12)；cs_rank(ts_min(qua5,24))

            """
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


def validate_expr(expr: str) -> str:
    e = expr.strip()
    e = re.sub(r'\s*,\s*', ',', e)
    e = re.sub(r'(\w+)\s*\(\s*', r'\1(', e)
    e = re.sub(r'\s*\)\s*', ')', e)
    e = re.sub(r',([0-9]+)\.0+\)', r',\1)', e)

    opens  = e.count('(')
    closes = e.count(')')
    if opens > closes:
        e = e + ')' * (opens - closes)
    elif closes > opens:
        e = '(' * (closes - opens) + e

    cnt = 0
    for ch in e:
        if ch == '(':
            cnt += 1
        elif ch == ')':
            cnt -= 1
        if cnt < 0:
            raise SyntaxError(f"修复后仍有多余右括号：{e}")
    if cnt != 0:
        raise SyntaxError(f"修复后括号不匹配：{e}")

    if re.match(r'^[^()]+,[^()]+$', e):
        parts = e.split(',', 1)
        e = f"add({parts[0]}, {parts[1]})"

    def _to_int(m):
        num = m.group(1)
        try:
            return ',' + str(int(float(num))) + ')'
        except:
            raise ValueError(f"窗口参数非整数：{expr} 中的 {num}")
    e = re.sub(r',([0-9\.]+)\)', _to_int, e)

    return e

if __name__ == "__main__":
    os.chdir(DATA_PATH)
    client = OpenAI(
        # api_key = DEEPSEEK_API_KEY,
        api_key = '',
        base_url= DS_BASE_URL
    )
    operator_df = pd.read_excel(OPERATOR_FILE)
    alpha_df = pd.read_excel(DEFINITION_FILE, sheet_name = '因子映射表')

    processed_fundamental_path = f'{FUNDAMENTAL_PROCESSED_PATH}/{INDUSTRY_L1_CODE}_fundamental.csv'
    if os.path.exists(processed_fundamental_path):
        fundamental_df = pd.read_csv(processed_fundamental_path)
    else:
        fundamental_df = process_fundamental_data()

    evaluator = FactorEvaluator(fundamental_df)    
    alpha_df = alpha_df[alpha_df['因子代码'].isin(fundamental_df.columns[10:])].reset_index(drop=True)
    prev_factors  = None
    # gen = 1
    keep_cols = ['FDate','SecCode','TotalMV','Amount','PRICE']
    good_factors_df = fundamental_df[keep_cols].copy()
    # best_factors_df = fundamental_df[keep_cols].copy()
    good_factors = []
    # best_factors = []
    all_factors = []
    # for gen in range(1, generations + 1):
    for gen in range(1,21):
        print(f"\n=== Generation {gen} ===")
        prompt = build_prompt("食品饮料", operator_df, alpha_df, prev_factors, '信息系数(IC)表现')
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
        all_factors_name = []
        candidates = []
        for i in range(len(optimized_factors['优化因子列表']) - 1, -1, -1):
            entry = optimized_factors['优化因子列表'][i]
            raw_expr = entry['因子']
            try:
                expr = validate_expr(raw_expr)
            except Exception as err:
                print(f"表达式 `{raw_expr}` 校验失败，将删除本条：{err}")
                optimized_factors['优化因子列表'].pop(i)
                continue
            entry['因子'] = expr
            if expr in all_factors_name:
                optimized_factors['优化因子列表'].pop(i)
                continue  
            try:
                result_df[expr] = evaluator.calculate(expr)
            except Exception as err:
                print(f"因子 {expr} 计算失败，将删除本条：{err}")
                # result_df[expr] = np.nan
                optimized_factors['优化因子列表'].pop(i)
                continue
            candidates.append(expr)
            bt = Backtest(result_df[['FDate','SecCode','PRICE', expr]])
            rets, ics = bt.run(expr, top_quantile=0.1)

            last_ret = round(np.cumsum(rets).tolist()[-1], 4)
            # last_ic  = round(ics[-1], 4) if pd.notna(ics[-1]) else np.nan
            icir = np.mean(ics)/np.std(ics, ddof=1)

            entry['累计回报(%)'] = last_ret
            # entry['最新一期IC信息系数'] = last_ic
            entry['信息系数ICIR'] = icir
            entry['迭代次数'] = gen

            all_factors.append(entry)
            all_factors_name.append(entry['因子'])
            if last_ret > 1:
                good_factors.append(entry)
                good_factors_df[expr] = (
                    result_df
                    .set_index(['FDate','SecCode'])[expr]
                    .values
                )
                print(f"factor:{expr}\ncum_mon_return:{last_ret}\nicir:{icir}")
        
        # # 缺失值处理：transform 保证索引对齐
        # filled = result_df.groupby('SecCode')[candidates] \
        #             .transform(lambda x: x.ffill().bfill())
        # result_df[candidates] = filled.fillna(0)

        factor_path = TRAINED_MODEL_DIR + f"/{INDUSTRY_L1_CODE}_new_factors_{gen}.csv"
        result_df.to_csv(factor_path, index=False)            

        json_path = TRAINED_MODEL_DIR + f'/{INDUSTRY_L1_CODE}_factors_with_perf_{gen}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_factors, f, ensure_ascii=False, indent=4)


        # prev_factors = good_factors.copy()
        prev_factors = all_factors.copy()
        gen += 1

    with open(f"{TRAINED_MODEL_DIR}/{INDUSTRY_L1_CODE}_good_factors_final.json","w",encoding="utf-8") as f:
        json.dump({"优化因子列表": good_factors}, f, ensure_ascii=False, indent=4)
    # with open(f"{TRAINED_MODEL_DIR}/{INDUSTRY_L1_CODE}_best_factors_final.json","w",encoding="utf-8") as f:
    #     json.dump({"优化因子列表": best_factors}, f, ensure_ascii=False, indent=4)
    good_factors_df.to_csv(TRAINED_MODEL_DIR + f'/{INDUSTRY_L1_CODE}_good_factors.csv', index = False)
    # best_factors_df.to_csv(TRAINED_MODEL_DIR + f'/{INDUSTRY_L1_CODE}_best_factors.csv')
    print("good_factors_final.json saved。")