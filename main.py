import os
import re
import json
import pandas as pd
import numpy as np
from config import *
from openai import OpenAI
from alpha import FactorEvaluator
from backtest import Backtest


def build_prompt(industry_l1_name, operator_df, alpha_df, prev_factors = None) -> str:
    operator_records = [tuple(x) for x in operator_df.itertuples(index=False, name=None)]

    # 3. 把 alpha_df 转成 list of tuples
    alpha_records = [tuple(x) for x in alpha_df.itertuples(index=False, name=None)]
    prompt = f"""
            你是一名专业的量化分析师，你的专长在于深入分析及优化因子以提升行业内TOP10%多头收益。
            当前研究范围：申万一级行业“{industry_l1_name}”。

            用户将提供基本面基础数据和特定的算子。 你的任务是对这些基础数据和算子进行细致的分析，理解其逻辑，并用基础数据和算子构建新的因子。
            目标是在维持因子结构相对简洁的前提下，通过组合算子和数据来增强因子行业内TOP10%多头收益。

            以下是基本面数据及其构建定义和经济学经验判断的方向与类别：
            {alpha_records}。
            对基本面数据，请注意：1.大多季度更新，具体更新时段存在个股差异；
                                2.储存为月度数据，具体为取月尾最后一天的数值。可能存在季度间数值相同的情况，请在计算因子的逻辑中考虑到这一特殊情况，避免产生误差；
                                3.数据缺失处填充了0进行处理，使用算子时计算因子时，尽量避免inf和-inf，以防对分TOP10%组产生影响。
            
            以下是可用算子以及其定义和适用数据类型：{operator_records}。
            请用这些算子和基础数据，生成可解释的月度多头调仓因子表达式，并说明每个因子背后的逻辑。

            因子具体格式例如：sub(cs_rank(ts_pct(perf26,12)),cs_rank(ts_delta(lia2,24)));
                        add(cs_rank(ts_max(perf13,12)),cs_rank(ts_min(qua5,24)))

            """
#ts_min_max_diff(perf17, 8),ts_delta(div(perf1, lia2), 4),div(ts_delta(div(perf1, lia2), 4), cs_minmax(ts_mean(occ5, 12)))
    if prev_factors:
        prompt += "\n—— 上一代因子及其回测表现 ——\n"
        prev_text = json.dumps(prev_factors, ensure_ascii=False, indent=2)
        prompt += prev_text
        prompt += """ 
                    请基于以上信息，改进生成新的因子, 目标是提升行业内TOP10%多头收益。
                    改进方法可以是:
                    ## 算子替换: 例如 add -> sub, ts_mean-> ts_max等
                    ## 数据替换： 例如 occ1-> qua11; 1 -> 2等
                    ## 算子嵌套结构/层数
                    注意这里我只是举了几个例子，具体如何替换根据你自己的理解进行。
                    请确保给出新的改进因子,不要给出已有因子,在改进过程中注重实效性与因子的可解释性，避免不必要的复杂度增加。完成优化后，直接
                    """
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
        api_key = DEEPSEEK_API_KEY,
        base_url= DS_BASE_URL
    )

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
    for col in fundamental_df.columns:
        if fundamental_df[col].dtype.kind in 'fi':
            fundamental_df[col] = (
                fundamental_df[col]
                .fillna(method='ffill')
                .fillna(0)
            )

    evaluator = FactorEvaluator(fundamental_df)

    operator_df = pd.read_excel(OPERATOR_FILE)
    alpha_df = pd.read_excel(DEFINITION_FILE, sheet_name = '因子映射表')

    prev_factors  = None
    gen = 1
    keep_cols = ['FDate','SecCode','TotalMV','Amount','PRICE']
    good_factors_df = fundamental_df[keep_cols].copy()
    best_factors_df = fundamental_df[keep_cols].copy()
    good_factors = []
    best_factors = []
    all_factors = []
    # for gen in range(1, generations + 1):
    while True:
        print(f"\n=== Generation {gen} ===")
        prompt = build_prompt("食品饮料", operator_df, alpha_df, prev_factors)
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

        optimized_factors = json.loads(json_str)

        cleaned_list = []
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
            cleaned_list.append(expr)

        candidates = cleaned_list

        result_df = fundamental_df[keep_cols].copy()

        for expr in candidates:
            try:
                result_df[expr] = evaluator.calculate(expr)
            except Exception as e:
                print(f"因子 {expr} 计算失败：{e}")
                result_df[expr] = np.nan
        
        factor_cols = candidates
        # 缺失值处理：transform 保证索引对齐
        filled = result_df.groupby('SecCode')[factor_cols] \
                    .transform(lambda x: x.ffill().bfill())
        result_df[factor_cols] = filled.fillna(0)

        factor_path = TRAINED_MODEL_DIR + f"/{INDUSTRY_L1_CODE}_new_factors_{gen}.csv"
        result_df.to_csv(factor_path, index=False)
        # print(f"新因子已保存到：{factor_path}.csv")

        bt = Backtest(result_df)

        monthly_perfs = {}
        for entry in optimized_factors['优化因子列表']:
            expr = entry['因子']
            perf_list = bt.run(expr, top_quantile=0.1)
            monthly_perfs[expr] = perf_list

        for entry in optimized_factors['优化因子列表']:
            expr = entry['因子']
            # entry['月度回测表现'] = monthly_perfs.get(expr, [])
            entry['累计回测表现'] = np.cumsum(monthly_perfs.get(expr, [])).tolist()

        json_path = TRAINED_MODEL_DIR + f'/{INDUSTRY_L1_CODE}_factors_with_perf_{gen}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_factors, f, ensure_ascii=False, indent=4)

        # print(f"包含月度回测表现的JSON已保存到：{json_path}.json")
        print('---factors with fine performance---')
        for entry in optimized_factors['优化因子列表']:
            last_cum = entry['累计回测表现'][-1] if entry['累计回测表现'] else 0

            new_entry = entry.copy()
            new_entry['累计回测表现'] = round(last_cum, 2)
            # new_entry['累计回测表现'] = [round(last_cum, 2)]

            all_factors.append(new_entry)
            if last_cum > 1:
                good_factors.append(entry)
                good_factors_df[expr] = result_df.set_index(['FDate','SecCode'])[expr].values
                print(f"factor:{entry['因子']}\ncum_mon_return:{entry['累计回测表现'][-1]}")
            if last_cum > 2.5:
                best_factors.append(entry)
                best_factors_df[expr] = result_df.set_index(['FDate','SecCode'])[expr].values

        unique_best = {}
        for e in best_factors:
            unique_best[e['因子']] = e
        best_factors = list(unique_best.values())[:10]

        top_returns = [e['累计回测表现'][-1] for e in best_factors] or [0]
        if len(best_factors) >= 10 or max(top_returns) >= 4 or gen >= 20:
            print("break loop")
            break

        # prev_factors = good_factors.copy()
        prev_factors = all_factors.copy()
        gen += 1

    with open(f"{TRAINED_MODEL_DIR}/{INDUSTRY_L1_CODE}_good_factors_final.json","w",encoding="utf-8") as f:
        json.dump({"优化因子列表": good_factors}, f, ensure_ascii=False, indent=4)
    with open(f"{TRAINED_MODEL_DIR}/{INDUSTRY_L1_CODE}_best_factors_final.json","w",encoding="utf-8") as f:
        json.dump({"优化因子列表": best_factors}, f, ensure_ascii=False, indent=4)
    print("good_factors_final.json & best_factors_final.json saved。")
    good_factors_df.to_csv(TRAINED_MODEL_DIR + f'/{INDUSTRY_L1_CODE}_good_factors.csv')
    best_factors_df.to_csv(TRAINED_MODEL_DIR + f'/{INDUSTRY_L1_CODE}_best_factors.csv')