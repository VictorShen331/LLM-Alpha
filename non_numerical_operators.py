# new_operator.py
"""
算子解析与实现模块
根据 LLM 返回的因子表达式动态实例化算子，并计算每期选股池掩码
"""
import pandas as pd

class OperatorBase:
    """
    算子基类，所有算子需继承此类并实现 apply 方法
    """
    name = None
    description = None

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """
        对截面数据 df（包含 SecCode 和所需因子列）返回 Boolean Series
        """
        raise NotImplementedError

# ----- 算子实现 -----
class DouFactorChoose(OperatorBase):
    name = 'dou_factor_choose'
    description = '因子1 & 因子2 同向'

    def __init__(self, f1, f2, *_):
        self.f1 = f1
        self.f2 = f2

    def apply(self, df):
        return (df[self.f1] > 0) & (df[self.f2] > 0)

class FactorDivGroup(OperatorBase):
    name = 'factor_div_group'
    description = '行业内分域后：高区按因子2取多头，低区按因子3取多头'

    def __init__(self, grp, f_high, f_low, *_):
        self.grp = grp
        self.f_high = f_high
        self.f_low = f_low
        self.q = 0.2

    def apply(self, df):
        # 按 grp 中位数分高低两组
        median = df[self.grp].median()
        high_mask = (df[self.grp] >= median) & (df[self.f_high] >= df[self.f_high].quantile(1-self.q))
        low_mask  = (df[self.grp] <  median) & (df[self.f_low] >= df[self.f_low].quantile(1-self.q))
        return high_mask | low_mask

class DouFactorChooseSig(OperatorBase):
    name = 'dou_factor_choose_sig'
    description = '因子1 & 因子2 & 因子3 同向'

    def __init__(self, f1, f2, f3, *_):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

    def apply(self, df):
        return (df[self.f1] > 0) & (df[self.f2] > 0) & (df[self.f3] > 0)

class DouFactorChooseDrop(OperatorBase):
    name = 'dou_factor_choose_drop'
    description = '因子1 & 因子2 同向，剔除因子3 弱势底部20%'

    def __init__(self, f1, f2, f3, *_):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.q = 0.2

    def apply(self, df):
        mask = (df[self.f1] > 0) & (df[self.f2] > 0)
        thresh = df[self.f3].quantile(self.q)
        return mask & (df[self.f3] >= thresh)

class DouFactorChooseSigOne(OperatorBase):
    name = 'dou_factor_choose_sig_one'
    description = '因子1 & 因子2 同向，因子3 单向'

    def __init__(self, f1, f2, f3, *_):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

    def apply(self, df):
        mask_12 = (df[self.f1] > 0) & (df[self.f2] > 0)
        # 因子3 单向：仅取正向
        mask_3 = df[self.f3] > 0
        return mask_12 & mask_3

class FactorGroupRank(OperatorBase):
    name = 'factor_group_rank'
    description = '对因子进行两步分组：f1 top20%，再在组内选 f2 top20%'

    def __init__(self, f1, f2, *_):
        self.f1 = f1
        self.f2 = f2
        self.q = 0.2

    def apply(self, df):
        sub1 = df[['SecCode', self.f1]].dropna().copy()
        sub1['r1'] = sub1[self.f1].rank(pct=True)
        codes1 = sub1.loc[sub1['r1'] >= 1-self.q, 'SecCode']
        sub2 = df[df['SecCode'].isin(codes1)][['SecCode', self.f2]].dropna().copy()
        sub2['r2'] = sub2[self.f2].rank(pct=True)
        codes2 = sub2.loc[sub2['r2'] >= 1-self.q, 'SecCode']
        return df['SecCode'].isin(codes2)

class FactorOrNeg(OperatorBase):
    name = 'factor_or_neg'
    description = '因子1 多头 or 因子2 空头'

    def __init__(self, f1, f2, *_):
        self.f1 = f1
        self.f2 = f2

    def apply(self, df):
        return (df[self.f1] > 0) | (df[self.f2] < 0)

class FactorRankNeg(OperatorBase):
    name = 'factor_rank_neg'
    description = '因子1 top20% or 因子2 bottom20%'

    def __init__(self, f1, f2, *_):
        self.f1 = f1
        self.f2 = f2
        self.q = 0.2

    def apply(self, df):
        sub1 = df[['SecCode', self.f1]].dropna().copy()
        sub1['r1'] = sub1[self.f1].rank(pct=True)
        sub2 = df[['SecCode', self.f2]].dropna().copy()
        sub2['r2'] = sub2[self.f2].rank(pct=True)

        codes1 = sub1.loc[sub1['r1'] >= 1-self.q, 'SecCode']
        codes2 = sub2.loc[sub2['r2'] <= self.q, 'SecCode']

        # 先把 Series 转成 Index，然后调用 union
        union_idx = pd.Index(codes1).union(pd.Index(codes2))
        return df['SecCode'].isin(union_idx)

class FactorGroupDropFactor(OperatorBase):
    name = 'factor_group_drop_factor'
    description = '先按 f1 top20%，剔除组内 f2 bottom20%'

    def __init__(self, f1, f2, *_):
        self.f1 = f1
        self.f2 = f2
        self.q = 0.2

    def apply(self, df):
        sub1 = df[['SecCode', self.f1]].dropna().copy()
        sub1['r1'] = sub1[self.f1].rank(pct=True)
        codes1 = sub1.loc[sub1['r1'] >= 1-self.q, 'SecCode']
        sub2 = df[df['SecCode'].isin(codes1)][['SecCode', self.f2]].dropna().copy()
        sub2['r2'] = sub2[self.f2].rank(pct=True)
        codes_drop = sub2.loc[sub2['r2'] <= self.q, 'SecCode']
        sel = set(codes1) - set(codes_drop)
        return df['SecCode'].isin(sel)

# 算子名称到类的映射
ALL_OPERATORS = [
    DouFactorChoose,
    FactorDivGroup,
    DouFactorChooseSig,
    DouFactorChooseDrop,
    DouFactorChooseSigOne,
    FactorGroupRank,
    FactorOrNeg,
    FactorRankNeg,
    FactorGroupDropFactor,
]

NAME_TO_CLASS = {op.name: op for op in ALL_OPERATORS}

# —— 新增：通用表达式算子 ——
from alpha import FactorEvaluator

class ExpressionOperator(OperatorBase):
    """
    通用算子：对任意表达式调用 FactorEvaluator 计算，选取结果 > 0 的标的
    """
    name = 'expression'
    description = '基于任意表达式的通用算子，使用 FactorEvaluator 计算后阈值 > 0'

    def __init__(self, expr, *args):
        self.expr = expr

    def apply(self, df):
        ev = FactorEvaluator(df)
        ser = ev.calculate(self.expr)
        return ser.fillna(0) > 0

# —— 解析函数 ——
def parse_operator(expr: str, alpha_list: list) -> OperatorBase:
    """
    从表达式字符串解析并返回对应算子实例。
    如果算子名称未在预定义列表中，使用 ExpressionOperator 处理任意表达式。
    """
    # 拆分函数名与参数
    if '(' in expr and expr.strip().endswith(')'):
        name, argstr = expr.split('(', 1)
        name = name.strip()
        args = [a.strip().strip("'\"") for a in argstr.rstrip(')').split(',')]
    else:
        # 不是函数形式，直接作为表达式
        # return ExpressionOperator(expr)
        return False

    if not set(args).issubset(alpha_list):
        return False

    cls = NAME_TO_CLASS.get(name)
    if cls:
        return cls(*args)
    else:
        # 未命中的算子名称，转为 ExpressionOperator
        # return ExpressionOperator(expr)
        return False
