import numpy as np
import pandas as pd
from numpy.linalg import lstsq

class FactorOperators:
    """
    元素算子和基础数组计算实现，返回 numpy 数组
    """
    @staticmethod
    def add(m1, m2): return m1 + m2
    @staticmethod
    def sub(m1, m2): return m1 - m2
    @staticmethod
    def div(m1, m2): return m1 / m2
    @staticmethod
    def mul(m1, m2): return m1 * m2

    @staticmethod
    def ts_slope_array(arr, v):
        """返回 arr 以长度 v 窗口的滚动斜率数组"""
        N = len(arr)
        slopes = np.full(N, np.nan)
        x = np.arange(v)
        for i in range(v-1, N):
            window = arr[i-v+1:i+1]
            mask = ~np.isnan(window)
            if mask.sum() < 2:
                continue
            A = np.vstack([x[mask], np.ones(mask.sum())]).T
            slopes[i], _ = lstsq(A, window[mask], rcond=None)[0]
        return slopes

    @staticmethod
    def ts_regression_slope_array(arr1, arr2, v):
        """返回 arr1 对 arr2 的滚动回归斜率数组"""
        N = len(arr1)
        slopes = np.full(N, np.nan)
        for i in range(v-1, N):
            y = arr2[i-v+1:i+1]
            x = arr1[i-v+1:i+1]
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 2:
                continue
            A = np.vstack([x[mask], np.ones(mask.sum())]).T
            slopes[i], _ = lstsq(A, y[mask], rcond=None)[0]
        return slopes

    @staticmethod
    def ts_regression_resi_array(arr1, arr2, v):
        """返回 arr1 对 arr2 的滚动回归残差（最后点）数组"""
        N = len(arr1)
        res = np.full(N, np.nan)
        for i in range(v-1, N):
            y = arr2[i-v+1:i+1]
            x = arr1[i-v+1:i+1]
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 2:
                continue
            A = np.vstack([x[mask], np.ones(mask.sum())]).T
            slope, intercept = lstsq(A, y[mask], rcond=None)[0]
            res[i] = y[-1] - (slope * x[-1] + intercept)
        return res

    @staticmethod
    def ts_regression_rsquare_array(arr1, arr2, v):
        """返回 arr1 对 arr2 的滚动回归 R^2 数组"""
        N = len(arr1)
        r2 = np.full(N, np.nan)
        for i in range(v-1, N):
            y = arr2[i-v+1:i+1]
            x = arr1[i-v+1:i+1]
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 2:
                continue
            A = np.vstack([x[mask], np.ones(mask.sum())]).T
            slope, intercept = lstsq(A, y[mask], rcond=None)[0]
            fitted = slope * x[mask] + intercept
            ss_res = np.sum((y[mask] - fitted) ** 2)
            ss_tot = np.sum((y[mask] - np.nanmean(y[mask])) ** 2)
            r2[i] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return r2

class FactorEvaluator:
    """
    Three categories:
      - 元素算子 (add, sub, div, mul)
      - 时序算子 (ts_*, yoy, qoq, ttm, ts_regression_*): groupby SecCode-rolling/shift
      - 横截面算子 (cs_*): groupby FDate-rank/norm/minmax
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        self.group_sec = self.df['SecCode']
        self.group_date = self.df['FDate']

    def calculate(self, expr: str) -> pd.Series:
        env = {}
        for col in self.df.columns:
            env[col] = self.df[col]
        # 元素算子
        for op in ['add', 'sub', 'div', 'mul']:
            env[op] = getattr(FactorOperators, op)
        # 时序算子
        env.update({
            'ts_mean': lambda m, v: m.groupby(self.group_sec).transform(lambda x: x.rolling(v).mean()),
            'ts_std': lambda m, v: m.groupby(self.group_sec).transform(lambda x: x.rolling(v).std()),
            'ts_delay': lambda m, v: m.groupby(self.group_sec).shift(v),
            'ts_delta': lambda m, v: m - m.groupby(self.group_sec).shift(v),
            'ts_pct': lambda m, v: (m - m.groupby(self.group_sec).shift(v)) / m.groupby(self.group_sec).shift(v),
            'ts_max': lambda m, v: m.groupby(self.group_sec).transform(lambda x: x.rolling(v).max()),
            'ts_min': lambda m, v: m.groupby(self.group_sec).transform(lambda x: x.rolling(v).min()),
            'ts_min_max_diff': lambda m, v: env['ts_max'](m, v) - env['ts_min'](m, v),
            'yoy': lambda m: (m - m.groupby(self.group_sec).shift(12)) / m.groupby(self.group_sec).shift(12),
            'qoq': lambda m: (m - m.groupby(self.group_sec).shift(4)) / m.groupby(self.group_sec).shift(4),
            'ttm': lambda m: m.groupby(self.group_sec).transform(lambda x: x.rolling(4).sum()),
            'ts_slope': lambda m, v: m.groupby(self.group_sec).transform(
                lambda x: FactorOperators.ts_slope_array(x.values, v)
            ),
            'ts_resi': lambda m, v: m.groupby(self.group_sec).transform(
                lambda x: x.values - (FactorOperators.ts_slope_array(x.values, v) * np.arange(len(x)))
            ),
            'ts_rsquare': lambda m, v: m.groupby(self.group_sec).transform(
                lambda x: pd.Series(
                    FactorOperators.ts_regression_rsquare_array(x.values, x.values, v), index=x.index
                )
            ),
            'ts_regression_slope': lambda m1, m2, v: pd.Series(
                FactorOperators.ts_regression_slope_array(
                    m1.values, m2.values, v
                ), index=m1.index
            ),
            'ts_regression_resi': lambda m1, m2, v: pd.Series(
                FactorOperators.ts_regression_resi_array(
                    m1.values, m2.values, v
                ), index=m1.index
            ),
            'ts_regression_rsquare': lambda m1, m2, v: pd.Series(
                FactorOperators.ts_regression_rsquare_array(
                    m1.values, m2.values, v
                ), index=m1.index
            ),
        })
        # 横截面算子
        env.update({
            'cs_norm': lambda m: m.groupby(self.group_date).transform(lambda x: (x - x.mean()) / x.std()),
            'cs_minmax': lambda m: m.groupby(self.group_date).transform(lambda x: (x - x.min()) / (x.max() - x.min())),
            'cs_rank': lambda m: m.groupby(self.group_date).transform(lambda x: x.rank(pct=True)),
        })
        result = eval(expr, {}, env)
        # to Series match lenth
        if isinstance(result, pd.Series):
            return result.reset_index(drop=True)
        if isinstance(result, np.ndarray):
            return pd.Series(result)
        return pd.Series([result] * len(self.df))
