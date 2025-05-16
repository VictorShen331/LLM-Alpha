import pandas as pd
import numpy as np

class Backtest:
    """
    Monthly rebalance, long top_quantile stocks, equally weighted
    DataFrame['FDate', 'SecCode', 'PRICE', <factor_cols>]
    """
    def __init__(self, df: pd.DataFrame,
                 date_col: str = 'FDate',
                 asset_col: str = 'SecCode',
                 price_col: str = 'PRICE'):
        self.date_col = date_col
        self.asset_col = asset_col
        self.price_col = price_col
        self.df = df.copy().sort_values([date_col, asset_col]).reset_index(drop=True)
        self.df['return'] = (
            self.df.groupby(self.asset_col)[self.price_col]
               .pct_change()
        )
        self.df['forward_return'] = (
            self.df.groupby(self.asset_col)['return']
               .shift(-1)
        )

    def run(self, factor: str, top_quantile: float = 0.1) -> list:
        """
        Calculate return
        """
        dates = sorted(self.df[self.date_col].unique())
        monthly_rets = []
        monthly_ics = []
        for dt in dates[:-1]:
            sub = self.df[self.df[self.date_col] == dt]
            sub = sub.dropna(subset=[factor, 'forward_return'])
            mask = np.isfinite(sub[factor])
            sub = sub.loc[mask]
            if sub.empty:
                monthly_rets.append(0.0)
                continue
            n = max(int(len(sub) * top_quantile), 1)
            top = sub.nlargest(n, factor)
            ret = top['forward_return'].mean()
            monthly_rets.append(ret if pd.notna(ret) else 0.0)
            
            ic = sub[factor].corr(sub['forward_return'])
            monthly_ics.append(ic)
        return monthly_rets, monthly_ics
