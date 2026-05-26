"""
metrics.py
==========
Performance metrics for comparing hedging strategies.

Key metric: Hedge Efficiency
    = Variance reduction vs unhedged / Total hedge cost incurred
    Higher is better — you want maximum risk reduction per dollar spent.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest_engine import BacktestResult


@dataclass
class StrategyMetrics:
    name:              str
    total_pnl:         float
    total_hedge_cost:  float
    sharpe:            float
    max_drawdown:      float
    hedge_efficiency:  float
    avg_hedge_ratio:   float
    pnl_vol:           float
    win_rate:          float


def _sharpe(pnl: pd.Series, periods_per_year: int = 8_760) -> float:
    if pnl.std() == 0:
        return 0.0
    return (pnl.mean() / pnl.std()) * np.sqrt(periods_per_year)


def _max_drawdown(cum_pnl: pd.Series) -> float:
    roll_max = cum_pnl.cummax()
    dd = cum_pnl - roll_max
    return float(dd.min())


def _hedge_efficiency(
    result: BacktestResult,
    unhedged_var: float,
) -> float:
    """
    Var reduction achieved per unit of hedge cost.
    Normalised so values across strategies are comparable.
    """
    hedged_var  = result.position_pnl.var()
    var_reduced = max(unhedged_var - hedged_var, 0)
    total_cost  = result.hedge_cost.sum()
    if total_cost == 0:
        return 0.0
    return var_reduced / total_cost


def compute_metrics(
    result: BacktestResult,
    unhedged_var: float,
) -> StrategyMetrics:
    pnl = result.pnl
    return StrategyMetrics(
        name             = result.strategy_name,
        total_pnl        = float(pnl.sum()),
        total_hedge_cost = float(result.hedge_cost.sum()),
        sharpe           = _sharpe(pnl),
        max_drawdown     = _max_drawdown(result.cumulative_pnl),
        hedge_efficiency = _hedge_efficiency(result, unhedged_var),
        avg_hedge_ratio  = float(result.hedge_ratio.mean()),
        pnl_vol          = float(pnl.std()),
        win_rate         = float((pnl > 0).mean()),
    )


def summarise(
    results: dict[str, BacktestResult],
    unhedged_pnl: pd.Series,
) -> pd.DataFrame:
    unhedged_var = unhedged_pnl.var()
    rows = [compute_metrics(r, unhedged_var) for r in results.values()]
    df = pd.DataFrame([vars(r) for r in rows]).set_index("name")
    df = df.round(4)
    return df
