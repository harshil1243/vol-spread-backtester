"""
backtest_engine.py
==================
Walk-forward backtesting of two hedging strategies.

Strategy A — Static Hedge
    Always hedge 100% of net position at the prevailing spread.

Strategy B — Vol-Regime Hedge  (our hypothesis)
    Dynamically adjust hedge ratio based on the current volatility regime:
        Low  vol regime → 50%  hedge  (reduce carry cost, accept residual risk)
        Med  vol regime → 85%  hedge
        High vol regime → 120% hedge  (over-hedge for adverse-move protection)

Value Proposition
-----------------
If volatility clusters (i.e. high-vol hours follow high-vol hours), then:
    - Regime B hedges more aggressively *before* the worst spread widening
    - Regime B hedges less in quiet periods → saves hedging cost
    - Net effect: lower hedging cost per unit of risk reduced

P&L Accounting (simplified market-maker frame)
----------------------------------------------
For each bar t:

  position_pnl(t)   = (price_change) × notional × (1 − hedge_ratio)
                       (unhedged residual exposure)

  hedge_cost(t)     = hedge_ratio × spread_pips(t) × pip_value × notional
                       + transaction_cost × |Δhedge_ratio| × notional
                       (entry cost + rebalancing friction)

  net_pnl(t)        = −|position_pnl(t)| − hedge_cost(t)
                       (negative because cost frame: we minimise losses)

  hedge_efficiency  = variance_reduction_vs_unhedged / total_hedge_cost
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PIP_VALUE = 10.0   # USD per pip per 100k lot (standard for EUR/USD)


@dataclass
class BacktestResult:
    strategy_name:    str
    pnl:              pd.Series
    hedge_cost:       pd.Series
    position_pnl:     pd.Series
    hedge_ratio:      pd.Series
    cumulative_pnl:   pd.Series = field(init=False)

    def __post_init__(self):
        self.cumulative_pnl = self.pnl.cumsum()


def _run_strategy(
    df: pd.DataFrame,
    hedge_ratios: Dict[int, float],
    notional: float,
    tx_cost_pips: float,
    name: str,
) -> BacktestResult:
    """
    Iterate bar-by-bar to avoid lookahead.
    hedge_ratios : mapping {regime_label → hedge_fraction}
                   Pass {0: static, 1: static, 2: static} for the baseline.
    """
    n = len(df)
    pos_pnl    = np.zeros(n)
    h_cost     = np.zeros(n)
    net_pnl    = np.zeros(n)
    h_ratio    = np.zeros(n)

    prev_ratio = list(hedge_ratios.values())[1]   # start at medium hedge

    for i, (idx, row) in enumerate(df.iterrows()):
        regime = int(row["vol_regime"])
        ratio  = hedge_ratios.get(regime, 1.0)

        # Unhedged position P&L (absolute price move on residual exposure)
        price_move_pips = abs(row["log_return"]) / 1e-4   # rough pip-equivalent
        residual        = 1.0 - ratio
        p_pnl           = -(residual * price_move_pips * PIP_VALUE * (notional / 100_000))

        # Hedge cost: spread × hedge size + rebalancing friction
        spread_cost = ratio * row["spread_pips"] * PIP_VALUE * (notional / 100_000)
        rebal_cost  = abs(ratio - prev_ratio) * tx_cost_pips * PIP_VALUE * (notional / 100_000)
        h_c         = spread_cost + rebal_cost

        pos_pnl[i] = p_pnl
        h_cost[i]  = h_c
        net_pnl[i] = p_pnl - h_c
        h_ratio[i] = ratio
        prev_ratio  = ratio

    idx = df.index
    return BacktestResult(
        strategy_name = name,
        pnl           = pd.Series(net_pnl,  index=idx, name="pnl"),
        hedge_cost    = pd.Series(h_cost,   index=idx, name="hedge_cost"),
        position_pnl  = pd.Series(pos_pnl,  index=idx, name="position_pnl"),
        hedge_ratio   = pd.Series(h_ratio,  index=idx, name="hedge_ratio"),
    )


def run_backtest(
    df: pd.DataFrame,
    dynamic_hedge_ratios: Dict[int, float],
    static_hedge_ratio: float,
    notional: float,
    tx_cost_pips: float,
) -> Dict[str, BacktestResult]:
    static_map = {k: static_hedge_ratio for k in dynamic_hedge_ratios}

    results = {
        "Static Hedge": _run_strategy(
            df, static_map, notional, tx_cost_pips, "Static Hedge"
        ),
        "Vol-Regime Hedge": _run_strategy(
            df, dynamic_hedge_ratios, notional, tx_cost_pips, "Vol-Regime Hedge"
        ),
    }
    logger.info("Backtest complete — %d bars.", len(df))
    return results
