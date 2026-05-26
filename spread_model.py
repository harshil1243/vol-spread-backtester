"""
spread_model.py
===============
Models the EUR/USD interbank spread as a linear function of volatility.

Motivation
----------
Market makers (like IG) widen their quoted spread during high-volatility periods
to compensate for adverse-selection risk (Kyle 1985, Glosten-Milgrom 1985).
The relationship is approximately:

    spread(t) = base_spread + α · RV(t) + β · GARCH_vol(t) + ε(t)

When real bid/ask tick data is unavailable we calibrate α and β so that the
simulated spread falls in the realistic 0.5–3.0 pip range observed in live
EUR/USD markets.  With actual order-book data these coefficients would be
estimated via OLS / WLS regression.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PIP = 1e-4


def simulate_spread(
    df: pd.DataFrame,
    base_pips: float,
    alpha: float,
    beta: float,
    seed: int = 0,
) -> pd.Series:
    """
    Simulate interbank spread in pips.

    Parameters
    ----------
    base_pips : minimum spread in quiet markets.
    alpha     : sensitivity to realised vol (annualised, price terms).
    beta      : sensitivity to GARCH conditional vol.
    """
    rng  = np.random.default_rng(seed)
    noise = rng.normal(0, 0.05, len(df))   # small idiosyncratic component

    spread = (
        base_pips
        + alpha * df["realized_vol"]
        + beta  * df["garch_vol"]
        + noise
    ).clip(lower=0.3)   # floor at 0.3 pips (tight algorithmic market)

    spread.name = "spread_pips"
    logger.info(
        "Spread stats — mean: %.3f pips  std: %.3f  max: %.3f",
        spread.mean(), spread.std(), spread.max()
    )
    return spread


def attach_spread(
    df: pd.DataFrame,
    base_pips: float,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    out = df.copy()
    out["spread_pips"] = simulate_spread(out, base_pips, alpha, beta)
    # Express spread as fraction of mid-price (basis points equivalent)
    out["spread_frac"] = (out["spread_pips"] * PIP) / out["close"]
    return out
