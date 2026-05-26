"""
data_loader.py
==============
Primary source : yfinance  (EUR/USD hourly OHLCV).
Fallback       : Synthetic 3-state Markov-switching GBM so the repo runs
                 offline and in CI without any API dependency.

The synthetic generator deliberately embeds volatility clustering via a
persistence-biased Markov transition matrix — letting us verify our models
against known ground truth.
"""

from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PIP = 1e-4   # EUR/USD pip size


# ── yfinance ─────────────────────────────────────────────────────────────────

def _load_yfinance(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end,
                         interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Empty DataFrame.")
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index.name = "datetime"
        logger.info("yfinance: loaded %d rows (%s).", len(df), ticker)
        return df
    except Exception as exc:
        logger.warning("yfinance unavailable (%s). Using synthetic data.", exc)
        return None


# ── Synthetic EUR/USD ─────────────────────────────────────────────────────────

def generate_synthetic(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    3-state Markov-switching GBM for EUR/USD hourly bars.

    States
    ------
    0 – Low vol   σ ≈ 4 pips/hr   (quiet Asian session)
    1 – Med vol   σ ≈ 8 pips/hr   (London open)
    2 – High vol  σ ≈ 16 pips/hr  (macro news / risk-off)

    Transition matrix is persistence-heavy to embed genuine vol clustering.
    """
    rng = np.random.default_rng(seed)

    # Persistence-biased transition matrix
    P = np.array([
        [0.91, 0.07, 0.02],   # from Low
        [0.08, 0.82, 0.10],   # from Med
        [0.03, 0.14, 0.83],   # from High
    ])

    vol_by_state = np.array([4.0, 8.0, 16.0]) * PIP   # σ per hour

    # Simulate Markov chain
    states = np.empty(n, dtype=int)
    states[0] = 1
    for t in range(1, n):
        states[t] = rng.choice(3, p=P[states[t - 1]])

    sigma   = vol_by_state[states]
    returns = sigma * rng.standard_normal(n)
    mid     = 1.1000 * np.cumprod(1 + returns)

    noise  = sigma * 0.4
    close_ = mid   + rng.normal(0, noise * 0.3, n)
    open_  = mid   + rng.normal(0, noise * 0.3, n)
    high_  = np.maximum(open_, close_) + np.abs(rng.normal(0, noise, n))
    low_   = np.minimum(open_, close_) - np.abs(rng.normal(0, noise, n))

    idx = pd.date_range("2022-01-03 00:00", periods=n, freq="h")
    df  = pd.DataFrame(
        {"open": open_, "high": high_, "low": low_,
         "close": close_, "volume": rng.integers(500, 5000, n).astype(float),
         "true_regime": states},
        index=idx,
    )
    df.index.name = "datetime"
    logger.info("Synthetic: generated %d EUR/USD bars.", n)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(
    ticker: str, start: str, end: str, interval: str,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    if not force_synthetic:
        df = _load_yfinance(ticker, start, end, interval)
        if df is not None:
            return df
    return generate_synthetic()


def train_test_split(
    df: pd.DataFrame, train_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
