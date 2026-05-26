"""
statistical_tests.py
====================
Three tests that underpin the core hypothesis:

    "EUR/USD volatility clusters, and lagged realised volatility
     Granger-causes spread widening."

Test 1 — Ljung-Box  :  Autocorrelation in squared returns (ARCH effects)
Test 2 — Granger    :  Does RV(t-k) help predict spread(t)?
Test 3 — Regime     :  Conditional spread distribution by vol regime
"""

from __future__ import annotations
import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


# ── 1. Ljung-Box (ARCH effects / vol clustering) ──────────────────────────────

def test_arch_effects(returns: pd.Series, lags: int = 12) -> pd.DataFrame:
    """
    H0: No autocorrelation in squared returns (no vol clustering).
    Rejecting H0 supports the clustering hypothesis.
    """
    sq_rets = returns.dropna() ** 2
    lb = acorr_ljungbox(sq_rets, lags=lags, return_df=True)
    lb.index.name = "lag"
    logger.info("Ljung-Box on squared returns:\n%s", lb.to_string())
    return lb


# ── 2. Granger Causality ──────────────────────────────────────────────────────

def test_granger(
    df: pd.DataFrame,
    max_lag: int,
    significance: float,
) -> Dict[int, float]:
    """
    Test whether lagged RV Granger-causes spread_pips.

    Returns a dict {lag → p-value (F-test)}.
    """
    data = df[["spread_pips", "realized_vol"]].dropna()
    results_raw = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    p_values: Dict[int, float] = {}
    for lag, res in results_raw.items():
        p = res[0]["ssr_ftest"][1]   # p-value from F-test
        p_values[lag] = round(p, 6)
        sig = "✓ SIGNIFICANT" if p < significance else "  not significant"
        logger.info("Granger lag %d: p=%.4f  %s", lag, p, sig)

    return p_values


# ── 3. Conditional Spread by Regime ──────────────────────────────────────────

def regime_spread_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean / std / median spread for each vol regime.
    Tests whether high-regime is statistically wider than low-regime (t-test).
    """
    stats_rows = []
    for regime, grp in df.groupby("vol_regime"):
        label = {0: "Low", 1: "Medium", 2: "High"}.get(regime, str(regime))
        stats_rows.append({
            "regime":       label,
            "count":        len(grp),
            "mean_spread":  grp["spread_pips"].mean(),
            "std_spread":   grp["spread_pips"].std(),
            "median_spread": grp["spread_pips"].median(),
        })
    result = pd.DataFrame(stats_rows).set_index("regime")

    # One-way ANOVA: are spread means different across regimes?
    groups = [df[df["vol_regime"] == r]["spread_pips"].dropna() for r in sorted(df["vol_regime"].unique())]
    f_stat, p_anova = stats.f_oneway(*groups)
    logger.info("ANOVA across regimes: F=%.3f  p=%.6f", f_stat, p_anova)
    result.attrs["anova_f"] = f_stat
    result.attrs["anova_p"] = p_anova

    return result.round(4)


# ── Regime Persistence (Transition) Matrix ────────────────────────────────────

def regime_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """
    Empirical Markov transition matrix.
    High diagonal values confirm volatility clustering.
    """
    labels = {0: "Low", 1: "Med", 2: "High"}
    r      = regimes.dropna().astype(int)
    counts = pd.crosstab(r.map(labels), r.shift(-1).map(labels), normalize="index")
    counts = counts.round(3)
    logger.info("Regime transition matrix:\n%s", counts.to_string())
    return counts
