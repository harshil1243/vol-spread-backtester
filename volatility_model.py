"""
volatility_model.py
===================
Realised Volatility  →  Parkinson Estimator  →  GARCH(1,1)  →  Regime Labels

All thresholds are computed on the training set and applied to the test set
without lookahead to preserve walk-forward integrity.
"""

from __future__ import annotations
import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANNUALISE = np.sqrt(8_760)   # hourly bars → annualised vol


# ── Returns ───────────────────────────────────────────────────────────────────

def log_returns(close: pd.Series) -> pd.Series:
    r = np.log(close / close.shift(1))
    r.name = "log_return"
    return r


# ── Realised Volatility ───────────────────────────────────────────────────────

def realised_vol(returns: pd.Series, window: int) -> pd.Series:
    """Close-to-close rolling RV, annualised."""
    rv = returns.rolling(window).std() * ANNUALISE
    rv.name = "realized_vol"
    return rv


def parkinson_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Parkinson (1980) high-low estimator.
    More efficient than close-to-close; captures intrabar range.
    """
    hl2 = np.log(df["high"] / df["low"]) ** 2
    pv  = np.sqrt(hl2.rolling(window).mean() / (4 * np.log(2))) * ANNUALISE
    pv.name = "parkinson_vol"
    return pv


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> pd.Series:
    """
    Fit GARCH(p,q) via the arch library.
    Falls back to realised vol if arch is not installed.
    """
    try:
        from arch import arch_model
        scaled = returns * 10_000   # scale to basis points for numerical stability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(scaled, vol="Garch", p=p, q=q,
                             rescale=False).fit(disp="off", show_warning=False)
        cond_vol = (res.conditional_volatility / 10_000) * ANNUALISE
        cond_vol.name = "garch_vol"
        cond_vol = cond_vol.reindex(returns.index)
        logger.info("GARCH(%d,%d) fit — AIC %.1f  BIC %.1f", p, q, res.aic, res.bic)
        return cond_vol
    except Exception as exc:
        logger.warning("GARCH failed (%s). Falling back to RV.", exc)
        return realised_vol(returns, window=24).rename("garch_vol")


# ── Regime Classification ─────────────────────────────────────────────────────

def compute_thresholds(vol: pd.Series, quantile_bins: list[float]) -> np.ndarray:
    """Derive cut-points from a training vol series."""
    thresholds = vol.dropna().quantile(quantile_bins[1:-1]).values
    logger.info("Regime thresholds (annualised): %s", np.round(thresholds, 4))
    return thresholds


def apply_thresholds(vol: pd.Series, thresholds: np.ndarray) -> pd.Series:
    regimes = pd.Series(
        np.digitize(vol.ffill(), bins=thresholds).astype(int),
        index=vol.index,
        name="vol_regime",
    )
    return regimes


# ── Full Feature Pipeline ─────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    rv_window: int,
    garch_p: int,
    garch_q: int,
    quantile_bins: list[float],
    train_thresholds: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Attach vol features and regime labels to df.

    Returns
    -------
    df_out         : DataFrame with all vol columns added.
    thresholds     : The quantile cut-points used (from training if supplied).
    """
    out  = df.copy()
    rets = log_returns(out["close"])

    out["log_return"]    = rets
    out["realized_vol"]  = realised_vol(rets, rv_window)
    out["parkinson_vol"] = parkinson_vol(out, rv_window)
    out["garch_vol"]     = fit_garch(rets.reindex(out.index).fillna(0), garch_p, garch_q)

    if train_thresholds is None:
        thresholds = compute_thresholds(out["realized_vol"], quantile_bins)
    else:
        thresholds = train_thresholds

    out["vol_regime"] = apply_thresholds(out["realized_vol"], thresholds)

    # Lagged features for predictive modelling
    for lag in [1, 2, 3, 6]:
        out[f"rv_lag{lag}"]     = out["realized_vol"].shift(lag)
        out[f"regime_lag{lag}"] = out["vol_regime"].shift(lag).astype("Int64")

    out.dropna(inplace=True)
    logger.info("Feature build complete: %d rows, %d columns.", *out.shape)
    return out, thresholds
