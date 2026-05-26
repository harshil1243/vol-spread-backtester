"""
config.py — Central configuration for the EUR/USD Volatility-Spread Backtester.
"""

# ── Data ─────────────────────────────────────────────────────────────────────
TICKER      = "EURUSD=X"
START_DATE  = "2022-01-01"
END_DATE    = "2024-12-31"
INTERVAL    = "1h"

# ── Volatility model ─────────────────────────────────────────────────────────
RV_WINDOW        = 24          # rolling hours for realised volatility
GARCH_P          = 1
GARCH_Q          = 1
VOL_REGIME_BINS  = [0, 0.33, 0.67, 1.0]   # quantile cut-points: Low / Med / High

# ── Spread model ─────────────────────────────────────────────────────────────
BASE_SPREAD_PIPS     = 0.8    # EUR/USD typical interbank spread
VOL_SPREAD_ALPHA     = 18.0   # sensitivity to realised vol  (tuned to realistic pip range)
GARCH_SPREAD_BETA    = 10.0   # sensitivity to GARCH cond. vol

# ── Backtest ─────────────────────────────────────────────────────────────────
TRAIN_RATIO      = 0.70
NOTIONAL         = 100_000    # 1 standard FX lot
TRANSACTION_COST = 0.15       # pips per hedge trade (one-way)

# Hedge ratio per regime  {0=Low, 1=Medium, 2=High}
HEDGE_RATIOS = {0: 0.50, 1: 0.85, 2: 1.20}
STATIC_HEDGE = 1.00

# ── Statistical tests ────────────────────────────────────────────────────────
GRANGER_MAX_LAG    = 6
SIGNIFICANCE_LEVEL = 0.05

# ── Output dirs ──────────────────────────────────────────────────────────────
FIGURES_DIR = "figures"
RESULTS_DIR = "results"
