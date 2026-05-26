"""
main.py
=======
Orchestrates the full pipeline:

    1. Load EUR/USD hourly data
    2. Build volatility features (RV, Parkinson, GARCH, regimes)
    3. Attach simulated spread
    4. Run statistical tests (Ljung-Box, Granger, regime ANOVA)
    5. Run backtest — Static Hedge vs Vol-Regime Hedge
    6. Compute performance metrics
    7. Generate figures and save results
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

import config
from data_loader        import load_data, train_test_split
from volatility_model   import build_features
from spread_model        import attach_spread
from statistical_tests   import test_arch_effects, test_granger, regime_spread_stats, regime_transition_matrix
from backtest_engine     import run_backtest
from metrics             import summarise
from visualisation       import (
    plot_regime_overview,
    plot_spread_by_regime,
    plot_granger,
    plot_cumulative_pnl,
    plot_hedge_ratio,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    Path(config.FIGURES_DIR).mkdir(exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading EUR/USD data")
    raw = load_data(
        ticker=config.TICKER,
        start=config.START_DATE,
        end=config.END_DATE,
        interval=config.INTERVAL,
    )
    train_raw, test_raw = train_test_split(raw, config.TRAIN_RATIO)
    logger.info("Train: %d rows  |  Test: %d rows", len(train_raw), len(test_raw))

    # ── 2. Volatility features ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Building volatility features")
    train, thresholds = build_features(
        train_raw,
        rv_window    = config.RV_WINDOW,
        garch_p      = config.GARCH_P,
        garch_q      = config.GARCH_Q,
        quantile_bins= config.VOL_REGIME_BINS,
    )
    test, _ = build_features(
        test_raw,
        rv_window       = config.RV_WINDOW,
        garch_p         = config.GARCH_P,
        garch_q         = config.GARCH_Q,
        quantile_bins   = config.VOL_REGIME_BINS,
        train_thresholds= thresholds,   # no lookahead on test
    )

    # ── 3. Spread ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Attaching spread model")
    train = attach_spread(train, config.BASE_SPREAD_PIPS,
                          config.VOL_SPREAD_ALPHA, config.GARCH_SPREAD_BETA)
    test  = attach_spread(test,  config.BASE_SPREAD_PIPS,
                          config.VOL_SPREAD_ALPHA, config.GARCH_SPREAD_BETA)

    # ── 4. Statistical tests ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Statistical tests (train set)")

    logger.info("  [4a] Ljung-Box (ARCH effects / vol clustering)")
    lb = test_arch_effects(train["log_return"], lags=12)
    lb.to_csv(f"{config.RESULTS_DIR}/ljungbox.csv")

    logger.info("  [4b] Granger causality — RV → Spread")
    p_vals = test_granger(train, config.GRANGER_MAX_LAG, config.SIGNIFICANCE_LEVEL)
    pd.Series(p_vals, name="p_value").to_csv(f"{config.RESULTS_DIR}/granger.csv")

    logger.info("  [4c] Spread stats by regime (ANOVA)")
    regime_stats = regime_spread_stats(train)
    regime_stats.to_csv(f"{config.RESULTS_DIR}/regime_spread_stats.csv")
    print("\n── Conditional Spread by Regime (train) ──")
    print(regime_stats.to_string())
    print(f"  ANOVA  F={regime_stats.attrs['anova_f']:.2f}  "
          f"p={regime_stats.attrs['anova_p']:.2e}\n")

    logger.info("  [4d] Empirical regime transition matrix")
    trans = regime_transition_matrix(train["vol_regime"])
    trans.to_csv(f"{config.RESULTS_DIR}/transition_matrix.csv")
    print("── Regime Transition Matrix ──")
    print(trans.to_string(), "\n")

    # ── 5. Backtest ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 — Backtest on test set")
    results = run_backtest(
        df                  = test,
        dynamic_hedge_ratios= config.HEDGE_RATIOS,
        static_hedge_ratio  = config.STATIC_HEDGE,
        notional            = config.NOTIONAL,
        tx_cost_pips        = config.TRANSACTION_COST,
    )

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 — Performance metrics")
    # Unhedged baseline P&L for hedge efficiency calculation
    unhedged_pnl = -(abs(test["log_return"]) / 1e-4) * 10.0 * (config.NOTIONAL / 100_000)
    metrics_df = summarise(results, unhedged_pnl)
    metrics_df.to_csv(f"{config.RESULTS_DIR}/metrics.csv")
    print("\n── Strategy Performance (out-of-sample) ──")
    print(metrics_df.to_string(), "\n")

    # ── 7. Figures ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7 — Generating figures")
    combined = pd.concat([train, test])

    plot_regime_overview(combined,      f"{config.FIGURES_DIR}/01_regime_overview.png")
    plot_spread_by_regime(combined,     f"{config.FIGURES_DIR}/02_spread_by_regime.png")
    plot_granger(p_vals, config.SIGNIFICANCE_LEVEL,
                                        f"{config.FIGURES_DIR}/03_granger.png")
    plot_cumulative_pnl(results,        f"{config.FIGURES_DIR}/04_cumulative_pnl.png")
    plot_hedge_ratio(results, test,     f"{config.FIGURES_DIR}/05_hedge_ratio.png")

    logger.info("=" * 60)
    logger.info("Pipeline complete.  Figures → %s/   Results → %s/",
                config.FIGURES_DIR, config.RESULTS_DIR)


if __name__ == "__main__":
    main()
