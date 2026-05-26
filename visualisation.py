"""
visualisation.py
================
All charts for the research summary and README.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = {
    "Low":    "#2ecc71",
    "Med":    "#f39c12",
    "High":   "#e74c3c",
    "static": "#3498db",
    "regime": "#e74c3c",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── 1. Price + Vol Regime overview ────────────────────────────────────────────

def plot_regime_overview(df: pd.DataFrame, out: str) -> None:
    regime_colors = {0: PALETTE["Low"], 1: PALETTE["Med"], 2: PALETTE["High"]}
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Price
    axes[0].plot(df.index, df["close"], lw=0.8, color="#2c3e50")
    axes[0].set_ylabel("EUR/USD")
    axes[0].set_title("EUR/USD Price, Volatility Regimes, and Spread", fontsize=13)

    # Realised vol coloured by regime
    for i, (idx, row) in enumerate(df.iterrows()):
        axes[1].axvline(idx, color=regime_colors.get(int(row["vol_regime"]), "grey"),
                        alpha=0.3, lw=1.2)
    axes[1].plot(df.index, df["realized_vol"], lw=0.9, color="#2c3e50", zorder=2)
    axes[1].set_ylabel("Realised Vol (ann.)")
    for label, color in [("Low", PALETTE["Low"]), ("Med", PALETTE["Med"]),
                          ("High", PALETTE["High"])]:
        axes[1].plot([], [], color=color, lw=4, alpha=0.6, label=f"{label} regime")
    axes[1].legend(loc="upper right", fontsize=8)

    # Spread
    axes[2].fill_between(df.index, df["spread_pips"], color="#c0392b", alpha=0.4)
    axes[2].plot(df.index, df["spread_pips"], lw=0.7, color="#c0392b")
    axes[2].set_ylabel("Simulated Spread (pips)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()
    _save(fig, out)


# ── 2. Spread distribution by regime ─────────────────────────────────────────

def plot_spread_by_regime(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    for regime_id, label in sorted(label_map.items()):
        subset = df[df["vol_regime"] == regime_id]["spread_pips"].dropna()
        sns.kdeplot(subset, ax=ax, label=f"{label} Vol Regime",
                    color=list(PALETTE.values())[regime_id], fill=True, alpha=0.25)
    ax.set_xlabel("Spread (pips)")
    ax.set_ylabel("Density")
    ax.set_title("Spread Distribution by Volatility Regime", fontsize=13)
    ax.legend()
    _save(fig, out)


# ── 3. Granger causality p-values ────────────────────────────────────────────

def plot_granger(p_values: dict[int, float], sig: float, out: str) -> None:
    lags = list(p_values.keys())
    pvals = list(p_values.values())
    colors = [PALETTE["regime"] if p < sig else PALETTE["static"] for p in pvals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, pvals, color=colors, edgecolor="white", width=0.6)
    ax.axhline(sig, color="black", ls="--", lw=1.2, label=f"α = {sig}")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("p-value (F-test)")
    ax.set_title("Granger Causality: Does RV(t−k) predict Spread(t)?", fontsize=13)
    ax.legend()
    _save(fig, out)


# ── 4. Cumulative P&L comparison ─────────────────────────────────────────────

def plot_cumulative_pnl(results: dict, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in results.items():
        color = PALETTE["static"] if "Static" in name else PALETTE["regime"]
        axes[0].plot(res.cumulative_pnl.index, res.cumulative_pnl,
                     label=name, color=color, lw=1.5)
    axes[0].set_title("Cumulative Net P&L", fontsize=13)
    axes[0].set_ylabel("USD")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()

    # Hedge cost comparison
    costs = {name: res.hedge_cost.sum() for name, res in results.items()}
    bars  = axes[1].bar(costs.keys(), costs.values(),
                         color=[PALETTE["static"], PALETTE["regime"]], edgecolor="white")
    axes[1].set_title("Total Hedge Cost (USD)", fontsize=13)
    axes[1].set_ylabel("USD")
    for bar, val in zip(bars, costs.values()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"${val:,.0f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Strategy Comparison — Out-of-Sample (Test Set)", fontsize=14, y=1.01)
    _save(fig, out)


# ── 5. Hedge ratio over time ──────────────────────────────────────────────────

def plot_hedge_ratio(results: dict, df: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    res_dyn = results["Vol-Regime Hedge"]
    axes[0].plot(res_dyn.hedge_ratio.index, res_dyn.hedge_ratio,
                 lw=0.9, color=PALETTE["regime"], alpha=0.8, label="Dynamic ratio")
    axes[0].axhline(1.0, ls="--", lw=1, color="#7f8c8d", label="Static = 100%")
    axes[0].set_ylabel("Hedge Ratio")
    axes[0].set_title("Dynamic Hedge Ratio vs Vol Regime", fontsize=13)
    axes[0].legend(fontsize=9)

    axes[1].plot(df.index, df["realized_vol"], lw=0.8, color="#2c3e50")
    axes[1].set_ylabel("Realised Vol")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()
    _save(fig, out)
