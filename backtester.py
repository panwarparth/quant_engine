"""
Backtester Module

This module implements a simple event‑driven backtester for single‑asset
strategies. It evaluates a strategy by iterating over historical data,
generating signals, computing positions, and tracking the portfolio equity
curve. Key performance metrics (cumulative return, annualized return, Sharpe
ratio, maximum drawdown, and Value at Risk) are calculated to assess the
strategy's risk‑adjusted performance.

Extending this backtester to support multi‑asset portfolios, transaction costs,
broker slippage, or intraday bar spacing is encouraged. See the `TODO` notes
for ideas on future enhancements.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Dict, Any


@dataclass
class Backtester:
    """Backtest a trading strategy on historical price data."""
    prices: pd.Series
    signals: pd.Series
    initial_capital: float = 10000.0
    transaction_cost: float = 0.0  # flat commission per trade, ignored for now

    def run(self) -> Dict[str, Any]:
        """Run the backtest and compute portfolio performance.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the equity curve, returns, and performance metrics.
        """
        # Align signals with prices
        signals = self.signals.reindex(self.prices.index).fillna(0)
        # compute daily returns
        returns = self.prices.pct_change().fillna(0)
        # compute strategy returns: position * daily return
        strat_returns = signals * returns
        # compute cumulative wealth
        equity_curve = (1 + strat_returns).cumprod() * self.initial_capital
        # compute metrics
        metrics = compute_performance_metrics(strat_returns)
        return {
            'equity_curve': equity_curve,
            'strategy_returns': strat_returns,
            'signals': signals,
            'metrics': metrics
        }

    def plot(self, results: Dict[str, Any], title: str = "Equity Curve") -> None:
        """Plot the equity curve of the backtest.

        Parameters
        ----------
        results : Dict[str, Any]
            Dictionary returned by `run()` containing 'equity_curve'.
        title : str
            Title of the plot.
        """
        equity_curve = results['equity_curve']
        plt.figure(figsize=(10, 4))
        plt.plot(equity_curve.index, equity_curve.values)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.tight_layout()
        # Instead of showing, we return the figure. Users can save or display.
        return plt.gcf()


def compute_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """Compute common performance metrics for a series of returns.

    Parameters
    ----------
    returns : pandas.Series
        Series of periodic strategy returns (not cumulative). Assumes daily frequency.
    risk_free_rate : float
        Daily risk‑free rate used in Sharpe ratio calculation. Default 0.

    Returns
    -------
    Dict[str, float]
        Dictionary containing metrics: total_return, annualized_return,
        annualized_volatility, sharpe_ratio, max_drawdown, value_at_risk
        (95% one‑day VaR).
    """
    # Cumulative return
    cumulative = (1 + returns).prod() - 1
    # Annualization factor (approx. 252 trading days)
    ann_factor = 252
    # Annualized return
    avg_daily_return = returns.mean()
    ann_return = (1 + avg_daily_return) ** ann_factor - 1
    # Annualized volatility
    ann_volatility = returns.std(ddof=1) * np.sqrt(ann_factor)
    # Sharpe ratio
    # If volatility is zero (flat returns), Sharpe ratio is 0
    if ann_volatility != 0:
        sharpe = (avg_daily_return - risk_free_rate) / returns.std(ddof=1) * np.sqrt(ann_factor)
    else:
        sharpe = 0.0
    # Max drawdown
    cumulative_return_series = (1 + returns).cumprod()
    peak = cumulative_return_series.cummax()
    drawdowns = (cumulative_return_series - peak) / peak
    max_drawdown = drawdowns.min()
    # Value at Risk (VaR) at 95% confidence (one‑day horizon)
    # Using historical simulation: the 5th percentile of the return distribution
    var_95 = np.percentile(returns.dropna(), 5)
    return {
        'total_return': cumulative,
        'annualized_return': ann_return,
        'annualized_volatility': ann_volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'value_at_risk': var_95
    }