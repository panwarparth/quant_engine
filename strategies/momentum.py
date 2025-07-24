"""
Momentum Strategy Implementation

This module defines a basic moving average crossover strategy. It generates
long/flat positions based on the relationship between a short‑term and a
long‑term moving average. When the short moving average crosses above the long
moving average, the strategy goes long (+1). When the short moving average
crosses below the long moving average, the position is closed (0). For a
purely academic demonstration you could invert the logic or allow short
positions; this implementation keeps it simple.

Usage:
    from quant_engine.strategies.momentum import MomentumStrategy
    import quant_engine.data_handler as dh

    handler = dh.DataHandler.synthetic(n_days=252)
    strategy = MomentumStrategy(short_window=20, long_window=50)
    signals = strategy.generate_signals(handler.get_prices())

    # signals is a pandas.Series of positions aligned with the input prices
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Union


@dataclass
class MomentumStrategy:
    """Simple moving average crossover strategy."""
    short_window: int = 20
    long_window: int = 50

    def generate_signals(self, prices: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Generate trading signals based on moving average crossovers.

        Parameters
        ----------
        prices : Union[pd.Series, pd.DataFrame]
            Price series or DataFrame containing a 'close' column.

        Returns
        -------
        pandas.Series
            A series of positions (1 for long, 0 for flat) indexed like the input.
        """
        # Accept either a Series of prices or DataFrame with 'close'
        if isinstance(prices, pd.DataFrame):
            price_series = prices['close']
        else:
            price_series = prices
        # compute moving averages
        short_ma = price_series.rolling(window=self.short_window, min_periods=1).mean()
        long_ma = price_series.rolling(window=self.long_window, min_periods=1).mean()
        # generate raw signals: long when short_ma > long_ma
        signal = (short_ma > long_ma).astype(int)
        # ensure no lookahead bias: shift the signal forward by 1 day to execute
        shifted_signal = signal.shift(1).fillna(0)
        return shifted_signal