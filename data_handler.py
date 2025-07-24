"""
Data Handler module for the Quantitative Trading Engine.

This module provides a simple interface to ingest historical price data into the
backtesting system. In a production environment, you would replace or extend
this module to pull data from real‑time feeds (e.g. broker APIs, WebSockets) or
to parse a wide variety of file formats (CSV, HDF5, Parquet). For the purposes
of this project we keep the interface lean and focus on readability.

Usage:
    from quant_engine.data_handler import DataHandler

    # load a CSV file of daily OHLCV data
    handler = DataHandler.from_csv("./sample_data/spy.csv")
    prices = handler.get_prices()
    returns = handler.get_returns()

    # alternatively, generate synthetic data for testing
    synthetic_handler = DataHandler.synthetic(n_days=252*2, seed=42)
    synthetic_prices = synthetic_handler.get_prices()
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class DataHandler:
    """Class to manage historical price data for backtesting.

    Attributes
    ----------
    data : pandas.DataFrame
        DataFrame indexed by datetime with a 'close' column (others are optional).
        Additional columns such as 'open', 'high', 'low', 'volume' can be present.
    symbol : str
        Name of the instrument or symbol represented by the data.
    """

    data: pd.DataFrame
    symbol: str = field(default="UNKNOWN")

    @classmethod
    def from_csv(cls, filepath: str, symbol: Optional[str] = None, parse_dates: bool = True) -> "DataHandler":
        """Create a DataHandler instance from a CSV file.

        The CSV should contain at least a 'Date' column and a 'Close' column.
        Additional OHLCV columns are preserved. Dates are parsed into the index.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        symbol : Optional[str]
            Optional symbol name for the dataset. Defaults to the filename stem.
        parse_dates : bool
            Whether to parse the 'Date' column as datetimes.

        Returns
        -------
        DataHandler
            A new DataHandler instance containing the loaded data.
        """
        df = pd.read_csv(filepath)
        # Normalize column names
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        if parse_dates and ('date' in df.columns or 'datetime' in df.columns):
            date_col = 'date' if 'date' in df.columns else 'datetime'
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
        # Ensure 'close' column exists
        if 'close' not in df.columns:
            raise ValueError("CSV must contain a 'Close' column (case‑insensitive).")
        if symbol is None:
            from pathlib import Path
            symbol = Path(filepath).stem
        return cls(data=df, symbol=symbol)

    @classmethod
    def synthetic(cls, n_days: int = 252, seed: Optional[int] = None, symbol: str = "SYNTH") -> "DataHandler":
        """Generate synthetic log‑normally distributed price data.

        Creates a random walk with drift to mimic equity returns. The drift and
        volatility parameters are heuristically chosen. You can customize this
        method to generate different market regimes or multiple symbols.

        Parameters
        ----------
        n_days : int
            Number of trading days to simulate.
        seed : Optional[int]
            Seed for the random number generator to ensure reproducibility.
        symbol : str
            Symbol name for the synthetic series.

        Returns
        -------
        DataHandler
            A new DataHandler instance containing the synthetic price series.
        """
        rng = np.random.default_rng(seed)
        # assume 252 trading days per year; set annualized drift and volatility
        mu_annual = 0.10  # 10% annual drift
        sigma_annual = 0.20  # 20% annual volatility
        dt = 1/252  # daily time step
        mu = (mu_annual - 0.5 * sigma_annual ** 2) * dt
        sigma = sigma_annual * np.sqrt(dt)
        # generate log returns
        log_returns = mu + sigma * rng.standard_normal(n_days)
        # convert to price series starting at 100
        prices = 100 * np.exp(np.cumsum(log_returns))
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
        df = pd.DataFrame({'close': prices}, index=dates)
        return cls(data=df, symbol=symbol)

    def get_prices(self) -> pd.Series:
        """Return the close price series.

        Returns
        -------
        pandas.Series
            Series of close prices indexed by datetime.
        """
        return self.data['close']

    def get_returns(self) -> pd.Series:
        """Compute daily returns from the close price series.

        Returns
        -------
        pandas.Series
            Daily percentage returns (pct_change) with the first value NaN.
        """
        returns = self.get_prices().pct_change().dropna()
        return returns

    def get_ohlc(self) -> pd.DataFrame:
        """Return OHLCV data if available.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing available columns from the dataset.
        """
        return self.data.copy()