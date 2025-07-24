"""Initialize the quant_engine package.

Exposes key components at the top level for convenience. You can import
DataHandler, MomentumStrategy, Backtester, and Kelly Criterion functions directly
from this package:

    from quant_engine import DataHandler, MomentumStrategy, Backtester

This init also sets up the package structure for Python.
"""

from .data_handler import DataHandler
from .strategies.momentum import MomentumStrategy
from .backtester import Backtester
from .risk.kelly_criterion import kelly_fraction_win_loss, kelly_fraction_continuous

__all__ = [
    "DataHandler",
    "MomentumStrategy",
    "Backtester",
    "kelly_fraction_win_loss",
    "kelly_fraction_continuous",
]