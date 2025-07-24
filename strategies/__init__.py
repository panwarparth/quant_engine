"""Strategy subpackage for quant_engine.

This package groups different trading strategies. New strategies can be added
as separate modules and exposed here.
"""

from .momentum import MomentumStrategy

__all__ = ["MomentumStrategy"]