"""
Example script to run a simple backtest using the Quantitative Trading Engine.

This script demonstrates how to stitch together the data handler, strategy,
and backtester. It generates synthetic data, applies a moving average
crossover strategy, runs the backtest, prints performance metrics, and
saves a plot of the equity curve. Use this as a starting point to experiment
with different strategies or real datasets.
"""

from __future__ import annotations
# Ensure the parent directory is in sys.path so quant_engine can be imported
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # Use non‑interactive backend for headless environments
import matplotlib.pyplot as plt

from quant_engine.data_handler import DataHandler
from quant_engine.strategies.momentum import MomentumStrategy
from quant_engine.backtester import Backtester


def main():
    # Generate synthetic price data for two years (~504 trading days)
    handler = DataHandler.synthetic(n_days=504, seed=123, symbol="SYNTH")
    prices = handler.get_prices()

    # Instantiate the strategy
    strategy = MomentumStrategy(short_window=20, long_window=50)
    signals = strategy.generate_signals(prices)

    # Run the backtest
    bt = Backtester(prices=prices, signals=signals, initial_capital=10000.0)
    results = bt.run()

    # Print performance metrics
    print("Performance Metrics:\n")
    for k, v in results['metrics'].items():
        print(f"{k:20s}: {v:.4f}")

    # Plot and save the equity curve
    fig = bt.plot(results, title="Synthetic Momentum Strategy Equity Curve")
    output_path = Path(__file__).resolve().parent / "equity_curve.png"
    fig.savefig(output_path)
    print(f"Equity curve saved to: {output_path}")


if __name__ == "__main__":
    main()