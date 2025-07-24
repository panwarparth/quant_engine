"""
Kelly Criterion Module

This module contains functions to compute optimal bet sizes using the Kelly
criterion. The Kelly criterion maximizes the expected logarithm of wealth and
is widely used in risk management and position sizing. Given an expected
return and variance of returns, the Kelly fraction indicates the proportion of
capital that should be allocated to a bet or investment. For discrete win/loss
outcomes with known probabilities, use the closed‑form `kelly_fraction`.
For continuous returns, the ratio of mean to variance is used as an
approximation.
"""

from __future__ import annotations
import numpy as np
from typing import Iterable


def kelly_fraction_win_loss(p: float, b: float) -> float:
    """Compute the Kelly fraction for a binary win/loss scenario.

    The Kelly criterion for a bet that returns b:1 on a win with probability p
    (and loses 1 unit otherwise) is given by f* = (b * p - (1-p)) / b. When
    b=1 (even odds) this simplifies to f* = 2p - 1.

    Parameters
    ----------
    p : float
        Probability of winning (0 < p < 1).
    b : float
        Net odds received on a win. For example, if you win $1 for every $1
        bet, b=1. For a bet that returns $2 for every $1, b=2.

    Returns
    -------
    float
        Optimal fraction of capital to wager.
    """
    if not 0 < p < 1:
        raise ValueError("Probability p must be between 0 and 1.")
    if b <= 0:
        raise ValueError("Odds b must be positive.")
    return (b * p - (1 - p)) / b


def kelly_fraction_continuous(returns: Iterable[float]) -> float:
    """Approximate Kelly fraction from a series of returns.

    This function estimates the Kelly fraction by assuming returns are IID and
    applying the continuous Kelly formula: f* = mean(return) / variance(return).
    It is intended for illustration; real applications require careful
    consideration of distributional assumptions and drawdowns.

    Parameters
    ----------
    returns : Iterable[float]
        Sequence of return values (e.g., daily returns) expressed in decimal form.

    Returns
    -------
    float
        Approximate Kelly fraction.
    """
    r = np.asarray(list(returns))
    mu = np.mean(r)
    var = np.var(r)
    if var == 0:
        return 0.0
    return mu / var