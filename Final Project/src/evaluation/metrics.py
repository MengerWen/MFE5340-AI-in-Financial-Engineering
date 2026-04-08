"""Metric names and small dependency-light metric helpers."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt


PRIMARY_METRICS: tuple[str, ...] = (
    "oos_r2",
    "rank_ic",
    "pricing_error",
    "long_short_return",
    "long_only_return",
    "sharpe_ratio",
    "max_drawdown",
    "turnover",
)


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean for non-empty numeric sequences."""

    if not values:
        raise ValueError("values must be non-empty")
    return sum(values) / len(values)


def annualized_sharpe(monthly_returns: Sequence[float]) -> float:
    """Compute a simple monthly-to-annual Sharpe ratio without risk-free adjustment."""

    if len(monthly_returns) < 2:
        raise ValueError("at least two monthly returns are required")
    avg = mean(monthly_returns)
    variance = sum((value - avg) ** 2 for value in monthly_returns) / (len(monthly_returns) - 1)
    volatility = sqrt(variance)
    if volatility == 0:
        raise ValueError("return volatility is zero")
    return sqrt(12) * avg / volatility
