"""Evaluation metrics implemented with pandas, numpy, and scikit-learn."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


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


def metric_catalog() -> pd.DataFrame:
    """Return the planned metric catalog as a pandas DataFrame."""

    return pd.DataFrame(
        [
            {"metric": "oos_r2", "stage": "prediction", "description": "Out-of-sample R2 for return forecasts"},
            {"metric": "rank_ic", "stage": "prediction", "description": "Monthly Spearman rank correlation"},
            {"metric": "pricing_error", "stage": "asset_pricing", "description": "Cross-sectional pricing error"},
            {"metric": "long_short_return", "stage": "portfolio", "description": "Long-short realized portfolio return"},
            {"metric": "long_only_return", "stage": "portfolio", "description": "Long-only realized portfolio return"},
            {"metric": "sharpe_ratio", "stage": "portfolio", "description": "Annualized Sharpe ratio"},
            {"metric": "max_drawdown", "stage": "portfolio", "description": "Maximum drawdown"},
            {"metric": "turnover", "stage": "portfolio", "description": "Portfolio turnover"},
        ]
    )


def as_series(values: Iterable[float] | pd.Series, name: str | None = None) -> pd.Series:
    """Convert numeric inputs to a clean pandas Series."""

    series = values.copy() if isinstance(values, pd.Series) else pd.Series(list(values), dtype="float64")
    if name is not None:
        series.name = name
    return pd.to_numeric(series, errors="coerce").dropna()


def annualized_sharpe(monthly_returns: Iterable[float] | pd.Series) -> float:
    """Compute annualized Sharpe ratio from monthly returns with pandas/numpy."""

    returns = as_series(monthly_returns, name="monthly_return")
    if returns.size < 2:
        raise ValueError("at least two monthly returns are required")
    volatility = returns.std(ddof=1)
    if np.isclose(volatility, 0.0):
        raise ValueError("return volatility is zero")
    return float(np.sqrt(12.0) * returns.mean() / volatility)


def out_of_sample_r2(y_true: Iterable[float] | pd.Series, y_pred: Iterable[float] | pd.Series) -> float:
    """Compute OOS R2 using scikit-learn after pandas alignment."""

    frame = pd.concat(
        [as_series(y_true, "y_true"), as_series(y_pred, "y_pred")],
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        raise ValueError("aligned y_true/y_pred data is empty")
    return float(r2_score(frame["y_true"], frame["y_pred"]))


def rank_ic_by_month(frame: pd.DataFrame, date_col: str, actual_col: str, forecast_col: str) -> pd.Series:
    """Compute Spearman rank IC for each month with pandas."""

    required = [date_col, actual_col, forecast_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return (
        frame[required]
        .dropna()
        .groupby(date_col, sort=True)
        .apply(lambda block: block[actual_col].corr(block[forecast_col], method="spearman"), include_groups=False)
        .rename("rank_ic")
    )


def max_drawdown(monthly_returns: Iterable[float] | pd.Series) -> float:
    """Compute maximum drawdown from a monthly return series."""

    returns = as_series(monthly_returns, name="monthly_return")
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())
