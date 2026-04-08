"""Portfolio backtesting helpers implemented with pandas/numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    """Configuration for later long-short and long-only portfolio tests."""

    long_short_quantile: float = 0.2
    long_only_count: int = 50
    rebalance_frequency: str = "monthly"
    report_turnover: bool = True


def validate_portfolio_config(config: PortfolioConfig) -> None:
    """Validate portfolio construction choices."""

    if not 0 < config.long_short_quantile < 0.5:
        raise ValueError("long_short_quantile must be between 0 and 0.5")
    if config.long_only_count <= 0:
        raise ValueError("long_only_count must be positive")
    if config.rebalance_frequency != "monthly":
        raise ValueError("Stage 2 panel protocol assumes monthly rebalancing")


def equal_weight_long_short(signals: pd.DataFrame, date_col: str, stock_col: str, score_col: str, quantile: float = 0.2) -> pd.DataFrame:
    """Create equal-weight long-short weights from monthly cross-sectional scores."""

    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")
    required = [date_col, stock_col, score_col]
    missing = [col for col in required if col not in signals.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    def _weights(block: pd.DataFrame) -> pd.DataFrame:
        clean = block[[stock_col, score_col]].dropna().copy()
        if clean.empty:
            clean["weight"] = pd.Series(dtype="float64")
            return clean[[stock_col, "weight"]]
        low = clean[score_col].quantile(quantile)
        high = clean[score_col].quantile(1.0 - quantile)
        clean["weight"] = 0.0
        long_mask = clean[score_col] >= high
        short_mask = clean[score_col] <= low
        if long_mask.any():
            clean.loc[long_mask, "weight"] = 1.0 / long_mask.sum()
        if short_mask.any():
            clean.loc[short_mask, "weight"] = -1.0 / short_mask.sum()
        return clean[[stock_col, "weight"]]

    weights = signals.groupby(date_col, sort=True).apply(_weights, include_groups=False).reset_index(level=0)
    return weights.rename(columns={date_col: "date", stock_col: "stock_id"})


def monthly_turnover(weights: pd.DataFrame, date_col: str = "date", stock_col: str = "stock_id", weight_col: str = "weight") -> pd.Series:
    """Compute one-way monthly turnover from a long-format weight panel."""

    wide = weights.pivot_table(index=date_col, columns=stock_col, values=weight_col, fill_value=0.0).sort_index()
    turnover = wide.diff().abs().sum(axis=1) / 2.0
    turnover.iloc[0] = np.nan
    return turnover.rename("turnover")


def run_backtest(*_args: object, config: PortfolioConfig | None = None, **_kwargs: object) -> object:
    """Validate settings; full portfolio backtesting is reserved for a later stage."""

    validate_portfolio_config(config or PortfolioConfig())
    raise NotImplementedError("Actual portfolio backtesting is reserved for the portfolio stage.")
