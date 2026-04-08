"""Portfolio backtesting configuration and Stage 1 placeholder."""

from __future__ import annotations

from dataclasses import dataclass


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
        raise ValueError("Stage 1 protocol assumes monthly rebalancing")


def run_backtest(*_args: object, config: PortfolioConfig | None = None, **_kwargs: object) -> object:
    """Placeholder for portfolio backtesting."""

    validate_portfolio_config(config or PortfolioConfig())
    raise NotImplementedError("Actual portfolio backtesting is outside the Stage 1 scope.")
