"""Validate pandas/numpy portfolio backtesting settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.backtest import PortfolioConfig, validate_portfolio_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate portfolio backtest settings.")
    parser.add_argument("--long-short-quantile", default=0.2, type=float)
    parser.add_argument("--long-only-count", default=50, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PortfolioConfig(
        long_short_quantile=args.long_short_quantile,
        long_only_count=args.long_only_count,
    )
    validate_portfolio_config(config)
    print(f"Portfolio config is valid for pandas/numpy backtesting helpers: {config}")
    print("Full portfolio backtesting is reserved for the portfolio stage.")


if __name__ == "__main__":
    main()
