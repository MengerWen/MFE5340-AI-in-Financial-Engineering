"""Preprocessing interfaces for the monthly asset-pricing panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for Stage 2 panel construction."""

    feature_universe: str = "features500"
    target_horizon_months: int = 1
    use_blacklist: bool = True
    use_untradable: bool = True
    use_csi500_mask: bool = True


def validate_preprocess_config(config: PreprocessConfig) -> None:
    """Validate high-level preprocessing choices before reading data."""

    if config.feature_universe not in {"features500", "features"}:
        raise ValueError("feature_universe must be 'features500' or 'features'")
    if config.target_horizon_months != 1:
        raise ValueError("Stage 1 protocol assumes a one-month-ahead target")


def build_analysis_panel(config: PreprocessConfig, output_dir: Path) -> Path:
    """Placeholder for the Stage 2 point-in-time panel builder."""

    validate_preprocess_config(config)
    raise NotImplementedError(
        "Stage 2 should inspect data schemas, align characteristics and returns, "
        "apply tradability filters, and save a point-in-time monthly panel."
    )
