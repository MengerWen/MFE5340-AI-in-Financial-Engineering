"""Build a monthly point-in-time stock panel for Stage 2.

The implementation is schema-driven for the observed data layout:
wide DataFrames indexed by month-end dates with stock identifiers as columns,
and daily MultiIndex flag files with ``date`` and ``asset`` levels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import yaml

from src.data.loaders import get_project_paths, list_feature_files


@dataclass(frozen=True)
class CleaningConfig:
    """Configuration for monthly panel construction and cleaning."""

    feature_universe: str = "features500"
    target_horizon_months: int = 1
    use_excess_return: bool = True
    apply_blacklist_filter: bool = True
    apply_untradable_filter: bool = True
    apply_csi500_filter: bool = True
    flag_month_rule: str = "any_in_current_month"
    missing_feature_threshold: float = 0.95
    missing_row_threshold: float = 1.0
    impute_method: str = "cross_sectional_median"
    winsorize: bool = True
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    normalize: bool = True
    max_features: int | None = None
    output_panel: str = "outputs/panels/main_features500_panel.pkl"
    output_metadata: str = "outputs/metadata/main_features500_panel_metadata.json"


@dataclass(frozen=True)
class PanelBuildResult:
    """Paths and metadata returned by the panel builder."""

    panel_path: Path
    metadata_path: Path
    metadata: dict[str, Any]



# Backwards-compatible names for the Stage 1 validation script.
PreprocessConfig = CleaningConfig


def validate_preprocess_config(config: CleaningConfig) -> None:
    """Validate preprocessing settings without building the panel."""

    if config.feature_universe not in {"features500", "features"}:
        raise ValueError("feature_universe must be 'features500' or 'features'")
    if config.target_horizon_months != 1:
        raise ValueError("Stage 2 validates a one-month-ahead target")
    if config.flag_month_rule != "any_in_current_month":
        raise ValueError("Only any_in_current_month flag conversion is implemented")

def load_cleaning_config(path: Path) -> CleaningConfig:
    """Load a YAML cleaning config into a typed config object."""

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return CleaningConfig(**raw)


def _month_end_index(index: pd.Index) -> pd.DatetimeIndex:
    dates = pd.to_datetime(index)
    return dates.to_period("M").to_timestamp("M")


def _standardize_wide_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = _month_end_index(out.index)
    out.index.name = "date"
    out.columns = out.columns.astype(str)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index().sort_index(axis=1)


def _stack_on_index(df: pd.DataFrame, base_index: pd.MultiIndex, name: str) -> pd.Series:
    aligned = _standardize_wide_frame(df).reindex(
        index=base_index.get_level_values("date").unique(),
        columns=base_index.get_level_values("stock_id").unique(),
    )
    stacked = aligned.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["date", "stock_id"])
    return stacked.reindex(base_index).rename(name)


def _monthly_risk_free(data_dir: Path) -> pd.Series:
    rf = pd.read_csv(data_dir / "risk_free.csv")
    rf["date"] = pd.to_datetime(rf["date"])
    rf = rf.sort_values("date")
    monthly = rf.groupby(rf["date"].dt.to_period("M"))["RF"].apply(lambda x: float(np.prod(1.0 + x) - 1.0))
    monthly.index = monthly.index.to_timestamp("M")
    monthly.index.name = "date"
    return monthly.rename("rf_monthly")


def _daily_flags_to_monthly_index(path: Path, base_index: pd.MultiIndex, name: str) -> pd.Series:
    flags = pd.read_pickle(path)
    if not isinstance(flags.index, pd.MultiIndex):
        raise ValueError(f"{path.name} must have a MultiIndex")
    frame = flags.index.to_frame(index=False)
    frame.columns = ["date", "stock_id"]
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
    frame["stock_id"] = frame["stock_id"].astype(str)
    month_asset = pd.MultiIndex.from_frame(frame.drop_duplicates(["date", "stock_id"]))
    flag_series = pd.Series(True, index=month_asset, name=name)
    flag_series.index = flag_series.index.set_names(["date", "stock_id"])
    return flag_series.reindex(base_index, fill_value=False).astype(bool)


def _make_base_index(
    config: CleaningConfig,
    feature_dates: pd.DatetimeIndex,
    feature_columns: pd.Index,
    returns: pd.DataFrame,
    csi500_mask: pd.DataFrame | None,
) -> tuple[pd.MultiIndex, dict[str, Any]]:
    target = returns.shift(-config.target_horizon_months)
    target_dates = target.index[target.notna().any(axis=1)]
    common_dates = feature_dates.intersection(target_dates)

    metadata: dict[str, Any] = {
        "base_feature_dates": [str(feature_dates.min()), str(feature_dates.max())],
        "target_available_dates": [str(target_dates.min()), str(target_dates.max())],
    }

    if config.apply_csi500_filter and csi500_mask is not None:
        mask = _standardize_wide_frame(csi500_mask).astype(bool)
        common_dates = common_dates.intersection(mask.index)
        common_cols = feature_columns.intersection(mask.columns).intersection(returns.columns)
        mask = mask.reindex(index=common_dates, columns=common_cols, fill_value=False)
        stacked_mask = mask.stack(future_stack=True)
        stacked_mask.index = stacked_mask.index.set_names(["date", "stock_id"])
        base_index = stacked_mask[stacked_mask].index
        metadata.update(
            {
                "base_rule": "CSI 500 membership mask true at month t",
                "csi500_mask_dates": [str(mask.index.min()), str(mask.index.max())],
                "csi500_mask_columns": int(mask.shape[1]),
            }
        )
    else:
        common_cols = feature_columns.intersection(returns.columns)
        base_index = pd.MultiIndex.from_product([common_dates, common_cols], names=["date", "stock_id"])
        metadata["base_rule"] = "date-stock product over feature columns with available next-month target"

    return base_index.sort_values(), metadata


def _winsorize_by_month(panel: pd.DataFrame, feature_cols: list[str], lower_q: float, upper_q: float) -> None:
    for date, idx in panel.groupby("date").groups.items():
        block = panel.loc[idx, feature_cols]
        lower = block.quantile(lower_q)
        upper = block.quantile(upper_q)
        panel.loc[idx, feature_cols] = block.clip(lower=lower, upper=upper, axis=1)


def _impute_by_month(panel: pd.DataFrame, feature_cols: list[str]) -> None:
    global_median = panel[feature_cols].median(skipna=True)
    for date, idx in panel.groupby("date").groups.items():
        med = panel.loc[idx, feature_cols].median(skipna=True).fillna(global_median)
        panel.loc[idx, feature_cols] = panel.loc[idx, feature_cols].fillna(med)
    panel[feature_cols] = panel[feature_cols].fillna(global_median).fillna(0.0)


def _normalize_by_month(panel: pd.DataFrame, feature_cols: list[str]) -> None:
    for date, idx in panel.groupby("date").groups.items():
        block = panel.loc[idx, feature_cols]
        mean = block.mean(skipna=True)
        std = block.std(skipna=True).replace(0.0, 1.0).fillna(1.0)
        panel.loc[idx, feature_cols] = (block - mean) / std


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def build_monthly_panel(config: CleaningConfig, root: Path | None = None) -> PanelBuildResult:
    """Build and save the cleaned monthly panel.

    Feature values dated month ``t`` are paired with realized returns at
    ``t + target_horizon_months``. Cleaning statistics are computed within each
    month to avoid cross-time leakage.
    """

    paths = get_project_paths(root)
    data_dir = paths.data_dir

    if config.feature_universe not in {"features500", "features"}:
        raise ValueError("feature_universe must be 'features500' or 'features'")
    if config.target_horizon_months != 1:
        raise ValueError("Only one-month-ahead target construction is validated in Stage 2")
    if not 0.0 <= config.missing_feature_threshold <= 1.0:
        raise ValueError("missing_feature_threshold must be between 0 and 1")
    if config.flag_month_rule != "any_in_current_month":
        raise ValueError("Only any_in_current_month flag conversion is implemented")

    returns = _standardize_wide_frame(pd.read_pickle(data_dir / "monthly_returns.pkl"))
    mcap = _standardize_wide_frame(pd.read_pickle(data_dir / "mcap.pkl"))
    csi500_mask = pd.read_pickle(data_dir / "csi500_mask_monthly.pkl") if (data_dir / "csi500_mask_monthly.pkl").exists() else None

    feature_files = list_feature_files(config.feature_universe, root=paths.root)
    if config.max_features is not None:
        feature_files = feature_files[: config.max_features]
    if not feature_files:
        raise FileNotFoundError(f"No feature files found for {config.feature_universe}")

    first_feature = _standardize_wide_frame(pd.read_pickle(feature_files[0]))
    base_index, base_metadata = _make_base_index(
        config=config,
        feature_dates=first_feature.index,
        feature_columns=first_feature.columns,
        returns=returns,
        csi500_mask=csi500_mask,
    )

    base_panel = pd.DataFrame(index=base_index)
    base_panel["csi500_member_t"] = True if config.apply_csi500_filter else pd.NA

    target_return = _stack_on_index(returns.shift(-config.target_horizon_months), base_index, "target_return")
    base_panel["target_return"] = target_return.to_numpy()

    if config.use_excess_return:
        rf_monthly = _monthly_risk_free(data_dir)
        next_rf = rf_monthly.shift(-config.target_horizon_months).reindex(base_index.get_level_values("date"))
        base_panel["rf_next_month"] = next_rf.to_numpy()
        base_panel["target_excess_return"] = base_panel["target_return"] - base_panel["rf_next_month"]
    else:
        base_panel["rf_next_month"] = np.nan
        base_panel["target_excess_return"] = np.nan

    base_panel["mcap_t"] = _stack_on_index(mcap, base_index, "mcap_t").to_numpy()
    base_panel["blacklisted_t"] = _daily_flags_to_monthly_index(data_dir / "BLACKLIST.pkl", base_index, "blacklisted_t").to_numpy()
    base_panel["untradable_t"] = _daily_flags_to_monthly_index(data_dir / "UNTRADABLE.pkl", base_index, "untradable_t").to_numpy()

    blacklisted_true_base = int(base_panel["blacklisted_t"].sum())
    untradable_true_base = int(base_panel["untradable_t"].sum())

    feature_names: list[str] = []
    feature_series: list[pd.Series] = []
    for feature_path in feature_files:
        name = feature_path.stem
        values = _stack_on_index(pd.read_pickle(feature_path), base_index, name)
        feature_series.append(values)
        feature_names.append(name)

    feature_matrix = pd.concat(feature_series, axis=1)
    raw_feature_missing = feature_matrix.isna().mean().astype(float).to_dict()
    panel = pd.concat([base_panel, feature_matrix], axis=1).reset_index()

    row_counts: dict[str, int] = {"base_rows": int(len(panel))}
    if config.apply_blacklist_filter:
        panel = panel.loc[~panel["blacklisted_t"]].copy()
    row_counts["after_blacklist_filter"] = int(len(panel))

    if config.apply_untradable_filter:
        panel = panel.loc[~panel["untradable_t"]].copy()
    row_counts["after_untradable_filter"] = int(len(panel))

    panel = panel.loc[panel["target_return"].notna()].copy()
    row_counts["after_drop_missing_target"] = int(len(panel))

    if config.use_excess_return:
        panel = panel.loc[panel["target_excess_return"].notna()].copy()
    row_counts["after_drop_missing_excess_target"] = int(len(panel))

    panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_missing_after_filters = panel[feature_names].isna().mean().sort_values(ascending=False)
    kept_features = feature_missing_after_filters[feature_missing_after_filters <= config.missing_feature_threshold].index.tolist()
    dropped_features = [name for name in feature_names if name not in kept_features]
    panel.drop(columns=dropped_features, inplace=True)

    if config.missing_row_threshold < 1.0 and kept_features:
        row_missing = panel[kept_features].isna().mean(axis=1)
        panel = panel.loc[row_missing <= config.missing_row_threshold].copy()
    row_counts["after_feature_missing_filters"] = int(len(panel))

    if kept_features and config.winsorize:
        _winsorize_by_month(panel, kept_features, config.winsor_lower, config.winsor_upper)
    if kept_features and config.impute_method == "cross_sectional_median":
        _impute_by_month(panel, kept_features)
    elif kept_features and config.impute_method not in {"none", "cross_sectional_median"}:
        raise ValueError("impute_method must be 'cross_sectional_median' or 'none'")
    if kept_features and config.normalize:
        _normalize_by_month(panel, kept_features)

    panel.sort_values(["date", "stock_id"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    stocks_per_month = panel.groupby("date")["stock_id"].nunique()
    metadata: dict[str, Any] = {
        "config": asdict(config),
        "input_schema": {
            "returns_shape": list(returns.shape),
            "mcap_shape": list(mcap.shape),
            "first_feature_file": feature_files[0].name,
            "first_feature_shape": list(first_feature.shape),
            "feature_file_count_loaded": len(feature_files),
        },
        "alignment": {
            **base_metadata,
            "target_logic": "feature row at month t uses monthly_returns shifted -1, so target_return is realized return at t+1",
            "excess_return_logic": "target_excess_return = target_return - compounded risk_free.csv RF over month t+1",
            "flag_logic": "BLACKLIST and UNTRADABLE daily rows are converted to month-end flags for the current month t only",
        },
        "row_counts": row_counts,
        "panel_shape": list(panel.shape),
        "date_min": str(panel["date"].min()) if len(panel) else None,
        "date_max": str(panel["date"].max()) if len(panel) else None,
        "n_months": int(panel["date"].nunique()) if len(panel) else 0,
        "n_stocks": int(panel["stock_id"].nunique()) if len(panel) else 0,
        "stocks_per_month": {
            "min": int(stocks_per_month.min()) if len(stocks_per_month) else 0,
            "median": float(stocks_per_month.median()) if len(stocks_per_month) else 0.0,
            "max": int(stocks_per_month.max()) if len(stocks_per_month) else 0,
            "mean": float(stocks_per_month.mean()) if len(stocks_per_month) else 0.0,
        },
        "features": {
            "raw_count": len(feature_names),
            "kept_count": len(kept_features),
            "dropped_count": len(dropped_features),
            "kept_features": kept_features,
            "dropped_features": dropped_features,
            "raw_missing_top10": dict(sorted(raw_feature_missing.items(), key=lambda item: item[1], reverse=True)[:10]),
            "post_filter_missing_top10": feature_missing_after_filters.head(10).to_dict(),
        },
        "target_summary": {
            "target_return_missing_fraction_final": float(panel["target_return"].isna().mean()) if len(panel) else None,
            "target_excess_return_missing_fraction_final": float(panel["target_excess_return"].isna().mean()) if len(panel) else None,
            "target_return_mean": float(panel["target_return"].mean()) if len(panel) else None,
            "target_return_std": float(panel["target_return"].std()) if len(panel) else None,
            "target_excess_return_mean": float(panel["target_excess_return"].mean()) if len(panel) else None,
            "target_excess_return_std": float(panel["target_excess_return"].std()) if len(panel) else None,
        },
        "filter_summary": {
            "blacklisted_true_base": blacklisted_true_base,
            "untradable_true_base": untradable_true_base,
        },
        "caveats": [
            "Feature disclosure lags are not observable from the stored feature matrices; Stage 2 uses month-end feature timestamps as available-at-t labels and documents this limitation.",
            "BLACKLIST and UNTRADABLE are daily zero-column MultiIndex DataFrames; current-month aggregation is a conservative implemented rule but exact trading-time semantics should be verified if possible.",
            "features500 appears CSI 500-filtered by missingness outside the mask, but files retain broad stock-code columns with NaN values outside membership.",
        ],
    }

    panel_path = (paths.root / config.output_panel).resolve()
    metadata_path = (paths.root / config.output_metadata).resolve()
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(panel_path)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False, default=_json_default)

    return PanelBuildResult(panel_path=panel_path, metadata_path=metadata_path, metadata=metadata)


